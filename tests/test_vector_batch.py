import unittest
import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _install_vector_stubs():
    nonebot = sys.modules.get("nonebot") or types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    sys.modules["nonebot"] = nonebot

    httpx = sys.modules.get("httpx") or types.ModuleType("httpx")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

    httpx.Client = Client
    sys.modules["httpx"] = httpx

    openai = sys.modules.get("openai") or types.ModuleType("openai")

    class OpenAI:
        pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai

    config = types.ModuleType("vector_test.config")
    config.plugin_config = {"embedding": {}, "rerank": {}}
    config.get_memory_endpoint_settings = lambda: {
        "model": "BAAI/bge-m3",
        "base_url": "https://api.siliconflow.cn/v1",
        "timeout": 30.0,
        "rerank_base_url": "https://api.siliconflow.cn/v1/rerank",
        "rerank_timeout": 10.0,
    }
    sys.modules.setdefault("vector_test.config", config)

    class DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    backup_lock = sys.modules.get("vector_test.database.backup_lock") or types.ModuleType("vector_test.database.backup_lock")
    backup_lock.BACKUP_IO_LOCK = DummyLock()
    sys.modules["vector_test.database.backup_lock"] = backup_lock

    database = types.ModuleType("vector_test.database")
    database.__path__ = []
    sys.modules.setdefault("vector_test.database", database)

    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = object
    sys.modules.setdefault("chromadb", chromadb)
    chromadb_api = types.ModuleType("chromadb.api")
    chromadb_types = types.ModuleType("chromadb.api.types")
    chromadb_types.Documents = list
    chromadb_types.Embeddings = list

    class EmbeddingFunction:
        pass

    chromadb_types.EmbeddingFunction = EmbeddingFunction
    sys.modules.setdefault("chromadb.api", chromadb_api)
    sys.modules.setdefault("chromadb.api.types", chromadb_types)


def _load_vector_module():
    _install_vector_stubs()
    package = types.ModuleType("vector_test.memory")
    package.__path__ = []
    sys.modules.setdefault("vector_test", types.ModuleType("vector_test"))
    sys.modules.setdefault("vector_test.memory", package)
    spec = importlib.util.spec_from_file_location(
        "vector_test.memory.vector",
        PLUGIN_DIR / "memory" / "vector.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vector_test.memory.vector"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCollection:
    def __init__(self):
        self.query_calls = []
        self.add_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "distances": [
                [0.05],
                [0.4],
            ]
        }

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeRetrievalCollection:
    def __init__(self):
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "documents": [["memory alpha", "memory beta"]],
            "metadatas": [[{"source": "memory"}, {"source": "preset"}]],
            "distances": [[0.1, 0.7]],
        }


class FakeClient:
    def __init__(self):
        self.deleted = []
        self.created = []

    def delete_collection(self, name):
        self.deleted.append(name)

    def get_or_create_collection(self, **kwargs):
        self.created.append(kwargs)
        return "new-collection"


class VectorBatchTests(unittest.TestCase):
    def test_add_memories_with_dedup_batches_query_and_add(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeCollection()

        result = memory.add_memories_with_dedup([
            ("重复记忆", {"source": "memory", "user_id": "1"}),
            ("新的记忆", {"source": "memory", "user_id": "2"}),
            ("   ", {"source": "memory", "user_id": "3"}),
        ], threshold=0.9)

        self.assertEqual({"added": 1, "skipped_empty": 1, "skipped_dedup": 1}, result)
        self.assertEqual(1, len(memory.collection.query_calls))
        self.assertEqual(["重复记忆", "新的记忆"], memory.collection.query_calls[0]["query_texts"])
        self.assertEqual(
            {
                "$or": [
                    {"source": {"$eq": "memory"}},
                    {"source": {"$eq": "preset"}},
                ]
            },
            memory.collection.query_calls[0]["where"],
        )
        self.assertEqual(1, len(memory.collection.add_calls))
        self.assertEqual(["新的记忆"], memory.collection.add_calls[0]["documents"])

    def test_retrieve_attaches_distance_score_without_rerank(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeRetrievalCollection()
        memory.reranker = None

        result = memory.retrieve(["alpha"], k=2, use_rerank=False)

        self.assertEqual(["memory alpha", "memory beta"], [item["content"] for item in result])
        self.assertEqual(0.9, result[0]["metadata"]["retrieval_score"])
        self.assertEqual(0.3, round(result[1]["metadata"]["retrieval_score"], 2))

    def test_retrieve_with_decay_records_aggregate_stats(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeRetrievalCollection()
        memory.reranker = None

        result = memory.retrieve_with_decay(["alpha"], k=1, use_rerank=False, decay_rate=0)
        stats = memory.last_retrieval_stats

        self.assertEqual(["memory alpha"], [item["content"] for item in result])
        self.assertEqual(2, stats["candidate_count"])
        self.assertEqual(1, stats["returned_count"])
        self.assertEqual("rerank_disabled", stats["fallback_reason"])
        self.assertEqual(0.3, round(stats["adjusted_score_min"], 2))
        self.assertEqual(0.9, stats["adjusted_score_max"])

    def test_clear_recreates_collection_with_cosine_metadata(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.client = FakeClient()
        memory.emb_fn = object()

        memory.clear()

        self.assertEqual([module.MEMORY_COLLECTION_NAME], memory.client.deleted)
        self.assertEqual(module.MEMORY_COLLECTION_NAME, memory.client.created[0]["name"])
        self.assertEqual(module.MEMORY_COLLECTION_METADATA, memory.client.created[0]["metadata"])
        self.assertEqual("new-collection", memory.collection)


if __name__ == "__main__":
    unittest.main()
