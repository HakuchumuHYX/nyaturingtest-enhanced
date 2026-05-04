import unittest
import importlib.util
import sys
import types
from unittest.mock import patch
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

    config = types.ModuleType("vector_http_test.config")
    config.plugin_config = {
        "embedding": {
            "model": "custom-embedding",
            "base_url": "https://embedding.example/v1",
            "timeout": 12,
        },
        "rerank": {
            "base_url": "https://rerank.example/v1/rerank",
            "timeout": 8,
        },
    }
    config.get_memory_endpoint_settings = lambda: {
        "model": config.plugin_config["embedding"]["model"],
        "base_url": config.plugin_config["embedding"]["base_url"],
        "timeout": float(config.plugin_config["embedding"]["timeout"]),
        "rerank_base_url": config.plugin_config["rerank"]["base_url"],
        "rerank_timeout": float(config.plugin_config["rerank"]["timeout"]),
    }
    sys.modules.setdefault("vector_http_test.config", config)

    class DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    backup_lock = sys.modules.get("vector_http_test.database.backup_lock") or types.ModuleType("vector_http_test.database.backup_lock")
    backup_lock.BACKUP_IO_LOCK = DummyLock()
    sys.modules["vector_http_test.database.backup_lock"] = backup_lock

    database = types.ModuleType("vector_http_test.database")
    database.__path__ = []
    sys.modules.setdefault("vector_http_test.database", database)

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
    package = types.ModuleType("vector_http_test.memory")
    package.__path__ = []
    sys.modules.setdefault("vector_http_test", types.ModuleType("vector_http_test"))
    sys.modules.setdefault("vector_http_test.memory", package)
    spec = importlib.util.spec_from_file_location(
        "vector_http_test.memory.vector",
        PLUGIN_DIR / "memory" / "vector.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vector_http_test.memory.vector"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [{"embedding": [0.1, 0.2]}],
            "results": [{"index": 0, "relevance_score": 0.9}],
        }


class CountingClient:
    created = 0

    def __init__(self, *args, **kwargs):
        CountingClient.created += 1

    def post(self, *args, **kwargs):
        return FakeResponse()


class FakeEmbedding:
    def __init__(self):
        self.embedding = [0.1, 0.2]


class FakeEmbeddingsResource:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(data=[FakeEmbedding()])


class CountingOpenAI:
    created = 0
    instances = []

    def __init__(self, *args, **kwargs):
        CountingOpenAI.created += 1
        self.kwargs = kwargs
        self.embeddings = FakeEmbeddingsResource()
        CountingOpenAI.instances.append(self)

    def close(self):
        return None


class VectorHttpClientTests(unittest.TestCase):
    def test_embedding_function_uses_openai_sdk_client(self):
        vector = _load_vector_module()
        CountingClient.created = 0
        CountingOpenAI.created = 0
        CountingOpenAI.instances = []
        with patch.object(vector, "OpenAI", CountingOpenAI), \
             patch.object(vector.httpx, "Client", CountingClient):
            emb = vector.SiliconFlowEmbeddingFunction("key", "session")
            emb(["hello"])
            emb(["world"])

        self.assertEqual(1, CountingOpenAI.created)
        self.assertEqual(0, CountingClient.created)
        self.assertEqual("https://embedding.example/v1", CountingOpenAI.instances[0].kwargs["base_url"])
        self.assertEqual(2, len(CountingOpenAI.instances[0].embeddings.calls))
        self.assertEqual("custom-embedding", CountingOpenAI.instances[0].embeddings.calls[0]["model"])

    def test_reranker_reuses_http_client(self):
        vector = _load_vector_module()
        CountingClient.created = 0
        with patch.object(vector.httpx, "Client", CountingClient):
            reranker = vector.SiliconFlowReranker("key", "model")
            reranker.rerank("q", ["a"])
            reranker.rerank("q", ["b"])

        self.assertEqual(1, CountingClient.created)
        self.assertEqual("https://rerank.example/v1/rerank", reranker.api_url)


if __name__ == "__main__":
    unittest.main()
