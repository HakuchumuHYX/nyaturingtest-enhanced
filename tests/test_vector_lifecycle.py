import tempfile
import unittest
from datetime import datetime, timedelta

from test_vector_batch import _load_vector_module


class FakeIdsCollection:
    metadata = {"hnsw:space": "cosine"}

    def __init__(self):
        self.query_calls = []

    def get(self, **kwargs):
        return {"ids": []}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "ids": [["id-alpha"]],
            "documents": [["memory alpha"]],
            "metadatas": [[{"source": "memory"}]],
            "distances": [[0.1]],
        }


class FakeLifecycleCollection:
    metadata = {"hnsw:space": "cosine"}

    def __init__(self, rows):
        self.rows = {item_id: dict(metadata) for item_id, metadata in rows}
        self.updated = []
        self.deleted = []

    def get(self, **kwargs):
        return {
            "ids": list(self.rows.keys()),
            "metadatas": [dict(metadata) for metadata in self.rows.values()],
        }

    def update(self, *, ids, metadatas):
        self.updated.append((list(ids), [dict(metadata) for metadata in metadatas]))
        for item_id, metadata in zip(ids, metadatas):
            self.rows[item_id] = dict(metadata)

    def delete(self, *, ids):
        self.deleted.append(list(ids))
        for item_id in ids:
            self.rows.pop(item_id, None)


class VectorLifecycleTests(unittest.TestCase):
    def test_ids_probe_treats_top_level_ids_key_as_supported(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeIdsCollection()
        memory.persist_directory = "/tmp/vector-test"

        self.assertTrue(memory._probe_ids_support())

    def test_retrieve_attaches_memory_ref_from_query_ids(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeIdsCollection()
        memory.reranker = None

        result = memory.retrieve(["alpha"], k=1, use_rerank=False)

        self.assertEqual("id-alpha", result[0]["metadata"]["memory_ref"])

    def test_backfill_active_status_updates_missing_status_until_verified(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        with tempfile.TemporaryDirectory() as tmpdir:
            memory.persist_directory = tmpdir
            memory.collection = FakeLifecycleCollection([
                ("old", {"source": "memory", "type": "event"}),
                ("new", {"source": "memory", "type": "event", "status": "active"}),
            ])

            dry_run = memory.backfill_active_status(dry_run=True)
            result = memory.backfill_active_status(dry_run=False, batch_size=1)

        self.assertEqual(1, dry_run["missing_status_count"])
        self.assertTrue(result["complete"])
        self.assertEqual("active", memory.collection.rows["old"]["status"])
        self.assertEqual([["old"]], [ids for ids, _ in memory.collection.updated])

    def test_cleanup_deletes_expired_memory_in_batches_and_keeps_preset(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        old_date = int((datetime.now() - timedelta(days=120)).strftime("%Y%m%d"))
        memory.collection = FakeLifecycleCollection([
            ("event-old", {"source": "memory", "type": "event", "date": old_date}),
            ("preset-old", {"source": "preset", "type": "rule", "date": old_date}),
            ("superseded-old", {"source": "memory", "type": "profile", "status": "superseded", "date": old_date}),
        ])

        memory.cleanup(days_retention=90)

        deleted = [item_id for batch in memory.collection.deleted for item_id in batch]
        self.assertIn("event-old", deleted)
        self.assertIn("superseded-old", deleted)
        self.assertNotIn("preset-old", deleted)


if __name__ == "__main__":
    unittest.main()
