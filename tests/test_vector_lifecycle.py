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
        requested_ids = kwargs.get("ids")
        row_ids = list(requested_ids) if requested_ids is not None else list(self.rows.keys())
        row_ids = [item_id for item_id in row_ids if item_id in self.rows]
        return {
            "ids": row_ids,
            "metadatas": [dict(self.rows[item_id]) for item_id in row_ids],
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

    def test_retrieve_with_decay_filters_non_active_status_but_keeps_legacy_and_preset(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "superseded memory", "metadata": {"source": "memory", "type": "event", "status": "superseded", "retrieval_score": 0.99}},
                {"content": "archived memory", "metadata": {"source": "memory", "type": "event", "status": "archived", "retrieval_score": 0.98}},
                {"content": "legacy active memory", "metadata": {"source": "memory", "type": "event", "retrieval_score": 0.90}},
                {"content": "preset memory", "metadata": {"source": "preset", "type": "rule", "retrieval_score": 0.50}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=4, use_rerank=False, decay_rate=0)

        self.assertEqual(["legacy active memory", "preset memory"], [item["content"] for item in result])
        self.assertEqual("active", result[0]["metadata"]["status"])
        self.assertEqual("active", result[1]["metadata"]["status"])
        self.assertEqual(2, memory.last_retrieval_stats["returned_count"])

    def test_retrieve_with_decay_applies_confidence_and_importance_with_legacy_neutral_defaults(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "legacy neutral", "metadata": {"source": "memory", "type": "event", "date": today, "retrieval_score": 0.80}},
                {"content": "important update", "metadata": {"source": "memory", "type": "event", "date": today, "retrieval_score": 0.70, "confidence": 1.0, "importance": 1.0}},
                {"content": "zero confidence", "metadata": {"source": "memory", "type": "event", "date": today, "retrieval_score": 1.0, "confidence": 0.0, "importance": 1.0}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=3, use_rerank=False, decay_rate=0)

        self.assertEqual(["important update", "legacy neutral", "zero confidence"], [item["content"] for item in result])
        self.assertEqual(1.0, result[1]["metadata"]["confidence_weight"])
        self.assertEqual(1.0, result[1]["metadata"]["importance_weight"])
        self.assertEqual(0.0, result[2]["metadata"]["adjusted_score"])

    def test_get_and_update_metadata_by_id_for_supersede(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeLifecycleCollection([
            ("mem-1", {"source": "memory", "type": "preference", "status": "active"}),
        ])

        self.assertEqual(
            {"source": "memory", "type": "preference", "status": "active"},
            memory.get_metadata_by_id("mem-1"),
        )
        self.assertIsNone(memory.get_metadata_by_id("missing"))

        memory.update_metadata_by_id("mem-1", {"source": "memory", "type": "preference", "status": "superseded"})

        self.assertEqual("superseded", memory.collection.rows["mem-1"]["status"])
        self.assertEqual([(["mem-1"], [{"source": "memory", "type": "preference", "status": "superseded"}])], memory.collection.updated)


if __name__ == "__main__":
    unittest.main()
