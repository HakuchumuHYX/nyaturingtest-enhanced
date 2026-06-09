import unittest
from datetime import datetime
from pathlib import Path

from test_vector_batch import _load_vector_module


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class VectorDecayTests(unittest.TestCase):
    def test_retrieve_with_decay_uses_real_date_delta(self):
        source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("datetime.strptime", source)
        self.assertNotIn("month_diff * 30", source)

    def test_retrieve_with_decay_uses_retrieval_score_without_rerank(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "low", "metadata": {"date": today, "retrieval_score": 0.2}},
                {"content": "high", "metadata": {"date": today, "retrieval_score": 0.9}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=2, use_rerank=False, decay_rate=0)

        self.assertEqual(["high", "low"], [item["content"] for item in result])

    def test_preset_records_do_not_receive_missing_date_decay(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "preset", "metadata": {"source": "preset", "retrieval_score": 0.9}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=1, use_rerank=False, decay_rate=1)

        self.assertEqual("preset", result[0]["content"])
        self.assertEqual(0, result[0]["metadata"]["days_ago"])
        self.assertEqual(1.0, result[0]["metadata"]["decay_factor"])
        self.assertEqual(0.85, result[0]["metadata"]["source_type_weight"])

    def test_memory_event_uses_memory_weight_not_preset_event_weight(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "memory", "metadata": {"source": "memory", "type": "event", "date": today, "retrieval_score": 0.75}},
                {"content": "preset", "metadata": {"source": "preset", "type": "rule", "retrieval_score": 0.75}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=2, use_rerank=False, decay_rate=0)

        self.assertEqual(["memory", "preset"], [item["content"] for item in result])
        self.assertEqual(1.0, result[0]["metadata"]["source_type_weight"])
        self.assertEqual(0.85, result[1]["metadata"]["source_type_weight"])

    def test_candidate_k_none_preserves_old_double_k_behavior(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        calls = []

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            calls.append(k)
            return []

        memory.retrieve = fake_retrieve

        memory.retrieve_with_decay(["query"], k=7, use_rerank=False)

        self.assertEqual([14], calls)

    def test_candidate_k_overrides_internal_double_k(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        calls = []

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            calls.append(k)
            return []

        memory.retrieve = fake_retrieve

        memory.retrieve_with_decay(["query"], k=7, candidate_k=9, use_rerank=False)

        self.assertEqual([9], calls)

    def test_active_user_scope_downweights_other_users_and_weights_active_memories(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "other user", "metadata": {"source": "memory", "type": "event", "user_id": "2", "date": today, "retrieval_score": 0.99}},
                {"content": "global memory", "metadata": {"source": "memory", "type": "event", "user_id": "", "date": today, "retrieval_score": 0.75}},
                {"content": "active user", "metadata": {"source": "memory", "type": "event", "user_id": "1", "date": today, "retrieval_score": 0.70}},
                {"content": "preset", "metadata": {"source": "preset", "type": "rule", "retrieval_score": 0.60}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=4, use_rerank=False, decay_rate=0, active_user_ids={"1"})

        self.assertEqual(["active user", "global memory", "other user", "preset"], [item["content"] for item in result])
        self.assertEqual(1.10, result[0]["metadata"]["scope_weight"])
        self.assertEqual(1.0, result[1]["metadata"]["scope_weight"])
        self.assertEqual("legacy_subject", result[2]["metadata"]["scope"])
        self.assertEqual(0.75, result[2]["metadata"]["scope_weight"])
        self.assertEqual(1.0, result[3]["metadata"]["scope_weight"])

    def test_old_callers_without_active_user_ids_keep_user_specific_memories(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [
                {"content": "user specific", "metadata": {"source": "memory", "type": "event", "user_id": "2", "date": today, "retrieval_score": 0.90}},
            ]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(["query"], k=1, use_rerank=False, decay_rate=0)

        self.assertEqual(["user specific"], [item["content"] for item in result])

    def test_where_adapter_uses_or_without_in_operator(self):
        module = _load_vector_module()

        where = module.where_any("source", ["preset", "memory"])

        self.assertEqual({"$or": [{"source": {"$eq": "preset"}}, {"source": {"$eq": "memory"}}]}, where)
        self.assertNotIn("$in", str(where))


if __name__ == "__main__":
    unittest.main()
