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


if __name__ == "__main__":
    unittest.main()
