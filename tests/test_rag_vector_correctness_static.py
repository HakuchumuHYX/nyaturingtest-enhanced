import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagVectorCorrectnessStaticTests(unittest.TestCase):
    def test_query_dedupe_preserves_order_in_session_and_vector(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        rag_query_source = (PLUGIN_DIR / "core" / "rag_query.py").read_text(encoding="utf-8")
        vector_source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("build_chat_rag_queries(", session_source)
        self.assertIn("def _dedupe_preserve_order", rag_query_source)
        self.assertIn("def _dedupe_preserve_order", vector_source)
        self.assertIn("return _dedupe_preserve_order(effective_queries)", rag_query_source)
        self.assertIn("_dedupe_preserve_order([q for q in queries", vector_source)
        self.assertNotIn("list(set([q for q in queries", session_source)
        self.assertNotIn("list(set(", rag_query_source)
        self.assertNotIn("list(set([q for q in queries", vector_source)

    def test_vector_has_separate_preset_and_memory_weight_tables(self):
        source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("PRESET_TYPE_WEIGHT", source)
        self.assertIn('"legacy_rule": 0.85', source)
        self.assertIn("MEMORY_TYPE_WEIGHT", source)
        self.assertIn('"event": 1.0', source)
        self.assertIn('PRESET_TYPE_WEIGHT.get(subtype, PRESET_TYPE_WEIGHT["legacy_rule"])', source)
        self.assertIn("MEMORY_TYPE_WEIGHT.get(memory_type, 1.0)", source)

    def test_metric_check_is_deduplicated_per_persist_directory(self):
        source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("_metric_check_done: set[str]", source)
        self.assertIn("def _check_collection_metric_once", source)
        self.assertIn("self.persist_directory in _metric_check_done", source)


if __name__ == "__main__":
    unittest.main()
