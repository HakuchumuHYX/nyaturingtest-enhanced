import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagObservabilityStaticTests(unittest.TestCase):
    def test_rag_search_is_emitted_only_from_session_search_stage(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        vector_source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertEqual(1, session_source.count('log_event("rag_search"'))
        search_start = session_source.index("async def search_stage")
        feedback_start = session_source.index("async def feedback_stage", search_start)
        search_stage_source = session_source[search_start:feedback_start]

        self.assertIn('log_event("rag_search"', search_stage_source)
        self.assertNotIn('log_event("rag_search"', vector_source)

    def test_rag_prompt_budget_is_emitted_from_chat_stage_after_prompt_build(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertEqual(1, session_source.count('log_event("rag_prompt_budget"'))
        search_start = session_source.index("async def search_stage")
        feedback_start = session_source.index("async def feedback_stage", search_start)
        search_stage_source = session_source[search_start:feedback_start]
        chat_start = session_source.index("async def chat_stage")
        chat_source = session_source[chat_start:]

        self.assertNotIn("chat_prompt_total_chars", search_stage_source)
        self.assertIn("prompt = get_chat_prompt", chat_source)
        self.assertLess(chat_source.index("prompt = get_chat_prompt"), chat_source.index('log_event("rag_prompt_budget"'))
        self.assertIn("chat_prompt_total_chars=len(prompt)", chat_source)

    def test_rag_search_fields_and_preview_policy_are_fixed(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        for field in [
            '"skip_reason": "none"',
            '"fallback_reason": "none"',
            '"candidate_count": 0',
            '"returned_count": 0',
            '"injected_count": 0',
            '"injected_chars": 0',
            '"adjusted_score_p50": None',
            '"adjusted_score_p90": None',
        ]:
            self.assertIn(field, session_source)

        self.assertIn("[q[:40] for q in queries[:3]]", session_source)
        self.assertIn('"low_willingness"', session_source)
        self.assertIn('"no_queries"', session_source)
        self.assertIn('"result_debug"', session_source)
        self.assertIn("runtime_settings[\"rag_debug_log\"]", session_source)

    def test_rag_debug_log_is_runtime_configured(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        example_source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")

        self.assertIn('"rag_debug_log": False', config_source)
        self.assertIn('"rag_debug_log": flag("rag_debug_log", False)', config_source)
        self.assertIn('"rag_debug_log": false', example_source)

    def test_vector_memory_exposes_stats_without_rag_logging(self):
        vector_source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("def last_retrieval_stats", vector_source)
        self.assertIn("fallback_reason", vector_source)
        self.assertIn("_score_distribution", vector_source)
        self.assertNotIn('log_event("rag_search"', vector_source)


if __name__ == "__main__":
    unittest.main()
