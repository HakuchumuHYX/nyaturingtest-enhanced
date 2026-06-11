import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagSearchServiceStaticTests(unittest.TestCase):
    def test_services_define_rag_search_service_with_three_entrypoints_and_normalized_records(self):
        source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")

        for snippet in [
            "class RagSearchService",
            "def normalize_record",
            "async def search_for_chat",
            "async def search_for_user_profile",
            "async def search_for_debug",
            '"score"',
            '"memory_ref"',
            '"preview"',
            'meta.get("adjusted_score")',
            'meta.get("rerank_score")',
            'meta.get("retrieval_score")',
        ]:
            self.assertIn(snippet, source)

    def test_chat_rag_uses_service_without_changing_prompt_formatting(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("RagSearchService(self.long_term_memory).search_for_chat", session_source)
        self.assertIn('prefix = f"【设定/{subtype}】"', session_source)
        self.assertIn('prefix = f"【记忆/d:{date_str}】"', session_source)
        self.assertIn('max_len = runtime_settings["rag_memory_char_budget"]', session_source)

    def test_query_memory_uses_service_and_keeps_long_term_vad_semantics(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        self.assertIn("RagSearchService(long_term_memory).search_for_user_profile", source)
        self.assertIn("dynamic_k = calculate_dynamic_k", source)
        self.assertIn("_summarize_long_term_vad", source)
        self.assertNotIn("Profile.emotion", source)


if __name__ == "__main__":
    unittest.main()
