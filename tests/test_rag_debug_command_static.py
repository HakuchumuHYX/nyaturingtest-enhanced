import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagDebugCommandStaticTests(unittest.TestCase):
    def test_rag_debug_command_is_group_only_superuser_command(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        self.assertIn("from nonebot.permission import SUPERUSER", source)
        self.assertIn('rag_debug = on_command("rag_debug"', source)
        self.assertIn('aliases={"记忆诊断"}', source)
        self.assertIn("rule=is_group_message", source)
        self.assertIn("permission=SUPERUSER", source)

        query_definition = source[
            source.index('query_memory = on_command("查询记忆"'):
            source.index("_LONG_TERM_VAD_CACHE_TTL_SECONDS")
        ]
        self.assertNotIn("permission=SUPERUSER", query_definition)

    def test_rag_debug_formatter_uses_preview_not_full_content(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        self.assertIn("def _format_rag_debug_record", source)
        formatter = source[
            source.index("def _format_rag_debug_record"):
            source.index("@rag_debug.handle()")
        ]

        self.assertIn('record.get("preview"', formatter)
        self.assertIn('preview.replace("\\n", " ")[:80]', formatter)
        self.assertNotIn('record.get("content"', formatter)
        self.assertNotIn('record["content"]', formatter)
        self.assertNotIn("json.dumps(record", formatter)

    def test_rag_debug_formatter_includes_subject_and_speaker_short_fields(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")
        formatter_block = source[
            source.index("def _format_rag_debug_record"):
            source.index("async def _summarize_long_term_vad")
        ]

        self.assertIn("subject=", formatter_block)
        self.assertIn("speaker=", formatter_block)
        self.assertNotIn('record.get("content")', formatter_block)

    def test_rag_debug_handler_reports_counts_scores_and_fallback(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")
        handler = source[source.index("@rag_debug.handle()"):]

        for snippet in [
            "search_for_debug",
            'where_any("source", ["preset", "memory"])',
            '"candidate_count"',
            '"returned_count"',
            '"fallback_reason"',
            '"adjusted_score"',
            '"retrieval_score"',
            '"rerank_score"',
            "top_records",
        ]:
            self.assertIn(snippet, handler)

    def test_command_help_lists_rag_debug(self):
        source = (PLUGIN_DIR / "handlers" / "command_meta.py").read_text(encoding="utf-8")

        self.assertIn('CommandMeta("rag_debug <query>"', source)
        self.assertIn("诊断 RAG 记忆检索", source)


if __name__ == "__main__":
    unittest.main()
