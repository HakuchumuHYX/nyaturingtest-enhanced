import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MemoryCommandPublicGroupOnlyTests(unittest.TestCase):
    def test_query_memory_is_public_but_group_only(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        self.assertIn("async def is_group_message", source)
        self.assertIn('query_memory = on_command("查询记忆", aliases={"memory"}, rule=is_group_message', source)
        query_definition = source[
            source.index('query_memory = on_command("查询记忆"'):
            source.index("_LONG_TERM_VAD_CACHE_TTL_SECONDS")
        ]
        self.assertNotIn("permission=SUPERUSER", query_definition)

    def test_query_memory_sends_progress_before_expensive_work(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")
        query_handler = source[source.index("@query_memory.handle()"):]

        progress_index = query_handler.index('await query_memory.send("正在回溯记忆深处...")')
        lock_index = query_handler.index("async with state.session_lock")
        vector_index = query_handler.index("search_for_user_profile")
        vad_index = query_handler.index("long_term_vad = await _summarize_long_term_vad")

        self.assertLess(progress_index, lock_index)
        self.assertLess(progress_index, vector_index)
        self.assertLess(progress_index, vad_index)

    def test_query_memory_final_generation_uses_chat_thinking_settings(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")
        generation_source = source[source.index("# 6. 调用 LLM"):]

        self.assertIn("get_chat_thinking_settings", source)
        self.assertIn("query_memory_chat_thinking = get_chat_thinking_settings()", generation_source)
        self.assertIn("query_memory_chat_provider = get_effective_chat_provider()", generation_source)
        self.assertIn('query_memory_chat_provider == "deepseek_official"', generation_source)
        self.assertIn("query_memory_chat_extra_body = None", generation_source)
        self.assertIn('"type": "enabled" if query_memory_chat_thinking.get("enabled") else "disabled"', generation_source)
        self.assertIn('reasoning_effort=query_memory_chat_thinking.get("reasoning_effort", "high") if query_memory_use_deepseek_thinking else None', generation_source)
        self.assertIn('temperature=None if query_memory_use_deepseek_thinking else 0.8', generation_source)
        self.assertNotIn("attempt * 0.2", generation_source)
        self.assertNotIn('extra_body={"thinking": {"type": "disabled"}}', generation_source)

    def test_query_memory_separates_unscoped_memories_from_target_memories(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")
        query_handler = source[source.index("@query_memory.handle()"):]

        self.assertIn("target_vector_records", query_handler)
        self.assertIn("unscoped_vector_records", query_handler)
        self.assertIn("目标用户长期记忆碎片", query_handler)
        self.assertIn("全局/未标记记忆碎片", query_handler)
        self.assertIn("低优先级", query_handler)


if __name__ == "__main__":
    unittest.main()
