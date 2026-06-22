import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RetryBoundariesStaticTests(unittest.TestCase):
    def test_openai_sdk_retries_are_disabled_at_construction_sites(self):
        construction_sites = [
            PLUGIN_DIR / "core" / "state_manager.py",
            PLUGIN_DIR / "core" / "session.py",
            PLUGIN_DIR / "llm" / "vlm.py",
            PLUGIN_DIR / "memory" / "vector.py",
        ]

        for path in construction_sites:
            source = path.read_text(encoding="utf-8")
            self.assertIn(
                "max_retries=0",
                source,
                f"{path.relative_to(PLUGIN_DIR)} should disable hidden OpenAI SDK retries",
            )

    def test_chat_and_feedback_do_not_wrap_llm_calls_in_stage_retries(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        feedback_start = source.index("async def _run_feedback_llm")
        feedback_end = source.index("    def _apply_sediment", feedback_start)
        feedback_source = source[feedback_start:feedback_end]
        chat_start = source.index("async def chat_stage")
        chat_end = source.index("    # 提高插话阈值", chat_start)
        chat_source = source[chat_start:chat_end]

        self.assertNotIn("for attempt in range", feedback_source)
        self.assertNotIn("for attempt in range", chat_source)
        self.assertEqual(1, feedback_source.count("await llm_func("))
        self.assertEqual(1, chat_source.count("await llm_func("))

    def test_query_memory_does_not_add_a_command_level_llm_retry_loop(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        start = source.index("# 6. 调用 LLM")
        end = source.index('await query_memory.finish("大脑处理过载', start)
        command_llm_source = source[start:end]

        self.assertNotIn("max_retries", command_llm_source)
        self.assertNotIn("for attempt in range", command_llm_source)
        self.assertEqual(1, command_llm_source.count("await llm_response("))


if __name__ == "__main__":
    unittest.main()
