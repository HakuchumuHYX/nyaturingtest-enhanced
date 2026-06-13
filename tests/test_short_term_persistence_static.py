import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ShortTermPersistenceStaticTests(unittest.TestCase):
    def test_config_defines_short_term_buffer_size(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        self.assertIn('"short_term_buffer_size": 200', config_source)
        self.assertIn(
            '"short_term_buffer_size": number("short_term_buffer_size", 200, int, minimum=1)',
            config_source,
        )

    def test_example_config_defines_short_term_buffer_size(self):
        example = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        self.assertIn('"short_term_buffer_size"', example)

    def test_save_session_persists_full_snapshot(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        # 保存时必须用 snapshot()（全量）而不是 access()（最近 N 条）
        self.assertIn("self.global_memory.snapshot()", source)
        save_start = source.index("async def save_session")
        save_end = source.index("async def load_session")
        save_block = source[save_start:save_end]
        self.assertNotIn("self.global_memory.access().messages", save_block)

    def test_memory_constructed_with_runtime_buffer(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("buffer_size=", source)
        self.assertIn("short_term_buffer_size", source)


if __name__ == "__main__":
    unittest.main()
