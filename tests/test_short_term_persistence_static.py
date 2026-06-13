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

    def test_sync_messages_writes_every_new_message(self):
        source = (PLUGIN_DIR / "database" / "message_repository.py").read_text(encoding="utf-8")
        # 必须遍历 recent_msgs 全量构建 bulk，而非截断
        self.assertIn("for msg in recent_msgs:", source)
        self.assertIn("bulk_create(bulk_msgs)", source)
        self.assertNotIn("[-10:]", source)

    def test_load_uses_runtime_buffer_limit(self):
        source = (PLUGIN_DIR / "database" / "session_repository.py").read_text(encoding="utf-8")
        self.assertIn("short_term_buffer_size", source)
        self.assertNotIn(".limit(50)", source)


if __name__ == "__main__":
    unittest.main()
