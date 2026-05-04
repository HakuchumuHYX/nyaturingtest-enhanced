import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MemoryQueryLockScopeTests(unittest.TestCase):
    def test_vector_retrieval_runs_outside_session_lock(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        lock_start = source.index("async with state.session_lock:")
        lock_end = source.index("\n    # --- 用户画像与交互统计逻辑 ---", lock_start)
        locked_block = source[lock_start:lock_end]

        self.assertNotIn("retrieve_with_decay", locked_block)
        self.assertNotIn("count_by_user", locked_block)
        self.assertNotIn("get_recent_messages_by_user", locked_block)


if __name__ == "__main__":
    unittest.main()
