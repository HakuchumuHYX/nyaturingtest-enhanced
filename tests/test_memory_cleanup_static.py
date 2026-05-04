import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MemoryCleanupStaticTests(unittest.TestCase):
    def test_short_term_memory_has_no_dead_llm_or_compress_callback(self):
        source = (PLUGIN_DIR / "memory" / "short_term.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertNotIn("llm_client", source)
        self.assertNotIn("after_compress", source)
        self.assertNotIn("llm_client=", session_source)


if __name__ == "__main__":
    unittest.main()
