import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class UsageRecorderTests(unittest.TestCase):
    def test_shared_usage_recorder_module_exists(self):
        source = (PLUGIN_DIR / "core" / "usage.py").read_text(encoding="utf-8")

        self.assertIn("def record_token_usage", source)
        self.assertIn("def make_usage_recorder", source)
        self.assertIn("TokenUsageRepository.log_token_usage", source)

    def test_hot_paths_use_shared_usage_recorder(self):
        logic = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")
        memory = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")

        self.assertIn("make_usage_recorder", logic)
        self.assertIn("make_usage_recorder", memory)
        self.assertNotIn("def _make_usage_recorder", memory)
        self.assertNotIn("def make_vlm_recorder", logic)


if __name__ == "__main__":
    unittest.main()
