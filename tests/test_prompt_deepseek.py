import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class PromptDeepSeekTests(unittest.TestCase):
    def test_templates_do_not_request_visible_think_tags(self):
        source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")

        self.assertNotIn("<think>", source)
        self.assertNotIn("think_protocol", source)
        self.assertIn("内部完成分析", source)
        self.assertIn("只包含一个合法 JSON 对象", source)

    def test_templates_use_profiles_and_canonical_json(self):
        source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")

        self.assertIn("related_profiles_json", source)
        self.assertIn("sort_keys=True", source)
        self.assertIn('separators=(",", ":")', source)


if __name__ == "__main__":
    unittest.main()
