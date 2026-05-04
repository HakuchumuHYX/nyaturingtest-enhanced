import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MessageRenderingHelperTests(unittest.TestCase):
    def test_image_placeholder_and_resolution_gate_are_helpers(self):
        source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("def _image_placeholder", source)
        self.assertIn("def _should_resolve_image", source)
        self.assertLessEqual(source.count('plugin_config.get("vlm", {}).get("enabled", True)'), 2)
        self.assertLessEqual(source.count('return "\\n[表情包]\\n" if is_sticker else "\\n[图片]\\n"'), 1)


if __name__ == "__main__":
    unittest.main()
