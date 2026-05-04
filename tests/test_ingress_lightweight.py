import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class IngressLightweightTests(unittest.TestCase):
    def test_auto_chat_uses_vlm_enabled_for_image_resolution(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertNotIn("resolve_images=False", source)
        self.assertIn("resolve_images=plugin_config.get(\"vlm\", {}).get(\"enabled\", True)", source)

    def test_image_sticker_sub_type_accepts_string_or_int(self):
        source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("def _is_sticker_segment_data", source)
        self.assertIn('str(data.get("sub_type", "")) == "1"', source)


if __name__ == "__main__":
    unittest.main()
