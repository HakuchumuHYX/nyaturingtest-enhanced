import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class IngressLightweightTests(unittest.TestCase):
    def test_auto_chat_uses_effective_image_routes(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertNotIn("resolve_images=False", source)
        self.assertIn("_resolve_images = should_use_standalone_vlm()", source)
        self.assertIn("attach_native_images=native_vision_enabled()", source)
        self.assertIn("image_inputs_out=image_inputs", source)

    def test_queue_pressure_drops_low_priority_before_message_conversion(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertIn("pre_queue_priority", source)
        self.assertIn('decision="drop_pre_conversion"', source)
        self.assertLess(
            source.index("pre_queue_priority"),
            source.index("message_content, image_meta = await message2BotMessage"),
        )

    def test_image_sticker_sub_type_accepts_string_or_int(self):
        source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("def _is_sticker_segment_data", source)
        self.assertIn('str(data.get("sub_type", "")) == "1"', source)


if __name__ == "__main__":
    unittest.main()
