import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class PresetSubtypeStaticTests(unittest.TestCase):
    def test_load_preset_writes_subtype_metadata_by_section(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        load_start = source.index("async def load_preset")
        status_start = source.index("    def status", load_start)
        load_source = source[load_start:status_start]

        self.assertIn('preset_items.extend((item, "knowledge") for item in preset.knowledges)', load_source)
        self.assertIn('preset_items.extend((item, "relationship") for item in preset.relationships)', load_source)
        self.assertIn('preset_items.extend((item, "event") for item in preset.events)', load_source)
        self.assertIn('preset_items.extend((item, "bot_self") for item in preset.bot_self)', load_source)
        self.assertIn('"subtype": subtype', load_source)
        self.assertNotIn('[{"source": "preset", "type": "rule"} for _ in to_add]', load_source)

    def test_search_formatter_outputs_preset_subtype(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        search_start = source.index("async def search_stage")
        feedback_start = source.index("async def feedback_stage", search_start)
        search_source = source[search_start:feedback_start]

        self.assertIn('subtype = str(meta.get("subtype") or "legacy_rule")', search_source)
        self.assertIn('prefix = f"【设定/{subtype}】"', search_source)


if __name__ == "__main__":
    unittest.main()
