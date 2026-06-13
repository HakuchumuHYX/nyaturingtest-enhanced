import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ConsolidationSchemaTests(unittest.TestCase):
    def test_session_model_has_consolidation_watermark(self):
        source = (PLUGIN_DIR / "models" / "database.py").read_text(encoding="utf-8")
        self.assertIn("last_consolidated_time", source)

    def test_migration_adds_consolidation_column(self):
        source = (PLUGIN_DIR / "database" / "migrations.py").read_text(encoding="utf-8")
        self.assertIn("SCHEMA_VERSION = 3", source)
        self.assertIn("last_consolidated_time", source)

    def test_repository_persists_watermark(self):
        source = (PLUGIN_DIR / "database" / "session_repository.py").read_text(encoding="utf-8")
        self.assertIn('"last_consolidated_time"', source)

    def test_feedback_split_into_three_phases(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("async def _run_feedback_llm", source)
        self.assertIn("def _apply_sediment", source)
        self.assertIn("def _apply_decision", source)
        # feedback_stage 仍存在且现在由三段组合
        self.assertIn("async def feedback_stage", source)


if __name__ == "__main__":
    unittest.main()
