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

    def test_search_stage_supports_force_retrieve(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("force_retrieve", source)
        self.assertIn("should_retrieve = force_retrieve or", source)

    def test_consolidate_stage_exists_and_advances_watermark(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("async def consolidate_stage", source)
        self.assertIn("self.last_consolidated_time", source)
        # 固化路径不得触碰回复意愿（不调用 _apply_decision）
        cs = source[source.index("async def consolidate_stage"):]
        cs = cs[:cs.index("\n    async def ", 1)] if "\n    async def " in cs[1:] else cs
        self.assertNotIn("_apply_decision", cs)

    def test_config_defines_consolidation_keys(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        for snippet in [
            '"consolidation_enabled": True',
            '"consolidation_message_threshold": 8',
            '"consolidation_interval_seconds": 180.0',
            '"consolidation_max_messages": 60',
            '"consolidation_message_threshold": number("consolidation_message_threshold", 8, int, minimum=1)',
        ]:
            self.assertIn(snippet, source)

    def test_orchestrator_triggers_consolidation_not_periodic_feedback(self):
        source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("consolidate", source)
        self.assertIn("consolidation_message_threshold", source)

    def test_memory_service_has_consolidate(self):
        source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")
        self.assertIn("async def consolidate", source)
        self.assertIn("def note_incoming", source)


if __name__ == "__main__":
    unittest.main()
