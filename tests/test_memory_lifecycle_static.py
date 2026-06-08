import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MemoryLifecycleStaticTests(unittest.TestCase):
    def test_long_term_memory_writes_lifecycle_metadata(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        save_start = session_source.index("async def save_long_term_memory")
        save_source = session_source[save_start:]

        for snippet in [
            '"status": "active"',
            '"category": category',
            '"confidence": confidence',
            '"importance": importance',
            '"ttl_days": runtime_settings["rag_default_event_ttl_days"]',
            'action = str(item.get("action") or "add")',
            'if action == "ignore"',
        ]:
            self.assertIn(snippet, save_source)

    def test_runtime_config_has_default_event_ttl(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        example_source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")

        self.assertIn('"rag_default_event_ttl_days": 90', config_source)
        self.assertIn('"rag_default_event_ttl_days": number("rag_default_event_ttl_days", 90, int, minimum=1)', config_source)
        self.assertIn('"rag_default_event_ttl_days": 90', example_source)

    def test_long_term_memory_supersede_path_has_whitelist_and_rejection_logs(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        save_start = session_source.index("async def save_long_term_memory")
        save_source = session_source[save_start:]

        for snippet in [
            "supersede_candidates",
            "allowed_supersede_refs",
            '"rag_action_hallucination"',
            '"rag_action_rejected"',
            'updated_target_metadata["status"] = "superseded"',
            'metadata["supersedes"] = target_ref',
            "get_metadata_by_id",
            "update_metadata_by_id",
            "add_texts)([content]",
        ]:
            self.assertIn(snippet, save_source)


if __name__ == "__main__":
    unittest.main()
