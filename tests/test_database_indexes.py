import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class DatabaseIndexTests(unittest.TestCase):
    def test_message_unique_constraint_and_common_indexes_exist(self):
        model_source = (PLUGIN_DIR / "models" / "database.py").read_text(encoding="utf-8")
        migration_source = (PLUGIN_DIR / "database" / "migrations.py").read_text(encoding="utf-8")

        self.assertIn('unique_together = (("session", "msg_id"),)', model_source)
        self.assertIn("idx_messages_session_time", migration_source)
        self.assertIn("idx_token_usage_model_time", migration_source)
        self.assertIn("idx_interactions_timestamp", migration_source)


if __name__ == "__main__":
    unittest.main()
