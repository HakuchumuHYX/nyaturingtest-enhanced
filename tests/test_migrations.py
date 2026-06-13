import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MigrationTests(unittest.TestCase):
    def test_schema_version_migration_exists(self):
        source = (PLUGIN_DIR / "database" / "migrations.py").read_text(encoding="utf-8")

        self.assertIn("nyabot_schema_version", source)
        self.assertIn("SCHEMA_VERSION = 3", source)
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_session_msg_id", source)


if __name__ == "__main__":
    unittest.main()
