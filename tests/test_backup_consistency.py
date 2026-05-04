import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class BackupConsistencyTests(unittest.TestCase):
    def test_backup_uses_sqlite_backup_api_and_staging_dir(self):
        source = (PLUGIN_DIR / "database" / "backup.py").read_text(encoding="utf-8")

        self.assertIn("sqlite3", source)
        self.assertIn(".backup(", source)
        self.assertIn("TemporaryDirectory", source)
        self.assertIn("nyabot.sqlite", source)

    def test_vector_writes_share_backup_lock(self):
        source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("BACKUP_IO_LOCK", source)
        self.assertIn("with BACKUP_IO_LOCK", source)


if __name__ == "__main__":
    unittest.main()
