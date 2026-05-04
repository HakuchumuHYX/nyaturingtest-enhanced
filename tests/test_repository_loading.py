import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RepositoryLoadingTests(unittest.TestCase):
    def test_load_full_session_data_avoids_per_user_interaction_query(self):
        source = (PLUGIN_DIR / "database" / "session_repository.py").read_text(encoding="utf-8")

        self.assertIn("InteractionLogModel.filter(", source)
        self.assertIn("user_id__in=user_ids", source)
        self.assertIn("timestamp__gte=recent_interaction_cutoff", source)
        self.assertNotIn("user_db.interactions.all()", source)


if __name__ == "__main__":
    unittest.main()
