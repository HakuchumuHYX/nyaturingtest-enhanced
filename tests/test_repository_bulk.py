import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RepositoryBulkTests(unittest.TestCase):
    def test_interaction_and_token_usage_have_bulk_apis(self):
        source = "\n".join([
            (PLUGIN_DIR / "database" / "profile_repository.py").read_text(encoding="utf-8"),
            (PLUGIN_DIR / "database" / "token_repository.py").read_text(encoding="utf-8"),
        ])

        self.assertIn("async def log_interactions", source)
        self.assertIn("async def log_token_usages", source)
        self.assertIn("InteractionLogModel.bulk_create", source)
        self.assertIn("TokenUsageModel.bulk_create", source)


if __name__ == "__main__":
    unittest.main()
