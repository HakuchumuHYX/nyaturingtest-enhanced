import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ProfileDirtyTests(unittest.TestCase):
    def test_profile_model_exposes_dirty_tracking(self):
        source = (PLUGIN_DIR / "models" / "profile.py").read_text(encoding="utf-8")

        self.assertIn("is_dirty", source)
        self.assertIn("mark_clean", source)
        self.assertIn("mark_dirty", source)

    def test_save_session_filters_clean_profiles(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("dirty_profiles", source)
        self.assertIn("profile.is_dirty", source)
        self.assertIn("profile.mark_clean", source)


if __name__ == "__main__":
    unittest.main()
