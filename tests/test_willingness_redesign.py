import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


class WillingnessConfigTests(unittest.TestCase):
    def test_softened_defaults_and_new_keys(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        for snippet in [
            '"relevance_willingness_floor": 0.7',
            '"speak_willingness_retain_factor": 0.55',
            '"interest_topic_willingness_floor": 0.45',
            '"passive_growth_min_factor": 0.3',
            '"passive_growth_max_factor": 2.0',
            '"willingness_reply_threshold": 0.4',
            '"speak_willingness_retain_factor": ratio("speak_willingness_retain_factor", 0.55)',
        ]:
            self.assertIn(snippet, source)


if __name__ == "__main__":
    unittest.main()
