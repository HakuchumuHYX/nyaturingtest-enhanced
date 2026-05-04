import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class VectorDecayTests(unittest.TestCase):
    def test_retrieve_with_decay_uses_real_date_delta(self):
        source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertIn("datetime.strptime", source)
        self.assertNotIn("month_diff * 30", source)


if __name__ == "__main__":
    unittest.main()
