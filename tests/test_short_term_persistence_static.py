import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ShortTermPersistenceStaticTests(unittest.TestCase):
    def test_config_defines_short_term_buffer_size(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        self.assertIn('"short_term_buffer_size": 200', config_source)
        self.assertIn(
            '"short_term_buffer_size": number("short_term_buffer_size", 200, int, minimum=1)',
            config_source,
        )

    def test_example_config_defines_short_term_buffer_size(self):
        example = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        self.assertIn('"short_term_buffer_size"', example)


if __name__ == "__main__":
    unittest.main()
