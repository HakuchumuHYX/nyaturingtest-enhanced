import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MemoryWriteResilienceConfigTests(unittest.TestCase):
    def test_config_defines_retry_and_wal_keys(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        self.assertIn('"memory_write_max_retries": 3', source)
        self.assertIn('"memory_write_retry_base_delay": 0.5', source)
        self.assertIn(
            '"memory_write_max_retries": number("memory_write_max_retries", 3, int, minimum=0)',
            source,
        )

    def test_example_config_defines_retry_and_wal_keys(self):
        source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        self.assertIn('"memory_write_max_retries": 3', source)
        self.assertIn('"memory_write_retry_base_delay": 0.5', source)


if __name__ == "__main__":
    unittest.main()
