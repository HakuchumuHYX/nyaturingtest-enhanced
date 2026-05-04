import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class DomainModelTests(unittest.TestCase):
    def test_unused_domain_package_has_been_removed(self):
        domain_dir = PLUGIN_DIR / "domain"

        self.assertFalse((domain_dir / "__init__.py").exists())
        self.assertEqual([], [path.name for path in domain_dir.glob("*.py")])


if __name__ == "__main__":
    unittest.main()
