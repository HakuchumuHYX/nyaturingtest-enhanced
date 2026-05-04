import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ImportCleanupStaticTests(unittest.TestCase):
    def test_image_and_vector_unused_imports_are_removed(self):
        image_source = (PLUGIN_DIR / "memory" / "image.py").read_text(encoding="utf-8")
        vector_source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")

        self.assertNotIn("import numpy", image_source)
        self.assertNotIn("ImageSequence", image_source)
        self.assertNotIn("import asyncio", vector_source)

    def test_llm_response_uses_structured_generate(self):
        source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("result = await client.generate(", source)
        self.assertNotIn("client.generate_response(", source)


if __name__ == "__main__":
    unittest.main()
