import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagSearchResultFlowStaticTests(unittest.TestCase):
    def test_search_result_is_returned_and_explicitly_passed(self):
        orchestrator_source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        services_source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("search_result = await self.memory_service.search", orchestrator_source)
        self.assertIn("search_result=search_result", orchestrator_source)

        self.assertIn("return await self.session.search_stage", services_source)
        self.assertIn("search_result=None", services_source)
        self.assertIn("search_result=search_result", services_source)

        self.assertIn("return search_result", session_source)
        self.assertIn("search_result: _SearchResult | None = None", session_source)

    def test_session_no_longer_uses_shared_search_result_slot(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertNotIn("__search_result", session_source)


if __name__ == "__main__":
    unittest.main()
