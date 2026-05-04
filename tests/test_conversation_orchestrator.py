import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class ConversationOrchestratorTests(unittest.TestCase):
    def test_orchestrator_and_services_are_defined(self):
        orchestrator = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        services = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")

        self.assertIn("class ConversationOrchestrator", orchestrator)
        self.assertIn("async def process_chunk", orchestrator)
        self.assertIn("class FeedbackService", services)
        self.assertIn("class ChatService", services)
        self.assertIn("class MemoryService", services)
        self.assertNotIn("_Session__", services)

    def test_session_update_delegates_to_orchestrator(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("ConversationOrchestrator", source)
        self.assertIn("process_chunk", source)
        self.assertLess(source.count("async def update("), 2)


if __name__ == "__main__":
    unittest.main()
