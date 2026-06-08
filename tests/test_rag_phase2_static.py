import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagPhase2StaticTests(unittest.TestCase):
    def test_runtime_has_explicit_chat_rag_k_settings(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        example_source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")

        self.assertIn('"rag_final_k": 20', config_source)
        self.assertIn('"rag_candidate_k": 40', config_source)
        self.assertIn('"rag_final_k": number("rag_final_k", 20, int, minimum=1)', config_source)
        self.assertIn('"rag_candidate_k": number("rag_candidate_k", 40, int, minimum=1)', config_source)
        self.assertIn('"rag_final_k": 20', example_source)
        self.assertIn('"rag_candidate_k": 40', example_source)

    def test_orchestrator_services_session_support_active_users_double_track(self):
        orchestrator_source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        services_source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn('"user_id": str(msg.user_id or "")', orchestrator_source)
        self.assertIn('"user_name": msg.user_name', orchestrator_source)
        self.assertIn("active_users=active_users", orchestrator_source)

        self.assertIn("active_user_names: list[str] | None = None", services_source)
        self.assertIn("active_users: list[dict] | None = None", services_source)
        self.assertIn("active_users=active_users", services_source)

        self.assertIn("def _active_user_query_names", session_source)
        self.assertIn("active_users: list[dict] | None = None", session_source)
        self.assertIn('key = f"id:{user_id}" if user_id else f"name:{user_name}"', session_source)

    def test_chat_path_passes_explicit_candidate_and_final_k(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn('k=runtime_settings["rag_final_k"]', session_source)
        self.assertIn('candidate_k=runtime_settings["rag_candidate_k"]', session_source)

    def test_chat_path_passes_active_user_ids_for_scope_ranking(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("active_scope_user_ids", session_source)
        self.assertIn("active_user_ids=active_scope_user_ids", session_source)
        self.assertIn('where_any("source", ["preset", "memory"])', session_source)


if __name__ == "__main__":
    unittest.main()
