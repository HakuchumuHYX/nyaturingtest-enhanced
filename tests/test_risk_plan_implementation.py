import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PLUGIN_DIR.parents[1]


class RiskPlanImplementationTests(unittest.TestCase):
    def test_feedback_service_persists_after_processing(self):
        source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")

        self.assertIn("class FeedbackService", source)
        self.assertIn("finally:", source)
        self.assertIn("await self.session.save_session(expected_generation=expected_generation)", source)

    def test_short_term_memory_has_extended_context_api(self):
        memory_source = (PLUGIN_DIR / "memory" / "short_term.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("def access_context", memory_source)
        self.assertIn("access_context(", session_source)

    def test_summary_empty_string_keeps_session_and_memory_consistent(self):
        memory_source = (PLUGIN_DIR / "memory" / "short_term.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("if new_summary is not None", memory_source)
        self.assertIn('summary = response_dict.get("summary")', session_source)
        self.assertIn("if summary is not None:", session_source)
        self.assertNotIn('str(response_dict.get("summary", self.chat_summary))', session_source)

    def test_low_willingness_observation_is_documented_as_consolidation(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        example_source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        readme_source = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
        memory_analysis_source = (WORKSPACE_DIR / "docs" / "memory-analysis.md").read_text(encoding="utf-8")
        orchestrator_source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")

        self.assertNotIn('"low_willingness_observe_interval":', config_source)
        self.assertNotIn("low_willingness_observe_interval", example_source)
        self.assertNotIn("low_willingness_observe_interval", readme_source)
        self.assertNotIn("low_willingness_observe_interval", memory_analysis_source)
        self.assertNotIn("_passive_observe_skips", orchestrator_source)
        self.assertIn("await self.memory_service.consolidate", orchestrator_source)
        self.assertIn("consolidation_message_threshold", readme_source)
        self.assertIn("consolidation_message_threshold", memory_analysis_source)
        self.assertIn("被动记忆固化", readme_source)

    def test_self_message_fallback_id_exists(self):
        logic_source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("def _build_self_message_id", logic_source)
        self.assertIn("chunk_bot = state.bot", logic_source)
        self.assertIn("append_self_message(sent_content, msg_id, str(chunk_bot.self_id))", logic_source)

    def test_deepseek_v4_rp_marker_is_configurable(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        prompt_source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("deepseek_v4_roleplay", config_source)
        self.assertIn("build_deepseek_v4_rp_marker", prompt_source)
        self.assertIn("rp_style=get_chat_thinking_settings().get", session_source)
        self.assertNotIn('"thought"', prompt_source)

    def test_gemini_rp_style_uses_openai_compatible_path(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        prompt_source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")
        logic_source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("gemini_3_flash_roleplay", config_source)
        self.assertIn("build_gemini_3_flash_rp_marker", prompt_source)
        self.assertIn("chat_provider == \"deepseek_official\"", logic_source)
        self.assertIn("get_effective_feedback_provider() == \"deepseek_official\"", logic_source)
        self.assertIn("extra_body=chat_extra_body", logic_source)
        self.assertIn("extra_body=feedback_extra_body", logic_source)
        self.assertNotIn("google_ai_studio", prompt_source)
        self.assertNotIn("_build_gemini_payload", logic_source)

    def test_role_injection_has_length_limits(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        logic_source = (PLUGIN_DIR / "core" / "logic.py").read_text(encoding="utf-8")

        self.assertIn("role_max_chars", config_source)
        self.assertIn("examples_max_chars", config_source)
        self.assertIn("_limit_role_text", session_source)
        self.assertNotIn("current_role = state.session.role()", logic_source)
        self.assertIn("角色资料只来自用户消息中的 <profile> 区块", logic_source)

    def test_chat_rag_uses_decay_retrieval(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        services_source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")

        self.assertIn("search_for_chat", session_source)
        self.assertIn("retrieve_with_decay", services_source)

    def test_runtime_resources_are_closed(self):
        vector_source = (PLUGIN_DIR / "memory" / "vector.py").read_text(encoding="utf-8")
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        state_source = (PLUGIN_DIR / "core" / "state_manager.py").read_text(encoding="utf-8")

        self.assertIn("def close(self):", vector_source)
        self.assertIn("async def close(self):", session_source)
        self.assertIn("async def drain_background_tasks", session_source)
        self.assertIn("await state.session.drain_background_tasks", state_source)
        self.assertIn("await state.session.close()", state_source)

    def test_recent_interactions_query_is_bounded(self):
        source = (PLUGIN_DIR / "database" / "session_repository.py").read_text(encoding="utf-8")

        self.assertIn("recent_interaction_cutoff", source)
        self.assertIn("timestamp__gte", source)

    def test_ingress_priority_uses_structured_segments(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertIn("def _is_priority_message", source)
        self.assertIn('seg.type == "at"', source)
        self.assertIn('seg.type == "reply"', source)
        self.assertIn("pre_queue_priority = _is_priority_message(", source)
        self.assertIn("event.original_message,", source)


if __name__ == "__main__":
    unittest.main()
