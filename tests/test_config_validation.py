import importlib.util
import json
from tempfile import TemporaryDirectory
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_config_module():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("nonebot", nonebot)

    spec = importlib.util.spec_from_file_location("nyaturingtest_config", PLUGIN_DIR / "config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConfigValidationTests(unittest.TestCase):
    def test_removed_provider_fails_fast(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["chat"]["provider"] = "google" + "_ai_studio"

        with self.assertRaises(RuntimeError):
            module.normalize_config(cfg)

    def test_settings_validate_required_deepseek_fields(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["chat"]["api_key"] = ""

        with self.assertRaises(RuntimeError):
            module.build_settings(cfg, require_api_keys=True)

    def test_config_example_exists_without_real_secrets(self):
        example = PLUGIN_DIR / "config.example.json"
        self.assertTrue(example.exists())
        text = example.read_text(encoding="utf-8")

        self.assertIn("deepseek-v4-flash", text)
        self.assertNotIn("low_willingness_observe_interval", text)
        self.assertIn("role_max_chars", text)
        self.assertIn("examples_max_chars", text)
        self.assertIn("short_context_limit", text)
        self.assertIn("consolidation_enabled", text)
        self.assertIn("consolidation_message_threshold", text)
        self.assertIn("consolidation_interval_seconds", text)
        self.assertIn("consolidation_max_messages", text)
        self.assertIn("interaction_log_recent_days", text)
        self.assertIn("history_recall_limit", text)
        self.assertIn("active_to_bubble_threshold", text)
        self.assertIn('"embedding"', text)
        self.assertIn('"base_url": "https://api.siliconflow.cn/v1/rerank"', text)
        self.assertNotIn("sk-", text)
        self.assertNotIn("google" + "_api_key", text)

    def test_llm_output_budgets_default_to_65536(self):
        module = _load_config_module()
        cfg = module.get_default_config()

        self.assertEqual(65536, cfg["chat"]["max_tokens"])
        self.assertEqual(65536, cfg["feedback"]["max_tokens"])

        settings = module.build_settings(cfg)
        self.assertEqual(65536, settings.chat.max_tokens)
        self.assertEqual(65536, settings.feedback.max_tokens)

        module.plugin_config = {"chat": {}, "feedback": {}}
        self.assertEqual(65536, module.get_chat_max_tokens())
        self.assertEqual(65536, module.get_feedback_max_tokens())

    def test_config_example_uses_65536_llm_output_budgets(self):
        example = json.loads(
            (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        )

        self.assertEqual(65536, example["chat"]["max_tokens"])
        self.assertEqual(65536, example["feedback"]["max_tokens"])

    def test_readme_documents_restart_required_config_fields(self):
        readme = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")

        for snippet in [
            "需要重启",
            "chat.provider",
            "chat.base_url",
            "chat.api_key",
            "feedback.provider",
            "feedback.base_url",
            "feedback.api_key",
            "vlm.base_url",
            "embedding.base_url",
            "enabled_groups",
        ]:
            self.assertIn(snippet, readme)

    def test_legacy_config_is_normalized_to_current_schema(self):
        module = _load_config_module()
        legacy = {
            "chat": {
                "provider": "openai_compatible",
                "api_key": "chat-key",
                "base_url": "https://api.deepseek.com/v1",
                "model": "old-chat",
                "google" + "_api_key": "unused",
            },
            "feedback": {
                "provider": "openai_compatible",
                "api_key": "feedback-key",
                "base_url": "https://api.deepseek.com",
                "model": "old-feedback",
            },
            "vlm": {
                "enabled": True,
                "provider": "openai_compatible",
                "model": "old-vlm",
                "openai_api_key": "vlm-key",
                "openai_base_url": "https://vlm.example/v1",
            },
        }

        normalized = module.normalize_config(legacy)

        self.assertEqual(module.DEEPSEEK_OFFICIAL, normalized["chat"]["provider"])
        self.assertEqual(module.DEEPSEEK_OFFICIAL, normalized["feedback"]["provider"])
        self.assertEqual("vlm-key", normalized["vlm"]["api_key"])
        self.assertEqual("https://vlm.example/v1", normalized["vlm"]["base_url"])
        self.assertIn("thinking", normalized["chat"])
        self.assertIn("runtime", normalized)
        self.assertNotIn("openai_api_key", normalized["vlm"])
        self.assertNotIn("google" + "_api_key", normalized["chat"])

    def test_legacy_low_willingness_observe_interval_is_accepted_but_not_exposed(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["runtime"]["low_willingness_observe_interval"] = 0
        normalized = module.normalize_config(cfg)
        module.plugin_config = cfg

        self.assertIn("low_willingness_observe_interval", normalized["runtime"])
        self.assertNotIn("low_willingness_observe_interval", module.get_default_config()["runtime"])
        self.assertNotIn("low_willingness_observe_interval", module.get_runtime_settings())

    def test_runtime_exposes_recent_context_and_repository_limits(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["runtime"]["short_context_limit"] = 24
        cfg["runtime"]["interaction_log_recent_days"] = 90
        cfg["runtime"]["history_recall_limit"] = 12
        cfg["runtime"]["rag_per_query_recall_k"] = 18
        cfg["runtime"]["rag_merged_candidate_cap"] = 32
        cfg["runtime"]["rag_memory_char_budget"] = 1200
        cfg["runtime"]["speak_cooldown_seconds"] = 3
        cfg["runtime"]["rerank_willingness_threshold"] = 0.8
        cfg["runtime"]["low_willingness_skip_threshold"] = 0.2
        cfg["runtime"]["post_feedback_skip_threshold"] = 0.25
        cfg["runtime"]["active_to_bubble_threshold"] = 0.45
        module.plugin_config = cfg

        settings = module.get_runtime_settings()

        self.assertEqual(24, settings["short_context_limit"])
        self.assertEqual(90, settings["interaction_log_recent_days"])
        self.assertEqual(12, settings["history_recall_limit"])
        self.assertEqual(18, settings["rag_per_query_recall_k"])
        self.assertEqual(32, settings["rag_merged_candidate_cap"])
        self.assertEqual(1200, settings["rag_memory_char_budget"])
        self.assertEqual(3.0, settings["speak_cooldown_seconds"])
        self.assertEqual(0.8, settings["rerank_willingness_threshold"])
        self.assertEqual(0.2, settings["low_willingness_skip_threshold"])
        self.assertEqual(0.25, settings["post_feedback_skip_threshold"])
        self.assertEqual(0.45, settings["active_to_bubble_threshold"])

    def test_memory_endpoint_settings_are_configurable(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["embedding"]["model"] = "custom-embedding"
        cfg["embedding"]["base_url"] = "https://embedding.example/v1"
        cfg["embedding"]["timeout"] = 12
        cfg["rerank"]["base_url"] = "https://rerank.example/v1/rerank"
        cfg["rerank"]["timeout"] = 8
        module.plugin_config = cfg

        self.assertEqual(
            {
                "model": "custom-embedding",
                "base_url": "https://embedding.example/v1",
                "timeout": 12.0,
                "rerank_base_url": "https://rerank.example/v1/rerank",
                "rerank_timeout": 8.0,
            },
            module.get_memory_endpoint_settings(),
        )

    def test_runtime_settings_are_clamped_to_safe_ranges(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["runtime"]["queue_max_size"] = 0
        cfg["runtime"]["max_reply_messages"] = -3
        cfg["runtime"]["debounce_seconds"] = -1
        cfg["runtime"]["speak_cooldown_seconds"] = -10
        cfg["runtime"]["low_willingness_skip_threshold"] = 2
        cfg["runtime"]["post_feedback_skip_threshold"] = -1
        cfg["runtime"]["active_to_bubble_threshold"] = 2
        cfg["runtime"]["rerank_willingness_threshold"] = 1.5
        cfg["runtime"]["short_context_limit"] = 0
        cfg["runtime"]["interaction_log_recent_days"] = 0
        cfg["runtime"]["history_recall_limit"] = -5
        cfg["runtime"]["rag_per_query_recall_k"] = 0
        cfg["runtime"]["rag_merged_candidate_cap"] = -3
        cfg["runtime"]["rag_memory_char_budget"] = 0
        module.plugin_config = cfg

        settings = module.get_runtime_settings()

        self.assertEqual(1, settings["queue_max_size"])
        self.assertEqual(1, settings["max_reply_messages"])
        self.assertEqual(0.0, settings["debounce_seconds"])
        self.assertEqual(0.0, settings["speak_cooldown_seconds"])
        self.assertEqual(1.0, settings["low_willingness_skip_threshold"])
        self.assertEqual(0.0, settings["post_feedback_skip_threshold"])
        self.assertEqual(1.0, settings["active_to_bubble_threshold"])
        self.assertEqual(1.0, settings["rerank_willingness_threshold"])
        self.assertEqual(1, settings["short_context_limit"])
        self.assertEqual(1, settings["interaction_log_recent_days"])
        self.assertEqual(1, settings["history_recall_limit"])
        self.assertEqual(1, settings["rag_per_query_recall_k"])
        self.assertEqual(1, settings["rag_merged_candidate_cap"])
        self.assertEqual(1, settings["rag_memory_char_budget"])

    def test_load_plugin_config_fails_fast_for_invalid_json_without_last_known_good(self):
        module = _load_config_module()
        module._plugin_config = {}
        with TemporaryDirectory() as tmp:
            bad_config = Path(tmp) / "config.json"
            bad_config.write_text("{ bad json", encoding="utf-8")
            module.CONFIG_FILE = bad_config

            with self.assertRaises(json.JSONDecodeError):
                module.load_plugin_config()
            status = module.get_config_load_status()

        self.assertEqual("invalid", status.source)
        self.assertFalse(status.ok)
        self.assertIn("JSONDecodeError", status.error_type)
        self.assertIn("config.json", status.path)

    def test_invalid_config_keeps_last_known_good_config_when_available(self):
        module = _load_config_module()
        previous = module.get_default_config()
        previous["chat"]["model"] = "known-good-model"
        module._plugin_config = previous
        with TemporaryDirectory() as tmp:
            bad_config = Path(tmp) / "config.json"
            bad_config.write_text("{ bad json", encoding="utf-8")
            module.CONFIG_FILE = bad_config

            loaded = module.load_plugin_config()
            status = module.get_config_load_status()

        self.assertEqual("known-good-model", loaded["chat"]["model"])
        self.assertEqual("last_known_good", status.source)
        self.assertFalse(status.ok)
        self.assertIn("JSONDecodeError", status.error_type)

    def test_status_command_reports_config_load_diagnostics(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertIn("get_config_load_status", source)
        self.assertIn("Config:", source)


if __name__ == "__main__":
    unittest.main()
