import importlib.util
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
        self.assertIn("low_willingness_observe_interval", text)
        self.assertIn("role_max_chars", text)
        self.assertIn("examples_max_chars", text)
        self.assertIn("short_context_limit", text)
        self.assertIn("interaction_log_recent_days", text)
        self.assertIn("history_recall_limit", text)
        self.assertIn('"embedding"', text)
        self.assertIn('"base_url": "https://api.siliconflow.cn/v1/rerank"', text)
        self.assertNotIn("sk-", text)
        self.assertNotIn("google" + "_api_key", text)

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

    def test_low_willingness_observe_interval_allows_zero(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["runtime"]["low_willingness_observe_interval"] = 0
        module.plugin_config = cfg

        self.assertEqual(0, module.get_runtime_settings()["low_willingness_observe_interval"])

    def test_runtime_exposes_recent_context_and_repository_limits(self):
        module = _load_config_module()
        cfg = module.get_default_config()
        cfg["runtime"]["short_context_limit"] = 24
        cfg["runtime"]["interaction_log_recent_days"] = 90
        cfg["runtime"]["history_recall_limit"] = 12
        cfg["runtime"]["speak_cooldown_seconds"] = 3
        cfg["runtime"]["rerank_willingness_threshold"] = 0.8
        cfg["runtime"]["low_willingness_skip_threshold"] = 0.2
        cfg["runtime"]["post_feedback_skip_threshold"] = 0.25
        module.plugin_config = cfg

        settings = module.get_runtime_settings()

        self.assertEqual(24, settings["short_context_limit"])
        self.assertEqual(90, settings["interaction_log_recent_days"])
        self.assertEqual(12, settings["history_recall_limit"])
        self.assertEqual(3.0, settings["speak_cooldown_seconds"])
        self.assertEqual(0.8, settings["rerank_willingness_threshold"])
        self.assertEqual(0.2, settings["low_willingness_skip_threshold"])
        self.assertEqual(0.25, settings["post_feedback_skip_threshold"])

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
        cfg["runtime"]["rerank_willingness_threshold"] = 1.5
        cfg["runtime"]["short_context_limit"] = 0
        cfg["runtime"]["interaction_log_recent_days"] = 0
        cfg["runtime"]["history_recall_limit"] = -5
        module.plugin_config = cfg

        settings = module.get_runtime_settings()

        self.assertEqual(1, settings["queue_max_size"])
        self.assertEqual(1, settings["max_reply_messages"])
        self.assertEqual(0.0, settings["debounce_seconds"])
        self.assertEqual(0.0, settings["speak_cooldown_seconds"])
        self.assertEqual(1.0, settings["low_willingness_skip_threshold"])
        self.assertEqual(0.0, settings["post_feedback_skip_threshold"])
        self.assertEqual(1.0, settings["rerank_willingness_threshold"])
        self.assertEqual(1, settings["short_context_limit"])
        self.assertEqual(1, settings["interaction_log_recent_days"])
        self.assertEqual(1, settings["history_recall_limit"])

    def test_load_plugin_config_records_fallback_diagnostic_for_invalid_json(self):
        module = _load_config_module()
        with TemporaryDirectory() as tmp:
            bad_config = Path(tmp) / "config.json"
            bad_config.write_text("{ bad json", encoding="utf-8")
            module.CONFIG_FILE = bad_config

            loaded = module.load_plugin_config()
            status = module.get_config_load_status()

        self.assertEqual(module.get_default_config(), loaded)
        self.assertEqual("fallback", status.source)
        self.assertFalse(status.ok)
        self.assertIn("JSONDecodeError", status.error_type)
        self.assertIn("config.json", status.path)

    def test_status_command_reports_config_load_diagnostics(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertIn("get_config_load_status", source)
        self.assertIn("Config:", source)


if __name__ == "__main__":
    unittest.main()
