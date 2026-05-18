import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class TokenUsageSchemaTests(unittest.TestCase):
    def test_token_usage_model_has_deepseek_observability_fields(self):
        source = (PLUGIN_DIR / "models" / "database.py").read_text(encoding="utf-8")
        for field_name in [
            "provider",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
            "finish_reason",
        ]:
            with self.subTest(field_name=field_name):
                self.assertIn(field_name, source)

    def test_repository_records_and_aggregates_deepseek_usage_fields(self):
        source = (PLUGIN_DIR / "database" / "token_repository.py").read_text(encoding="utf-8")
        for token in [
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
            "finish_reason",
            "total_reasoning",
        ]:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_token_stats_output_includes_cache_miss_and_hit_ratio(self):
        repository = (PLUGIN_DIR / "database" / "token_repository.py").read_text(encoding="utf-8")
        renderer = (PLUGIN_DIR / "utils.py").read_text(encoding="utf-8")

        self.assertIn('"cache_miss":', repository)
        self.assertIn('"cache_hit_ratio":', repository)
        self.assertIn("Cache miss", renderer)
        self.assertIn("Hit ratio", renderer)

    def test_token_stats_command_filters_to_current_configured_models(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertIn("get_token_stats_model_names", source)
        self.assertIn("model_names=get_token_stats_model_names()", source)
        self.assertNotIn("model_names=None", source)

    def test_config_exposes_current_token_stats_model_names(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")

        self.assertIn("def get_token_stats_model_names", source)
        self.assertIn("get_effective_chat_model()", source)
        self.assertIn("get_effective_feedback_model()", source)
        self.assertIn("get_effective_vlm_model()", source)
        self.assertIn('plugin_config.get("vlm", {}).get("enabled", True)', source)


if __name__ == "__main__":
    unittest.main()
