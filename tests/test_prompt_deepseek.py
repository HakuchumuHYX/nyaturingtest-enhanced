import importlib.util
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
DYNAMIC_MARKER = "---- DYNAMIC INPUT ----"


def _load_templates_module():
    spec = importlib.util.spec_from_file_location(
        "nyaturingtest_prompt_templates",
        PLUGIN_DIR / "prompts" / "templates.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stable_prefix(prompt: str) -> str:
    marker_index = prompt.index(DYNAMIC_MARKER)
    return prompt[:marker_index]


class PromptDeepSeekTests(unittest.TestCase):
    def test_templates_do_not_request_visible_think_tags(self):
        source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")

        self.assertNotIn("<think>", source)
        self.assertNotIn("think_protocol", source)
        self.assertIn("内部完成分析", source)
        self.assertIn("只包含一个合法 JSON 对象", source)

    def test_templates_use_profiles_and_canonical_json(self):
        source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")

        self.assertIn("related_profiles", source)
        self.assertIn("sort_keys=True", source)
        self.assertIn('separators=(",", ":")', source)

    def test_feedback_prompt_has_stable_prefix(self):
        templates = _load_templates_module()
        base_prompt = templates.get_feedback_prompt(
            "bot",
            "role A",
            0.25,
            1,
            "old summary",
            ["old msg"],
            ["new msg"],
            {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
            '[{"user_name":"u1"}]',
            ["memory A"],
            "last topic",
            is_relevant=False,
            time_info="2026年05月18日 12:00 周一 [中午] [工作日]",
        )
        changed_prompt = templates.get_feedback_prompt(
            "bot2",
            "role B",
            0.95,
            2,
            "changed summary",
            ["changed old msg"],
            ["changed new msg"],
            {"valence": -0.5, "arousal": 0.8, "dominance": -0.2},
            '[{"user_name":"u2"}]',
            ["memory B"],
            "changed topic",
            is_relevant=True,
            time_info="2026年05月18日 23:59 周一 [深夜] [工作日]",
        )

        prefix = _stable_prefix(base_prompt)
        self.assertEqual(prefix, _stable_prefix(changed_prompt))
        self.assertNotIn("role A", prefix)
        self.assertNotIn("new msg", prefix)
        self.assertNotIn("2026年", prefix)
        self.assertIn("new msg", base_prompt)
        self.assertIn("2026年", base_prompt)

    def test_chat_prompt_has_stable_prefix(self):
        templates = _load_templates_module()
        base_prompt = templates.get_chat_prompt(
            "bot",
            "role A",
            1,
            "old summary",
            ["old msg"],
            ["new msg"],
            {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
            '[{"user_name":"u1"}]',
            ["memory A"],
            "topic A",
            examples_text="User: hi\nbot: hello",
            recalled_history="none",
            time_info="2026年05月18日 12:00 周一 [中午] [工作日]",
            rp_style="deepseek_v4_roleplay",
        )
        changed_prompt = templates.get_chat_prompt(
            "bot2",
            "role B",
            2,
            "changed summary",
            ["changed old msg"],
            ["changed new msg"],
            {"valence": -0.5, "arousal": 0.8, "dominance": -0.2},
            '[{"user_name":"u2"}]',
            ["memory B"],
            "topic B",
            examples_text="User: bye\nbot2: see you",
            recalled_history="historical event",
            time_info="2026年05月18日 23:59 周一 [深夜] [工作日]",
            rp_style="deepseek_v4_roleplay",
        )

        prefix = _stable_prefix(base_prompt)
        self.assertEqual(prefix, _stable_prefix(changed_prompt))
        self.assertIn("deepseek_v4_roleplay_instruct", prefix)
        self.assertNotIn("role A", prefix)
        self.assertNotIn("new msg", prefix)
        self.assertNotIn("2026年", prefix)
        self.assertIn("new msg", base_prompt)
        self.assertIn("2026年", base_prompt)


if __name__ == "__main__":
    unittest.main()
