import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


class WillingnessConfigTests(unittest.TestCase):
    def test_softened_defaults_and_new_keys(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        for snippet in [
            '"relevance_willingness_floor": 0.7',
            '"speak_willingness_retain_factor": 0.55',
            '"interest_topic_willingness_floor": 0.45',
            '"passive_growth_min_factor": 0.3',
            '"passive_growth_max_factor": 2.0',
            '"willingness_reply_threshold": 0.4',
            '"speak_willingness_retain_factor": ratio("speak_willingness_retain_factor", 0.55)',
        ]:
            self.assertIn(snippet, source)


class InterestScoreTests(unittest.TestCase):
    def _fn(self):
        import importlib.util
        import sys
        import types

        nb = sys.modules.get("nonebot") or types.ModuleType("nonebot")
        nb.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                                          error=lambda *a, **k: None, debug=lambda *a, **k: None)
        sys.modules["nonebot"] = nb
        plugins_pkg = sys.modules.get("plugins") or types.ModuleType("plugins")
        plugins_pkg.__path__ = [str(PLUGIN_DIR.parent)]
        sys.modules["plugins"] = plugins_pkg
        plugin_pkg = sys.modules.get("plugins.nyaturingtest") or types.ModuleType("plugins.nyaturingtest")
        plugin_pkg.__path__ = [str(PLUGIN_DIR)]
        sys.modules["plugins.nyaturingtest"] = plugin_pkg
        spec = importlib.util.spec_from_file_location(
            "plugins.nyaturingtest.utils",
            PLUGIN_DIR / "utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "plugins.nyaturingtest"
        spec.loader.exec_module(module)
        return module.score_message_interest

    def test_question_scores_higher_than_filler(self):
        fn = self._fn()
        q = fn(["这个怎么搞？"], bot_name="喵", aliases=[], lo=0.3, hi=2.0)
        filler = fn(["哈哈哈哈"], bot_name="喵", aliases=[], lo=0.3, hi=2.0)
        self.assertGreater(q, filler)

    def test_name_mention_boosts(self):
        fn = self._fn()
        named = fn(["喵你觉得呢"], bot_name="喵", aliases=[], lo=0.3, hi=2.0)
        plain = fn(["天气不错"], bot_name="喵", aliases=[], lo=0.3, hi=2.0)
        self.assertGreater(named, plain)

    def test_clamped_to_bounds(self):
        fn = self._fn()
        value = fn(["？？？喵喵喵在吗在吗"], bot_name="喵", aliases=[], lo=0.3, hi=2.0)
        self.assertLessEqual(value, 2.0)
        self.assertGreaterEqual(value, 0.3)


class WillingnessGrowthTests(unittest.TestCase):
    def test_passive_growth_uses_interest_score(self):
        source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("score_message_interest", source)
        self.assertIn("interest_topic_willingness_floor", source)


class WillingnessDeductionTests(unittest.TestCase):
    def test_proportional_deduction_and_configurable_load(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("speak_willingness_retain_factor", source)
        self.assertNotIn("self.willingness - 0.5", source)
        self.assertIn('"willingness_load_value"', source)


class WillingnessDecayTests(unittest.TestCase):
    def test_decay_uses_dedicated_timer_not_activity_reset(self):
        source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("_last_decay_time", source)
        # 活跃/空闲速率应由「距上次发言」判断
        self.assertIn("_last_speak_time", source)


if __name__ == "__main__":
    unittest.main()
