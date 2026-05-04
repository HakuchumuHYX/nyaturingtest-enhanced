import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _install_stubs():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("nonebot", nonebot)

    short_term = types.ModuleType("memory.short_term")

    class Message:
        def __init__(self, content, user_name="u", user_id="", time=None, id=""):
            self.content = content
            self.user_name = user_name
            self.user_id = user_id
            self.time = time or datetime.now()
            self.id = id

    short_term.Message = Message
    sys.modules.setdefault("nyaturingtest_utils.memory.short_term", short_term)


def _load_module(name: str, path: Path):
    _install_stubs()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RuntimePolicyTests(unittest.TestCase):
    def test_send_policy_preserves_sentence_punctuation_and_limits_parts(self):
        module = _load_module("message_sender_policy", PLUGIN_DIR / "core" / "message_sender.py")

        parts = module.build_send_parts("第一句。第二句！第三句？", max_messages=2)

        self.assertEqual(["第一句。", "第二句！"], parts)

    def test_image_cache_key_rejects_path_traversal_and_long_values(self):
        module = _load_module("image_policy", PLUGIN_DIR / "memory" / "image_policy.py")

        self.assertIsNone(module.sanitize_image_cache_key("../x"))
        self.assertIsNone(module.sanitize_image_cache_key("a/b"))
        self.assertIsNone(module.sanitize_image_cache_key("x" * 256))
        self.assertEqual("abc_123-OK", module.sanitize_image_cache_key("abc_123-OK"))

    def test_vad_clamp_rejects_out_of_range_values(self):
        module = _load_module("emotion_policy", PLUGIN_DIR / "models" / "emotion.py")

        self.assertEqual(1.0, module.clamp_vad_value(99, -1.0, 1.0))
        self.assertEqual(0.0, module.clamp_vad_value("bad", -1.0, 1.0, default=0.0))

    def test_relevance_does_not_trigger_on_one_character_alias(self):
        source = (PLUGIN_DIR / "utils.py").read_text(encoding="utf-8")

        self.assertIn("len(t.strip()) >= 2", source)


if __name__ == "__main__":
    unittest.main()
