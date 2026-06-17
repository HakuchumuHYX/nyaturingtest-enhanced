import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_short_term():
    nonebot = sys.modules.get("nonebot") or types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None, debug=lambda *a, **k: None,
    )
    sys.modules["nonebot"] = nonebot
    spec = importlib.util.spec_from_file_location(
        "short_term_image_meta_test", PLUGIN_DIR / "memory" / "short_term.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MessageImageMetaTests(unittest.TestCase):
    def test_image_meta_roundtrip(self):
        module = _load_short_term()
        meta = {
            "primary": {"entities": [{"name": "初音未来", "type": "character", "confidence": 0.85}],
                        "ocr_text": "我装的", "pragmatic_intent": "否认",
                        "affect": {"valence": 0.7, "arousal": 0.5, "dominance": 0.2},
                        "temporal": [], "is_sticker": True},
        }
        msg = module.Message(time=datetime.now(), user_name="u", content="[表情包|...]",
                             id="1", user_id="100", image_meta=meta)
        js = msg.to_json()
        self.assertEqual(meta, js["image_meta"])
        restored = module.Message.from_json(js)
        self.assertEqual(meta, restored.image_meta)

    def test_image_meta_default_none(self):
        module = _load_short_term()
        msg = module.Message(time=datetime.now(), user_name="u", content="hi", id="1", user_id="100")
        self.assertIsNone(msg.image_meta)
        js = msg.to_json()
        self.assertIsNone(js["image_meta"])
        restored = module.Message.from_json(js)
        self.assertIsNone(restored.image_meta)

    def test_old_json_without_image_meta_compatible(self):
        """旧数据缺 image_meta 字段，from_json 兼容为 None。"""
        module = _load_short_term()
        old = {
            "time": datetime.now().isoformat(),
            "user_name": "u", "content": "hi", "id": "1", "user_id": "100",
        }
        restored = module.Message.from_json(old)
        self.assertIsNone(restored.image_meta)
        self.assertEqual("hi", restored.content)


if __name__ == "__main__":
    unittest.main()
