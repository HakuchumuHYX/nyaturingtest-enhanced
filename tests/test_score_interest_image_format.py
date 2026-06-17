import importlib.util
import sys
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_utils():
    # utils.py 用相对 import (from .memory.short_term import Message)，需以包名加载并 stub 依赖
    nonebot = sys.modules.get("nonebot") or types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None, debug=lambda *a, **k: None,
    )
    sys.modules.setdefault("nonebot", nonebot)

    short_term = types.ModuleType("memory.short_term")

    class Message:
        def __init__(self, content, user_name="u", user_id="", time=None, id=""):
            self.content = content
            self.user_name = user_name
            self.user_id = user_id
            self.time = time
            self.id = id

    short_term.Message = Message
    sys.modules.setdefault("nyaturingtest_utils_score.memory.short_term", short_term)
    # 同时注册无包前缀版本，兼容 utils.py 内其他可能的解析
    pkg = types.ModuleType("nyaturingtest_utils_score")
    pkg.memory = types.ModuleType("nyaturingtest_utils_score.memory")
    pkg.memory.short_term = short_term
    sys.modules.setdefault("nyaturingtest_utils_score", pkg)
    sys.modules.setdefault("nyaturingtest_utils_score.memory", pkg.memory)

    spec = importlib.util.spec_from_file_location(
        "nyaturingtest_utils_score.utils",
        PLUGIN_DIR / "utils.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    # 确保父包已注册，使 utils.py 内的相对 import (from .memory.short_term) 能解析
    parent = types.ModuleType("nyaturingtest_utils_score")
    parent.__path__ = [str(PLUGIN_DIR)]
    sys.modules.setdefault("nyaturingtest_utils_score", parent)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nyaturingtest_utils_score.utils"] = module
    spec.loader.exec_module(module)
    return module


class ScoreInterestImageFormatTests(unittest.TestCase):
    """固化渲染格式变更对意愿系统的影响（计划 2.10 决策：带语义标签不惩罚）。"""

    def setUp(self):
        self.utils = _load_utils()
        self.score = self.utils.score_message_interest

    def test_pure_placeholder_still_penalized(self):
        """resolve_images=False 时 _image_placeholder 产出的纯占位仍被 -0.5 惩罚。"""
        s_placeholder = self.score(["[表情包]"])
        s_normal = self.score(["普通的一句话消息"])
        # 纯 [表情包] 命中 -0.5；普通消息不命中
        self.assertLess(s_placeholder, s_normal)

    def test_pure_image_placeholder_penalized(self):
        s = self.score(["[图片]"])
        # [图片] 纯占位：1.0 - 0.5 = 0.5（无其他加减）
        self.assertEqual(0.5, s)

    def test_semantic_sticker_label_not_penalized(self):
        """带语义的管道标签 [表情包|...] 不命中纯占位惩罚。"""
        label = "[表情包|实体:初音未来(0.85)|配字:我装的|意图:否认|情感:V0.70,A0.50,D0.20|画面:绿发女孩比耶]"
        s = self.score([label])
        # 不命中 -0.5；len>=15 命中 +0.2 -> 1.2
        self.assertEqual(1.2, s)
        # 明显高于纯占位
        self.assertGreater(s, self.score(["[表情包]"]))

    def test_semantic_image_label_not_penalized(self):
        label = "[图片|实体:|配字:无|意图:卖萌|情感:V0.60,A0.40,D0.10|画面:橘猫趴在键盘上]"
        s = self.score([label])
        self.assertEqual(1.2, s)

    def test_empty_entities_label_still_treated_as_content(self):
        """即使实体/配字为空，管道标签仍因画面描述有内容而不被当纯占位惩罚。"""
        label = "[表情包|实体:|配字:无|意图:无|情感:V0.00,A0.00,D0.00|画面:一个圆形的白色物体]"
        s = self.score([label])
        self.assertGreater(s, self.score(["[表情包]"]))

    def test_question_mark_in_label_still_adds(self):
        """管道标签里若含问号仍触发 +0.6（与文本消息一致）。"""
        label = "[图片|实体:|配字:这是啥？|意图:求助|情感:V0.10,A0.30,D0.00|画面:一个奇怪的东西]"
        s = self.score([label])
        # +0.6(问号) +0.2(len>=15) = 1.8
        self.assertEqual(1.8, s)


if __name__ == "__main__":
    unittest.main()
