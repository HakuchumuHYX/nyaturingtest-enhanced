import importlib.util
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_rag_query():
    spec = importlib.util.spec_from_file_location(
        "nyaturingtest_rag_query_image_format", PLUGIN_DIR / "core" / "rag_query.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RagQueryImageFormatTests(unittest.TestCase):
    """固化渲染格式变更对 RAG query 过滤的影响（计划 2.11 决策：带语义标签放行）。"""

    def setUp(self):
        self.mod = _load_rag_query()
        self.is_low = self.mod.is_low_value_rag_query

    def test_pure_placeholder_filtered(self):
        """resolve_images=False 时纯占位 [表情包] 被 startswith 规则过滤。
        注意：纯 [图片] 在既有代码里本就不被特殊过滤（无 startswith("[图片]") 规则），
        这是改造前就存在的行为，本计划不改变它。"""
        self.assertTrue(self.is_low("[表情包]"))
        # [图片] 既有行为：不被过滤（长 4 字符、非 emoji-only、不在 noise set）
        self.assertFalse(self.is_low("[图片]"))

    def test_semantic_sticker_label_passes(self):
        """带语义的管道标签 [表情包|...] 不被过滤，参与 RAG 检索。"""
        label = "[表情包|实体:初音未来(0.85)|配字:我装的|意图:否认|情感:V0.70,A0.50,D0.20|画面:绿发女孩比耶]"
        self.assertFalse(self.is_low(label))

    def test_semantic_image_label_passes(self):
        label = "[图片|实体:|配字:无|意图:卖萌|情感:V0.60,A0.40,D0.10|画面:橘猫趴在键盘上]"
        self.assertFalse(self.is_low(label))

    def test_short_empty_label_filtered(self):
        """极短的管道标签（<4 字）仍被 len<4 过滤。"""
        self.assertTrue(self.is_low("[图"))

    def test_build_queries_keeps_semantic_label(self):
        """端到端：带语义标签进入 effective_queries。"""
        queries = self.mod.build_chat_rag_queries(
            ["[表情包|实体:猫|配字:无|意图:卖萌|情感:V0.5,A0.4,D0.1|画面:一只猫]",
             "[表情包]",
             "我在聊宠物"],
            chat_summary="旧话题",
            active_user_names=[],
            active_users=[],
        )
        # 带语义标签保留，纯 [表情包] 被过滤
        self.assertTrue(any("一只猫" in q for q in queries))
        self.assertNotIn("[表情包]", queries)

    def test_build_queries_drops_pure_placeholder(self):
        """纯 [表情包] 被过滤；纯 [图片] 既有行为不被过滤（本计划未改变此既有行为）。"""
        queries = self.mod.build_chat_rag_queries(
            ["[图片]", "[表情包]"],
            chat_summary="旧话题",
            active_user_names=[],
            active_users=[],
        )
        self.assertNotIn("[表情包]", queries)
        # [图片] 既有行为：保留（未被过滤）
        self.assertIn("[图片]", queries)


if __name__ == "__main__":
    unittest.main()
