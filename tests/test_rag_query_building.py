import unittest
from pathlib import Path

from core.rag_query import build_chat_rag_queries


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class RagQueryBuildingTests(unittest.TestCase):
    def test_chat_rag_queries_drop_short_reactions_and_expression_markers(self):
        queries = build_chat_rag_queries(
            ["？", "[表情包]", "我现在喜欢喝乌龙茶"],
            chat_summary="旧话题摘要",
            active_user_names=[],
            active_users=[],
        )

        self.assertEqual(["我现在喜欢喝乌龙茶", "旧话题摘要"], queries[:2])
        self.assertNotIn("？", queries)
        self.assertNotIn("[表情包]", queries)

    def test_chat_rag_queries_keep_effective_message_before_summary_and_names(self):
        queries = build_chat_rag_queries(
            ["哈哈哈", "明天我要去上海考试"],
            chat_summary="群友在闲聊考试安排",
            active_user_names=["Alice"],
            active_users=[{"user_id": "100", "user_name": "Alice"}],
        )

        self.assertEqual(["明天我要去上海考试", "群友在闲聊考试安排", "关于Alice"], queries)

    def test_chat_rag_queries_keep_latest_two_effective_messages_in_recent_order(self):
        queries = build_chat_rag_queries(
            ["昨天我买了新键盘", "？", "这个键盘是静音轴"],
            chat_summary="群友在聊外设",
            active_user_names=[],
            active_users=[],
        )

        self.assertEqual(["昨天我买了新键盘", "这个键盘是静音轴", "群友在聊外设"], queries)

    def test_session_search_stage_uses_chat_rag_query_builder(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("from .rag_query import build_chat_rag_queries", session_source)
        search_stage_source = session_source[
            session_source.index("async def search_stage"):
            session_source.index("async def feedback_stage")
        ]
        self.assertIn("build_chat_rag_queries(", search_stage_source)


if __name__ == "__main__":
    unittest.main()
