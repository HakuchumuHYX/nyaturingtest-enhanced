import importlib.util
import json
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
DYNAMIC_MARKER = "---- DYNAMIC INPUT ----"


def _load_templates_module():
    spec = importlib.util.spec_from_file_location(
        "nyaturingtest_prompt_templates_image_meta",
        PLUGIN_DIR / "prompts" / "templates.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_payload(prompt_text: str) -> dict:
    """从 prompt 文本里抠出 DYNAMIC INPUT 后的 JSON payload。"""
    idx = prompt_text.index(DYNAMIC_MARKER) + len(DYNAMIC_MARKER)
    return json.loads(prompt_text[idx:].strip())


class PromptPayloadImageMetaTests(unittest.TestCase):
    def setUp(self):
        self.tm = _load_templates_module()

    def test_feedback_payload_carries_image_meta(self):
        meta = {
            "primary": {"entities": [{"name": "初音未来", "type": "character", "confidence": 0.85}],
                        "ocr_text": "我装的", "pragmatic_intent": "否认",
                        "affect": {"valence": 0.7, "arousal": 0.5, "dominance": 0.2},
                        "temporal": [], "is_sticker": True},
        }
        new_msgs = [
            {"id": "100", "name": "Alice", "content": "你看这个", "image_meta": None},
            {"id": "101", "name": "Bob", "content": "[表情包|实体:初音未来(0.85)|...]", "image_meta": meta},
        ]
        prompt = self.tm.get_feedback_prompt(
            bot_name="Nya", role="r", willingness=0.5, chat_state_value=1,
            history_summary="", recent_msgs=[], new_msgs_formatted=new_msgs,
            emotion={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            related_profiles_json="[]", search_result=[], last_summary="",
            is_relevant=False, time_info="",
        )
        payload = _extract_payload(prompt)
        self.assertEqual(2, len(payload["new_msgs"]))
        self.assertIsNone(payload["new_msgs"][0]["image_meta"])
        self.assertEqual(meta, payload["new_msgs"][1]["image_meta"])

    def test_feedback_payload_image_meta_none_does_not_break(self):
        new_msgs = [{"id": "1", "name": "A", "content": "hi", "image_meta": None}]
        prompt = self.tm.get_feedback_prompt(
            bot_name="Nya", role="r", willingness=0.5, chat_state_value=1,
            history_summary="", recent_msgs=[], new_msgs_formatted=new_msgs,
            emotion={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            related_profiles_json="[]", search_result=[], last_summary="",
        )
        payload = _extract_payload(prompt)
        self.assertIsNone(payload["new_msgs"][0]["image_meta"])

    def test_chat_payload_carries_image_meta(self):
        meta = {"primary": {"entities": [{"name": "X", "type": "meme", "confidence": 0.6}],
                            "ocr_text": "", "pragmatic_intent": "嘲讽",
                            "affect": {"valence": -0.2, "arousal": 0.4, "dominance": 0.1},
                            "temporal": [], "is_sticker": True}}
        new_msgs = [{"id": "1", "name": "A", "content": "[表情包|...]", "image_meta": meta}]
        prompt = self.tm.get_chat_prompt(
            bot_name="Nya", role="r", chat_state_value=1,
            history_summary="", recent_msgs=[], new_msgs_formatted=new_msgs,
            emotion={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            related_profiles_json="[]", search_result=[], chat_summary="",
        )
        payload = _extract_payload(prompt)
        self.assertEqual(meta, payload["new_msgs"][0]["image_meta"])

    def test_feedback_prompt_mentions_image_meta_schema(self):
        prompt = self.tm.get_feedback_prompt(
            bot_name="Nya", role="r", willingness=0.5, chat_state_value=1,
            history_summary="", recent_msgs=[], new_msgs_formatted=[],
            emotion={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            related_profiles_json="[]", search_result=[], last_summary="",
        )
        self.assertIn("image_meta", prompt)
        self.assertIn("affect", prompt)

    def test_chat_prompt_mentions_image_meta(self):
        prompt = self.tm.get_chat_prompt(
            bot_name="Nya", role="r", chat_state_value=1,
            history_summary="", recent_msgs=[], new_msgs_formatted=[],
            emotion={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            related_profiles_json="[]", search_result=[], chat_summary="",
        )
        self.assertIn("image_meta", prompt)
        self.assertIn("entities", prompt)


if __name__ == "__main__":
    unittest.main()
