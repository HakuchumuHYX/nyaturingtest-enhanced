import importlib.util
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_templates_module():
    spec = importlib.util.spec_from_file_location(
        "nyaturingtest_prompt_templates_actions",
        PLUGIN_DIR / "prompts" / "templates.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FeedbackMemoryActionTests(unittest.TestCase):
    def test_feedback_memory_action_schema_includes_subject_and_speaker_fields(self):
        source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")

        self.assertIn('"subject_user_id"', source)
        self.assertIn('"subject_user_name"', source)
        self.assertIn('"speaker_user_id"', source)
        self.assertIn('"speaker_user_name"', source)
        self.assertIn('"related_user_id"', source)

    def test_feedback_prompt_dynamic_payload_includes_new_msg_speakers(self):
        source = (PLUGIN_DIR / "prompts" / "templates.py").read_text(encoding="utf-8")

        self.assertIn('"new_msg_speakers"', source)
        self.assertIn("- new_msg_speakers:", source)

    def test_feedback_prompt_includes_supersede_schema_when_memory_refs_are_supported(self):
        templates = _load_templates_module()

        prompt = templates.get_feedback_prompt(
            "bot",
            "role",
            0.5,
            1,
            "summary",
            [],
            ["[ID:100] Alice: '我现在喜欢茶了'"],
            {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            "[]",
            ["memory"],
            "last",
            existing_related_memories=[
                {
                    "memory_ref": "mem-1",
                    "content_preview": "Alice 以前喜欢咖啡",
                    "source": "memory",
                    "type": "preference",
                    "subtype": "preference",
                    "category": "preference",
                    "confidence": 0.7,
                }
            ],
            allow_memory_supersede=True,
        )

        self.assertIn("existing_related_memories", prompt)
        self.assertIn('"memory_ref":"mem-1"', prompt)
        self.assertIn('"action":"supersede"', prompt)
        self.assertIn('"target_ref"', prompt)
        self.assertIn('"action":"add"', prompt)
        self.assertIn('"action":"ignore"', prompt)

    def test_feedback_prompt_omits_supersede_schema_and_refs_when_ids_are_not_supported(self):
        templates = _load_templates_module()

        prompt = templates.get_feedback_prompt(
            "bot",
            "role",
            0.5,
            1,
            "summary",
            [],
            ["[ID:100] Alice: '我现在喜欢茶了'"],
            {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            "[]",
            ["memory"],
            "last",
            existing_related_memories=[
                {
                    "memory_ref": "mem-1",
                    "content_preview": "Alice 以前喜欢咖啡",
                    "source": "memory",
                    "type": "preference",
                    "subtype": "preference",
                    "category": "preference",
                    "confidence": 0.7,
                }
            ],
            allow_memory_supersede=False,
        )

        self.assertIn("existing_related_memories", prompt)
        self.assertNotIn('"memory_ref":"mem-1"', prompt)
        self.assertNotIn('"action":"supersede"', prompt)
        self.assertNotIn('"target_ref"', prompt)
        self.assertIn('"action":"add"', prompt)
        self.assertIn('"action":"ignore"', prompt)

    def test_session_builds_existing_related_memories_for_feedback_prompt(self):
        session_source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")

        self.assertIn("def _existing_related_memories", session_source)
        self.assertIn("content_preview", session_source)
        self.assertIn("memory_ref", session_source)
        self.assertIn("ids_supported", session_source)
        self.assertIn("existing_related_memories=existing_related_memories", session_source)
        self.assertIn("allow_memory_supersede=allow_memory_supersede", session_source)


if __name__ == "__main__":
    unittest.main()
