import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class FeedbackSubjectSpeakerMetadataTests(unittest.TestCase):
    def test_save_long_term_memory_writes_subject_speaker_metadata_fields(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        save_block = source[
            source.index("async def save_long_term_memory"):
            source.index("async def chat_stage")
        ]

        self.assertIn('item.get("subject_user_id"', save_block)
        self.assertIn('item.get("subject_user_name"', save_block)
        self.assertIn('item.get("speaker_user_id"', save_block)
        self.assertIn('item.get("speaker_user_name"', save_block)
        self.assertIn('"schema_version": 2', save_block)
        self.assertIn('"subject_user_id": subject_user_id', save_block)
        self.assertIn('"speaker_user_id": speaker_user_id', save_block)
        self.assertIn('"user_id": subject_user_id', save_block)

    def test_feedback_stage_passes_new_msg_speakers_to_prompt(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        feedback_block = source[
            source.index("async def feedback_stage"):
            source.index("async def save_long_term_memory")
        ]

        self.assertIn("new_msg_speakers", feedback_block)
        self.assertIn('"user_id": str(msg.user_id or "")', feedback_block)
        self.assertIn('"user_name": msg.user_name', feedback_block)


if __name__ == "__main__":
    unittest.main()
