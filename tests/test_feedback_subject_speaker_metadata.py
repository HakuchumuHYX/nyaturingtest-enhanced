import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class FeedbackSubjectSpeakerMetadataTests(unittest.TestCase):
    def test_existing_related_memories_filters_by_subject_user_id(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        helper_block = source[
            source.index("def _existing_related_memories"):
            source.index("class _ChattingState")
        ]

        self.assertIn("subject_user_id", helper_block)
        self.assertIn('meta.get("subject_user_id") or meta.get("user_id")', helper_block)

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
            source.index("async def _run_feedback_llm"):
            source.index("def _apply_sediment")
        ]

        self.assertIn("new_msg_speakers", feedback_block)
        self.assertIn('"user_id": str(msg.user_id or "")', feedback_block)
        self.assertIn('"user_name": msg.user_name', feedback_block)

    def test_query_memory_keeps_user_id_as_subject_filter(self):
        source = (PLUGIN_DIR / "handlers" / "memory.py").read_text(encoding="utf-8")
        query_block = source[source.index("@query_memory.handle()"):]

        self.assertIn('{"user_id": {"$eq": target_id}}', query_block)
        self.assertNotIn('{"speaker_user_id": {"$eq": target_id}}', query_block)


if __name__ == "__main__":
    unittest.main()
