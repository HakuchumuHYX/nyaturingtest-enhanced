import unittest
from datetime import datetime

from test_vector_batch import _load_vector_module


class VectorSubjectSpeakerScopeTests(unittest.TestCase):
    def test_normalized_metadata_maps_legacy_user_id_to_subject_alias(self):
        module = _load_vector_module()

        meta = module._normalized_metadata({
            "source": "memory",
            "type": "event",
            "user_id": "10001",
        })

        self.assertEqual("10001", meta["user_id"])
        self.assertEqual("10001", meta["subject_user_id"])
        self.assertEqual("", meta["subject_user_name"])
        self.assertEqual("", meta["speaker_user_id"])
        self.assertEqual("", meta["speaker_user_name"])
        self.assertEqual(1, meta["schema_version"])

    def test_normalized_metadata_keeps_user_id_as_subject_alias_for_v2_records(self):
        module = _load_vector_module()

        meta = module._normalized_metadata({
            "schema_version": 2,
            "source": "memory",
            "type": "event",
            "user_id": "wrong-legacy-value",
            "subject_user_id": "10001",
            "subject_user_name": "A",
            "speaker_user_id": "20002",
            "speaker_user_name": "B",
        })

        self.assertEqual(2, meta["schema_version"])
        self.assertEqual("10001", meta["user_id"])
        self.assertEqual("10001", meta["subject_user_id"])
        self.assertEqual("A", meta["subject_user_name"])
        self.assertEqual("20002", meta["speaker_user_id"])
        self.assertEqual("B", meta["speaker_user_name"])

    def test_mentioned_subject_survives_when_subject_is_not_active_speaker(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [{
                "content": "B 说小明最近在准备考试",
                "metadata": {
                    "schema_version": 2,
                    "source": "memory",
                    "type": "event",
                    "user_id": "10001",
                    "subject_user_id": "10001",
                    "subject_user_name": "小明",
                    "speaker_user_id": "20002",
                    "speaker_user_name": "B",
                    "date": today,
                    "retrieval_score": 0.80,
                },
            }]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(
            ["C 和 D 在讨论小明的考试"],
            k=3,
            use_rerank=False,
            decay_rate=0,
            active_user_ids={"30003", "40004"},
        )

        self.assertEqual(["B 说小明最近在准备考试"], [item["content"] for item in result])
        self.assertEqual("mentioned_subject", result[0]["metadata"]["scope"])
        self.assertGreater(result[0]["metadata"]["scope_weight"], 1.0)

    def test_single_character_subject_name_does_not_get_mentioned_subject_boost(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [{
                "content": "B 说 A 最近在准备考试",
                "metadata": {
                    "schema_version": 2,
                    "source": "memory",
                    "type": "event",
                    "subject_user_id": "10001",
                    "subject_user_name": "A",
                    "date": today,
                    "retrieval_score": 0.80,
                },
            }]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(
            ["C 和 D 在讨论 A 的考试"],
            k=3,
            use_rerank=False,
            decay_rate=0,
            active_user_ids={"30003", "40004"},
        )

        self.assertEqual(["B 说 A 最近在准备考试"], [item["content"] for item in result])
        self.assertEqual("other_subject", result[0]["metadata"]["scope"])
        self.assertEqual(0.5, result[0]["metadata"]["scope_weight"])

    def test_other_subject_is_downweighted_not_filtered(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [{
                "content": "E 喜欢夜跑",
                "metadata": {
                    "schema_version": 2,
                    "source": "memory",
                    "type": "event",
                    "subject_user_id": "50005",
                    "subject_user_name": "E",
                    "date": today,
                    "retrieval_score": 0.80,
                },
            }]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(
            ["C 和 D 在讨论 A"],
            k=3,
            use_rerank=False,
            decay_rate=0,
            active_user_ids={"30003", "40004"},
        )

        self.assertEqual(["E 喜欢夜跑"], [item["content"] for item in result])
        self.assertEqual("other_subject", result[0]["metadata"]["scope"])
        self.assertEqual(0.5, result[0]["metadata"]["scope_weight"])
        self.assertEqual(0, memory.last_retrieval_stats["other_user_filtered_count"])
        self.assertEqual(1, memory.last_retrieval_stats["other_subject_downweighted_count"])

    def test_legacy_subject_without_subject_name_uses_legacy_subject_weight(self):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        today = int(datetime.now().strftime("%Y%m%d"))

        def fake_retrieve(queries, k=5, where=None, use_rerank=True):
            return [{
                "content": "A 最近在准备考试",
                "metadata": {
                    "source": "memory",
                    "type": "event",
                    "user_id": "10001",
                    "date": today,
                    "retrieval_score": 0.80,
                },
            }]

        memory.retrieve = fake_retrieve

        result = memory.retrieve_with_decay(
            ["C 和 D 在讨论 A 的考试"],
            k=3,
            use_rerank=False,
            decay_rate=0,
            active_user_ids={"30003", "40004"},
        )

        self.assertEqual(["A 最近在准备考试"], [item["content"] for item in result])
        self.assertEqual("legacy_subject", result[0]["metadata"]["scope"])
        self.assertEqual(0.75, result[0]["metadata"]["scope_weight"])


if __name__ == "__main__":
    unittest.main()
