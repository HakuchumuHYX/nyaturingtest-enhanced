import unittest

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


if __name__ == "__main__":
    unittest.main()
