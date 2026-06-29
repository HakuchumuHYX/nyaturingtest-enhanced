import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class MemoryWriteResilienceConfigTests(unittest.TestCase):
    def test_config_defines_retry_and_wal_keys(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        self.assertIn('"memory_write_max_retries": 3', source)
        self.assertIn('"memory_write_retry_base_delay": 0.5', source)
        self.assertIn(
            '"memory_write_max_retries": number("memory_write_max_retries", 3, int, minimum=0)',
            source,
        )

    def test_example_config_defines_retry_and_wal_keys(self):
        source = (PLUGIN_DIR / "config.example.json").read_text(encoding="utf-8")
        self.assertIn('"memory_write_max_retries": 3', source)
        self.assertIn('"memory_write_retry_base_delay": 0.5', source)


class AddTextsWALTests(unittest.TestCase):
    def test_failed_add_appends_to_wal(self):
        import json
        import os
        import tempfile

        from test_vector_batch import _load_vector_module

        module = _load_vector_module()
        mem = object.__new__(module.VectorMemory)
        tmp = tempfile.mkdtemp()
        mem.persist_directory = tmp

        class BoomCollection:
            def add(self, **kwargs):
                raise RuntimeError("embedding down")

        mem.collection = BoomCollection()

        result = mem.add_texts(["记住这件事"], metadatas=[{"source": "memory"}])

        wal = os.path.join(tmp, "pending_memories.jsonl")
        self.assertTrue(os.path.exists(wal))
        with open(wal, encoding="utf-8") as handle:
            line = json.loads(handle.readline())
        self.assertEqual("记住这件事", line["content"])
        self.assertEqual({"added": 0, "queued_wal": 1, "failed": 0}, result)

    def test_successful_add_returns_confirmed_write_result(self):
        import tempfile

        from test_vector_batch import _load_vector_module

        module = _load_vector_module()
        mem = object.__new__(module.VectorMemory)
        mem.persist_directory = tempfile.mkdtemp()
        added = []

        class OkCollection:
            def add(self, **kwargs):
                added.extend(kwargs["documents"])

        mem.collection = OkCollection()

        result = mem.add_texts(["确认写入的记忆"], metadatas=[{"source": "memory"}])

        self.assertEqual(["确认写入的记忆"], added)
        self.assertEqual({"added": 1, "queued_wal": 0, "failed": 0}, result)


class WALReplayTests(unittest.TestCase):
    def test_replay_reads_and_clears_wal(self):
        import json
        import os
        import tempfile

        from test_vector_batch import _load_vector_module

        module = _load_vector_module()
        mem = object.__new__(module.VectorMemory)
        tmp = tempfile.mkdtemp()
        mem.persist_directory = tmp
        added = []

        class OkCollection:
            def add(self, **kwargs):
                added.extend(kwargs["documents"])

        mem.collection = OkCollection()

        with open(os.path.join(tmp, "pending_memories.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"content": "旧的待写记忆", "metadata": {"source": "memory"}}) + "\n")

        mem.replay_pending()

        self.assertIn("旧的待写记忆", added)
        self.assertFalse(os.path.exists(os.path.join(tmp, "pending_memories.jsonl")))


class DedupWALTests(unittest.TestCase):
    def test_dedup_query_failure_appends_valid_memories_to_wal(self):
        import json
        import os
        import tempfile

        from test_vector_batch import _load_vector_module

        module = _load_vector_module()
        mem = object.__new__(module.VectorMemory)
        tmp = tempfile.mkdtemp()
        mem.persist_directory = tmp

        class FailingQueryCollection:
            def query(self, **kwargs):
                raise RuntimeError("embedding query down")

            def add(self, **kwargs):
                raise AssertionError("dedup failure should not write directly")

        mem.collection = FailingQueryCollection()

        result = mem.add_memories_with_dedup([
            ("Alice 讨厌香菜", {"source": "memory", "type": "preference"}),
        ])

        self.assertEqual(1, result["dedup_errors"])
        wal = os.path.join(tmp, "pending_memories.jsonl")
        self.assertTrue(os.path.exists(wal))
        with open(wal, encoding="utf-8") as handle:
            line = json.loads(handle.readline())
        self.assertEqual("Alice 讨厌香菜", line["content"])

    def test_dedup_add_failure_does_not_count_wal_as_added(self):
        import json
        import os
        import tempfile

        from test_vector_batch import _load_vector_module

        module = _load_vector_module()
        mem = object.__new__(module.VectorMemory)
        tmp = tempfile.mkdtemp()
        mem.persist_directory = tmp

        class FailingAddCollection:
            def query(self, **kwargs):
                return {"distances": [[]]}

            def add(self, **kwargs):
                raise RuntimeError("embedding add down")

        mem.collection = FailingAddCollection()

        result = mem.add_memories_with_dedup([
            ("Alice 讨厌香菜", {"source": "memory", "type": "preference"}),
        ])

        self.assertEqual(0, result["added"])
        self.assertEqual(0, result["dedup_errors"])
        wal = os.path.join(tmp, "pending_memories.jsonl")
        self.assertTrue(os.path.exists(wal))
        with open(wal, encoding="utf-8") as handle:
            line = json.loads(handle.readline())
        self.assertEqual("Alice 讨厌香菜", line["content"])


class DrainTimeoutTests(unittest.TestCase):
    def test_drain_timeout_configurable(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        self.assertIn('"memory_drain_timeout_seconds"', config_source)
        state_source = (PLUGIN_DIR / "core" / "state_manager.py").read_text(encoding="utf-8")
        self.assertIn("memory_drain_timeout_seconds", state_source)


if __name__ == "__main__":
    unittest.main()
