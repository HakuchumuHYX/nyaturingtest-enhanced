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

        mem.add_texts(["记住这件事"], metadatas=[{"source": "memory"}])

        wal = os.path.join(tmp, "pending_memories.jsonl")
        self.assertTrue(os.path.exists(wal))
        with open(wal, encoding="utf-8") as handle:
            line = json.loads(handle.readline())
        self.assertEqual("记住这件事", line["content"])


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


class DrainTimeoutTests(unittest.TestCase):
    def test_drain_timeout_configurable(self):
        config_source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        self.assertIn('"memory_drain_timeout_seconds"', config_source)
        state_source = (PLUGIN_DIR / "core" / "state_manager.py").read_text(encoding="utf-8")
        self.assertIn("memory_drain_timeout_seconds", state_source)


if __name__ == "__main__":
    unittest.main()
