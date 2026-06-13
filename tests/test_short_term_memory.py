import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_short_term():
    nonebot = sys.modules.get("nonebot") or types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None, debug=lambda *a, **k: None,
    )
    sys.modules["nonebot"] = nonebot
    spec = importlib.util.spec_from_file_location(
        "short_term_under_test", PLUGIN_DIR / "memory" / "short_term.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShortTermMemoryTests(unittest.TestCase):
    def _msgs(self, module, n):
        return [
            module.Message(time=datetime.now(), user_name="u", content=str(i), id=str(i))
            for i in range(n)
        ]

    def test_snapshot_returns_all_buffered_messages(self):
        module = _load_short_term()
        mem = module.Memory(context_limit=20, buffer_size=200)

        import asyncio
        asyncio.get_event_loop().run_until_complete(mem.update(self._msgs(module, 60)))

        snap = mem.snapshot()
        self.assertEqual(60, len(snap))
        self.assertEqual([str(i) for i in range(60)], [m.id for m in snap])

    def test_buffer_size_caps_retention(self):
        module = _load_short_term()
        mem = module.Memory(context_limit=20, buffer_size=50)

        import asyncio
        asyncio.get_event_loop().run_until_complete(mem.update(self._msgs(module, 80)))

        self.assertEqual(50, len(mem.snapshot()))


if __name__ == "__main__":
    unittest.main()
