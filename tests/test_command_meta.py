import importlib.util
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_command_meta():
    spec = importlib.util.spec_from_file_location(
        "command_meta",
        PLUGIN_DIR / "handlers" / "command_meta.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CommandMetaTests(unittest.TestCase):
    def test_help_is_generated_from_command_metadata(self):
        module = _load_command_meta()

        self.assertIn("reset confirm", module.render_group_help())
        self.assertIn("set_role <群号> <角色名> <角色设定>", module.render_private_help())

    def test_private_group_id_parsing_has_value_error_guard(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")

        self.assertIn("async def _parse_group_id_or_finish", source)
        self.assertIn("except ValueError", source)
        for unsafe in [
            "int(arg)",
            "int(preset_args[0])",
            "int(role_args[0])",
            "int(parts[0])",
        ]:
            self.assertNotIn(unsafe, source)


if __name__ == "__main__":
    unittest.main()
