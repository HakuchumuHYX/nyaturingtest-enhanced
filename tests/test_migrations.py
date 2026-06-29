import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class _FakeConnection:
    def __init__(self, *, current_version=1, failures=None):
        self.current_version = current_version
        self.failures = failures or {}
        self.executed = []

    async def execute_query(self, statement):
        self.executed.append(statement)
        for pattern, exc in self.failures.items():
            if pattern in statement:
                raise exc
        if "SET version=2" in statement:
            self.current_version = 2
        elif "SET version=3" in statement:
            self.current_version = 3
        elif "VALUES (1, 1)" in statement:
            self.current_version = 1
        return None

    async def execute_query_dict(self, statement):
        if self.current_version == 0:
            return []
        return [{"version": self.current_version}]


def _load_migrations_module(connection):
    previous = {}

    def install(name, module):
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module

    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    install("nonebot", nonebot)

    tortoise = types.ModuleType("tortoise")
    tortoise.Tortoise = types.SimpleNamespace(get_connection=lambda name: connection)
    install("tortoise", tortoise)

    module_name = "migrations_under_test"
    previous[module_name] = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            PLUGIN_DIR / "database" / "migrations.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class MigrationTests(unittest.TestCase):
    def test_schema_version_migration_exists(self):
        source = (PLUGIN_DIR / "database" / "migrations.py").read_text(encoding="utf-8")

        self.assertIn("nyabot_schema_version", source)
        self.assertIn("SCHEMA_VERSION = 3", source)
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_session_msg_id", source)

    def test_non_duplicate_v2_failure_raises_and_does_not_advance_version(self):
        conn = _FakeConnection(
            current_version=1,
            failures={
                "ADD COLUMN provider": RuntimeError("disk full"),
            },
        )
        module = _load_migrations_module(conn)

        with self.assertRaises(RuntimeError):
            asyncio.run(module.ensure_schema_version())

        self.assertEqual(1, conn.current_version)
        self.assertFalse(any("SET version=2" in statement for statement in conn.executed))

    def test_duplicate_column_is_ignored_and_version_advances(self):
        conn = _FakeConnection(
            current_version=1,
            failures={
                "ADD COLUMN provider": RuntimeError("duplicate column name: provider"),
                "CREATE UNIQUE INDEX": RuntimeError("index uq_messages_session_msg_id already exists"),
            },
        )
        module = _load_migrations_module(conn)

        asyncio.run(module.ensure_schema_version())

        self.assertEqual(3, conn.current_version)
        self.assertTrue(any("SET version=2" in statement for statement in conn.executed))
        self.assertTrue(any("SET version=3" in statement for statement in conn.executed))

    def test_non_duplicate_v3_failure_raises_and_does_not_advance_version(self):
        conn = _FakeConnection(
            current_version=2,
            failures={
                "last_consolidated_time": RuntimeError("database is locked"),
            },
        )
        module = _load_migrations_module(conn)

        with self.assertRaises(RuntimeError):
            asyncio.run(module.ensure_schema_version())

        self.assertEqual(2, conn.current_version)
        self.assertFalse(any("SET version=3" in statement for statement in conn.executed))


if __name__ == "__main__":
    unittest.main()
