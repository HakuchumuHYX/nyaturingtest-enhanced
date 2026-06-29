import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from tortoise import Tortoise


PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "sqlite_repo_under_test"
_MISSING = object()


def _restore_modules(saved):
    for name, module in saved.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_repository_modules():
    module_names = [
        "nonebot",
        PACKAGE_ROOT,
        f"{PACKAGE_ROOT}.database",
        f"{PACKAGE_ROOT}.memory",
        f"{PACKAGE_ROOT}.models",
        f"{PACKAGE_ROOT}.config",
        f"{PACKAGE_ROOT}.memory.short_term",
        f"{PACKAGE_ROOT}.models.database",
        f"{PACKAGE_ROOT}.utils",
        f"{PACKAGE_ROOT}.database.message_repository",
        f"{PACKAGE_ROOT}.database.profile_repository",
        f"{PACKAGE_ROOT}.database.session_repository",
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)

        nonebot = types.ModuleType("nonebot")
        nonebot.logger = types.SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )
        sys.modules["nonebot"] = nonebot

        for package in [
            PACKAGE_ROOT,
            f"{PACKAGE_ROOT}.database",
            f"{PACKAGE_ROOT}.memory",
            f"{PACKAGE_ROOT}.models",
        ]:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module

        config = types.ModuleType(f"{PACKAGE_ROOT}.config")
        config.get_runtime_settings = lambda: {
            "interaction_log_recent_days": 180,
            "short_term_buffer_size": 200,
        }
        sys.modules[f"{PACKAGE_ROOT}.config"] = config

        modules = {}
        modules["short_term"] = _load_module(
            f"{PACKAGE_ROOT}.memory.short_term",
            "memory/short_term.py",
        )
        modules["models"] = _load_module(
            f"{PACKAGE_ROOT}.models.database",
            "models/database.py",
        )
        modules["utils"] = _load_module(
            f"{PACKAGE_ROOT}.utils",
            "utils.py",
        )
        modules["message_repository"] = _load_module(
            f"{PACKAGE_ROOT}.database.message_repository",
            "database/message_repository.py",
        )
        modules["profile_repository"] = _load_module(
            f"{PACKAGE_ROOT}.database.profile_repository",
            "database/profile_repository.py",
        )
        modules["session_repository"] = _load_module(
            f"{PACKAGE_ROOT}.database.session_repository",
            "database/session_repository.py",
        )
        return modules, saved
    except Exception:
        _restore_modules(saved)
        raise


class SQLiteRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if os.environ.get("NYATURINGTEST_RUN_SQLITE_INTEGRATION") != "1":
            self.skipTest("set NYATURINGTEST_RUN_SQLITE_INTEGRATION=1 to run real aiosqlite/Tortoise tests")
        self.modules, self.saved_modules = _install_repository_modules()
        await Tortoise.init(
            db_url="sqlite://:memory:",
            modules={"models": [f"{PACKAGE_ROOT}.models.database"]},
        )
        await Tortoise.generate_schemas()

    async def asyncTearDown(self):
        await Tortoise.close_connections()
        Tortoise.apps.clear()
        _restore_modules(self.saved_modules)

    async def test_message_sync_uses_final_ids_for_real_sqlite_dedup(self):
        models = self.modules["models"]
        message_repo = self.modules["message_repository"].MessageRepository
        Message = self.modules["short_term"].Message
        await models.SessionModel.create(id="group-1")
        timestamp = datetime(2026, 6, 29, 1, 0, 0)
        messages = [
            Message(time=timestamp, user_name="Alice", user_id="1001", content="same text"),
            Message(time=timestamp, user_name="Bob", user_id="1002", content="same text"),
        ]

        await message_repo.sync_messages("group-1", messages)
        await message_repo.sync_messages("group-1", messages)

        rows = await models.GlobalMessageModel.filter(session_id="group-1").order_by("user_id")
        self.assertEqual(2, len(rows))
        self.assertEqual(["1001", "1002"], [row.user_id for row in rows])
        self.assertEqual(2, len({row.msg_id for row in rows}))

    async def test_profile_update_missing_session_raises_on_real_sqlite(self):
        profile_repo = self.modules["profile_repository"].ProfileRepository
        profile = SimpleNamespace(
            emotion=SimpleNamespace(valence=0.1, arousal=0.2, dominance=0.3)
        )

        with self.assertRaises(RuntimeError):
            await profile_repo.update_user_profiles("missing-session", {"1001": profile})

    async def test_delete_session_data_clears_raw_rows_but_keeps_session(self):
        models = self.modules["models"]
        session_repo = self.modules["session_repository"].SessionStateRepository
        session = await models.SessionModel.create(id="group-1")
        profile = await models.UserProfileModel.create(session=session, user_id="1001")
        await models.InteractionLogModel.create(
            user=profile,
            timestamp=datetime(2026, 6, 29, 1, 0, 0),
            delta_valence=0.1,
            delta_arousal=0.2,
            delta_dominance=0.3,
        )
        await models.GlobalMessageModel.create(
            session=session,
            user_name="Alice",
            user_id="1001",
            content="hello",
            time=datetime(2026, 6, 29, 1, 1, 0),
            msg_id="msg-1",
        )

        await session_repo.delete_session_data("group-1")

        self.assertIsNotNone(await models.SessionModel.get_or_none(id="group-1"))
        self.assertEqual(0, await models.UserProfileModel.filter(session=session).count())
        self.assertEqual(0, await models.InteractionLogModel.all().count())
        self.assertEqual(0, await models.GlobalMessageModel.filter(session=session).count())


if __name__ == "__main__":
    unittest.main()
