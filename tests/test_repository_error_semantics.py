import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
STUB_MODULES = [
    "nonebot",
    "plugins",
    "plugins.nyaturingtest",
    "plugins.nyaturingtest.database",
    "plugins.nyaturingtest.memory",
    "plugins.nyaturingtest.models",
    "plugins.nyaturingtest.memory.short_term",
    "plugins.nyaturingtest.utils",
    "plugins.nyaturingtest.config",
    "plugins.nyaturingtest.models.database",
    "plugins.nyaturingtest.database.profile_repository",
    "plugins.nyaturingtest.database.message_repository",
    "plugins.nyaturingtest.database.session_repository",
]
_MISSING = object()


def _install_repository_stubs():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["nonebot"] = nonebot

    for package in [
        "plugins",
        "plugins.nyaturingtest",
        "plugins.nyaturingtest.database",
        "plugins.nyaturingtest.memory",
        "plugins.nyaturingtest.models",
    ]:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module

    short_term = types.ModuleType("plugins.nyaturingtest.memory.short_term")

    class Message:
        def __init__(self, time, user_name, content, id="", user_id=""):
            self.time = time
            self.user_name = user_name
            self.content = content
            self.id = id
            self.user_id = user_id

    short_term.Message = Message
    sys.modules["plugins.nyaturingtest.memory.short_term"] = short_term

    utils = types.ModuleType("plugins.nyaturingtest.utils")
    utils.sanitize_text = lambda value: str(value or "")
    sys.modules["plugins.nyaturingtest.utils"] = utils

    config = types.ModuleType("plugins.nyaturingtest.config")
    config.get_runtime_settings = lambda: {
        "interaction_log_recent_days": 30,
        "short_term_buffer_size": 200,
    }
    sys.modules["plugins.nyaturingtest.config"] = config

    models = types.ModuleType("plugins.nyaturingtest.models.database")

    class _RaisingSessionModel:
        @staticmethod
        async def get_or_none(id):
            raise RuntimeError("db unavailable")

        @staticmethod
        async def update_or_create(*args, **kwargs):
            raise RuntimeError("db unavailable")

    class _UnusedModel:
        pass

    models.SessionModel = _RaisingSessionModel
    models.GlobalMessageModel = _UnusedModel
    models.UserProfileModel = _UnusedModel
    models.InteractionLogModel = _UnusedModel
    sys.modules["plugins.nyaturingtest.models.database"] = models


def _load_repository(module_filename: str, module_name: str):
    modules_to_restore = list(dict.fromkeys([*STUB_MODULES, module_name]))
    saved = {name: sys.modules.get(name, _MISSING) for name in modules_to_restore}
    try:
        for name in modules_to_restore:
            sys.modules.pop(name, None)
        _install_repository_stubs()
        spec = importlib.util.spec_from_file_location(
            module_name,
            PLUGIN_DIR / "database" / module_filename,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class RepositoryErrorSemanticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_update_errors_propagate_to_caller(self):
        module = _load_repository(
            "profile_repository.py",
            "plugins.nyaturingtest.database.profile_repository",
        )

        with self.assertRaises(RuntimeError):
            await module.ProfileRepository.update_user_profiles("group-1", {"u": object()})

    async def test_message_sync_errors_propagate_to_caller(self):
        module = _load_repository(
            "message_repository.py",
            "plugins.nyaturingtest.database.message_repository",
        )

        with self.assertRaises(RuntimeError):
            await module.MessageRepository.sync_messages("group-1", [])

    async def test_session_state_save_errors_propagate_to_caller(self):
        module = _load_repository(
            "session_repository.py",
            "plugins.nyaturingtest.database.session_repository",
        )

        with self.assertRaises(RuntimeError):
            await module.SessionStateRepository.save_session_state("group-1", {})

    async def test_session_delete_errors_propagate_to_caller(self):
        module = _load_repository(
            "session_repository.py",
            "plugins.nyaturingtest.database.session_repository",
        )

        with self.assertRaises(RuntimeError):
            await module.SessionStateRepository.delete_session_data("group-1")


class MessageFinalIdTests(unittest.TestCase):
    def test_final_message_id_uses_existing_message_id(self):
        module = _load_repository(
            "message_repository.py",
            "plugins.nyaturingtest.database.message_repository",
        )
        msg = types.SimpleNamespace(id="qq-1", content="same", time=datetime.fromtimestamp(1), user_id="u1", user_name="A")

        self.assertEqual("qq-1", module.MessageRepository._message_final_id(msg))

    def test_no_id_fallback_includes_user_identity(self):
        module = _load_repository(
            "message_repository.py",
            "plugins.nyaturingtest.database.message_repository",
        )
        timestamp = datetime.fromtimestamp(1)
        first = types.SimpleNamespace(id="", content="same", time=timestamp, user_id="u1", user_name="A")
        second = types.SimpleNamespace(id="", content="same", time=timestamp, user_id="u2", user_name="B")

        self.assertNotEqual(
            module.MessageRepository._message_final_id(first),
            module.MessageRepository._message_final_id(second),
        )


class MessageSyncFinalIdQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_messages_queries_existing_rows_with_final_fallback_id(self):
        module = _load_repository(
            "message_repository.py",
            "plugins.nyaturingtest.database.message_repository",
        )
        timestamp = datetime.fromtimestamp(1)
        msg = types.SimpleNamespace(id="", content="same", time=timestamp, user_id="u1", user_name="A")
        final_id = module.MessageRepository._message_final_id(msg)

        class FakeSessionModel:
            @staticmethod
            async def get_or_none(id):
                return object()

        class FakeQuery:
            async def values_list(self, *args, **kwargs):
                return [final_id]

        class FakeGlobalMessageModel:
            queried_ids = None
            bulk_create_called = False

            @classmethod
            def filter(cls, **kwargs):
                cls.queried_ids = list(kwargs["msg_id__in"])
                return FakeQuery()

            @classmethod
            async def bulk_create(cls, rows):
                cls.bulk_create_called = True

        module.SessionModel = FakeSessionModel
        module.GlobalMessageModel = FakeGlobalMessageModel

        await module.MessageRepository.sync_messages("group-1", [msg])

        self.assertEqual([final_id], FakeGlobalMessageModel.queried_ids)
        self.assertFalse(FakeGlobalMessageModel.bulk_create_called)


if __name__ == "__main__":
    unittest.main()
