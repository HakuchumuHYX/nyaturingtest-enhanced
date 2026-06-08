import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class _FakeMatcher:
    def handle(self):
        def decorator(func):
            return func

        return decorator

    async def send(self, *args, **kwargs):
        return None

    async def finish(self, *args, **kwargs):
        return None


def _install_stub_modules(package_name: str):
    previous = {}

    def install(name: str, module):
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module

    nonebot = types.ModuleType("nonebot")
    nonebot.on_command = lambda *args, **kwargs: _FakeMatcher()
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    install("nonebot", nonebot)

    onebot = types.ModuleType("nonebot.adapters.onebot.v11")
    onebot.Bot = object
    onebot.Event = object
    onebot.GroupMessageEvent = type("GroupMessageEvent", (), {})
    onebot.Message = list
    install("nonebot.adapters", types.ModuleType("nonebot.adapters"))
    install("nonebot.adapters.onebot", types.ModuleType("nonebot.adapters.onebot"))
    install("nonebot.adapters.onebot.v11", onebot)

    params = types.ModuleType("nonebot.params")
    params.CommandArg = lambda: None
    install("nonebot.params", params)

    permission = types.ModuleType("nonebot.permission")
    permission.SUPERUSER = object()
    install("nonebot.permission", permission)

    utils = types.ModuleType("nonebot.utils")
    utils.run_sync = lambda func: func
    install("nonebot.utils", utils)

    exception = types.ModuleType("nonebot.exception")
    exception.FinishedException = type("FinishedException", (Exception,), {})
    install("nonebot.exception", exception)

    root = types.ModuleType(package_name)
    root.__path__ = [str(PLUGIN_DIR)]
    install(package_name, root)

    handlers = types.ModuleType(f"{package_name}.handlers")
    handlers.__path__ = [str(PLUGIN_DIR / "handlers")]
    install(f"{package_name}.handlers", handlers)

    for subpackage in ["core", "database", "memory"]:
        module = types.ModuleType(f"{package_name}.{subpackage}")
        module.__path__ = [str(PLUGIN_DIR / subpackage)]
        install(f"{package_name}.{subpackage}", module)

    state_manager = types.ModuleType(f"{package_name}.core.state_manager")
    state_manager.ensure_group_state = lambda *args, **kwargs: None
    install(f"{package_name}.core.state_manager", state_manager)

    package_utils = types.ModuleType(f"{package_name}.utils")
    package_utils.extract_and_parse_json = lambda text: json.loads(text)
    package_utils.calculate_dynamic_k = lambda *args, **kwargs: 1
    package_utils.should_store_memory = lambda content: True
    install(f"{package_name}.utils", package_utils)

    message_repository = types.ModuleType(f"{package_name}.database.message_repository")
    message_repository.MessageRepository = object
    install(f"{package_name}.database.message_repository", message_repository)

    profile_repository = types.ModuleType(f"{package_name}.database.profile_repository")
    profile_repository.ProfileRepository = object
    install(f"{package_name}.database.profile_repository", profile_repository)

    logic = types.ModuleType(f"{package_name}.core.logic")

    async def missing_llm_response(*args, **kwargs):
        raise AssertionError("test must patch llm_response")

    logic.llm_response = missing_llm_response
    install(f"{package_name}.core.logic", logic)

    usage = types.ModuleType(f"{package_name}.core.usage")
    usage.make_usage_recorder = lambda *args, **kwargs: (lambda usage: None)
    install(f"{package_name}.core.usage", usage)

    services = types.ModuleType(f"{package_name}.core.services")
    services.RagSearchService = object
    install(f"{package_name}.core.services", services)

    vector = types.ModuleType(f"{package_name}.memory.vector")
    vector.where_any = lambda field, values: {"$or": [{field: {"$eq": value}} for value in values]}
    install(f"{package_name}.memory.vector", vector)

    config = types.ModuleType(f"{package_name}.config")
    config.get_effective_chat_model = lambda: "chat-model"
    config.get_effective_feedback_model = lambda: "feedback-model"
    config.get_runtime_settings = lambda: {"rag_final_k": 20, "rag_candidate_k": 40}
    config.get_chat_thinking_settings = lambda: {"enabled": False}
    config.get_chat_max_tokens = lambda: 1024
    config.get_chat_timeout = lambda: 30
    config.get_feedback_max_tokens = lambda: 512
    config.get_feedback_timeout = lambda: 20
    install(f"{package_name}.config", config)

    return previous


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_memory_module():
    package_name = "vad_cache_under_test"
    previous = _install_stub_modules(package_name)
    try:
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.handlers.memory",
            PLUGIN_DIR / "handlers" / "memory.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(previous)


class _FakeSession:
    id = "session-1"


class _FakeState:
    feedback_client = object()
    session = _FakeSession()


class LongTermVadCacheTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_memory_module()
        cache = getattr(self.module, "_LONG_TERM_VAD_CACHE", None)
        if cache is not None:
            cache.clear()

    def _install_llm_response(self, responses):
        calls = []

        async def fake_llm_response(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            index = min(len(calls) - 1, len(responses) - 1)
            return responses[index]

        self.module.llm_response = fake_llm_response
        return calls

    def _summarize(self, *, bot_role="role", records=None):
        return self.module._summarize_long_term_vad(
            state=_FakeState(),
            bot_name="bot",
            bot_role=bot_role,
            target_name="target",
            target_id="target-1",
            vector_records=records or ["target likes tea"],
        )

    def test_long_term_vad_cache_hit_skips_llm(self):
        calls = self._install_llm_response([
            '{"valence": 0.4, "arousal": 0.2, "dominance": -0.1}',
        ])

        first = asyncio.run(self._summarize())
        first["valence"] = -1.0
        second = asyncio.run(self._summarize())

        self.assertEqual(1, len(calls))
        self.assertEqual(
            {"valence": 0.4, "arousal": 0.2, "dominance": -0.1},
            second,
        )

    def test_long_term_vad_cache_miss_when_records_change(self):
        calls = self._install_llm_response([
            '{"valence": 0.1, "arousal": 0.2, "dominance": 0.3}',
            '{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}',
        ])

        first = asyncio.run(self._summarize(records=["target likes tea"]))
        second = asyncio.run(self._summarize(records=["target dislikes tea"]))

        self.assertEqual(2, len(calls))
        self.assertEqual({"valence": 0.1, "arousal": 0.2, "dominance": 0.3}, first)
        self.assertEqual({"valence": 0.4, "arousal": 0.5, "dominance": 0.6}, second)

    def test_long_term_vad_cache_miss_when_role_changes(self):
        calls = self._install_llm_response([
            '{"valence": 0.1, "arousal": 0.2, "dominance": 0.3}',
            '{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}',
        ])

        first = asyncio.run(self._summarize(bot_role="role-a"))
        second = asyncio.run(self._summarize(bot_role="role-b"))

        self.assertEqual(2, len(calls))
        self.assertEqual({"valence": 0.1, "arousal": 0.2, "dominance": 0.3}, first)
        self.assertEqual({"valence": 0.4, "arousal": 0.5, "dominance": 0.6}, second)

    def test_long_term_vad_cache_does_not_cache_invalid_result(self):
        calls = self._install_llm_response([
            "null",
            '{"valence": 0.4, "arousal": 0.5, "dominance": 0.6}',
        ])

        first = asyncio.run(self._summarize())
        second = asyncio.run(self._summarize())

        self.assertIsNone(first)
        self.assertEqual({"valence": 0.4, "arousal": 0.5, "dominance": 0.6}, second)
        self.assertEqual(2, len(calls))


if __name__ == "__main__":
    unittest.main()
