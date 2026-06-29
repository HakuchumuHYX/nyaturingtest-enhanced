import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class _FakeMatcher:
    def __init__(self):
        self.sent = []
        self.finished = []

    def handle(self):
        def decorator(func):
            return func

        return decorator

    async def send(self, message, *args, **kwargs):
        self.sent.append(message)

    async def finish(self, message, *args, **kwargs):
        self.finished.append(message)


class _FakeSender:
    card = "Alice"
    nickname = "Alice"


class _FakeEvent:
    group_id = 1001
    user_id = 2002
    sender = _FakeSender()


class _AsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    id = "session-1"
    profiles = {}

    async def load_session(self):
        return None

    def name(self):
        return "Nya"

    def role(self):
        return "role"


class _FakeLongTermMemory:
    def count_by_user(self, target_id):
        return 1


class _FakeSessionWithLongTermMemory(_FakeSession):
    long_term_memory = _FakeLongTermMemory()


class _FakeState:
    client = object()
    feedback_client = object()
    session = _FakeSession()
    session_lock = _AsyncLock()


def _install_stub_modules(package_name: str):
    previous = {}
    matchers = []

    async def async_recent_messages(*args, **kwargs):
        return ["hello"]

    async def async_interaction_count(*args, **kwargs):
        return 1

    async def async_first_interaction_time(*args, **kwargs):
        return None

    def install(name: str, module):
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module

    nonebot = types.ModuleType("nonebot")

    def fake_on_command(*args, **kwargs):
        matcher = _FakeMatcher()
        matchers.append(matcher)
        return matcher

    nonebot.on_command = fake_on_command
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

    def run_sync(func):
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    utils.run_sync = run_sync
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
    message_repository.MessageRepository = types.SimpleNamespace(
        get_recent_messages_by_user=async_recent_messages,
    )
    install(f"{package_name}.database.message_repository", message_repository)

    profile_repository = types.ModuleType(f"{package_name}.database.profile_repository")
    profile_repository.ProfileRepository = types.SimpleNamespace(
        get_interaction_count=async_interaction_count,
        get_first_interaction_time=async_first_interaction_time,
    )
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
    config.get_effective_chat_provider = lambda: "openai_compatible"
    config.get_effective_feedback_provider = lambda: "openai_compatible"
    config.get_runtime_settings = lambda: {
        "rag_final_k": 20,
        "rag_candidate_k": 40,
        "rag_merged_candidate_cap": 80,
    }
    config.get_chat_thinking_settings = lambda: {"enabled": False, "reasoning_effort": "low"}
    config.get_chat_max_tokens = lambda: 1024
    config.get_chat_timeout = lambda: 30
    config.get_feedback_max_tokens = lambda: 512
    config.get_feedback_timeout = lambda: 20
    install(f"{package_name}.config", config)

    return previous, matchers


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_memory_module():
    package_name = "memory_provider_gating_under_test"
    previous, matchers = _install_stub_modules(package_name)
    module_name = f"{package_name}.handlers.memory"
    previous[module_name] = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            PLUGIN_DIR / "handlers" / "memory.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module._test_matchers = matchers
        return module
    finally:
        _restore_modules(previous)


class MemoryCommandsProviderGatingTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_memory_module()
        cache = getattr(self.module, "_LONG_TERM_VAD_CACHE", None)
        if cache is not None:
            cache.clear()

    def _install_llm_response(self, response):
        calls = []

        async def fake_llm_response(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return response

        self.module.llm_response = fake_llm_response
        return calls

    def test_long_term_vad_openai_compatible_omits_deepseek_extra_body(self):
        self.module.get_effective_feedback_provider = lambda: "openai_compatible"
        calls = self._install_llm_response(
            '{"valence": 0.2, "arousal": 0.3, "dominance": 0.4}'
        )

        result = asyncio.run(self.module._summarize_long_term_vad(
            state=_FakeState(),
            bot_name="Nya",
            bot_role="role",
            target_name="Alice",
            target_id="2002",
            vector_records=["Alice likes tea"],
        ))

        self.assertEqual({"valence": 0.2, "arousal": 0.3, "dominance": 0.4}, result)
        self.assertIsNone(calls[0]["kwargs"].get("extra_body"))

    def test_long_term_vad_deepseek_official_disables_thinking(self):
        self.module.get_effective_feedback_provider = lambda: "deepseek_official"
        calls = self._install_llm_response(
            '{"valence": 0.2, "arousal": 0.3, "dominance": 0.4}'
        )

        asyncio.run(self.module._summarize_long_term_vad(
            state=_FakeState(),
            bot_name="Nya",
            bot_role="role",
            target_name="Alice",
            target_id="2002",
            vector_records=["Alice likes tea"],
        ))

        self.assertEqual(
            {"thinking": {"type": "disabled"}},
            calls[0]["kwargs"].get("extra_body"),
        )

    def test_query_memory_openai_compatible_omits_deepseek_extra_body(self):
        self.module.ensure_group_state = lambda group_id: _FakeState()
        self.module.get_effective_chat_provider = lambda: "openai_compatible"
        self.module.get_chat_thinking_settings = lambda: {
            "enabled": True,
            "reasoning_effort": "high",
        }
        calls = self._install_llm_response('{"description": "desc", "emotion": "calm"}')

        asyncio.run(self.module.handle_query_memory(object(), _FakeEvent(), []))

        self.assertEqual(1, len(calls))
        self.assertIsNone(calls[0]["kwargs"].get("extra_body"))
        self.assertIsNone(calls[0]["kwargs"].get("reasoning_effort"))
        self.assertEqual(0.8, calls[0]["kwargs"].get("temperature"))

    def test_query_memory_deepseek_official_preserves_chat_thinking_body(self):
        self.module.ensure_group_state = lambda group_id: _FakeState()
        self.module.get_effective_chat_provider = lambda: "deepseek_official"
        self.module.get_chat_thinking_settings = lambda: {
            "enabled": True,
            "reasoning_effort": "high",
        }
        calls = self._install_llm_response('{"description": "desc", "emotion": "calm"}')

        asyncio.run(self.module.handle_query_memory(object(), _FakeEvent(), []))

        self.assertEqual(
            {"thinking": {"type": "enabled"}},
            calls[0]["kwargs"].get("extra_body"),
        )
        self.assertEqual("high", calls[0]["kwargs"].get("reasoning_effort"))
        self.assertIsNone(calls[0]["kwargs"].get("temperature"))

    def test_query_memory_deepseek_official_disabled_thinking_sends_disabled_body(self):
        self.module.ensure_group_state = lambda group_id: _FakeState()
        self.module.get_effective_chat_provider = lambda: "deepseek_official"
        self.module.get_chat_thinking_settings = lambda: {
            "enabled": False,
            "reasoning_effort": "high",
        }
        calls = self._install_llm_response('{"description": "desc", "emotion": "calm"}')

        asyncio.run(self.module.handle_query_memory(object(), _FakeEvent(), []))

        self.assertEqual(
            {"thinking": {"type": "disabled"}},
            calls[0]["kwargs"].get("extra_body"),
        )
        self.assertIsNone(calls[0]["kwargs"].get("reasoning_effort"))
        self.assertEqual(0.8, calls[0]["kwargs"].get("temperature"))

    def test_query_memory_vector_retrieval_uses_runtime_candidate_cap(self):
        class StateWithLongTermMemory(_FakeState):
            session = _FakeSessionWithLongTermMemory()

        class FakeRagSearchService:
            calls = []

            def __init__(self, memory):
                self.memory = memory

            async def search_for_user_profile(self, *args, **kwargs):
                self.calls.append(kwargs)
                return [{
                    "content": "Alice 明确表示自己喜欢薄荷巧克力冰淇淋",
                    "metadata": {"subject_user_id": "2002", "user_id": "2002"},
                }]

        async def no_vad_summary(*args, **kwargs):
            return None

        self.module.ensure_group_state = lambda group_id: StateWithLongTermMemory()
        self.module.RagSearchService = FakeRagSearchService
        self.module._summarize_long_term_vad = no_vad_summary
        self._install_llm_response('{"description": "desc", "emotion": "calm"}')

        asyncio.run(self.module.handle_query_memory(object(), _FakeEvent(), []))

        self.assertEqual(1, len(FakeRagSearchService.calls))
        self.assertEqual(80, FakeRagSearchService.calls[0]["merged_candidate_cap"])


if __name__ == "__main__":
    unittest.main()
