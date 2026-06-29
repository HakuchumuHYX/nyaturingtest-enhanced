import importlib.util
import asyncio
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODULE_NAME = "plugins.nyaturingtest.core.logic"
STUB_MODULES = [
    "nonebot",
    "nonebot.adapters.onebot.v11",
    "nonebot.adapters.onebot.v11.exception",
    "plugins",
    "plugins.nyaturingtest",
    "plugins.nyaturingtest.core",
    "plugins.nyaturingtest.llm",
    "plugins.nyaturingtest.memory",
    "plugins.nyaturingtest.llm.client",
    "plugins.nyaturingtest.config",
    "plugins.nyaturingtest.memory.image",
    "plugins.nyaturingtest.memory.image_schema",
    "plugins.nyaturingtest.memory.short_term",
    "plugins.nyaturingtest.core.metrics",
    "plugins.nyaturingtest.core.message_sender",
    "plugins.nyaturingtest.core.structured_log",
    "plugins.nyaturingtest.core.state_manager",
    "plugins.nyaturingtest.core.usage",
    MODULE_NAME,
]
_MISSING = object()


def _install_logic_stubs():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["nonebot"] = nonebot

    onebot = types.ModuleType("nonebot.adapters.onebot.v11")
    onebot.Bot = object

    class OneBotMessage:
        def __init__(self, content=""):
            self.content = str(content or "")
            self.segments = []

        def insert(self, index, segment):
            self.segments.insert(index, segment)

        def extract_plain_text(self):
            return self.content

        def __len__(self):
            return len(self.segments) + (1 if self.content else 0)

        def __str__(self):
            return self.content

    onebot.Message = OneBotMessage
    onebot.MessageSegment = types.SimpleNamespace(reply=lambda msg_id: ("reply", msg_id))
    sys.modules["nonebot.adapters.onebot.v11"] = onebot

    onebot_exc = types.ModuleType("nonebot.adapters.onebot.v11.exception")
    onebot_exc.ActionFailed = type("ActionFailed", (Exception,), {})
    sys.modules["nonebot.adapters.onebot.v11.exception"] = onebot_exc

    for package in [
        "plugins",
        "plugins.nyaturingtest",
        "plugins.nyaturingtest.core",
        "plugins.nyaturingtest.llm",
        "plugins.nyaturingtest.memory",
    ]:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module

    llm_client = types.ModuleType("plugins.nyaturingtest.llm.client")
    llm_client.LLMClient = object
    sys.modules["plugins.nyaturingtest.llm.client"] = llm_client

    config = types.ModuleType("plugins.nyaturingtest.config")
    config.plugin_config = {"vlm": {"enabled": False}}
    config.get_effective_chat_model = lambda: "chat"
    config.get_effective_chat_provider = lambda: "openai_compatible"
    config.get_effective_feedback_model = lambda: "feedback"
    config.get_effective_feedback_provider = lambda: "openai_compatible"
    config.get_chat_thinking_settings = lambda: {}
    config.get_chat_max_tokens = lambda: 100
    config.get_chat_timeout = lambda: 10
    config.get_feedback_max_tokens = lambda: 100
    config.get_feedback_timeout = lambda: 10
    config.get_runtime_settings = lambda: {
        "debounce_seconds": 0,
        "humanized_delay_seconds": 0,
        "max_reply_messages": 3,
        "send_strategy": "sequential",
    }
    sys.modules["plugins.nyaturingtest.config"] = config

    image = types.ModuleType("plugins.nyaturingtest.memory.image")
    image.image_manager = object()
    sys.modules["plugins.nyaturingtest.memory.image"] = image

    image_schema = types.ModuleType("plugins.nyaturingtest.memory.image_schema")
    image_schema.merge_segment_metas = lambda metas: None
    sys.modules["plugins.nyaturingtest.memory.image_schema"] = image_schema

    short_term = types.ModuleType("plugins.nyaturingtest.memory.short_term")

    class Message:
        def __init__(self, user_id="", id="", content=""):
            self.user_id = user_id
            self.id = id
            self.content = content

    short_term.Message = Message
    sys.modules["plugins.nyaturingtest.memory.short_term"] = short_term

    metrics_mod = types.ModuleType("plugins.nyaturingtest.core.metrics")
    metrics_mod.metrics = types.SimpleNamespace(llm_success=0, llm_failure=0)
    sys.modules["plugins.nyaturingtest.core.metrics"] = metrics_mod

    sender = types.ModuleType("plugins.nyaturingtest.core.message_sender")
    sender.build_send_parts = lambda content, max_messages=0, strategy="": [content]
    sys.modules["plugins.nyaturingtest.core.message_sender"] = sender

    structured_log = types.ModuleType("plugins.nyaturingtest.core.structured_log")
    structured_log.log_event = lambda *args, **kwargs: None
    sys.modules["plugins.nyaturingtest.core.structured_log"] = structured_log

    state_manager = types.ModuleType("plugins.nyaturingtest.core.state_manager")
    state_manager.GroupState = object
    state_manager.SELF_SENT_MSG_IDS = []
    state_manager.is_shutting_down = lambda: False
    sys.modules["plugins.nyaturingtest.core.state_manager"] = state_manager

    usage = types.ModuleType("plugins.nyaturingtest.core.usage")
    usage.make_usage_recorder = lambda *args, **kwargs: None
    sys.modules["plugins.nyaturingtest.core.usage"] = usage


def _load_logic_module():
    saved_modules = {name: sys.modules.get(name, _MISSING) for name in STUB_MODULES}
    try:
        for name in STUB_MODULES:
            sys.modules.pop(name, None)
        _install_logic_stubs()
        spec = importlib.util.spec_from_file_location(MODULE_NAME, PLUGIN_DIR / "core" / "logic.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class LocalSelfEchoFilterTests(unittest.TestCase):
    def test_mixed_chunk_removes_local_self_echo_and_keeps_user_message(self):
        module = _load_logic_module()
        module.SELF_SENT_MSG_IDS.append("echo-1")
        echo = types.SimpleNamespace(user_id="10000", id="echo-1", content="bot said this")
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")

        filtered, local_echoes = module._filter_local_self_echoes([echo, user], "10000")

        self.assertEqual([user], filtered)
        self.assertEqual([echo], local_echoes)

    def test_local_self_echo_only_chunk_is_removed(self):
        module = _load_logic_module()
        module.SELF_SENT_MSG_IDS.append("echo-1")
        echo = types.SimpleNamespace(user_id="10000", id="echo-1", content="bot said this")

        filtered, local_echoes = module._filter_local_self_echoes([echo], "10000")

        self.assertEqual([], filtered)
        self.assertEqual([echo], local_echoes)

    def test_non_local_bot_id_message_is_not_removed(self):
        module = _load_logic_module()
        other_bot_message = types.SimpleNamespace(user_id="10000", id="not-local", content="external")

        filtered, local_echoes = module._filter_local_self_echoes([other_bot_message], "10000")

        self.assertEqual([other_bot_message], filtered)
        self.assertEqual([], local_echoes)

    def test_user_message_with_self_sent_id_is_not_removed(self):
        module = _load_logic_module()
        module.SELF_SENT_MSG_IDS.append("echo-1")
        user = types.SimpleNamespace(user_id="20000", id="echo-1", content="same id somehow")

        filtered, local_echoes = module._filter_local_self_echoes([user], "10000")

        self.assertEqual([user], filtered)
        self.assertEqual([], local_echoes)

    def test_spawn_state_skips_local_self_echo_only_chunk(self):
        module = _load_logic_module()
        module.SELF_SENT_MSG_IDS.append("echo-1")
        echo = types.SimpleNamespace(user_id="10000", id="echo-1", content="bot said this")
        state = _FakeState([echo])

        asyncio.run(module.spawn_state(state))

        self.assertEqual(0, state.session.load_calls)
        self.assertEqual([], state.session.update_without_trigger_chunks)
        self.assertEqual([], state.session.update_chunks)

    def test_spawn_state_keeps_non_local_bot_id_echo_only_memory_path(self):
        module = _load_logic_module()
        other_bot_message = types.SimpleNamespace(user_id="10000", id="not-local", content="external")
        state = _FakeState([other_bot_message])

        asyncio.run(module.spawn_state(state))

        self.assertEqual(1, state.session.load_calls)
        self.assertEqual([[other_bot_message]], state.session.update_without_trigger_chunks)
        self.assertEqual([], state.session.update_chunks)

    def test_spawn_state_passes_filtered_mixed_chunk_to_session_update(self):
        module = _load_logic_module()
        module.SELF_SENT_MSG_IDS.append("echo-1")
        echo = types.SimpleNamespace(user_id="10000", id="echo-1", content="bot said this")
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")
        state = _FakeState([echo, user])

        asyncio.run(module.spawn_state(state))

        self.assertEqual([], state.session.update_without_trigger_chunks)
        self.assertEqual([[user]], state.session.update_chunks)

    def test_spawn_state_discards_replies_when_generation_changes_after_update(self):
        module = _load_logic_module()
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")
        state = _FakeState([user])
        state.session.update_response = [{"content": "stale reply"}]
        state.session.bump_during_update = True

        asyncio.run(module.spawn_state(state))

        self.assertEqual([], state.bot.sent_messages)
        self.assertEqual([], state.session.appended_self_messages)

    def test_spawn_state_sends_and_appends_self_message_when_generation_is_current(self):
        module = _load_logic_module()
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")
        state = _FakeState([user])
        state.session.update_response = [{"content": "fresh reply"}]

        asyncio.run(module.spawn_state(state))

        self.assertEqual(["fresh reply"], state.bot.sent_messages)
        self.assertEqual([("fresh reply", "sent-1", "10000")], state.session.appended_self_messages)

    def test_spawn_state_uses_chunk_bot_snapshot_for_send(self):
        module = _load_logic_module()
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")
        state = _FakeState([user])
        original_bot = state.bot
        replacement_bot = _FakeBot()

        async def update(messages_chunk, chat_llm_func, feedback_llm_func, publish=True, expected_generation=None):
            state.bot = replacement_bot
            return [{"content": "fresh reply"}]

        state.session.update = update

        asyncio.run(module.spawn_state(state))

        self.assertEqual(["fresh reply"], original_bot.sent_messages)
        self.assertEqual([], replacement_bot.sent_messages)
        self.assertEqual([("fresh reply", "sent-1", "10000")], state.session.appended_self_messages)

    def test_spawn_state_enforces_turn_level_max_reply_messages(self):
        module = _load_logic_module()
        module.build_send_parts = lambda content, max_messages=0, strategy="": str(content).split("|")[:max_messages]
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")
        state = _FakeState([user])
        state.session.update_response = [
            {"content": "one|two"},
            {"content": "three|four"},
        ]

        asyncio.run(module.spawn_state(state))

        self.assertEqual(["one", "two", "three"], state.bot.sent_messages)

    def test_spawn_state_rechecks_generation_inside_append_lock(self):
        module = _load_logic_module()
        user = types.SimpleNamespace(user_id="20000", id="user-1", content="hello")
        state = _FakeState([user])
        state.session.update_response = [{"content": "fresh reply"}]
        state.session_lock.bump_on_append_enter = True

        asyncio.run(module.spawn_state(state))

        self.assertEqual(["fresh reply"], state.bot.sent_messages)
        self.assertEqual([], state.session.appended_self_messages)


class _OneShotSignal:
    def __init__(self):
        self.waits = 0
        self.clears = 0

    async def wait(self):
        self.waits += 1
        if self.waits > 1:
            raise asyncio.CancelledError()
        return True

    def clear(self):
        self.clears += 1


class _AsyncLock:
    def __init__(self):
        self.bump_on_append_enter = False
        self.enters = 0
        self.session = None

    async def __aenter__(self):
        self.enters += 1
        if self.bump_on_append_enter and self.enters >= 2 and self.session is not None:
            self.session.generation += 1
            self.bump_on_append_enter = False
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    id = "group-1"

    def __init__(self):
        self.generation = 0
        self.load_calls = 0
        self.update_without_trigger_chunks = []
        self.update_chunks = []
        self.update_response = None
        self.bump_during_update = False
        self.appended_self_messages = []
        self.stale_logs = []

    async def load_session(self):
        self.load_calls += 1

    async def update_without_trigger(self, chunk):
        self.update_without_trigger_chunks.append(list(chunk))

    def is_generation_stale(self, expected_generation):
        return expected_generation is not None and self.generation != expected_generation

    def _log_stale_generation(self, stage, expected_generation):
        self.stale_logs.append((stage, expected_generation, self.generation))

    async def update(self, messages_chunk, chat_llm_func, feedback_llm_func, publish=True, expected_generation=None):
        self.update_chunks.append(list(messages_chunk))
        if self.bump_during_update:
            self.generation += 1
        return self.update_response

    async def append_self_message(self, content, msg_id, bot_user_id):
        self.appended_self_messages.append((content, msg_id, bot_user_id))


class _FakeBot:
    self_id = "10000"

    def __init__(self):
        self.sent_messages = []

    async def send(self, message, event):
        self.sent_messages.append(message.extract_plain_text())
        return {"message_id": "sent-1"}


class _FakeState:
    def __init__(self, messages):
        self.new_message_signal = _OneShotSignal()
        self.data_lock = _AsyncLock()
        self.session_lock = _AsyncLock()
        self.messages_chunk = list(messages)
        self.bot = _FakeBot()
        self.event = object()
        self.session = _FakeSession()
        self.session_lock.session = self.session
        self.client = object()
        self.feedback_client = object()


if __name__ == "__main__":
    unittest.main()
