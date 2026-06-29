import asyncio
import importlib.util
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from threading import RLock


PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODULE_NAME = "plugins.nyaturingtest.core.session"
STUB_MODULES = [
    "nonebot",
    "nonebot_plugin_localstore",
    "nonebot.utils",
    "httpx",
    "openai",
    "plugins",
    "plugins.nyaturingtest",
    "plugins.nyaturingtest.core",
    "plugins.nyaturingtest.database",
    "plugins.nyaturingtest.llm",
    "plugins.nyaturingtest.memory",
    "plugins.nyaturingtest.models",
    "plugins.nyaturingtest.prompts",
    "plugins.nyaturingtest.llm.client",
    "plugins.nyaturingtest.config",
    "plugins.nyaturingtest.models.emotion",
    "plugins.nyaturingtest.memory.vector",
    "plugins.nyaturingtest.memory.validation",
    "plugins.nyaturingtest.models.impression",
    "plugins.nyaturingtest.memory.short_term",
    "plugins.nyaturingtest.prompts.presets",
    "plugins.nyaturingtest.models.profile",
    "plugins.nyaturingtest.utils",
    "plugins.nyaturingtest.prompts.templates",
    "plugins.nyaturingtest.database.message_repository",
    "plugins.nyaturingtest.database.profile_repository",
    "plugins.nyaturingtest.database.session_repository",
    "plugins.nyaturingtest.database.backup_lock",
    "plugins.nyaturingtest.core.services",
    "plugins.nyaturingtest.core.orchestrator",
    "plugins.nyaturingtest.core.structured_log",
    "plugins.nyaturingtest.core.rag_query",
    MODULE_NAME,
]
_MISSING = object()


def _install_session_stubs():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["nonebot"] = nonebot

    localstore = types.ModuleType("nonebot_plugin_localstore")
    localstore.get_plugin_data_dir = lambda: "/tmp/nyaturingtest-test"
    sys.modules["nonebot_plugin_localstore"] = localstore

    def run_sync(func):
        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    nonebot_utils = types.ModuleType("nonebot.utils")
    nonebot_utils.run_sync = run_sync
    sys.modules["nonebot.utils"] = nonebot_utils

    httpx = types.ModuleType("httpx")
    httpx.AsyncClient = object
    httpx.Limits = lambda *args, **kwargs: None
    sys.modules["httpx"] = httpx

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = object
    sys.modules["openai"] = openai

    for package in [
        "plugins",
        "plugins.nyaturingtest",
        "plugins.nyaturingtest.core",
        "plugins.nyaturingtest.database",
        "plugins.nyaturingtest.llm",
        "plugins.nyaturingtest.memory",
        "plugins.nyaturingtest.models",
        "plugins.nyaturingtest.prompts",
    ]:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module

    llm_client = types.ModuleType("plugins.nyaturingtest.llm.client")
    llm_client.LLMClient = object
    sys.modules["plugins.nyaturingtest.llm.client"] = llm_client

    config = types.ModuleType("plugins.nyaturingtest.config")
    config.get_chat_thinking_settings = lambda: {}
    config.get_runtime_settings = lambda: {
        "role_max_chars": 1000,
        "short_context_limit": 20,
        "short_term_buffer_size": 200,
    }
    sys.modules["plugins.nyaturingtest.config"] = config

    emotion = types.ModuleType("plugins.nyaturingtest.models.emotion")

    class EmotionState:
        def __init__(self):
            self.valence = 0.0
            self.arousal = 0.0
            self.dominance = 0.0

    emotion.EmotionState = EmotionState
    emotion.clamp_vad_value = lambda value, lo, hi, default=0.0: default
    sys.modules["plugins.nyaturingtest.models.emotion"] = emotion

    vector = types.ModuleType("plugins.nyaturingtest.memory.vector")
    vector.VectorMemory = object
    vector.where_any = lambda *args, **kwargs: None
    sys.modules["plugins.nyaturingtest.memory.vector"] = vector

    validation = types.ModuleType("plugins.nyaturingtest.memory.validation")
    validation.validate_memory_candidate = lambda *args, **kwargs: types.SimpleNamespace(valid=True, reason="ok")
    sys.modules["plugins.nyaturingtest.memory.validation"] = validation

    impression = types.ModuleType("plugins.nyaturingtest.models.impression")
    impression.Impression = object
    sys.modules["plugins.nyaturingtest.models.impression"] = impression

    short_term = types.ModuleType("plugins.nyaturingtest.memory.short_term")
    short_term.Memory = object
    short_term.Message = object
    sys.modules["plugins.nyaturingtest.memory.short_term"] = short_term

    presets = types.ModuleType("plugins.nyaturingtest.prompts.presets")
    presets.PRESETS = {}
    sys.modules["plugins.nyaturingtest.prompts.presets"] = presets

    profile = types.ModuleType("plugins.nyaturingtest.models.profile")

    class PersonProfile:
        pass

    profile.PersonProfile = PersonProfile
    sys.modules["plugins.nyaturingtest.models.profile"] = profile

    utils = types.ModuleType("plugins.nyaturingtest.utils")
    utils.extract_and_parse_json = lambda value: {}
    utils.check_relevance = lambda *args, **kwargs: False
    utils.sanitize_text = lambda value: str(value or "")
    utils.escape_for_prompt = lambda value: str(value or "")
    utils.get_time_description = lambda value: ""
    utils.should_store_memory = lambda value: True
    sys.modules["plugins.nyaturingtest.utils"] = utils

    templates = types.ModuleType("plugins.nyaturingtest.prompts.templates")
    templates.get_feedback_prompt = lambda *args, **kwargs: ""
    templates.get_chat_prompt = lambda *args, **kwargs: ""
    sys.modules["plugins.nyaturingtest.prompts.templates"] = templates

    for name, class_name in [
        ("plugins.nyaturingtest.database.message_repository", "MessageRepository"),
        ("plugins.nyaturingtest.database.profile_repository", "ProfileRepository"),
        ("plugins.nyaturingtest.database.session_repository", "SessionStateRepository"),
    ]:
        module = types.ModuleType(name)
        setattr(module, class_name, object)
        sys.modules[name] = module

    backup_lock = types.ModuleType("plugins.nyaturingtest.database.backup_lock")
    backup_lock.BACKUP_IO_LOCK = RLock()
    sys.modules["plugins.nyaturingtest.database.backup_lock"] = backup_lock

    services = types.ModuleType("plugins.nyaturingtest.core.services")
    services.RagSearchService = object
    sys.modules["plugins.nyaturingtest.core.services"] = services

    orchestrator = types.ModuleType("plugins.nyaturingtest.core.orchestrator")
    orchestrator.ConversationOrchestrator = object
    sys.modules["plugins.nyaturingtest.core.orchestrator"] = orchestrator

    structured_log = types.ModuleType("plugins.nyaturingtest.core.structured_log")
    structured_log.log_event = lambda *args, **kwargs: None
    sys.modules["plugins.nyaturingtest.core.structured_log"] = structured_log

    rag_query = types.ModuleType("plugins.nyaturingtest.core.rag_query")
    rag_query.build_chat_rag_queries = lambda *args, **kwargs: []
    sys.modules["plugins.nyaturingtest.core.rag_query"] = rag_query


def _load_session_module():
    saved = {name: sys.modules.get(name, _MISSING) for name in STUB_MODULES}
    try:
        for name in STUB_MODULES:
            sys.modules.pop(name, None)
        _install_session_stubs()
        spec = importlib.util.spec_from_file_location(MODULE_NAME, PLUGIN_DIR / "core" / "session.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class _FakeProfile:
    def __init__(self):
        self.is_dirty = True
        self.clean_calls = 0

    def mark_clean(self):
        self.clean_calls += 1
        self.is_dirty = False


class _FakeMemory:
    def __init__(self, messages=None):
        self._messages = list(messages or [])

    def snapshot(self):
        return list(self._messages)


class _FakeLongTermMemory:
    def cleanup(self, days_retention=90):
        return None


class _FailingProfileRepo:
    @staticmethod
    async def update_user_profiles(session_id, profiles):
        raise RuntimeError("profile write failed")


class _OkProfileRepo:
    @staticmethod
    async def update_user_profiles(session_id, profiles):
        return None


class _OkSessionRepo:
    @staticmethod
    async def save_session_state(session_id, data):
        return None


class _OkMessageRepo:
    @staticmethod
    async def sync_messages(session_id, recent_msgs):
        return None


class _SlowMessageRepo:
    active = 0
    max_active = 0

    @staticmethod
    async def sync_messages(session_id, recent_msgs):
        _SlowMessageRepo.active += 1
        _SlowMessageRepo.max_active = max(_SlowMessageRepo.max_active, _SlowMessageRepo.active)
        await asyncio.sleep(0.01)
        _SlowMessageRepo.active -= 1


def _make_session(module, *, profile=None, messages=None):
    session = module.Session.__new__(module.Session)
    session.id = "group-1"
    session._Session__name = "terminus"
    session._Session__role = "role"
    session._Session__aliases = []
    session.global_emotion = types.SimpleNamespace(valence=0.0, arousal=0.0, dominance=0.0)
    session.chat_summary = ""
    session._last_speak_time = datetime.min
    session.last_consolidated_time = None
    session._Session__chatting_state = module._ChattingState.IDLE
    session.profiles = {"u1": profile} if profile else {}
    session.global_memory = _FakeMemory(messages)
    session.long_term_memory = _FakeLongTermMemory()
    session._save_lock = asyncio.Lock()
    return session


class SaveSessionPersistenceTests(unittest.TestCase):
    def test_profile_write_failure_returns_false_and_preserves_dirty_profile(self):
        module = _load_session_module()
        profile = _FakeProfile()
        session = _make_session(module, profile=profile)
        module.SessionStateRepository = _OkSessionRepo
        module.ProfileRepository = _FailingProfileRepo
        module.MessageRepository = _OkMessageRepo

        result = asyncio.run(session.save_session())

        self.assertFalse(result)
        self.assertTrue(profile.is_dirty)
        self.assertEqual(0, profile.clean_calls)

    def test_successful_save_returns_true_and_marks_dirty_profile_clean(self):
        module = _load_session_module()
        profile = _FakeProfile()
        session = _make_session(module, profile=profile)
        module.SessionStateRepository = _OkSessionRepo
        module.ProfileRepository = _OkProfileRepo
        module.MessageRepository = _OkMessageRepo

        result = asyncio.run(session.save_session())

        self.assertTrue(result)
        self.assertFalse(profile.is_dirty)
        self.assertEqual(1, profile.clean_calls)

    def test_concurrent_save_session_calls_are_serialized(self):
        module = _load_session_module()
        session = _make_session(module, messages=[object()])
        module.SessionStateRepository = _OkSessionRepo
        module.ProfileRepository = _OkProfileRepo
        module.MessageRepository = _SlowMessageRepo
        _SlowMessageRepo.active = 0
        _SlowMessageRepo.max_active = 0

        async def run_saves():
            return await asyncio.gather(session.save_session(), session.save_session())

        results = asyncio.run(run_saves())

        self.assertEqual([True, True], results)
        self.assertEqual(1, _SlowMessageRepo.max_active)


if __name__ == "__main__":
    unittest.main()
