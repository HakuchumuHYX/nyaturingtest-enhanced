import asyncio
import importlib.util
import json
import sys
import types
import unittest
from threading import RLock
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
_MISSING = object()


def _restore_modules(saved):
    for name, module in saved.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_common_packages(package_root: str):
    for package in [
        package_root,
        f"{package_root}.core",
        f"{package_root}.database",
        f"{package_root}.llm",
        f"{package_root}.memory",
        f"{package_root}.models",
        f"{package_root}.prompts",
    ]:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module


def _install_logger():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["nonebot"] = nonebot


def _runtime_settings():
    return {
        "active_to_bubble_threshold": 0.4,
        "consolidation_enabled": False,
        "consolidation_interval_seconds": 9999,
        "consolidation_max_messages": 10,
        "consolidation_message_threshold": 9999,
        "examples_max_chars": 1000,
        "history_recall_limit": 10,
        "interest_topic_willingness_floor": 0.3,
        "low_willingness_skip_threshold": 0.0,
        "passive_growth_max_factor": 1.0,
        "passive_growth_min_factor": 1.0,
        "passive_willingness_growth_limit": 1.0,
        "passive_willingness_growth_per_message": 0.0,
        "post_feedback_skip_threshold": 0.0,
        "rag_default_event_ttl_days": 90,
        "relevance_willingness_floor": 0.8,
        "rerank_willingness_threshold": 0.7,
        "role_max_chars": 1000,
        "short_context_limit": 20,
        "short_term_buffer_size": 200,
        "speak_cooldown_seconds": 0,
        "speak_willingness_retain_factor": 0.5,
        "willingness_decay_rate_active": 0.0,
        "willingness_decay_rate_idle": 0.0,
        "willingness_idle_after_seconds": 9999,
        "willingness_reply_threshold": 0.1,
    }


def _load_orchestrator_module():
    module_name = "turn_generation_orchestrator_under_test.core.orchestrator"
    package_root = "turn_generation_orchestrator_under_test"
    stub_names = [
        "nonebot",
        package_root,
        f"{package_root}.core",
        f"{package_root}.config",
        f"{package_root}.memory",
        f"{package_root}.memory.short_term",
        f"{package_root}.utils",
        f"{package_root}.core.services",
        f"{package_root}.core.structured_log",
        module_name,
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in stub_names}
    try:
        for name in stub_names:
            sys.modules.pop(name, None)
        _install_logger()
        _install_common_packages(package_root)

        config = types.ModuleType(f"{package_root}.config")
        config.get_runtime_settings = _runtime_settings
        sys.modules[f"{package_root}.config"] = config

        short_term = types.ModuleType(f"{package_root}.memory.short_term")
        short_term.Message = object
        sys.modules[f"{package_root}.memory.short_term"] = short_term

        utils = types.ModuleType(f"{package_root}.utils")
        utils.check_relevance = lambda *args, **kwargs: False
        utils.score_message_interest = lambda *args, **kwargs: 1.0
        sys.modules[f"{package_root}.utils"] = utils

        services = types.ModuleType(f"{package_root}.core.services")
        services.ChatService = object
        services.FeedbackService = object
        services.MemoryService = object
        sys.modules[f"{package_root}.core.services"] = services

        structured_log = types.ModuleType(f"{package_root}.core.structured_log")
        structured_log.log_event = lambda *args, **kwargs: None
        sys.modules[f"{package_root}.core.structured_log"] = structured_log

        spec = importlib.util.spec_from_file_location(module_name, PLUGIN_DIR / "core" / "orchestrator.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, saved
    except Exception:
        _restore_modules(saved)
        raise


def _load_session_module():
    module_name = "turn_generation_session_under_test.core.session"
    package_root = "turn_generation_session_under_test"
    stub_names = [
        "nonebot",
        "nonebot_plugin_localstore",
        "nonebot.utils",
        "httpx",
        "openai",
        package_root,
        f"{package_root}.core",
        f"{package_root}.database",
        f"{package_root}.llm",
        f"{package_root}.memory",
        f"{package_root}.models",
        f"{package_root}.prompts",
        f"{package_root}.llm.client",
        f"{package_root}.config",
        f"{package_root}.models.emotion",
        f"{package_root}.memory.vector",
        f"{package_root}.memory.validation",
        f"{package_root}.models.impression",
        f"{package_root}.memory.short_term",
        f"{package_root}.prompts.presets",
        f"{package_root}.models.profile",
        f"{package_root}.utils",
        f"{package_root}.prompts.templates",
        f"{package_root}.database.message_repository",
        f"{package_root}.database.profile_repository",
        f"{package_root}.database.session_repository",
        f"{package_root}.database.backup_lock",
        f"{package_root}.core.services",
        f"{package_root}.core.orchestrator",
        f"{package_root}.core.structured_log",
        f"{package_root}.core.rag_query",
        module_name,
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in stub_names}
    try:
        for name in stub_names:
            sys.modules.pop(name, None)
        _install_logger()
        _install_common_packages(package_root)

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

        llm_client = types.ModuleType(f"{package_root}.llm.client")
        llm_client.LLMClient = object
        sys.modules[f"{package_root}.llm.client"] = llm_client

        config = types.ModuleType(f"{package_root}.config")
        config.get_chat_thinking_settings = lambda: {}
        config.get_runtime_settings = _runtime_settings
        sys.modules[f"{package_root}.config"] = config

        emotion = types.ModuleType(f"{package_root}.models.emotion")

        @dataclass
        class EmotionState:
            valence: float = 0.0
            arousal: float = 0.0
            dominance: float = 0.0

        emotion.EmotionState = EmotionState
        emotion.clamp_vad_value = lambda value, lo, hi, default=0.0: default
        sys.modules[f"{package_root}.models.emotion"] = emotion

        vector = types.ModuleType(f"{package_root}.memory.vector")
        vector.VectorMemory = object
        vector.where_any = lambda *args, **kwargs: None
        sys.modules[f"{package_root}.memory.vector"] = vector

        validation = types.ModuleType(f"{package_root}.memory.validation")
        validation.validate_memory_candidate = lambda *args, **kwargs: types.SimpleNamespace(valid=True, reason="ok")
        sys.modules[f"{package_root}.memory.validation"] = validation

        impression = types.ModuleType(f"{package_root}.models.impression")
        impression.Impression = object
        sys.modules[f"{package_root}.models.impression"] = impression

        short_term = types.ModuleType(f"{package_root}.memory.short_term")
        short_term.Memory = object
        short_term.Message = object
        sys.modules[f"{package_root}.memory.short_term"] = short_term

        presets = types.ModuleType(f"{package_root}.prompts.presets")
        presets.PRESETS = {}
        sys.modules[f"{package_root}.prompts.presets"] = presets

        profile = types.ModuleType(f"{package_root}.models.profile")

        class PersonProfile:
            def __init__(self, user_id=""):
                self.user_id = user_id
                self.emotion = EmotionState()

        profile.PersonProfile = PersonProfile
        sys.modules[f"{package_root}.models.profile"] = profile

        utils = types.ModuleType(f"{package_root}.utils")
        utils.extract_and_parse_json = lambda value: json.loads(value)
        utils.check_relevance = lambda *args, **kwargs: False
        utils.sanitize_text = lambda value: str(value or "")
        utils.escape_for_prompt = lambda value: str(value or "")
        utils.get_time_description = lambda value: ""
        utils.should_store_memory = lambda value: True
        sys.modules[f"{package_root}.utils"] = utils

        templates = types.ModuleType(f"{package_root}.prompts.templates")
        templates.get_feedback_prompt = lambda *args, **kwargs: ""
        templates.get_chat_prompt = lambda *args, **kwargs: "chat prompt"
        sys.modules[f"{package_root}.prompts.templates"] = templates

        message_repository = types.ModuleType(f"{package_root}.database.message_repository")
        message_repository.MessageRepository = types.SimpleNamespace(
            get_history_before=lambda *args, **kwargs: [],
        )
        sys.modules[f"{package_root}.database.message_repository"] = message_repository

        profile_repository = types.ModuleType(f"{package_root}.database.profile_repository")
        profile_repository.ProfileRepository = object
        sys.modules[f"{package_root}.database.profile_repository"] = profile_repository

        session_repository = types.ModuleType(f"{package_root}.database.session_repository")
        session_repository.SessionStateRepository = object
        sys.modules[f"{package_root}.database.session_repository"] = session_repository

        backup_lock = types.ModuleType(f"{package_root}.database.backup_lock")
        backup_lock.BACKUP_IO_LOCK = RLock()
        sys.modules[f"{package_root}.database.backup_lock"] = backup_lock

        services = types.ModuleType(f"{package_root}.core.services")
        services.RagSearchService = object
        sys.modules[f"{package_root}.core.services"] = services

        orchestrator = types.ModuleType(f"{package_root}.core.orchestrator")
        orchestrator.ConversationOrchestrator = object
        sys.modules[f"{package_root}.core.orchestrator"] = orchestrator

        structured_log = types.ModuleType(f"{package_root}.core.structured_log")
        structured_log.log_event = lambda *args, **kwargs: None
        sys.modules[f"{package_root}.core.structured_log"] = structured_log

        rag_query = types.ModuleType(f"{package_root}.core.rag_query")
        rag_query.build_chat_rag_queries = lambda *args, **kwargs: []
        sys.modules[f"{package_root}.core.rag_query"] = rag_query

        spec = importlib.util.spec_from_file_location(module_name, PLUGIN_DIR / "core" / "session.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, saved
    except Exception:
        _restore_modules(saved)
        raise


class _Message:
    def __init__(self):
        self.content = "hello"
        self.id = "m1"
        self.image_meta = None
        self.time = datetime.now()
        self.user_id = "user-1"
        self.user_name = "Alice"


class _ContextRecord:
    compressed_history = ""
    messages = []


class _Memory:
    def __init__(self):
        self.summary = ""
        self.cleared = False

    async def clear(self):
        self.cleared = True

    def access_context(self, limit=None):
        return _ContextRecord()

    def update_summary(self, summary):
        self.summary = summary


class _LongTermMemory:
    def __init__(self):
        self.cleared = False
        self.deleted_filters = []
        self.added = []

    def clear(self):
        self.cleared = True

    def delete_by_metadata(self, metadata):
        self.deleted_filters.append(metadata)

    def add_texts(self, texts, metadatas=None):
        self.added.append((list(texts), list(metadatas or [])))


class _FakeSessionForOrchestrator:
    id = "session-1"

    def __init__(self):
        self.generation = 0
        self.willingness = 1.0
        self._last_decay_time = datetime.now()
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        self._engaged = True
        self._messages_since_consolidation = 0
        self._last_consolidation_attempt = datetime.min
        self.last_consolidated_time = None
        self.save_calls = 0

    def name(self):
        return "Nya"

    def aliases(self):
        return []

    def bump_generation(self, reason: str = ""):
        self.generation += 1

    def is_generation_stale(self, expected_generation):
        return expected_generation is not None and self.generation != expected_generation

    async def save_session(self, *args, **kwargs):
        self.save_calls += 1
        return True


class _MemoryService:
    def __init__(self):
        self.updated = False

    async def update_short_term(self, messages_chunk):
        self.updated = True

    async def search(self, *args, **kwargs):
        return types.SimpleNamespace(mem_history=[], raw_records=[], stats={})

    async def consolidate(self, *args, **kwargs):
        return None


class _MemoryServiceBumpsAfterShortTerm(_MemoryService):
    async def update_short_term(self, messages_chunk):
        self.updated = True
        self.session.bump_generation("admin-change-after-short-term")


class _FeedbackBumpsGeneration:
    def __init__(self, session):
        self.session = session

    async def process(self, *args, **kwargs):
        self.session.bump_generation("feedback-returned-after-admin-change")
        return []


class _FeedbackOk:
    async def process(self, *args, **kwargs):
        return []


class _ChatBumpsGeneration:
    def __init__(self, session):
        self.session = session
        self.called = False

    async def plan_reply(self, *args, **kwargs):
        self.called = True
        self.session.bump_generation("chat-returned-after-admin-change")
        return [{"content": "stale reply"}]


class _ChatRecordsCall:
    def __init__(self):
        self.called = False

    async def plan_reply(self, *args, **kwargs):
        self.called = True
        return [{"content": "should not be called"}]


def _make_stage_session(module):
    session = module.Session.__new__(module.Session)
    session.id = "session-1"
    session.generation = 0
    session._Session__name = "Nya"
    session._Session__role = "role"
    session._Session__aliases = []
    session._Session__examples_str = ""
    session._Session__chatting_state = module._ChattingState.IDLE
    session.willingness = 1.0
    session.global_emotion = module.EmotionState()
    session.global_memory = _Memory()
    session.profiles = {}
    session.chat_summary = ""
    session.long_term_memory = _LongTermMemory()
    session._save_lock = asyncio.Lock()
    session._background_tasks = set()
    return session


class TurnGenerationInterleavingTests(unittest.TestCase):
    def test_set_role_bumps_generation_for_in_flight_turns(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)

            async def save_session():
                return True

            session.save_session = save_session

            asyncio.run(session.set_role("NewName", "new role"))

            self.assertEqual(1, session.generation)
        finally:
            _restore_modules(saved)

    def test_reset_bumps_generation_for_in_flight_turns(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)

            async def save_session():
                return True

            class FakeSessionRepo:
                @staticmethod
                async def delete_session_data(session_id):
                    return None

            session.save_session = save_session
            module.SessionStateRepository = FakeSessionRepo

            asyncio.run(session.reset())

            self.assertEqual(1, session.generation)
            self.assertTrue(session.global_memory.cleared)
            self.assertTrue(session.long_term_memory.cleared)
        finally:
            _restore_modules(saved)

    def test_stale_background_save_skips_after_generation_change(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)
            session._save_lock = asyncio.Lock()
            save_calls = []

            class FakeSessionRepo:
                @staticmethod
                async def save_session_state(session_id, data):
                    save_calls.append((session_id, data))

            module.SessionStateRepository = FakeSessionRepo

            async def scenario():
                await session._save_lock.acquire()
                task = asyncio.create_task(session.save_session(expected_generation=0))
                await asyncio.sleep(0)
                session.bump_generation("reset")
                session._save_lock.release()
                return await task

            result = asyncio.run(scenario())

            self.assertFalse(result)
            self.assertEqual([], save_calls)
        finally:
            _restore_modules(saved)

    def test_reset_deletes_and_final_saves_under_save_lock(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)
            session._save_lock = asyncio.Lock()
            delete_lock_states = []
            save_lock_states = []

            class FakeSessionRepo:
                @staticmethod
                async def delete_session_data(session_id):
                    delete_lock_states.append(session._save_lock.locked())

            async def save_session_locked(force_index=False):
                save_lock_states.append(session._save_lock.locked())
                return True

            module.SessionStateRepository = FakeSessionRepo
            session._save_session_locked = save_session_locked

            asyncio.run(session.reset())

            self.assertEqual([True], delete_lock_states)
            self.assertEqual([True], save_lock_states)
        finally:
            _restore_modules(saved)

    def test_stale_interaction_log_task_skips_after_generation_change(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)
            session._save_lock = asyncio.Lock()
            interactions = []

            class FakeProfileRepository:
                @staticmethod
                async def log_interaction(session_id, user_id, delta):
                    interactions.append((session_id, user_id, delta))

            module.ProfileRepository = FakeProfileRepository
            session.bump_generation("reset")

            asyncio.run(session._save_interaction_log(
                "user-1",
                {"valence": 0.5},
                expected_generation=0,
            ))

            self.assertEqual([], interactions)
        finally:
            _restore_modules(saved)

    def test_calm_down_bumps_generation_for_in_flight_turns(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)

            async def save_session():
                return True

            session.save_session = save_session

            asyncio.run(session.calm_down())

            self.assertEqual(1, session.generation)
        finally:
            _restore_modules(saved)

    def test_reset_emotion_bumps_generation_for_in_flight_turns(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)

            async def save_session():
                return True

            session.save_session = save_session

            asyncio.run(session.reset_emotion())

            self.assertEqual(1, session.generation)
        finally:
            _restore_modules(saved)

    def test_load_preset_bumps_generation_for_prompt_identity_change(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)

            async def save_session():
                return True

            session.save_session = save_session
            module.PRESETS = {
                "nya.json": types.SimpleNamespace(
                    name="Nya",
                    aliases=["nyan"],
                    role="cat role",
                    examples=[],
                    knowledges=["knows tea"],
                    relationships=[],
                    events=[],
                    bot_self=[],
                    hidden=False,
                )
            }

            loaded = asyncio.run(session.load_preset("nya"))

            self.assertTrue(loaded)
            self.assertEqual(1, session.generation)
            self.assertEqual("Nya", session.name())
            self.assertEqual(["nyan"], session.aliases())
        finally:
            _restore_modules(saved)

    def test_reset_command_bumps_generation_before_backup(self):
        source = (PLUGIN_DIR / "handlers" / "commands.py").read_text(encoding="utf-8")
        block = source[source.index("async def do_reset"):source.index("@get_status.handle()")]

        self.assertLess(block.index("state.session.bump_generation"), block.index("await backup_task()"))

    def test_feedback_stage_discards_stale_llm_result_before_sediment(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)
            sediment_calls = []
            decision_calls = []

            async def run_feedback_llm(self, *args, **kwargs):
                self.bump_generation("admin-change")
                return module._FeedbackContext(
                    response_dict={"willing": 0.9, "new_emotion": {"valence": 1.0}},
                    existing_related_memories=[],
                    allow_memory_supersede=False,
                    active_user_ids=set(),
                )

            def apply_sediment(self, *args, **kwargs):
                sediment_calls.append(True)

            async def apply_decision(self, *args, **kwargs):
                decision_calls.append(True)
                return ["history"]

            session._run_feedback_llm = types.MethodType(run_feedback_llm, session)
            session._apply_sediment = types.MethodType(apply_sediment, session)
            session._apply_decision = types.MethodType(apply_decision, session)

            result = asyncio.run(session.feedback_stage(
                [_Message()],
                lambda *args, **kwargs: "",
                expected_generation=0,
            ))

            self.assertEqual([], result)
            self.assertEqual([], sediment_calls)
            self.assertEqual([], decision_calls)
        finally:
            _restore_modules(saved)

    def test_chat_stage_discards_stale_reply_before_state_mutation(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)

            async def chat_llm(prompt, json_mode=False):
                session.bump_generation("admin-change")
                return '{"reply": [{"content": "stale reply"}]}'

            replies = asyncio.run(session.chat_stage(
                [_Message()],
                chat_llm,
                recalled_history=[],
                expected_generation=0,
            ))

            self.assertEqual([], replies)
            self.assertEqual(1.0, session.willingness)
            self.assertEqual(module._ChattingState.IDLE, session._Session__chatting_state)
        finally:
            _restore_modules(saved)

    def test_orchestrator_stops_after_feedback_if_generation_changed(self):
        module, saved = _load_orchestrator_module()
        try:
            session = _FakeSessionForOrchestrator()
            chat_service = _ChatRecordsCall()
            orchestrator = module.ConversationOrchestrator(
                session,
                memory_service=_MemoryService(),
                feedback_service=_FeedbackBumpsGeneration(session),
                chat_service=chat_service,
            )

            result = asyncio.run(orchestrator.process_chunk(
                [_Message()],
                lambda *args, **kwargs: "",
                lambda *args, **kwargs: "",
                expected_generation=0,
            ))

            self.assertIsNone(result)
            self.assertFalse(chat_service.called)
            self.assertEqual(0, session.save_calls)
        finally:
            _restore_modules(saved)

    def test_orchestrator_stops_before_state_mutation_if_generation_changes_after_short_term_update(self):
        module, saved = _load_orchestrator_module()
        try:
            session = _FakeSessionForOrchestrator()
            session.willingness = 0.5
            session._last_activity_time = datetime.min
            memory_service = _MemoryServiceBumpsAfterShortTerm()
            memory_service.session = session
            chat_service = _ChatRecordsCall()
            orchestrator = module.ConversationOrchestrator(
                session,
                memory_service=memory_service,
                feedback_service=_FeedbackOk(),
                chat_service=chat_service,
            )

            result = asyncio.run(orchestrator.process_chunk(
                [_Message()],
                lambda *args, **kwargs: "",
                lambda *args, **kwargs: "",
                expected_generation=0,
            ))

            self.assertIsNone(result)
            self.assertTrue(memory_service.updated)
            self.assertFalse(chat_service.called)
            self.assertEqual(0.5, session.willingness)
            self.assertEqual(datetime.min, session._last_activity_time)
        finally:
            _restore_modules(saved)

    def test_orchestrator_discards_replies_if_generation_changed_during_chat(self):
        module, saved = _load_orchestrator_module()
        try:
            session = _FakeSessionForOrchestrator()
            chat_service = _ChatBumpsGeneration(session)
            orchestrator = module.ConversationOrchestrator(
                session,
                memory_service=_MemoryService(),
                feedback_service=_FeedbackOk(),
                chat_service=chat_service,
            )

            result = asyncio.run(orchestrator.process_chunk(
                [_Message()],
                lambda *args, **kwargs: "",
                lambda *args, **kwargs: "",
                expected_generation=0,
            ))

            self.assertIsNone(result)
            self.assertTrue(chat_service.called)
        finally:
            _restore_modules(saved)

    def test_save_long_term_memory_rechecks_generation_before_vector_write(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)
            writes = []

            def add_memories_with_dedup(pending_memories):
                writes.extend(pending_memories)
                return {"added": len(pending_memories), "skipped_dedup": 0}

            session.long_term_memory.add_memories_with_dedup = add_memories_with_dedup

            def fake_run_sync(func):
                async def wrapper(*args, **kwargs):
                    session.bump_generation("admin-change")
                    return func(*args, **kwargs)
                return wrapper

            module.run_sync = fake_run_sync

            asyncio.run(session.save_long_term_memory(
                [{"content": "stale memory", "category": "event"}],
                expected_generation=0,
            ))

            self.assertEqual([], writes)
        finally:
            _restore_modules(saved)

    def test_save_long_term_memory_rechecks_generation_after_waiting_for_vector_gate(self):
        module, saved = _load_session_module()
        try:
            session = _make_stage_session(module)
            writes = []

            class BumpOnEnterLock:
                def __init__(self):
                    self.entered = 0

                def __enter__(self):
                    self.entered += 1
                    session.bump_generation("admin-change-while-vector-write-waits")
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            vector_gate = BumpOnEnterLock()
            module.BACKUP_IO_LOCK = vector_gate

            def add_memories_with_dedup(pending_memories):
                writes.extend(pending_memories)
                return {"added": len(pending_memories), "skipped_dedup": 0}

            session.long_term_memory.add_memories_with_dedup = add_memories_with_dedup

            asyncio.run(session.save_long_term_memory(
                [{"content": "stale memory", "category": "event"}],
                expected_generation=0,
            ))

            self.assertEqual(1, vector_gate.entered)
            self.assertEqual([], writes)
        finally:
            _restore_modules(saved)


if __name__ == "__main__":
    unittest.main()
