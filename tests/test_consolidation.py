import asyncio
import importlib.util
import sys
import types
import unittest
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


def _load_orchestrator_module(runtime_settings: dict):
    module_name = "consolidation_orchestrator_under_test.core.orchestrator"
    package_root = "consolidation_orchestrator_under_test"
    stub_names = [
        "nonebot",
        package_root,
        f"{package_root}.core",
        f"{package_root}.config",
        f"{package_root}.memory",
        f"{package_root}.memory.short_term",
        f"{package_root}.utils",
        f"{package_root}.core.services",
        module_name,
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in stub_names}
    try:
        for name in stub_names:
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
            package_root,
            f"{package_root}.core",
            f"{package_root}.memory",
        ]:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module

        config = types.ModuleType(f"{package_root}.config")
        config.get_runtime_settings = lambda: dict(runtime_settings)
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

        spec = importlib.util.spec_from_file_location(module_name, PLUGIN_DIR / "core" / "orchestrator.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, saved
    except Exception:
        _restore_modules(saved)
        raise


class _Message:
    content = "ordinary group chat"
    user_name = "Alice"
    user_id = "user-1"
    time = datetime.now()


class _LowWillingnessSession:
    id = "session-1"

    def __init__(self):
        self.generation = 0
        self.willingness = 0.0
        self._last_decay_time = datetime.now()
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.now()
        self._engaged = False
        self._messages_since_consolidation = 1
        self._last_consolidation_attempt = datetime.now()
        self.last_consolidated_time = None

    def name(self):
        return "Nya"

    def aliases(self):
        return []

    def is_generation_stale(self, expected_generation):
        return False


class _MemoryService:
    def __init__(self):
        self.consolidated_chunks = []

    async def update_short_term(self, messages_chunk):
        return None

    async def consolidate(self, messages_chunk, feedback_llm_func, *, expected_generation=None):
        self.consolidated_chunks.append(list(messages_chunk))


class _UnexpectedFeedbackService:
    async def process(self, *args, **kwargs):
        raise AssertionError("low-willingness consolidation path must not run Feedback decision")


class _UnexpectedChatService:
    async def plan_reply(self, *args, **kwargs):
        raise AssertionError("low-willingness consolidation path must not run Chat")


class ConsolidationSchemaTests(unittest.TestCase):
    def test_session_model_has_consolidation_watermark(self):
        source = (PLUGIN_DIR / "models" / "database.py").read_text(encoding="utf-8")
        self.assertIn("last_consolidated_time", source)

    def test_migration_adds_consolidation_column(self):
        source = (PLUGIN_DIR / "database" / "migrations.py").read_text(encoding="utf-8")
        self.assertIn("SCHEMA_VERSION = 3", source)
        self.assertIn("last_consolidated_time", source)

    def test_repository_persists_watermark(self):
        source = (PLUGIN_DIR / "database" / "session_repository.py").read_text(encoding="utf-8")
        self.assertIn('"last_consolidated_time"', source)

    def test_feedback_split_into_three_phases(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("async def _run_feedback_llm", source)
        self.assertIn("def _apply_sediment", source)
        self.assertIn("def _apply_decision", source)
        # feedback_stage 仍存在且现在由三段组合
        self.assertIn("async def feedback_stage", source)

    def test_search_stage_supports_force_retrieve(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("force_retrieve", source)
        self.assertIn("should_retrieve = force_retrieve or", source)

    def test_consolidate_stage_exists_and_advances_watermark(self):
        source = (PLUGIN_DIR / "core" / "session.py").read_text(encoding="utf-8")
        self.assertIn("async def consolidate_stage", source)
        self.assertIn("self.last_consolidated_time", source)
        # 固化路径不得触碰回复意愿（不调用 _apply_decision）
        cs = source[source.index("async def consolidate_stage"):]
        cs = cs[:cs.index("\n    async def ", 1)] if "\n    async def " in cs[1:] else cs
        self.assertNotIn("_apply_decision", cs)

    def test_config_defines_consolidation_keys(self):
        source = (PLUGIN_DIR / "config.py").read_text(encoding="utf-8")
        for snippet in [
            '"consolidation_enabled": True',
            '"consolidation_message_threshold": 8',
            '"consolidation_interval_seconds": 180.0',
            '"consolidation_max_messages": 60',
            '"consolidation_message_threshold": number("consolidation_message_threshold", 8, int, minimum=1)',
        ]:
            self.assertIn(snippet, source)

    def test_orchestrator_triggers_consolidation_not_periodic_feedback(self):
        source = (PLUGIN_DIR / "core" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("consolidate", source)
        self.assertIn("consolidation_message_threshold", source)

    def test_memory_service_has_consolidate(self):
        source = (PLUGIN_DIR / "core" / "services.py").read_text(encoding="utf-8")
        self.assertIn("async def consolidate", source)
        self.assertIn("def note_incoming", source)

    def test_low_willingness_non_relevant_chunk_can_trigger_consolidation(self):
        runtime_settings = {
            "active_to_bubble_threshold": 0.5,
            "consolidation_enabled": True,
            "consolidation_interval_seconds": 9999,
            "consolidation_max_messages": 60,
            "consolidation_message_threshold": 1,
            "interest_topic_willingness_floor": 0.45,
            "low_willingness_skip_threshold": 0.3,
            "passive_growth_max_factor": 1.0,
            "passive_growth_min_factor": 1.0,
            "passive_willingness_growth_limit": 0.7,
            "passive_willingness_growth_per_message": 0.0,
            "post_feedback_skip_threshold": 0.34,
            "relevance_willingness_floor": 0.7,
            "rerank_willingness_threshold": 0.68,
            "speak_cooldown_seconds": 16.0,
            "willingness_decay_rate_active": 0.0,
            "willingness_decay_rate_idle": 0.0,
            "willingness_idle_after_seconds": 300.0,
            "willingness_reply_threshold": 0.4,
        }
        module, saved = _load_orchestrator_module(runtime_settings)
        try:
            session = _LowWillingnessSession()
            memory_service = _MemoryService()
            orchestrator = module.ConversationOrchestrator(
                session,
                memory_service=memory_service,
                feedback_service=_UnexpectedFeedbackService(),
                chat_service=_UnexpectedChatService(),
            )

            result = asyncio.run(orchestrator.process_chunk(
                [_Message()],
                lambda *args, **kwargs: "",
                lambda *args, **kwargs: "",
                expected_generation=0,
            ))

            self.assertIsNone(result)
            self.assertEqual(1, len(memory_service.consolidated_chunks))
            self.assertEqual("ordinary group chat", memory_service.consolidated_chunks[0][0].content)
        finally:
            _restore_modules(saved)


if __name__ == "__main__":
    unittest.main()
