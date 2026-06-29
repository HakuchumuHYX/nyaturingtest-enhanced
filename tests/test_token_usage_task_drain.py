import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_usage_module(log_token_usage):
    previous = {}

    def install(name, module):
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module

    package_name = "usage_drain_under_test"
    root = types.ModuleType(package_name)
    root.__path__ = [str(PLUGIN_DIR)]
    install(package_name, root)

    core = types.ModuleType(f"{package_name}.core")
    core.__path__ = [str(PLUGIN_DIR / "core")]
    install(f"{package_name}.core", core)

    database = types.ModuleType(f"{package_name}.database")
    database.__path__ = [str(PLUGIN_DIR / "database")]
    install(f"{package_name}.database", database)

    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    install("nonebot", nonebot)

    token_repository = types.ModuleType(f"{package_name}.database.token_repository")
    token_repository.TokenUsageRepository = types.SimpleNamespace(
        log_token_usage=log_token_usage,
    )
    install(f"{package_name}.database.token_repository", token_repository)

    module_name = f"{package_name}.core.usage"
    previous[module_name] = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            PLUGIN_DIR / "core" / "usage.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, previous
    except Exception:
        _restore_modules(previous)
        raise


def _load_state_manager_module(drain_usage_tasks, events):
    previous = {}

    def install(name, module):
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module

    package_name = "state_usage_drain_under_test"
    root = types.ModuleType(package_name)
    root.__path__ = [str(PLUGIN_DIR)]
    install(package_name, root)

    for subpackage in ["core", "config", "memory", "database", "llm"]:
        module = types.ModuleType(f"{package_name}.{subpackage}")
        module.__path__ = [str(PLUGIN_DIR / subpackage)]
        install(f"{package_name}.{subpackage}", module)

    nonebot = types.ModuleType("nonebot")
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
    install("nonebot.adapters", types.ModuleType("nonebot.adapters"))
    install("nonebot.adapters.onebot", types.ModuleType("nonebot.adapters.onebot"))
    install("nonebot.adapters.onebot.v11", onebot)

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = lambda *args, **kwargs: object()
    install("openai", openai)

    tortoise = types.ModuleType("tortoise")

    async def close_connections():
        events.append("close_db")

    tortoise.Tortoise = types.SimpleNamespace(close_connections=close_connections)
    install("tortoise", tortoise)

    config = types.ModuleType(f"{package_name}.config")
    config.plugin_config = {}
    config.get_effective_chat_api_key = lambda: ""
    config.get_effective_chat_base_url = lambda: ""
    config.get_effective_chat_provider = lambda: "openai_compatible"
    config.get_chat_timeout = lambda: 30
    config.get_runtime_settings = lambda: {"memory_drain_timeout_seconds": 0.1}
    install(f"{package_name}.config", config)

    short_term = types.ModuleType(f"{package_name}.memory.short_term")
    short_term.Message = object
    install(f"{package_name}.memory.short_term", short_term)

    llm_client = types.ModuleType(f"{package_name}.llm.client")
    llm_client.LLMClient = lambda *args, **kwargs: object()
    install(f"{package_name}.llm.client", llm_client)

    utils = types.ModuleType(f"{package_name}.utils")
    utils.get_http_client = lambda: object()

    async def close_http_client():
        events.append("close_http")

    utils.close_http_client = close_http_client
    install(f"{package_name}.utils", utils)

    session_module = types.ModuleType(f"{package_name}.core.session")
    session_module.Session = lambda *args, **kwargs: object()
    install(f"{package_name}.core.session", session_module)

    enabled_group_repository = types.ModuleType(f"{package_name}.database.enabled_group_repository")
    enabled_group_repository.EnabledGroupRepository = types.SimpleNamespace(
        load_enabled_group_ids=lambda ids: ids,
    )
    install(f"{package_name}.database.enabled_group_repository", enabled_group_repository)

    usage = types.ModuleType(f"{package_name}.core.usage")
    usage.drain_usage_tasks = drain_usage_tasks
    install(f"{package_name}.core.usage", usage)

    async def close_vlm():
        events.append("close_vlm")

    image = types.ModuleType(f"{package_name}.memory.image")
    image.image_manager = types.SimpleNamespace(
        _vlm=types.SimpleNamespace(
            close=close_vlm
        )
    )
    install(f"{package_name}.memory.image", image)

    module_name = f"{package_name}.core.state_manager"
    previous[module_name] = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            PLUGIN_DIR / "core" / "state_manager.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, previous
    except Exception:
        _restore_modules(previous)
        raise


class TokenUsageTaskDrainTests(unittest.TestCase):
    def test_usage_recorder_registers_pending_task_and_removes_after_completion(self):
        async def scenario():
            release = asyncio.Event()
            started = asyncio.Event()

            async def log_token_usage(**kwargs):
                started.set()
                await release.wait()

            module, previous = _load_usage_module(log_token_usage)
            try:
                module.record_token_usage("session-1", "model-1", {"total_tokens": 3})
                await started.wait()

                self.assertEqual(1, len(module._PENDING_USAGE_TASKS))

                release.set()
                await module.drain_usage_tasks(timeout=1.0)

                self.assertEqual(0, len(module._PENDING_USAGE_TASKS))
            finally:
                _restore_modules(previous)

        asyncio.run(scenario())

    def test_drain_usage_tasks_waits_for_pending_task(self):
        async def scenario():
            release = asyncio.Event()
            completed = False

            async def log_token_usage(**kwargs):
                nonlocal completed
                await release.wait()
                completed = True

            module, previous = _load_usage_module(log_token_usage)
            try:
                module.record_token_usage("session-1", "model-1", {"total_tokens": 3})
                drain_task = asyncio.create_task(module.drain_usage_tasks(timeout=1.0))
                await asyncio.sleep(0)

                self.assertFalse(completed)
                self.assertFalse(drain_task.done())

                release.set()
                await drain_task

                self.assertTrue(completed)
                self.assertEqual(0, len(module._PENDING_USAGE_TASKS))
            finally:
                _restore_modules(previous)

        asyncio.run(scenario())

    def test_drain_usage_tasks_waits_for_tasks_registered_during_drain(self):
        async def scenario():
            first_started = asyncio.Event()
            first_release = asyncio.Event()
            second_started = asyncio.Event()
            second_release = asyncio.Event()
            second_completed = False
            calls = 0

            async def log_token_usage(**kwargs):
                nonlocal calls, second_completed
                calls += 1
                if calls == 1:
                    first_started.set()
                    await first_release.wait()
                    module.record_token_usage("session-1", "model-1", {"total_tokens": 4})
                    return
                second_started.set()
                await second_release.wait()
                second_completed = True

            module, previous = _load_usage_module(log_token_usage)
            try:
                module.record_token_usage("session-1", "model-1", {"total_tokens": 3})
                await first_started.wait()
                drain_task = asyncio.create_task(module.drain_usage_tasks(timeout=1.0))
                await asyncio.sleep(0)

                first_release.set()
                await second_started.wait()
                await asyncio.sleep(0.01)

                self.assertFalse(drain_task.done())
                self.assertEqual(1, len(module._PENDING_USAGE_TASKS))

                second_release.set()
                await drain_task

                self.assertTrue(second_completed)
                self.assertEqual(2, calls)
                self.assertEqual(0, len(module._PENDING_USAGE_TASKS))
            finally:
                second_release.set()
                _restore_modules(previous)

        asyncio.run(scenario())

    def test_task_exception_is_logged_and_removed(self):
        async def scenario():
            error_messages = []

            async def log_token_usage(**kwargs):
                raise RuntimeError("write failed")

            module, previous = _load_usage_module(log_token_usage)
            try:
                module.logger = types.SimpleNamespace(
                    error=lambda message: error_messages.append(message),
                    warning=lambda *args, **kwargs: None,
                )
                module.record_token_usage("session-1", "model-1", {"total_tokens": 3})
                await module.drain_usage_tasks(timeout=1.0)

                self.assertEqual(0, len(module._PENDING_USAGE_TASKS))
                self.assertTrue(any("write failed" in message for message in error_messages))
            finally:
                _restore_modules(previous)

        asyncio.run(scenario())

    def test_drain_usage_tasks_cancels_and_removes_unfinished_tasks_on_timeout(self):
        async def scenario():
            cancelled = False

            async def log_token_usage(**kwargs):
                nonlocal cancelled
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled = True
                    raise

            module, previous = _load_usage_module(log_token_usage)
            try:
                module.record_token_usage("session-1", "model-1", {"total_tokens": 3})
                await module.drain_usage_tasks(timeout=0.001)

                self.assertTrue(cancelled)
                self.assertEqual(0, len(module._PENDING_USAGE_TASKS))
            finally:
                _restore_modules(previous)

        asyncio.run(scenario())

    def test_cleanup_global_resources_drains_usage_before_database_close(self):
        async def scenario():
            events = []

            async def drain_usage_tasks(timeout=None):
                events.append("drain_usage")

            module, previous = _load_state_manager_module(drain_usage_tasks, events)
            try:
                await module.cleanup_global_resources()

                self.assertIn("drain_usage", events)
                self.assertIn("close_db", events)
                self.assertLess(events.index("drain_usage"), events.index("close_db"))
            finally:
                _restore_modules(previous)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
