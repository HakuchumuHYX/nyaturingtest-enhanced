import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODULE_NAME = "plugins.nyaturingtest.llm.client"
JSON_MODE_MODULE_NAME = "plugins.nyaturingtest.llm.json_mode"
STUB_MODULES = [
    "nonebot",
    "openai",
    "httpx",
    "plugins",
    "plugins.nyaturingtest",
    "plugins.nyaturingtest.llm",
    MODULE_NAME,
    JSON_MODE_MODULE_NAME,
]
_MISSING = object()


def _install_runtime_stubs():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    sys.modules["nonebot"] = nonebot

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = object
    openai.APIConnectionError = ConnectionError
    openai.APITimeoutError = TimeoutError
    sys.modules["openai"] = openai

    httpx = types.ModuleType("httpx")
    httpx.ConnectError = ConnectionError
    httpx.ReadTimeout = TimeoutError
    sys.modules["httpx"] = httpx


def _load_client_module():
    saved = {name: sys.modules.get(name, _MISSING) for name in STUB_MODULES}
    try:
        for name in STUB_MODULES:
            sys.modules.pop(name, None)
        _install_runtime_stubs()
        for package, path in [
            ("plugins", []),
            ("plugins.nyaturingtest", []),
            ("plugins.nyaturingtest.llm", [str(PLUGIN_DIR / "llm")]),
        ]:
            module = types.ModuleType(package)
            module.__path__ = path
            sys.modules[package] = module
        spec = importlib.util.spec_from_file_location(
            MODULE_NAME,
            PLUGIN_DIR / "llm" / "client.py",
        )
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


class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


class _JsonFallbackOpenAIClient:
    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError("Json mode is not supported for this model.")
        message = types.SimpleNamespace(content='{"reply":[]}', reasoning_content="")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=_FakeUsage(), model="compatible-model")


class _AlwaysRaisingOpenAIClient:
    def __init__(self, exc):
        self.calls = []
        self.exc = exc
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self.exc


class _JsonFallbackThenEmptyThenSuccessOpenAIClient:
    def __init__(self):
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("Json mode is not supported for this model.")
        if len(self.calls) == 2:
            message = types.SimpleNamespace(content="", reasoning_content="")
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            return types.SimpleNamespace(choices=[choice], usage=_FakeUsage(), model="compatible-model")
        message = types.SimpleNamespace(content='{"reply":[]}', reasoning_content="")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=_FakeUsage(), model="compatible-model")


class LLMClientJsonModeFallbackTests(unittest.TestCase):
    def test_client_uses_shared_json_mode_helper_when_loaded_as_package(self):
        module = _load_client_module()

        self.assertEqual(JSON_MODE_MODULE_NAME, module.is_json_mode_unsupported_error.__module__)

    def test_generate_retries_once_without_response_format_when_json_mode_unsupported(self):
        module = _load_client_module()
        fake = _JsonFallbackOpenAIClient()
        client = module.LLMClient(provider="openai_compatible", openai_client=fake)
        usage_rows = []

        response = asyncio.run(
            client.generate(
                prompt="p",
                model="compatible-model",
                response_format={"type": "json_object"},
                on_usage=usage_rows.append,
            )
        )

        self.assertEqual('{"reply":[]}', response.content)
        self.assertEqual(2, len(fake.calls))
        self.assertIn("response_format", fake.calls[0])
        self.assertNotIn("response_format", fake.calls[1])
        self.assertEqual(1, len(usage_rows))
        self.assertEqual("openai_compatible", usage_rows[0]["provider"])
        self.assertEqual([response.usage], usage_rows)

    def test_generate_does_not_fallback_for_unrelated_api_error(self):
        module = _load_client_module()
        fake = _AlwaysRaisingOpenAIClient(RuntimeError("bad request: missing required field"))
        client = module.LLMClient(provider="openai_compatible", openai_client=fake)

        response = asyncio.run(
            client.generate(
                prompt="p",
                model="compatible-model",
                response_format={"type": "json_object"},
            )
        )

        self.assertEqual("", response.content)
        self.assertEqual("api_error", response.usage["error_type"])
        self.assertEqual(1, len(fake.calls))
        self.assertIn("response_format", fake.calls[0])

    def test_generate_fallback_happens_only_once(self):
        module = _load_client_module()
        fake = _AlwaysRaisingOpenAIClient(RuntimeError("response_format is not supported"))
        client = module.LLMClient(provider="openai_compatible", openai_client=fake)

        response = asyncio.run(
            client.generate(
                prompt="p",
                model="compatible-model",
                response_format={"type": "json_object"},
            )
        )

        self.assertEqual("", response.content)
        self.assertEqual("api_error", response.usage["error_type"])
        self.assertEqual(2, len(fake.calls))
        self.assertIn("response_format", fake.calls[0])
        self.assertNotIn("response_format", fake.calls[1])

    def test_generate_keeps_response_format_removed_after_later_retry(self):
        module = _load_client_module()
        fake = _JsonFallbackThenEmptyThenSuccessOpenAIClient()
        client = module.LLMClient(provider="openai_compatible", openai_client=fake)

        response = asyncio.run(
            client.generate(
                prompt="p",
                model="compatible-model",
                response_format={"type": "json_object"},
            )
        )

        self.assertEqual('{"reply":[]}', response.content)
        self.assertEqual(3, len(fake.calls))
        self.assertIn("response_format", fake.calls[0])
        self.assertNotIn("response_format", fake.calls[1])
        self.assertNotIn("response_format", fake.calls[2])


if __name__ == "__main__":
    unittest.main()
