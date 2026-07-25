import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _install_runtime_stubs():
    nonebot = types.ModuleType("nonebot")
    nonebot.logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("nonebot", nonebot)

    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = object
    openai.APIConnectionError = RuntimeError
    openai.APITimeoutError = TimeoutError
    sys.modules.setdefault("openai", openai)

    httpx = types.ModuleType("httpx")
    httpx.ConnectError = ConnectionError
    httpx.ReadTimeout = TimeoutError
    sys.modules.setdefault("httpx", httpx)


def _load_client_module():
    _install_runtime_stubs()
    spec = importlib.util.spec_from_file_location(
        "nyaturingtest_llm_client",
        PLUGIN_DIR / "llm" / "client.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeUsage:
    def model_dump(self):
        return {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 12},
        }


class _FakeOpenAIClient:
    def __init__(self):
        self.last_kwargs = None
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = types.SimpleNamespace(
            content='{"reply":[]}',
            reasoning_content="internal reasoning",
        )
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(
            choices=[choice],
            usage=_FakeUsage(),
            model="deepseek-v4-flash",
        )


class _RaisingOpenAIClient:
    def __init__(self, exc):
        self.calls = 0
        self.exc = exc
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    async def create(self, **kwargs):
        self.calls += 1
        raise self.exc


class _FlakyOpenAIClient:
    def __init__(self, failures: int, exc):
        self.calls = 0
        self.failures = failures
        self.exc = exc
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        message = types.SimpleNamespace(content='{"reply":["ok"]}', reasoning_content="")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(
            choices=[choice],
            usage=_FakeUsage(),
            model="deepseek-v4-flash",
        )


class _StatusError(Exception):
    def __init__(self, status_code, text):
        super().__init__(text)
        self.status_code = status_code
        self.response = types.SimpleNamespace(status_code=status_code, text=text)


class DeepSeekLLMClientTests(unittest.TestCase):
    def test_generate_builds_native_multimodal_user_content(self):
        module = _load_client_module()
        fake = _FakeOpenAIClient()
        client = module.LLMClient(provider="openai_compatible", openai_client=fake)
        image = types.SimpleNamespace(
            to_openai_content=lambda: [
                {"type": "text", "text": "[当前消息图片 image_ref=primary:0:x]"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,abc",
                        "detail": "high",
                    },
                },
            ]
        )

        asyncio.run(
            client.generate(
                prompt="dynamic",
                model="vision-model",
                system_prompt="stable",
                images=[image],
            )
        )

        user_content = fake.last_kwargs["messages"][1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual({"type": "text", "text": "dynamic"}, user_content[0])
        self.assertEqual("primary:0:x", user_content[1]["text"].split("image_ref=")[1][:-1])
        self.assertEqual("image_url", user_content[2]["type"])
        self.assertEqual("high", user_content[2]["image_url"]["detail"])

    def test_generate_filters_sampling_when_thinking_is_enabled(self):
        module = _load_client_module()
        fake = _FakeOpenAIClient()
        client = module.LLMClient(provider="deepseek_official", openai_client=fake)

        response = asyncio.run(
            client.generate(
                prompt="dynamic",
                model="deepseek-v4-flash",
                temperature=0.8,
                system_prompt="stable",
                top_p=0.9,
                presence_penalty=0.2,
                frequency_penalty=0.2,
                reasoning_effort="high",
                response_format={"type": "json_object"},
                extra_body={
                    "thinking": {"type": "enabled"},
                },
            )
        )

        self.assertEqual('{"reply":[]}', response.content)
        self.assertEqual("internal reasoning", response.reasoning_content)
        self.assertEqual("stop", response.finish_reason)
        self.assertNotIn("temperature", fake.last_kwargs)
        self.assertNotIn("top_p", fake.last_kwargs)
        self.assertNotIn("presence_penalty", fake.last_kwargs)
        self.assertNotIn("frequency_penalty", fake.last_kwargs)
        self.assertEqual(
            {"type": "enabled"},
            fake.last_kwargs["extra_body"]["thinking"],
        )
        self.assertEqual("high", fake.last_kwargs["reasoning_effort"])

    def test_generate_reports_deepseek_cache_and_reasoning_usage(self):
        module = _load_client_module()
        fake = _FakeOpenAIClient()
        client = module.LLMClient(provider="deepseek_official", openai_client=fake)
        recorded = []

        response = asyncio.run(
            client.generate(
                prompt="dynamic",
                model="deepseek-v4-flash",
                system_prompt="stable",
                on_usage=recorded.append,
                extra_body={"thinking": {"type": "disabled"}},
            )
        )

        self.assertEqual("deepseek_official", response.provider)
        self.assertEqual("deepseek-v4-flash", response.model)
        self.assertEqual(80, response.usage["prompt_cache_hit_tokens"])
        self.assertEqual(20, response.usage["prompt_cache_miss_tokens"])
        self.assertEqual(12, response.usage["reasoning_tokens"])
        self.assertEqual("stop", response.usage["finish_reason"])
        self.assertEqual(response.usage, recorded[0])

    def test_generate_does_not_mutate_extra_body(self):
        module = _load_client_module()
        fake = _FakeOpenAIClient()
        client = module.LLMClient(provider="deepseek_official", openai_client=fake)
        extra_body = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }

        asyncio.run(
            client.generate(
                prompt="dynamic",
                model="deepseek-v4-flash",
                extra_body=extra_body,
            )
        )

        self.assertEqual(
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
            extra_body,
        )
        self.assertEqual({"thinking": {"type": "enabled"}}, fake.last_kwargs["extra_body"])
        self.assertEqual("high", fake.last_kwargs["reasoning_effort"])

    def test_429_sets_short_circuit_and_reports_provider_error(self):
        module = _load_client_module()
        fake = _RaisingOpenAIClient(_StatusError(429, "rate limit"))
        client = module.LLMClient(
            provider="deepseek_official",
            openai_client=fake,
            base_url="https://api.deepseek.com",
            api_key="same-key",
        )

        first = asyncio.run(client.generate("p", "deepseek-v4-flash"))
        second = asyncio.run(client.generate("p", "deepseek-v4-flash"))

        self.assertEqual("", first.content)
        self.assertEqual("rate_limit", first.usage["error_type"])
        self.assertEqual("circuit_open", second.usage["error_type"])
        self.assertEqual(1, fake.calls)
        self.assertEqual("rate_limit", client.provider_status.last_error_type)

    def test_429_creates_shared_advisory_backoff_without_global_hard_circuit(self):
        module = _load_client_module()
        sleep_calls = []
        original_sleep = module.asyncio.sleep

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        try:
            module.asyncio.sleep = fake_sleep
            first_fake = _RaisingOpenAIClient(_StatusError(429, "rate limit"))
            first_client = module.LLMClient(
                provider="deepseek_official",
                openai_client=first_fake,
                base_url="https://api.deepseek.com",
                api_key="same-key",
            )
            first = asyncio.run(first_client.generate("p", "deepseek-v4-flash"))

            second_fake = _FakeOpenAIClient()
            second_client = module.LLMClient(
                provider="deepseek_official",
                openai_client=second_fake,
                base_url="https://api.deepseek.com",
                api_key="same-key",
            )
            second = asyncio.run(second_client.generate("p", "deepseek-v4-flash"))
        finally:
            module.asyncio.sleep = original_sleep

        self.assertEqual("rate_limit", first.usage["error_type"])
        self.assertEqual('{"reply":[]}', second.content)
        self.assertTrue(second_fake.last_kwargs is not None)
        self.assertTrue(sleep_calls)
        self.assertLessEqual(max(sleep_calls), module.PROVIDER_ADVISORY_BACKOFF_MAX_SLEEP_SECONDS)
        self.assertNotEqual("circuit_open", second.usage.get("error_type"))

    def test_shared_advisory_backoff_key_includes_model(self):
        module = _load_client_module()
        sleep_calls = []
        original_sleep = module.asyncio.sleep

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        try:
            module.asyncio.sleep = fake_sleep
            first_fake = _RaisingOpenAIClient(_StatusError(429, "rate limit"))
            first_client = module.LLMClient(
                provider="openai_compatible",
                openai_client=first_fake,
                base_url="https://proxy.example/v1",
                api_key="same-key",
            )
            asyncio.run(first_client.generate("p", "model-a"))

            second_fake = _FakeOpenAIClient()
            second_client = module.LLMClient(
                provider="openai_compatible",
                openai_client=second_fake,
                base_url="https://proxy.example/v1",
                api_key="same-key",
            )
            response = asyncio.run(second_client.generate("p", "model-b"))
        finally:
            module.asyncio.sleep = original_sleep

        self.assertEqual('{"reply":[]}', response.content)
        self.assertEqual([], sleep_calls)

    def test_shared_advisory_backoff_key_isolates_base_url_and_api_key(self):
        module = _load_client_module()
        sleep_calls = []
        original_sleep = module.asyncio.sleep

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        try:
            module.asyncio.sleep = fake_sleep
            first_fake = _RaisingOpenAIClient(_StatusError(429, "rate limit"))
            first_client = module.LLMClient(
                provider="openai_compatible",
                openai_client=first_fake,
                base_url="https://proxy-a.example/v1",
                api_key="key-a",
            )
            asyncio.run(first_client.generate("p", "same-model"))

            other_base_url_fake = _FakeOpenAIClient()
            other_base_url_client = module.LLMClient(
                provider="openai_compatible",
                openai_client=other_base_url_fake,
                base_url="https://proxy-b.example/v1",
                api_key="key-a",
            )
            other_base_url = asyncio.run(other_base_url_client.generate("p", "same-model"))

            other_api_key_fake = _FakeOpenAIClient()
            other_api_key_client = module.LLMClient(
                provider="openai_compatible",
                openai_client=other_api_key_fake,
                base_url="https://proxy-a.example/v1",
                api_key="key-b",
            )
            other_api_key = asyncio.run(other_api_key_client.generate("p", "same-model"))
        finally:
            module.asyncio.sleep = original_sleep

        self.assertEqual('{"reply":[]}', other_base_url.content)
        self.assertEqual('{"reply":[]}', other_api_key.content)
        self.assertEqual([], sleep_calls)

    def test_content_filter_returns_empty_without_retry(self):
        module = _load_client_module()
        fake = _RaisingOpenAIClient(_StatusError(400, "content_filter"))
        client = module.LLMClient(provider="deepseek_official", openai_client=fake)

        response = asyncio.run(client.generate("p", "deepseek-v4-flash"))

        self.assertEqual("", response.content)
        self.assertEqual("content_filter", response.usage["error_type"])
        self.assertEqual(1, fake.calls)

    def test_network_errors_are_retried_by_llm_client_once_per_attempt(self):
        module = _load_client_module()
        sleep_calls = []
        original_sleep = module.asyncio.sleep

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        try:
            module.asyncio.sleep = fake_sleep
            fake = _FlakyOpenAIClient(failures=2, exc=module.httpx.ConnectError("temporary network"))
            client = module.LLMClient(provider="deepseek_official", openai_client=fake)

            response = asyncio.run(client.generate("p", "deepseek-v4-flash"))
        finally:
            module.asyncio.sleep = original_sleep

        self.assertEqual('{"reply":["ok"]}', response.content)
        self.assertEqual(3, fake.calls)
        self.assertEqual([2, 4], sleep_calls)


if __name__ == "__main__":
    unittest.main()
