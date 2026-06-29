import asyncio
from pathlib import Path
import sys
import unittest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

from llm.vlm import VLM


class FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 1, "completion_tokens": 2}


class FakeMessage:
    content = '{"description":"ok","emotion":"neutral"}'


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError("Error code: 400 - {'code': 20024, 'message': 'Json mode is not supported for this model.'}")
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class VLMJsonModeFallbackTests(unittest.TestCase):
    def test_glm_46v_does_not_send_response_format(self):
        vlm = object.__new__(VLM)
        vlm.model = "zai-org/GLM-4.6V"
        vlm.timeout = 30
        vlm.max_retries = 0
        vlm.retry_delay = 0
        vlm.client = FakeClient()

        result = asyncio.run(
            vlm.request(
                prompt="describe",
                image_base64="abc",
                image_format="jpeg",
                response_format={"type": "json_object"},
            )
        )

        calls = vlm.client.chat.completions.calls
        self.assertEqual('{"description":"ok","emotion":"neutral"}', result)
        self.assertEqual(1, len(calls))
        self.assertNotIn("response_format", calls[0])

    def test_request_retries_without_response_format_when_json_mode_is_unsupported(self):
        vlm = object.__new__(VLM)
        vlm.model = "other-vision-model"
        vlm.timeout = 30
        vlm.max_retries = 0
        vlm.retry_delay = 0
        vlm.client = FakeClient()

        usage_rows = []
        result = asyncio.run(
            vlm.request(
                prompt="describe",
                image_base64="abc",
                image_format="jpeg",
                response_format={"type": "json_object"},
                on_usage=usage_rows.append,
            )
        )

        calls = vlm.client.chat.completions.calls
        self.assertEqual('{"description":"ok","emotion":"neutral"}', result)
        self.assertEqual(2, len(calls))
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])
        self.assertEqual([
            {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "provider": "openai_compatible",
                "finish_reason": "",
            }
        ], usage_rows)


if __name__ == "__main__":
    unittest.main()
