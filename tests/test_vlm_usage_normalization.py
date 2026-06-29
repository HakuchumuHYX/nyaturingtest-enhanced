import asyncio
from pathlib import Path
import sys
import unittest


PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

from llm.vlm import VLM


class FakeUsage:
    def model_dump(self):
        return {
            "prompt_tokens": 7,
            "completion_tokens": 5,
        }


class FakeMessage:
    content = '{"description":"ok","emotion":"neutral"}'


class FakeChoice:
    message = FakeMessage()
    finish_reason = "stop"


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]


class FakeCompletions:
    async def create(self, **kwargs):
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class VLMUsageNormalizationTests(unittest.TestCase):
    def test_vlm_usage_callback_gets_provider_finish_reason_and_total_tokens(self):
        vlm = object.__new__(VLM)
        vlm.provider = "openai_compatible"
        vlm.model = "vision-model"
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
                on_usage=usage_rows.append,
            )
        )

        self.assertEqual('{"description":"ok","emotion":"neutral"}', result)
        self.assertEqual(
            [
                {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                    "provider": "openai_compatible",
                    "finish_reason": "stop",
                }
            ],
            usage_rows,
        )


if __name__ == "__main__":
    unittest.main()
