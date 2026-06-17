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
    content = '{"visual_description":"x"}'


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
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class VlmDetailPassthroughTests(unittest.TestCase):
    def _make_vlm(self):
        vlm = object.__new__(VLM)
        vlm.model = "some-vision-model"
        vlm.timeout = 30
        vlm.max_retries = 0
        vlm.retry_delay = 0
        vlm.client = FakeClient()
        return vlm

    def test_default_detail_is_low(self):
        vlm = self._make_vlm()
        asyncio.run(vlm.request(prompt="p", image_base64="abc", image_format="jpeg"))
        self.assertEqual(1, len(vlm.client.chat.completions.calls))
        self.assertEqual("low", vlm.client.chat.completions.calls[0]["messages"][0]["content"][0]["image_url"]["detail"])

    def test_detail_high_passthrough(self):
        vlm = self._make_vlm()
        asyncio.run(vlm.request(prompt="p", image_base64="abc", image_format="jpeg", detail="high"))
        self.assertEqual("high", vlm.client.chat.completions.calls[0]["messages"][0]["content"][0]["image_url"]["detail"])

    def test_detail_auto_passthrough(self):
        vlm = self._make_vlm()
        asyncio.run(vlm.request(prompt="p", image_base64="abc", image_format="jpeg", detail="auto"))
        self.assertEqual("auto", vlm.client.chat.completions.calls[0]["messages"][0]["content"][0]["image_url"]["detail"])


if __name__ == "__main__":
    unittest.main()
