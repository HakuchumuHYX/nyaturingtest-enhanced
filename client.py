# nyaturingtest/client.py
import httpx
from openai import AsyncOpenAI
from nonebot import logger


class LLMClient:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate_response(self, prompt: str, model: str, temperature: float = 0.7, **kwargs) -> str | None:
        """
        生成回复，支持透传参数 (如 response_format)
        """
        try:
            system_content = (
                "You are a helpful assistant designed to output JSON directly. "
                "You may use <think>...</think> tags for internal reasoning and planning. "
                "However, the final output MUST be a valid JSON object outside the tags."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            return None
