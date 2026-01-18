# nyaturingtest/client.py
import httpx
from openai import AsyncOpenAI
from nonebot import logger


class LLMClient:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate_response(self, prompt: str, model: str, temperature: float = 0.7, system_prompt: str = None, **kwargs) -> str | None:
        """
        生成回复，支持透传参数和自定义 System Prompt
        """
        try:
            if not system_prompt:
                system_content = (
                    "You are an intelligent agent. "
                    "Output the final response in JSON format."
                )
            else:
                system_content = system_prompt

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=60.0,  # 显式增加超时设置（单位：秒），防止大文本处理中断
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            return None
