# nyaturingtest/client.py
import asyncio
import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from nonebot import logger


class LLMClient:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate_response(self, prompt: str, model: str, temperature: float = 0.7, system_prompt: str = None,
                                **kwargs) -> str | None:
        """
        生成回复，支持透传参数和自定义 System Prompt，包含重试机制
        """
        # 1. 构造消息
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

        # 2. 重试配置
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                # 显式设置超时，覆盖默认值
                # timeout 可以是 float (总超时) 或 httpx.Timeout 对象
                request_timeout = kwargs.pop("timeout", 60.0)

                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=request_timeout,
                    **kwargs
                )
                return response.choices[0].message.content

            except (APIConnectionError, APITimeoutError, httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.warning(f"[LLM] 网络请求失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__} - {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))
                else:
                    logger.error(f"[LLM] 最终请求失败: {e}")
                    return None

            except Exception as e:
                logger.error(f"[LLM] API 调用发生不可重试错误: {e}")
                return None

        return None
