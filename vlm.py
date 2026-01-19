from openai import AsyncOpenAI
from nonebot import logger
import asyncio
import httpx 


class VLM:
    """
    通用视觉语言模型(VLM)适配器
    """

    def __init__(
            self,
            api_key: str,
            model: str,
            endpoint: str,
            timeout: int = 30,
            max_retries: int = 1,
            retry_delay: float = 1.0,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                timeout=timeout
            )
        )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def request(
            self,
            prompt: str,
            image_base64: str,
            image_format: str,
            **kwargs,  # 添加 **kwargs 接收额外参数
    ) -> str | None:
        """
        让vlm根据图片和文本提示词生成描述 (带重试机制)
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"},
                                },
                                {"type": "text", "text": f"{prompt}"},
                            ],
                        }
                    ],
                    timeout=self.timeout,
                    **kwargs  # 透传参数 (如 extra_body)
                )
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    logger.warning(f"VLM 返回内容为空 (尝试 {retries + 1}/{self.max_retries + 1})")

            except Exception as e:
                logger.warning(f"VLM 请求失败: {e} (尝试 {retries + 1}/{self.max_retries + 1})")

            retries += 1
            if retries <= self.max_retries:
                await asyncio.sleep(self.retry_delay)

        logger.error("VLM 请求最终失败，已跳过")
        return None
