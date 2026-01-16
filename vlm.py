from openai import AsyncOpenAI
from nonebot import logger
import asyncio
import httpx  # [新增] 用于配置连接池


class SiliconFlowVLM:
    """
    硅基流动视觉语言模型(VLM)适配器
    """

    def __init__(
            self,
            api_key: str,
            model: str = "Qwen/Qwen2.5-VL-72B-Instruct",  # 默认模型
            endpoint: str = "https://api.siliconflow.cn/v1",
            timeout: int = 30,
            max_retries: int = 1,
            retry_delay: float = 1.0,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            # 显式配置 HTTP 客户端，增加连接池大小，防止多图片并发阻塞
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                timeout=timeout  # 初始化时也传入超时作为保底
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
                    # 传入具体的请求超时
                    timeout=self.timeout
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
