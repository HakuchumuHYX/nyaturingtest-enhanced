from typing import Callable, Optional, List

import asyncio
import httpx
from openai import AsyncOpenAI
from nonebot import logger


class VLM:
    """
    通用视觉语言模型(VLM)适配器

    provider:
      - openai_compatible: OpenAI SDK compatible (/chat/completions) endpoint
      - google_ai_studio: Gemini Developer API official endpoint (generateContent)
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        endpoint: str,
        timeout: int = 60,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        *,
        provider: str = "openai_compatible",
        google_api_key: Optional[str] = None,
        google_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        proxy: Optional[str] = None,
    ):
        self.provider = (provider or "openai_compatible").strip().lower()
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.proxy = proxy

        # openai-compatible client
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            timeout=timeout,
        )
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            http_client=self._http_client,
        )

        # google official config
        self.google_api_key = (google_api_key or "").strip() or None
        self.google_base_url = (google_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    async def close(self):
        """关闭内部 HTTP 客户端，中断所有正在进行的请求"""
        try:
            await self._http_client.aclose()
        except Exception:
            pass

    @staticmethod
    def _parse_gemini_text_and_usage(data: dict) -> tuple[str, dict]:
        candidates = data.get("candidates") or []
        text_parts: List[str] = []
        if candidates:
            c0 = candidates[0] or {}
            content = (c0.get("content") or {})
            parts = content.get("parts") or []
            for p in parts:
                if isinstance(p, dict) and p.get("text"):
                    text_parts.append(str(p.get("text")))
        text = "\n".join(text_parts).strip()

        usage = data.get("usageMetadata") or {}
        usage_out = {
            "prompt_tokens": int(usage.get("promptTokenCount") or 0),
            "completion_tokens": int(usage.get("candidatesTokenCount") or 0),
            "total_tokens": int(usage.get("totalTokenCount") or 0),
        }
        return text, usage_out

    async def _request_google(
        self,
        *,
        prompt: str,
        image_base64: str,
        image_format: str,
        on_usage: Callable[[dict], None] | None,
        **kwargs,
    ) -> str | None:
        api_key = (self.google_api_key or "").strip()
        if not api_key:
            raise RuntimeError("google_api_key is required when provider=google_ai_studio")

        mime = f"image/{(image_format or 'jpeg').lower()}"
        payload: dict = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": mime, "data": image_base64}},
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": int(kwargs.pop("max_tokens", 1024) or 1024),
            },
        }

        # ignore OpenAI-only args
        kwargs.pop("extra_body", None)
        kwargs.pop("response_format", None)

        async with httpx.AsyncClient(
            base_url=self.google_base_url,
            proxy=self.proxy,
            timeout=float(kwargs.pop("timeout", self.timeout) or self.timeout),
        ) as client:
            resp = await client.post(
                f"/models/{self.model}:generateContent",
                params={"key": api_key},
                json=payload,
            )

        if resp.status_code != 200:
            raise Exception(f"API Error {resp.status_code}: {resp.text}")

        data = resp.json()
        text, usage = self._parse_gemini_text_and_usage(data)

        if on_usage:
            try:
                on_usage(usage)
            except Exception as ex:
                logger.warning(f"VLM Usage callback failed: {ex}")

        return text

    async def request(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        on_usage: Callable[[dict], None] | None = None,
        **kwargs,  # 透传参数 (如 extra_body)
    ) -> str | None:
        """
        让vlm根据图片和文本提示词生成描述 (带重试机制)
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                if self.provider == "google_ai_studio":
                    content = await self._request_google(
                        prompt=prompt,
                        image_base64=image_base64,
                        image_format=image_format,
                        on_usage=on_usage,
                        **kwargs,
                    )
                else:
                    # openai_compatible 模式下，过滤掉 Gemini 专属的 extra_body 字段，避免代理端报错
                    filtered_kwargs = dict(kwargs)
                    if "extra_body" in filtered_kwargs and isinstance(filtered_kwargs["extra_body"], dict):
                        filtered_kwargs["extra_body"] = {
                            k: v for k, v in filtered_kwargs["extra_body"].items()
                            if k != "google"
                        }
                        if not filtered_kwargs["extra_body"]:
                            del filtered_kwargs["extra_body"]
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
                        **filtered_kwargs,
                    )

                    if on_usage and response.usage:
                        try:
                            on_usage(response.usage.model_dump())
                        except Exception as ex:
                            logger.warning(f"VLM Usage callback failed: {ex}")

                    content = response.choices[0].message.content

                if content:
                    return content
                logger.warning(f"VLM 返回内容为空 (尝试 {retries + 1}/{self.max_retries + 1})")

            except Exception as e:
                logger.warning(f"VLM 请求失败: {e} (尝试 {retries + 1}/{self.max_retries + 1})")

            retries += 1
            if retries <= self.max_retries:
                await asyncio.sleep(self.retry_delay)

        logger.error("VLM 请求最终失败，已跳过")
        return None
