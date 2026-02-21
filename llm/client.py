# nyaturingtest/client.py
import asyncio
from typing import Callable, Any, Optional, List, Tuple

import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from nonebot import logger


class LLMClient:
    """
    Chat LLM client with pluggable provider.

    provider:
      - openai_compatible: use OpenAI SDK + /chat/completions compatible endpoint
      - google_ai_studio: use Gemini Developer API official endpoint (generateContent)
    """

    def __init__(
        self,
        *,
        provider: str = "openai_compatible",
        openai_client: Optional[AsyncOpenAI] = None,
        google_api_key: Optional[str] = None,
        google_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        proxy: Optional[str] = None,
    ):
        self.provider = (provider or "openai_compatible").strip().lower()
        self.openai_client = openai_client
        self.google_api_key = (google_api_key or "").strip() or None
        self.google_base_url = (google_base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.timeout = timeout
        self.proxy = proxy

    def _openai_client_required(self) -> AsyncOpenAI:
        if not self.openai_client:
            raise RuntimeError("openai_client is required when provider=openai_compatible")
        return self.openai_client

    @staticmethod
    def _build_gemini_payload(prompt: str, system_prompt: str | None) -> dict:
        system_text = (system_prompt or "").strip()
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        return payload

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

    async def _generate_response_google(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: str | None,
        on_usage: Callable[[dict], None] | None,
        **kwargs,
    ) -> str | None:
        api_key = (self.google_api_key or "").strip()
        if not api_key:
            raise RuntimeError("google_api_key is required when provider=google_ai_studio")

        payload = self._build_gemini_payload(prompt, system_prompt)
        payload["generationConfig"] = {
            "temperature": float(temperature),
            # keep a high ceiling; upstream will clamp as needed
            "maxOutputTokens": int(kwargs.pop("max_tokens", 4096) or 4096),
        }

        # ignore OpenAI-specific kwargs
        kwargs.pop("response_format", None)
        kwargs.pop("extra_body", None)
        kwargs.pop("top_p", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

        async with httpx.AsyncClient(
            base_url=self.google_base_url,
            proxy=self.proxy,
            timeout=float(kwargs.pop("timeout", self.timeout) or self.timeout),
        ) as client:
            resp = await client.post(
                f"/models/{model}:generateContent",
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
                logger.warning(f"Usage callback failed: {ex}")

        return text

    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        on_usage: Callable[[dict], None] | None = None,
        **kwargs,
    ) -> str | None:
        """
        生成回复，支持透传参数和自定义 System Prompt，包含重试机制
        """
        if not system_prompt:
            system_content = "You are an intelligent agent. Output the final response in JSON format."
        else:
            system_content = system_prompt

        # 重试配置
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                request_timeout = kwargs.pop("timeout", self.timeout)

                if self.provider == "google_ai_studio":
                    return await self._generate_response_google(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        system_prompt=system_content,
                        on_usage=on_usage,
                        timeout=request_timeout,
                        **kwargs,
                    )

                # openai-compatible path (legacy)
                client = self._openai_client_required()
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    timeout=request_timeout,
                    **kwargs,
                )

                if on_usage and response.usage:
                    try:
                        on_usage(response.usage.model_dump())
                    except Exception as ex:
                        logger.warning(f"Usage callback failed: {ex}")

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
