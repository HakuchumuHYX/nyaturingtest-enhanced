from typing import Callable

import asyncio
import httpx
from openai import AsyncOpenAI
from nonebot import logger

try:
    from .json_mode import is_json_mode_unsupported_error
except ImportError:
    from json_mode import is_json_mode_unsupported_error


def _model_supports_response_format(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return "glm-4.6v" not in normalized


def _is_json_mode_unsupported_error(exc: Exception) -> bool:
    return is_json_mode_unsupported_error(exc)


def _usage_to_dict(usage, *, provider: str, finish_reason: str) -> dict:
    if not usage:
        data = {}
    elif hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = dict(usage)
    else:
        data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    prompt_tokens = int(data.get("prompt_tokens") or 0)
    completion_tokens = int(data.get("completion_tokens") or 0)
    normalized = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(data.get("total_tokens") or prompt_tokens + completion_tokens),
        "provider": provider,
        "finish_reason": finish_reason or "",
    }
    return normalized


class VLM:
    """
    OpenAI-compatible vision-language adapter using chat.completions image_url.
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
    ):
        if (provider or "openai_compatible").strip().lower() != "openai_compatible":
            raise RuntimeError("VLM only supports OpenAI-compatible endpoints.")

        self.provider = "openai_compatible"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            timeout=timeout,
        )
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
            http_client=self._http_client,
            max_retries=0,
        )

    async def close(self):
        """关闭内部 HTTP 客户端，中断所有正在进行的请求"""
        try:
            await self._http_client.aclose()
        except Exception:
            pass

    async def request(
        self,
        prompt: str,
        image_base64: str,
        image_format: str,
        on_usage: Callable[[dict], None] | None = None,
        detail: str = "low",
        **kwargs,
    ) -> str | None:
        """
        让 VLM 根据图片和文本提示词生成描述。
        detail: OpenAI 视觉档位 "low"/"high"/"auto"，默认 "low"。
        """
        request_kwargs = dict(kwargs)
        if not _model_supports_response_format(self.model):
            request_kwargs.pop("response_format", None)

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
                                    "image_url": {
                                        "url": f"data:image/{image_format};base64,{image_base64}",
                                        "detail": detail,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    timeout=self.timeout,
                    **request_kwargs,
                )

                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", "") or ""

                if on_usage and response.usage:
                    try:
                        on_usage(_usage_to_dict(
                            response.usage,
                            provider=getattr(self, "provider", "openai_compatible"),
                            finish_reason=finish_reason,
                        ))
                    except Exception as ex:
                        logger.warning(f"VLM Usage callback failed: {ex}")

                content = choice.message.content
                if content:
                    return content
                logger.warning(f"VLM 返回内容为空 (尝试 {retries + 1}/{self.max_retries + 1})")

            except Exception as e:
                if "response_format" in request_kwargs and _is_json_mode_unsupported_error(e):
                    logger.warning("VLM 模型不支持 JSON mode，已降级为普通文本 JSON 提示重试")
                    request_kwargs.pop("response_format", None)
                    continue
                logger.warning(f"VLM 请求失败: {e} (尝试 {retries + 1}/{self.max_retries + 1})")

            retries += 1
            if retries <= self.max_retries:
                await asyncio.sleep(self.retry_delay)

        logger.error("VLM 请求最终失败，已跳过")
        return None
