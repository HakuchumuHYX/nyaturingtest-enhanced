import asyncio
import json
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from nonebot import logger
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from .config import EndpointSettings, get_app_settings
from .metrics import log_event, metrics, record_token_usage
from .prompts import JSON_SYSTEM_PROMPT


@dataclass(frozen=True)
class VisionInput:
    """发给 OpenAI 兼容多模态接口的一张图；只在本进程短暂持有，不落库。"""

    # 附图标签，写明是哪条消息的哪张图（如「消息 527151 引用的第 1 张图」），模型靠它和消息 id 对上号
    ref_id: str
    data_url: str = field(repr=False)
    is_sticker: bool = False
    detail: str = "auto"
    # 原始字节的 SHA-1，描述缓存的 key：同一张图换人发、换 fileid 也一样
    digest: str = ""

    def to_openai_content(self) -> list[dict]:
        return [
            {
                "type": "text",
                "text": f"[{self.ref_id}]",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self.data_url,
                    "detail": self.detail,
                },
            },
        ]


_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled HTTP client."""

    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.set_ciphers("ALL:@SECLEVEL=1")
        _HTTP_CLIENT = httpx.AsyncClient(
            verify=ssl_context,
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=100,
            ),
        )
    return _HTTP_CLIENT


async def close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None
        logger.info("全局 HTTP 客户端已关闭")


@dataclass
class ProviderStatus:
    last_error_type: str = ""
    last_error_message: str = ""
    circuit_until: float = 0.0

    @property
    def circuit_remaining_seconds(self) -> int:
        return max(0, int(self.circuit_until - time.time()))


LLM_MAX_ATTEMPTS = 3
LLM_RETRY_BASE_DELAY = 2.0
RATE_LIMIT_CIRCUIT_SECONDS = 30.0



def _usage_to_dict(usage: Any, finish_reason: str) -> dict[str, int | str]:
    data = usage.model_dump() if usage is not None else {}
    prompt_details = data.get("prompt_tokens_details") or {}
    completion_details = data.get("completion_tokens_details") or {}
    prompt_tokens = int(data.get("prompt_tokens") or 0)
    completion_tokens = int(data.get("completion_tokens") or 0)
    # 命中数来自 OpenAI 形状的 prompt_tokens_details.cached_tokens，未命中即剩余部分
    hit_tokens = min(int(prompt_details.get("cached_tokens") or 0), prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(data.get("total_tokens") or prompt_tokens + completion_tokens),
        "prompt_cache_hit_tokens": hit_tokens,
        "prompt_cache_miss_tokens": prompt_tokens - hit_tokens,
        "reasoning_tokens": int(
            completion_details.get("reasoning_tokens")
            or data.get("reasoning_tokens")
            or 0
        ),
        "finish_reason": finish_reason or "",
    }


def _classify_exception(exc: Exception) -> str:
    status_code = int(getattr(exc, "status_code", 0) or 0)
    text = str(exc).lower()
    if status_code == 429:
        return "rate_limit"
    if "content_filter" in text:
        return "content_filter"
    if "insufficient_system_resource" in text:
        return "insufficient_system_resource"
    if status_code >= 500:
        return "server_error"
    return "api_error"


class LLMClient:
    """一个端点的 OpenAI 兼容客户端：JSON 输出、重试、429 熔断、指标与 Token 记录。"""

    def __init__(self, settings: EndpointSettings):
        self.settings = settings
        self.openai_client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            http_client=get_http_client(),
            max_retries=0,
        )
        self.provider_status = ProviderStatus()

    def _fail(self, error_type: str, message: str) -> None:
        self.provider_status.last_error_type = error_type
        self.provider_status.last_error_message = message[:300]

    async def generate(
        self,
        prompt: str,
        *,
        session_id: str,
        temperature: float,
        system_prompt: str = JSON_SYSTEM_PROMPT,
        images: list[VisionInput] | None = None,
    ) -> str:
        """返回模型文本；失败返回空串，原因记在 provider_status。"""

        started_at = time.perf_counter()
        content = await self._request(
            prompt, session_id, temperature, system_prompt, images or []
        )
        if content:
            metrics.llm_success += 1
        else:
            metrics.llm_failure += 1
        log_event(
            "llm_success" if content else "llm_failure",
            "DEBUG" if content else "WARNING",
            model=self.settings.model,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            decision="content" if content else self.provider_status.last_error_type,
        )
        return content

    async def generate_json(
        self,
        prompt: str,
        *,
        session_id: str,
        temperature: float,
        system_prompt: str = JSON_SYSTEM_PROMPT,
        images: list[VisionInput] | None = None,
    ) -> dict | None:
        """返回模型输出的 JSON 对象；调用失败或输出不是对象返回 None。"""

        content = await self.generate(
            prompt,
            session_id=session_id,
            temperature=temperature,
            system_prompt=system_prompt,
            images=images,
        )
        return _parse_json_object(content) if content else None

    async def _request(
        self,
        prompt: str,
        session_id: str,
        temperature: float,
        system_prompt: str,
        images: list[VisionInput],
    ) -> str:
        user_content: str | list[dict] = prompt
        if images:
            user_content = [{"type": "text", "text": prompt}]
            for image in images:
                user_content.extend(image.to_openai_content())
        extra = {}
        if self.settings.reasoning_effort:
            extra["reasoning_effort"] = self.settings.reasoning_effort

        for attempt in range(LLM_MAX_ATTEMPTS):
            if self.provider_status.circuit_until > time.time():
                self._fail("circuit_open", "provider circuit breaker is open")
                return ""
            last_attempt = attempt == LLM_MAX_ATTEMPTS - 1
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.settings.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    max_tokens=self.settings.max_tokens,
                    timeout=self.settings.timeout,
                    response_format={"type": "json_object"},
                    **extra,
                )
            except (
                APIConnectionError,
                APITimeoutError,
                httpx.ConnectError,
                httpx.ReadTimeout,
            ) as e:
                logger.warning(
                    f"[LLM] 网络请求失败 (尝试 {attempt + 1}/{LLM_MAX_ATTEMPTS}): {type(e).__name__} - {e}"
                )
                self._fail("network_error", str(e))
                if last_attempt:
                    return ""
                await asyncio.sleep(LLM_RETRY_BASE_DELAY * (attempt + 1))
                continue
            except Exception as e:
                error_type = _classify_exception(e)
                logger.error(f"[LLM] API 调用失败 [{error_type}]: {e}")
                self._fail(error_type, str(e))
                if error_type == "rate_limit":
                    self.provider_status.circuit_until = (
                        time.time() + RATE_LIMIT_CIRCUIT_SECONDS
                    )
                    return ""
                retryable = error_type in {"insufficient_system_resource", "server_error"}
                if last_attempt or not retryable:
                    return ""
                await asyncio.sleep(LLM_RETRY_BASE_DELAY * (attempt + 1))
                continue

            choice = response.choices[0]
            finish_reason = choice.finish_reason or ""
            record_token_usage(
                session_id,
                self.settings.model,
                _usage_to_dict(response.usage, finish_reason),
            )
            if finish_reason == "length":
                self._fail("length", "finish_reason=length")
                return ""
            content = choice.message.content or ""
            if content.strip():
                return content
            self._fail("empty_content", "empty content from provider")
            if not last_attempt:
                await asyncio.sleep(0.2)
        return ""


# 所有群共用：限流与熔断本来就是按 API key 算的
chat_client = LLMClient(get_app_settings().chat)
feedback_client = LLMClient(get_app_settings().feedback)


def _parse_json_object(text: str) -> dict | None:
    """从模型输出里取出 JSON 对象：去掉 <think> 和代码块标记，解析失败再用 json_repair 修一次。"""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    payload = text[start : end + 1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            parsed = repair_json(payload, return_objects=True)
        except Exception as e:
            logger.warning(f"json_repair 失败: {e}")
            return None
    return parsed if isinstance(parsed, dict) else None
