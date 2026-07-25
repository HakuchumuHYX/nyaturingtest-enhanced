# nyaturingtest/client.py
import asyncio
from dataclasses import dataclass, field
import hashlib
import time
from typing import Callable, Any, Optional

import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from nonebot import logger

try:
    from .json_mode import is_json_mode_unsupported_error
except ImportError:
    def is_json_mode_unsupported_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "json mode is not supported" in text
            or "response_format" in text and "not supported" in text
        )


PROVIDER_ADVISORY_BACKOFF_SECONDS = 5.0
PROVIDER_ADVISORY_BACKOFF_MAX_SLEEP_SECONDS = 1.0
_SHARED_PROVIDER_BACKOFF_UNTIL: dict[str, float] = {}


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str = ""
    finish_reason: str = ""
    model: str = ""
    provider: str = "deepseek_official"
    usage: dict[str, int | str] = field(default_factory=dict)


@dataclass
class ProviderStatus:
    last_error_type: str = ""
    last_error_message: str = ""
    last_error_time: float = 0.0
    circuit_until: float = 0.0

    @property
    def circuit_remaining_seconds(self) -> int:
        return max(0, int(self.circuit_until - time.time()))


class LLMClient:
    """
    Chat LLM client for DeepSeek official and OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        *,
        provider: str = "deepseek_official",
        openai_client: Optional[AsyncOpenAI] = None,
        timeout: float = 60.0,
        base_url: str = "",
        api_key: str = "",
    ):
        self.provider = self._normalize_provider(provider)
        self.openai_client = openai_client
        self.timeout = timeout
        self.base_url = (base_url or "").strip().rstrip("/")
        self._api_key_hash = self._hash_secret(api_key)
        self.provider_status = ProviderStatus()

    @staticmethod
    def _normalize_provider(provider: str | None) -> str:
        value = (provider or "deepseek_official").strip().lower()
        if value == "openai_compatible":
            return value
        if value == "deepseek_official":
            return value
        raise ValueError(f"Unsupported LLM provider: {value}")

    def _openai_client_required(self) -> AsyncOpenAI:
        if not self.openai_client:
            raise RuntimeError("openai_client is required for LLMClient")
        return self.openai_client

    @staticmethod
    def _hash_secret(value: str | None) -> str:
        value = value or ""
        if not value:
            return ""
        return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:16]

    def _client_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return str(getattr(self.openai_client, "base_url", "") or "").strip().rstrip("/")

    def _client_api_key_hash(self) -> str:
        if self._api_key_hash:
            return self._api_key_hash
        return self._hash_secret(str(getattr(self.openai_client, "api_key", "") or ""))

    def _provider_backoff_key(self, model: str) -> str:
        return "|".join([
            self.provider,
            self._client_base_url(),
            self._client_api_key_hash(),
            str(model or ""),
        ])

    async def _sleep_for_shared_provider_backoff(self, model: str) -> None:
        key = self._provider_backoff_key(model)
        until = _SHARED_PROVIDER_BACKOFF_UNTIL.get(key, 0.0)
        remaining = until - time.time()
        if remaining <= 0:
            return
        delay = min(remaining, PROVIDER_ADVISORY_BACKOFF_MAX_SLEEP_SECONDS)
        logger.warning(f"[LLM] provider advisory backoff {delay:.2f}s for {self.provider}/{model}")
        await asyncio.sleep(delay)

    def _record_shared_provider_backoff(self, model: str) -> None:
        key = self._provider_backoff_key(model)
        _SHARED_PROVIDER_BACKOFF_UNTIL[key] = max(
            _SHARED_PROVIDER_BACKOFF_UNTIL.get(key, 0.0),
            time.time() + PROVIDER_ADVISORY_BACKOFF_SECONDS,
        )

    @staticmethod
    def _is_thinking_enabled(extra_body: dict[str, Any] | None) -> bool:
        thinking = (extra_body or {}).get("thinking")
        if isinstance(thinking, dict):
            return str(thinking.get("type", "")).lower() == "enabled"
        return False

    @staticmethod
    def _usage_to_dict(usage: Any, finish_reason: str) -> dict[str, int | str]:
        if not usage:
            data: dict[str, Any] = {}
        elif hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }

        completion_details = data.get("completion_tokens_details") or {}
        if hasattr(completion_details, "model_dump"):
            completion_details = completion_details.model_dump()

        prompt_tokens = int(data.get("prompt_tokens") or 0)
        completion_tokens = int(data.get("completion_tokens") or 0)
        hit_tokens = int(data.get("prompt_cache_hit_tokens") or 0)
        miss_tokens = int(data.get("prompt_cache_miss_tokens") or 0)
        reasoning_tokens = int(data.get("reasoning_tokens") or 0)
        if isinstance(completion_details, dict):
            reasoning_tokens = int(completion_details.get("reasoning_tokens") or reasoning_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(data.get("total_tokens") or prompt_tokens + completion_tokens),
            "prompt_cache_hit_tokens": hit_tokens,
            "prompt_cache_miss_tokens": miss_tokens,
            "reasoning_tokens": reasoning_tokens,
            "finish_reason": finish_reason or "",
        }

    @staticmethod
    def _status_code(exc: Exception) -> int:
        response = getattr(exc, "response", None)
        return int(getattr(response, "status_code", 0) or getattr(exc, "status_code", 0) or 0)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        return str(getattr(response, "text", "") or exc)

    def _classify_exception(self, exc: Exception) -> str:
        status_code = self._status_code(exc)
        text = self._error_text(exc).lower()
        if status_code == 429:
            return "rate_limit"
        if "content_filter" in text:
            return "content_filter"
        if "insufficient_system_resource" in text:
            return "insufficient_system_resource"
        if status_code >= 500:
            return "server_error"
        return "api_error"

    def _error_response(self, model: str, error_type: str, message: str = "") -> LLMResponse:
        self.provider_status.last_error_type = error_type
        self.provider_status.last_error_message = message[:300]
        self.provider_status.last_error_time = time.time()
        return LLMResponse(
            content="",
            model=model,
            provider=self.provider,
            usage={
                "provider": self.provider,
                "error_type": error_type,
                "error_message": message[:300],
            },
        )

    @staticmethod
    def _build_user_content(prompt: str, images: list[Any] | None) -> str | list[dict]:
        if not images:
            return prompt
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            builder = getattr(image, "to_openai_content", None)
            if not callable(builder):
                continue
            blocks = builder()
            if isinstance(blocks, list):
                content.extend(block for block in blocks if isinstance(block, dict))
        return content if len(content) > 1 else prompt

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float | None = None,
        system_prompt: str | None = None,
        on_usage: Callable[[dict], None] | None = None,
        images: list[Any] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a structured response. DeepSeek thinking mode is configured via
        extra_body={"thinking": {"type": "enabled"|"disabled"}, ...}.
        """
        system_content = system_prompt or "You are an intelligent agent. Output only valid JSON."
        max_retries = 3
        base_delay = 2
        json_mode_fallback_used = False

        for attempt in range(max_retries):
            if self.provider_status.circuit_until > time.time():
                return LLMResponse(
                    content="",
                    model=model,
                    provider=self.provider,
                    usage={
                        "provider": self.provider,
                        "error_type": "circuit_open",
                        "error_message": "provider circuit breaker is open",
                    },
                )
            await self._sleep_for_shared_provider_backoff(model)

            request_kwargs = dict(kwargs)
            request_timeout = request_kwargs.pop("timeout", self.timeout)
            extra_body = request_kwargs.get("extra_body")
            if isinstance(extra_body, dict):
                extra_body = dict(extra_body)
                request_kwargs["extra_body"] = extra_body
            thinking_enabled = self.provider == "deepseek_official" and self._is_thinking_enabled(extra_body)

            if isinstance(extra_body, dict) and "reasoning_effort" in extra_body:
                request_kwargs["reasoning_effort"] = extra_body.pop("reasoning_effort")

            if temperature is not None and not thinking_enabled:
                request_kwargs["temperature"] = temperature
            if thinking_enabled:
                request_kwargs.pop("temperature", None)
                request_kwargs.pop("top_p", None)
                request_kwargs.pop("presence_penalty", None)
                request_kwargs.pop("frequency_penalty", None)
            if json_mode_fallback_used:
                request_kwargs.pop("response_format", None)
            request_kwargs = {key: value for key, value in request_kwargs.items() if value is not None}

            while True:
                try:
                    client = self._openai_client_required()
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": self._build_user_content(prompt, images)},
                        ],
                        timeout=request_timeout,
                        **request_kwargs,
                    )

                    choice = response.choices[0]
                    message = choice.message
                    finish_reason = getattr(choice, "finish_reason", "") or ""
                    content = getattr(message, "content", "") or ""
                    reasoning_content = getattr(message, "reasoning_content", "") or ""
                    usage = self._usage_to_dict(getattr(response, "usage", None), finish_reason)
                    usage["provider"] = self.provider

                    result = LLMResponse(
                        content=content,
                        reasoning_content=reasoning_content,
                        finish_reason=finish_reason,
                        model=getattr(response, "model", "") or model,
                        provider=self.provider,
                        usage=usage,
                    )

                    if on_usage:
                        try:
                            on_usage(usage)
                        except Exception as ex:
                            logger.warning(f"Usage callback failed: {ex}")

                    if finish_reason == "length":
                        return self._error_response(model, "length", "finish_reason=length")
                    if not content.strip() and attempt < max_retries - 1:
                        self.provider_status.last_error_type = "empty_content"
                        self.provider_status.last_error_message = "empty content from provider"
                        self.provider_status.last_error_time = time.time()
                        await asyncio.sleep(0.2)
                        break

                    return result

                except (APIConnectionError, APITimeoutError, httpx.ConnectError, httpx.ReadTimeout) as e:
                    logger.warning(f"[LLM] 网络请求失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__} - {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (attempt + 1))
                        break
                    logger.error(f"[LLM] 最终请求失败: {e}")
                    return self._error_response(model, "network_error", str(e))

                except Exception as e:
                    if (
                        "response_format" in request_kwargs
                        and not json_mode_fallback_used
                        and is_json_mode_unsupported_error(e)
                    ):
                        logger.warning("LLM 模型不支持 JSON mode，已降级为普通文本 JSON 提示重试")
                        request_kwargs.pop("response_format", None)
                        json_mode_fallback_used = True
                        continue

                    error_type = self._classify_exception(e)
                    logger.error(f"[LLM] API 调用失败 [{error_type}]: {e}")
                    if error_type == "rate_limit":
                        self._record_shared_provider_backoff(model)
                        self.provider_status.circuit_until = time.time() + 30
                        return self._error_response(model, error_type, str(e))
                    if error_type in {"insufficient_system_resource", "server_error"} and attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (attempt + 1))
                        break
                    return self._error_response(model, error_type, str(e))

        return self._error_response(model, "retry_exhausted", "max retries exhausted")

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
        Backward-compatible string API. New call sites should use generate().
        """
        response = await self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            on_usage=on_usage,
            **kwargs,
        )
        return response.content
