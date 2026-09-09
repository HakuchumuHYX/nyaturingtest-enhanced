# 由多个模块合并而来：llm/json_mode.py, llm/vision.py, core/http_client.py, llm/client.py, core/turn_call_factory.py

from dataclasses import dataclass, field, replace
import ssl
import httpx
from nonebot import logger
import asyncio
from dataclasses import dataclass, field
import time
from typing import Callable, Any, Optional
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from ..config import (
    get_app_settings,
    get_reasoning_effort,
)
from .metrics import log_event
from .metrics import make_usage_recorder


# ======== from llm/json_mode.py ========
def is_json_mode_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "json mode is not supported" in text
        or "response_format" in text and "not supported" in text
    )

# ======== from llm/vision.py ========
VISION_DETAILS = {"low", "high", "auto"}


@dataclass(frozen=True)
class VisionInput:
    """Ephemeral image input for an OpenAI-compatible multimodal request."""

    ref_id: str
    data_url: str = field(repr=False)
    is_sticker: bool = False
    source: str = "primary"
    detail: str = "auto"

    def with_detail(self, detail: str) -> "VisionInput":
        normalized = str(detail or "auto").strip().lower()
        if normalized not in VISION_DETAILS:
            normalized = "auto"
        return replace(self, detail=normalized)

    def to_openai_content(self) -> list[dict]:
        source_label = "引用消息图片" if self.source == "referenced" else "当前消息图片"
        return [
            {
                "type": "text",
                "text": f"[{source_label} image_ref={self.ref_id}]",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self.data_url,
                    "detail": self.detail,
                },
            },
        ]

# ======== from core/http_client.py ========
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

# ======== from llm/client.py ========
# nyaturingtest/client.py






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
        """Generate a structured response using common OpenAI SDK parameters."""
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
            request_kwargs = dict(kwargs)
            request_timeout = request_kwargs.pop("timeout", self.timeout)

            if temperature is not None:
                request_kwargs["temperature"] = temperature
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
                        self.provider_status.circuit_until = time.time() + 30
                        return self._error_response(model, error_type, str(e))
                    if error_type in {"insufficient_system_resource", "server_error"} and attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (attempt + 1))
                        break
                    return self._error_response(model, error_type, str(e))

        return self._error_response(model, "retry_exhausted", "max retries exhausted")

# ======== from core/turn_call_factory.py ========
@dataclass(frozen=True)
class TurnCalls:
    chat: Callable
    feedback: Callable


CHAT_SYSTEM_PROMPT = (
    "你就是动态输入里的那个角色本人，正在群聊里用手机和人聊天。"
    "role 是你的性格与经历，examples_text 是你的说话习惯，search_result 是你的记忆，"
    "把它们当作自己的东西，不是别人给你的说明书。"
    "不要以 AI、助手、模型或角色扮演引擎的身份说话，不要解释设定。"
    "群聊回复要短、自然，像手机打字。"
    "请在内部完成分析，但最终输出只包含一个合法 JSON 对象，不要输出 Markdown、解释或思考过程。"
)

FEEDBACK_SYSTEM_PROMPT = (
    "你是一个对话分析引擎。你的输入是群聊消息日志和可选的群聊图片，"
    "输出是结构化的情感分析 JSON。"
    "这是一个纯数据处理任务：读取文本和图片 → 分析情感维度 → 输出 JSON。"
    "你不需要参与对话，不需要扮演任何角色，只需要做文本情感分析。"
    "你的输出必须包含 new_emotion 对象（含 valence、arousal、dominance 三个浮点数字段）。"
    "请在内部完成分析，但最终输出只包含一个合法 JSON 对象，不要输出 Markdown、解释或思考过程。"
)


class TurnCallFactory:
    def __init__(self, llm_response: Callable):
        self._llm_response = llm_response

    def build(
        self,
        *,
        state,
        session_id: str,
        chat_images: list,
        feedback_images: list,
    ) -> TurnCalls:
        def usage_recorder(model_name: str):
            def log_usage(usage: dict):
                log_event(
                    "token_usage",
                    session_id=session_id,
                    provider=usage.get("provider", ""),
                    model=model_name,
                    tokens=usage.get("total_tokens", 0),
                    decision=usage.get("finish_reason", ""),
                )

            return make_usage_recorder(
                session_id,
                model_name,
                event_logger=log_usage,
            )

        chat_model = get_app_settings().chat.model
        feedback_model = get_app_settings().feedback.model

        async def chat(message: str, json_mode: bool = False):
            return await self._llm_response(
                state.client,
                message,
                model=chat_model,
                temperature=0.7,
                reasoning_effort=get_reasoning_effort("chat"),
                json_mode=bool(json_mode),
                max_tokens=get_app_settings().chat.max_tokens,
                timeout=get_app_settings().chat.timeout,
                system_prompt=CHAT_SYSTEM_PROMPT,
                on_usage=usage_recorder(chat_model),
                images=chat_images,
            )

        async def feedback(message: str, json_mode: bool = False):
            return await self._llm_response(
                state.feedback_client,
                message,
                model=feedback_model,
                temperature=0.1,
                json_mode=bool(json_mode),
                reasoning_effort=get_reasoning_effort("feedback"),
                max_tokens=get_app_settings().feedback.max_tokens,
                timeout=get_app_settings().feedback.timeout,
                on_usage=usage_recorder(feedback_model),
                system_prompt=FEEDBACK_SYSTEM_PROMPT,
                images=feedback_images,
            )

        return TurnCalls(chat=chat, feedback=feedback)


def extract_and_parse_json(text: str) -> dict | list | None:
    """Extract a JSON object/array from a bounded LLM response."""

    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = re.sub(r"```json\s*|```\s*", "", text)
    object_start = text.find("{")
    array_start = text.find("[")
    if object_start != -1 and (array_start == -1 or object_start < array_start):
        end = text.rfind("}")
        payload = text[object_start:end + 1] if end != -1 else ""
    elif array_start != -1:
        end = text.rfind("]")
        payload = text[array_start:end + 1] if end != -1 else ""
    else:
        payload = ""
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json

        repaired = repair_json(payload, return_objects=True)
        return repaired if isinstance(repaired, (dict, list)) else None
    except Exception as e:
        logger.warning(f"json_repair 失败: {e}")
        return None
