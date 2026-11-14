import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from nonebot import logger
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError

from .config import EndpointSettings, get_app_settings
from .metrics import error_text, log_event, record_token_usage
from .prompts import JSON_SYSTEM_PROMPT
from .webconsole_store import record_llm_call


@dataclass(frozen=True)
class VisionInput:
    """发给 OpenAI 兼容多模态接口的一张图；只在本进程短暂持有，不落库。"""

    # 附图标签，写明是哪条消息的哪张图（如「消息 527151 引用的第 1 张图」），模型靠它和消息 id 对上号
    ref_id: str
    data_url: str = field(repr=False)
    is_sticker: bool = False
    # 原始字节的 SHA-1，描述缓存的 key：同一张图换人发、换 fileid 也一样
    digest: str = ""

    def to_openai_content(self) -> list[dict]:
        return [
            {
                "type": "text",
                "text": f"[{self.ref_id}]",
            },
            {"type": "image_url", "image_url": {"url": self.data_url}},
        ]


_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(
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
    if status_code == 429:
        return "rate_limit"
    if status_code >= 500:
        return "server_error"
    return "api_error"


class LLMClient:
    """一个端点的 OpenAI 兼容客户端：JSON 输出、重试、429 熔断、指标与 Token 记录。"""

    def __init__(self, name: str, settings: EndpointSettings):
        self.name = name
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

    async def generate_json(
        self,
        prompt: str,
        *,
        session_id: str,
        system_prompt: str = JSON_SYSTEM_PROMPT,
        images: list[VisionInput] | None = None,
    ) -> dict | None:
        """返回模型输出的 JSON 对象；调用失败或输出不是对象返回 None，原因记在 provider_status。"""

        started_at = time.perf_counter()
        content = await self._request(prompt, session_id, system_prompt, images or [])
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        log_event(
            "llm_success" if content else "llm_failure",
            "DEBUG" if content else "WARNING",
            model=self.settings.model,
            latency_ms=latency_ms,
            decision="content" if content else self.provider_status.last_error_type,
        )
        parsed = _parse_json_object(content) if content else None
        error_type = None
        if parsed is None:
            error_type = "invalid_json" if content else self.provider_status.last_error_type
        record_llm_call(
            self.name, self.settings.model, session_id, parsed is not None, latency_ms, error_type
        )
        return parsed

    async def _request(
        self,
        prompt: str,
        session_id: str,
        system_prompt: str,
        images: list[VisionInput],
    ) -> str:
        user_content: str | list[dict] = prompt
        if images:
            user_content = [{"type": "text", "text": prompt}]
            for image in images:
                user_content.extend(image.to_openai_content())
        # 按群固定：上游按这个值把请求分到同一台机器，前缀缓存才命中；不传时每次请求分到哪台都不定，线上量小几乎命中不了
        extra: dict[str, Any] = {"prompt_cache_key": f"nyaturingtest-{session_id}"}
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
                    f"{self.name} 网络请求失败（第 {attempt + 1}/{LLM_MAX_ATTEMPTS} 次）: {error_text(e)}"
                )
                self._fail("network_error", str(e))
                if last_attempt:
                    return ""
                await asyncio.sleep(LLM_RETRY_BASE_DELAY * (attempt + 1))
                continue
            except Exception as e:
                error_type = _classify_exception(e)
                logger.error(f"{self.name} 调用失败（{error_type}）: {error_text(e)}")
                self._fail(error_type, str(e))
                if error_type == "rate_limit":
                    self.provider_status.circuit_until = (
                        time.time() + RATE_LIMIT_CIRCUIT_SECONDS
                    )
                    return ""
                if last_attempt or error_type != "server_error":
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
chat_client = LLMClient("chat", get_app_settings().chat)
feedback_client = LLMClient("feedback", get_app_settings().feedback)
background_client = LLMClient("background", get_app_settings().background)

WEB_SEARCH_COUNT = 5
WEB_RESULT_CHARS = 400


async def web_search(query: str, ori_question: str, *, session_id: str) -> list[str]:
    """DeepSeek Search，每条结果一行「标题（站点，日期）：摘要」；失败当没搜到，Chat 照常回。"""

    settings = get_app_settings().web_search
    body: dict[str, Any] = {"query": query[:128], "count": WEB_SEARCH_COUNT}
    # 原问题帮它给结果排序；接口限 128 字
    if ori_question:
        body["ori_question"] = ori_question[:128]
    started = time.perf_counter()
    try:
        response = await get_http_client().post(
            f"{settings.base_url}/search",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=body,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        results = response.json()["results"]
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        record_llm_call("web_search", "search", session_id, False, latency_ms, type(e).__name__)
        log_event(
            "web_search_failed",
            "WARNING",
            session_id=session_id,
            query=query,
            error=error_text(e)[:200],
        )
        return []
    latency_ms = int((time.perf_counter() - started) * 1000)
    record_llm_call("web_search", "search", session_id, True, latency_ms)
    log_event(
        "web_search",
        session_id=session_id,
        query=query,
        latency_ms=latency_ms,
        results=len(results),
        top_score=round(results[0]["rerank_score"], 2) if results else None,
    )
    lines = []
    for item in results:
        date = (item.get("date") or "")[:10] or "日期不明"
        text = " ".join((item.get("summary") or item.get("snippet") or "").split())
        lines.append(
            f"{item.get('title') or ''}（{item.get('site_name') or ''}，{date}）："
            f"{text[:WEB_RESULT_CHARS]}"
        )
    return lines


# Open-Meteo：免费、不要 key；按坐标取，城市和坐标由状态模型给（它的按名查坐标常落错地方）
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 10.0
WEATHER_CACHE_SECONDS = 1800
_WMO_CODES = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴", 45: "雾", 48: "雾",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 56: "冻雨", 57: "冻雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "小雪", 80: "阵雨", 81: "阵雨", 82: "强阵雨",
    85: "阵雪", 86: "阵雪", 95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴冰雹",
}
_weather_cache: dict[tuple[float, float], tuple[float, dict]] = {}


async def _fetch_weather(lat: float, lon: float) -> dict | None:
    """当前天气和昨天到明天的逐日天气；失败返回 None。"""

    key = (round(lat, 1), round(lon, 1))
    cached = _weather_cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < WEATHER_CACHE_SECONDS:
        return cached[1]
    try:
        response = await get_http_client().get(
            WEATHER_URL,
            params={
                "latitude": key[0],
                "longitude": key[1],
                "current": "temperature_2m,apparent_temperature,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "past_days": 1,
                "forecast_days": 2,
            },
            timeout=WEATHER_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        log_event("weather_failed", "WARNING", lat=lat, lon=lon, error=error_text(e)[:200])
        return None
    _weather_cache[key] = (time.monotonic(), data)
    return data


def _day_weather(daily: dict, day: str) -> str:
    if day not in daily["time"]:
        return ""
    index = daily["time"].index(day)
    text = (
        f"{round(daily['temperature_2m_min'][index])}～{round(daily['temperature_2m_max'][index])}°C "
        f"{_WMO_CODES.get(daily['weather_code'][index], '')}"
    )
    rain = daily["precipitation_probability_max"][index]
    if rain:
        text += f"（降水概率 {rain}%）"
    return text


async def weather_now(city: str, lat: float, lon: float) -> str:
    """「北京，现在 9°C 晴，体感 6°C；今天 3～12°C 晴；明天 …」；取不到返回空串。"""

    data = await _fetch_weather(lat, lon)
    if data is None:
        return ""
    current, daily = data["current"], data["daily"]
    # timezone=auto 时日期是当地的；去外地时用当地的「今天」
    today = current["time"][:10]
    tomorrow = daily["time"][-1]
    return (
        f"{city}，现在 {round(current['temperature_2m'])}°C {_WMO_CODES.get(current['weather_code'], '')}，"
        f"体感 {round(current['apparent_temperature'])}°C；今天 {_day_weather(daily, today)}；"
        f"明天 {_day_weather(daily, tomorrow)}"
    )


async def weather_on(city: str, lat: float, lon: float, day: str) -> str:
    """「北京 3～12°C 晴」，给日记用；day 是 YYYY-MM-DD，只有昨天到明天取得到，取不到返回空串。"""

    data = await _fetch_weather(lat, lon)
    if data is None:
        return ""
    text = _day_weather(data["daily"], day)
    return f"{city} {text}" if text else ""


@dataclass(frozen=True)
class ImageResult:
    # ok / blocked（上游拒绝这条请求，多半是内容审核）/ failed（超时、上游故障）
    outcome: str
    image: bytes = field(default=b"", repr=False)
    error: str = ""


_image_client: AsyncOpenAI | None = None
# 出图动辄几十秒，全局限 2 张，一个群卡住不至于把上游占满
_IMAGE_SEMAPHORE = asyncio.Semaphore(2)


async def _request_image(prompt: str, reference: bytes | None) -> bytes:
    global _image_client
    settings = get_app_settings().image
    if _image_client is None:
        _image_client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            http_client=get_http_client(),
            # 超时的请求上游可能还在出图，重试会重复出图
            max_retries=0,
        )
    options: dict[str, Any] = {
        "model": settings.model,
        "prompt": prompt,
        "n": 1,
        "timeout": settings.timeout,
    }
    if settings.quality:
        options["quality"] = settings.quality
    if settings.size:
        options["size"] = settings.size
    extra_body = dict(settings.extra_body)
    if reference is None:
        response = await _image_client.images.generate(extra_body=extra_body, **options)
    elif settings.provider == "ark":
        # 方舟没有 images/edits，参考图以 data URL 放进 generations 的 image 字段
        extra_body["image"] = "data:image/jpeg;base64," + base64.b64encode(reference).decode()
        response = await _image_client.images.generate(extra_body=extra_body, **options)
    else:
        response = await _image_client.images.edit(
            image=("portrait.jpg", reference, "image/jpeg"), extra_body=extra_body, **options
        )
    if not response.data:
        raise ValueError("上游没有返回图片")
    item = response.data[0]
    if item.b64_json:
        return base64.b64decode(item.b64_json)
    if not item.url:
        raise ValueError("上游返回的图片既没有数据也没有地址")
    download = await get_http_client().get(item.url, timeout=settings.timeout)
    download.raise_for_status()
    return download.content


async def generate_image(prompt: str, reference: bytes | None, *, session_id: str) -> ImageResult:
    """没有参考图走 images/generations；有参考图时 openai 走 images/edits，ark 仍走 generations。
    不抛异常，结果分 ok / blocked / failed。"""

    model = get_app_settings().image.model
    started = time.perf_counter()
    try:
        async with _IMAGE_SEMAPHORE:
            image = await _request_image(prompt, reference)
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        outcome = "blocked" if isinstance(e, BadRequestError) else "failed"
        record_llm_call("image", model, session_id, False, latency_ms, type(e).__name__)
        return ImageResult(outcome, error=error_text(e)[:300])
    latency_ms = int((time.perf_counter() - started) * 1000)
    record_llm_call("image", model, session_id, True, latency_ms)
    return ImageResult("ok", image)


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
            logger.warning(f"json_repair 失败: {error_text(e)}")
            return None
    return parsed if isinstance(parsed, dict) else None
