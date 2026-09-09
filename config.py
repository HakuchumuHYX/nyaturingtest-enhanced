"""插件配置：只保留模型与端点。

运行时策略参数（意愿、RAG、Prompt 预算、发送、保留期等）不再作为配置项，
各自放在使用它们的模块里作为常量，避免「配置项与用途对不上」。
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nonebot import logger


PLUGIN_DIR = Path(__file__).parent
CONFIG_FILE = Path(
    os.environ.get("NYATURINGTEST_CONFIG_FILE", str(PLUGIN_DIR / "config.json"))
).expanduser()

WORKSPACE_ROOT = PLUGIN_DIR.resolve().parents[1]
DEFAULT_PRESET_DIR = WORKSPACE_ROOT / "config" / "nyaturingtest" / "nya_presets"


def get_data_dir() -> Path:
    """运行数据目录；仅此项支持环境变量覆盖，便于独立部署。"""

    value = os.environ.get("NYATURINGTEST_DATA_DIR", "").strip()
    if not value:
        return WORKSPACE_ROOT / "data" / "nyaturingtest"
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def get_cache_dir() -> Path:
    return WORKSPACE_ROOT / "cache" / "nyaturingtest"


def get_backup_dir() -> Path:
    return WORKSPACE_ROOT / "data" / "nyaturingtest_backups"


def get_preset_dir() -> Path:
    return DEFAULT_PRESET_DIR


def get_vector_dir(session_id: str) -> Path:
    return get_data_dir() / f"vector_index_{session_id}"


def get_image_cache_dir() -> Path:
    return get_cache_dir() / "image_cache"


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-v4-flash"
OPENAI_COMPATIBLE = "openai_compatible"
DEEPSEEK_OFFICIAL = "deepseek_official"

_plugin_config: dict[str, Any] = {}


@dataclass(frozen=True)
class EndpointSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float
    max_tokens: int = 0
    reasoning_effort: str = ""
    vision_enabled: bool = False
    vision_detail: str = "auto"


@dataclass(frozen=True)
class MemoryEndpointSettings:
    model: str
    base_url: str
    timeout: float
    rerank_base_url: str
    rerank_timeout: float


@dataclass(frozen=True)
class AppSettings:
    chat: EndpointSettings
    feedback: EndpointSettings
    vlm: EndpointSettings
    vlm_mode: str
    rerank_model: str
    rerank_threshold: float
    memory: MemoryEndpointSettings
    siliconflow_api_key: str


@dataclass(frozen=True)
class ConfigLoadStatus:
    ok: bool
    source: str
    path: str
    error_type: str = ""
    error_message: str = ""


_config_load_status = ConfigLoadStatus(ok=True, source="not_loaded", path=str(CONFIG_FILE))


def _set_config_load_status(
    *,
    ok: bool,
    source: str,
    error: Exception | None = None,
) -> None:
    global _config_load_status
    _config_load_status = ConfigLoadStatus(
        ok=ok,
        source=source,
        path=str(CONFIG_FILE),
        error_type=type(error).__name__ if error else "",
        error_message=str(error)[:300] if error else "",
    )


def get_config_load_status() -> ConfigLoadStatus:
    return _config_load_status


def get_default_config() -> dict:
    return {
        "chat": {
            "provider": DEEPSEEK_OFFICIAL,
            "api_key": "",
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_CHAT_MODEL,
            "reasoning_effort": "low",
            "max_tokens": 4096,
            "timeout": 180,
            "vision": {"enabled": False, "detail": "auto"},
        },
        "feedback": {
            "provider": DEEPSEEK_OFFICIAL,
            "api_key": "",
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_CHAT_MODEL,
            "reasoning_effort": "",
            "max_tokens": 2048,
            "timeout": 60,
            "vision": {"enabled": False, "detail": "low"},
        },
        "vlm": {
            "enabled": True,
            # fallback: 仅在 Chat/Feedback 至少一个不支持原生图片时调用
            # always: 始终生成文字观察；off: 完全不调用独立 VLM
            "mode": "fallback",
            "provider": OPENAI_COMPATIBLE,
            "api_key": "",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "zai-org/GLM-4.6V",
            "timeout": 60,
        },
        "siliconflow_api_key": "",
        "embedding": {
            "model": "BAAI/bge-m3",
            "base_url": "https://api.siliconflow.cn/v1",
            "timeout": 30,
        },
        "rerank": {
            "model": "Qwen/Qwen3-Reranker-4B",
            "base_url": "https://api.siliconflow.cn/v1/rerank",
            "timeout": 10,
            "threshold": 0.1,
        },
    }


def _deep_merge(default: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    result = dict(default)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_endpoint(section_name: str, section: dict[str, Any]) -> None:
    provider = str(section.get("provider") or "").strip().lower()
    base_url = str(section.get("base_url") or "").strip().rstrip("/")

    if section_name in {"chat", "feedback"}:
        if provider not in {DEEPSEEK_OFFICIAL, OPENAI_COMPATIBLE}:
            raise RuntimeError(f"Unsupported {section_name}.provider: {provider}")
    elif provider != OPENAI_COMPATIBLE:
        raise RuntimeError("vlm.provider only supports OpenAI-compatible endpoints.")


def _normalize_vision(section_name: str, section: dict[str, Any]) -> None:
    vision = section.get("vision")
    if not isinstance(vision, dict):
        vision = {}
        section["vision"] = vision
    vision["enabled"] = bool(vision.get("enabled", False))
    detail = str(vision.get("detail") or ("low" if section_name == "feedback" else "auto")).strip().lower()
    if detail not in {"low", "high", "auto"}:
        raise RuntimeError(f"{section_name}.vision.detail must be low, high, or auto.")
    vision["detail"] = detail


def _normalize_vlm_mode(vlm: dict[str, Any]) -> None:
    mode = str(vlm.get("mode") or "fallback").strip().lower()
    if mode not in {"fallback", "always", "off"}:
        raise RuntimeError("vlm.mode must be fallback, always, or off.")
    if not bool(vlm.get("enabled", True)):
        mode = "off"
    vlm["mode"] = mode


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = _deep_merge(get_default_config(), config)

    _normalize_endpoint("chat", merged["chat"])
    _normalize_endpoint("feedback", merged["feedback"])
    _normalize_endpoint("vlm", merged["vlm"])
    _normalize_vision("chat", merged["chat"])
    _normalize_vision("feedback", merged["feedback"])
    _normalize_vlm_mode(merged["vlm"])

    return merged


def _endpoint_vision_enabled(config: dict[str, Any], endpoint_name: str) -> bool:
    section = config.get(endpoint_name, {}) or {}
    vision = section.get("vision") or {}
    return bool(vision.get("enabled", False))


def _get_vlm_mode(config: dict[str, Any]) -> str:
    vlm = config.get("vlm", {}) or {}
    if not bool(vlm.get("enabled", True)):
        return "off"
    mode = str(vlm.get("mode") or "fallback").strip().lower()
    return mode if mode in {"fallback", "always", "off"} else "fallback"


def _should_use_standalone_vlm(config: dict[str, Any]) -> bool:
    mode = _get_vlm_mode(config)
    if mode == "off":
        return False
    if mode == "always":
        return True
    return not (
        _endpoint_vision_enabled(config, "chat")
        and _endpoint_vision_enabled(config, "feedback")
    )


def _require(value: str, field_name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise RuntimeError(f"Missing required config field: {field_name}")
    return value


def build_settings(config: dict[str, Any], *, require_api_keys: bool = False) -> AppSettings:
    cfg = normalize_config(config)

    def endpoint(name: str, *, require_key: bool, max_tokens_default: int = 0) -> EndpointSettings:
        section = cfg[name]
        api_key = section.get("api_key", "")
        if require_key:
            api_key = _require(api_key, f"{name}.api_key")
        base_url = _require(section.get("base_url", ""), f"{name}.base_url")
        model = _require(section.get("model", ""), f"{name}.model")
        vision = section.get("vision") or {}
        return EndpointSettings(
            provider=section.get("provider", ""),
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=float(section.get("timeout") or 60),
            max_tokens=int(section.get("max_tokens") or max_tokens_default),
            reasoning_effort=str(section.get("reasoning_effort") or ""),
            vision_enabled=bool(vision.get("enabled", False)),
            vision_detail=str(vision.get("detail") or ("low" if name == "feedback" else "auto")),
        )

    needs_standalone_vlm = _should_use_standalone_vlm(cfg)
    return AppSettings(
        chat=endpoint("chat", require_key=require_api_keys, max_tokens_default=4096),
        feedback=endpoint("feedback", require_key=require_api_keys, max_tokens_default=2048),
        vlm=endpoint("vlm", require_key=require_api_keys and needs_standalone_vlm),
        vlm_mode=_get_vlm_mode(cfg),
        rerank_model=str(cfg.get("rerank", {}).get("model") or ""),
        rerank_threshold=float(cfg.get("rerank", {}).get("threshold") or 0.0),
        memory=_build_memory_endpoint_settings(cfg),
        siliconflow_api_key=str(cfg.get("siliconflow_api_key") or ""),
    )


def describe_settings(settings: AppSettings) -> str:
    def endpoint(name: str, value: EndpointSettings) -> str:
        key_state = "set" if value.api_key else "missing"
        return f"{name}: provider={value.provider}, model={value.model}, base_url={value.base_url}, api_key={key_state}"

    return "; ".join([
        endpoint("chat", settings.chat),
        endpoint("feedback", settings.feedback),
        endpoint("vlm", settings.vlm),
        f"vision: chat={settings.chat.vision_enabled}, feedback={settings.feedback.vision_enabled}, vlm_mode={settings.vlm_mode}",
    ])


def load_plugin_config() -> dict:
    """从 config.json 加载插件配置。"""

    global _plugin_config

    if not CONFIG_FILE.exists():
        logger.warning(f"配置文件不存在，使用内置默认配置: {CONFIG_FILE}")
        _plugin_config = get_default_config()
        _set_config_load_status(ok=True, source="default")
        return _plugin_config

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        _plugin_config = normalize_config(loaded)
        settings = build_settings(_plugin_config, require_api_keys=False)
        _set_config_load_status(ok=True, source="file")
        logger.info(f"已加载插件配置: {CONFIG_FILE}")
        logger.info(f"插件配置摘要: {describe_settings(settings)}")
        return _plugin_config
    except RuntimeError as e:
        _set_config_load_status(ok=False, source="invalid", error=e)
        raise
    except Exception as e:
        _set_config_load_status(ok=False, source="invalid", error=e)
        raise


def get_app_settings() -> AppSettings:
    return _app_settings


def get_effective_chat_api_key() -> str:
    return get_app_settings().chat.api_key.strip()


def get_effective_chat_model() -> str:
    return get_app_settings().chat.model.strip()


def get_effective_chat_base_url() -> str:
    return get_app_settings().chat.base_url.strip()


def get_effective_chat_provider() -> str:
    return get_app_settings().chat.provider.strip().lower()


def get_reasoning_effort(endpoint_name: str) -> str | None:
    if endpoint_name not in {"chat", "feedback"}:
        raise ValueError(f"Unsupported reasoning endpoint: {endpoint_name}")
    return getattr(get_app_settings(), endpoint_name).reasoning_effort or None


def get_chat_timeout() -> float:
    return get_app_settings().chat.timeout


def get_chat_max_tokens() -> int:
    return get_app_settings().chat.max_tokens


def get_effective_feedback_api_key() -> str:
    return get_app_settings().feedback.api_key.strip()


def get_effective_feedback_model() -> str:
    return get_app_settings().feedback.model.strip()


def get_effective_feedback_base_url() -> str:
    return get_app_settings().feedback.base_url.strip()


def get_effective_feedback_provider() -> str:
    return get_app_settings().feedback.provider.strip().lower()


def get_feedback_timeout() -> float:
    return get_app_settings().feedback.timeout


def get_feedback_max_tokens() -> int:
    return get_app_settings().feedback.max_tokens


def get_vision_settings(endpoint_name: str) -> dict[str, Any]:
    if endpoint_name not in {"chat", "feedback"}:
        raise ValueError(f"Unsupported vision endpoint: {endpoint_name}")
    endpoint = getattr(get_app_settings(), endpoint_name)
    return {
        "enabled": endpoint.vision_enabled,
        "detail": endpoint.vision_detail,
    }


def get_effective_vlm_mode() -> str:
    return get_app_settings().vlm_mode


def should_use_standalone_vlm() -> bool:
    settings = get_app_settings()
    if settings.vlm_mode == "off":
        return False
    if settings.vlm_mode == "always":
        return True
    return not (settings.chat.vision_enabled and settings.feedback.vision_enabled)


def native_vision_enabled() -> bool:
    return (
        get_vision_settings("chat")["enabled"]
        or get_vision_settings("feedback")["enabled"]
    )


def get_effective_vlm_api_key() -> str:
    return get_app_settings().vlm.api_key.strip()


def get_effective_vlm_base_url() -> str:
    return get_app_settings().vlm.base_url.strip()


def get_effective_vlm_model() -> str:
    return get_app_settings().vlm.model.strip()


def get_siliconflow_api_key() -> str:
    return get_app_settings().siliconflow_api_key


def get_token_stats_model_names() -> list[str]:
    models = [
        get_effective_chat_model(),
        get_effective_feedback_model(),
    ]
    if should_use_standalone_vlm():
        models.append(get_effective_vlm_model())

    result = []
    seen = set()
    for model in models:
        if model and model not in seen:
            result.append(model)
            seen.add(model)
    return result


def _build_memory_endpoint_settings(
    config: dict[str, Any],
) -> MemoryEndpointSettings:
    embedding = config.get("embedding", {}) or {}
    rerank = config.get("rerank", {}) or {}
    return MemoryEndpointSettings(
        model=str(embedding.get("model") or "BAAI/bge-m3"),
        base_url=str(embedding.get("base_url") or "https://api.siliconflow.cn/v1").rstrip("/"),
        timeout=float(embedding.get("timeout") or 30),
        rerank_base_url=str(rerank.get("base_url") or "https://api.siliconflow.cn/v1/rerank").rstrip("/"),
        rerank_timeout=float(rerank.get("timeout") or 10),
    )


def get_memory_endpoint_settings() -> dict[str, str | float]:
    value = get_app_settings().memory
    return {
        "model": value.model,
        "base_url": value.base_url,
        "timeout": value.timeout,
        "rerank_base_url": value.rerank_base_url,
        "rerank_timeout": value.rerank_timeout,
    }


plugin_config = load_plugin_config()
_app_settings = build_settings(plugin_config, require_api_keys=False)
