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
PRESET_DIR = WORKSPACE_ROOT / "config" / "nyaturingtest" / "nya_presets"
BACKUP_DIR = WORKSPACE_ROOT / "data" / "nyaturingtest_backups"
IMAGE_CACHE_DIR = WORKSPACE_ROOT / "cache" / "nyaturingtest" / "image_cache"


def get_data_dir() -> Path:
    """运行数据目录；仅此项支持环境变量覆盖，便于独立部署。"""

    value = os.environ.get("NYATURINGTEST_DATA_DIR", "").strip()
    if not value:
        return WORKSPACE_ROOT / "data" / "nyaturingtest"
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


@dataclass(frozen=True)
class EndpointSettings:
    api_key: str
    base_url: str
    model: str
    timeout: float
    max_tokens: int
    reasoning_effort: str


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
    rerank_model: str
    rerank_threshold: float
    memory: MemoryEndpointSettings
    siliconflow_api_key: str


DEFAULT_CONFIG: dict[str, Any] = {
    "chat": {
        "api_key": "",
        "base_url": "",
        "model": "",
        "reasoning_effort": "low",
        "max_tokens": 4096,
        "timeout": 180,
    },
    "feedback": {
        "api_key": "",
        "base_url": "",
        "model": "",
        "reasoning_effort": "",
        "max_tokens": 2048,
        "timeout": 60,
    },
    "siliconflow_api_key": "",
    "embedding": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
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


def _require(section: dict, name: str, field_name: str) -> str:
    value = str(section.get(field_name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required config field: {name}.{field_name}")
    return value


def _endpoint(cfg: dict[str, Any], name: str) -> EndpointSettings:
    section = cfg[name]
    return EndpointSettings(
        api_key=str(section["api_key"] or ""),
        base_url=_require(section, name, "base_url"),
        model=_require(section, name, "model"),
        timeout=float(section["timeout"]),
        max_tokens=int(section["max_tokens"]),
        reasoning_effort=str(section["reasoning_effort"] or ""),
    )


def build_settings(config: dict[str, Any]) -> AppSettings:
    cfg = _deep_merge(DEFAULT_CONFIG, config)
    embedding = cfg["embedding"]
    rerank = cfg["rerank"]
    return AppSettings(
        chat=_endpoint(cfg, "chat"),
        feedback=_endpoint(cfg, "feedback"),
        rerank_model=str(rerank["model"] or ""),
        rerank_threshold=float(rerank["threshold"]),
        memory=MemoryEndpointSettings(
            model=str(embedding["model"]),
            base_url=str(embedding["base_url"]).rstrip("/"),
            timeout=float(embedding["timeout"]),
            rerank_base_url=str(rerank["base_url"]).rstrip("/"),
            rerank_timeout=float(rerank["timeout"]),
        ),
        siliconflow_api_key=str(cfg["siliconflow_api_key"] or ""),
    )


def _load_settings() -> AppSettings:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        settings = build_settings(json.load(f))
    logger.info(f"已加载插件配置: {CONFIG_FILE}")
    for name, endpoint in (("chat", settings.chat), ("feedback", settings.feedback)):
        key_state = "set" if endpoint.api_key else "missing"
        logger.info(
            f"{name}: model={endpoint.model}, base_url={endpoint.base_url}, "
            f"api_key={key_state}"
        )
    return settings


_app_settings = _load_settings()


def get_app_settings() -> AppSettings:
    return _app_settings


def get_token_stats_model_names() -> list[str]:
    models = [_app_settings.chat.model, _app_settings.feedback.model]
    return list(dict.fromkeys(model.strip() for model in models))
