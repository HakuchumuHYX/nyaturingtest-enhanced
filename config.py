# nyaturingtest/config.py
from nonebot import get_driver, get_plugin_config
from pydantic import BaseModel


class Config(BaseModel):
    # ===================== Chat LLM =====================
    # Generic chat model config (provider-agnostic)
    nyaturingtest_chat_api_key: str = ""
    nyaturingtest_chat_model: str = "Qwen/Qwen3-32B"
    nyaturingtest_chat_base_url: str = "https://api.siliconflow.cn/v1"

    # Deprecated legacy names (kept for backward compatibility / fallback)
    nyaturingtest_chat_openai_api_key: str = ""
    nyaturingtest_chat_openai_model: str = ""
    nyaturingtest_chat_openai_base_url: str = ""

    # Provider switch
    # - openai_compatible: OpenAI SDK + /chat/completions compatible endpoint
    # - google_ai_studio: Gemini Developer API official endpoint (generateContent)
    nyaturingtest_chat_provider: str = "openai_compatible"
    nyaturingtest_chat_google_api_key: str | None = None
    nyaturingtest_chat_google_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # ===================== VLM (Vision-Language Model) =====================
    # Enable/Disable VLM image understanding (legacy env: nyaturingtest_vlm_enabled)
    nyaturingtest_vlm_enabled: bool = True

    # Allow VLM to be configured independently from Chat.
    # If VLM openai_* is empty, it will fallback to Chat openai_* at runtime.
    nyaturingtest_vlm_provider: str = "openai_compatible"
    nyaturingtest_vlm_openai_api_key: str = ""
    nyaturingtest_vlm_openai_base_url: str = ""
    nyaturingtest_vlm_model: str = "gemini-3-flash-preview"
    nyaturingtest_vlm_google_api_key: str | None = None
    nyaturingtest_vlm_google_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # ===================== SiliconFlow / Rerank / Feedback (unchanged) =====================
    nyaturingtest_feedback_openai_model: str = "Qwen/Qwen2.5-7B-Instruct"
    nyaturingtest_siliconflow_api_key: str
    nyaturingtest_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    nyaturingtest_rerank_threshold: float = 0.05
    nyaturingtest_enabled_groups: list[int] = []


plugin_config: Config = get_plugin_config(Config)
global_config = get_driver().config


def get_effective_chat_api_key(cfg: Config | None = None) -> str:
    cfg = cfg or plugin_config
    return (cfg.nyaturingtest_chat_api_key or cfg.nyaturingtest_chat_openai_api_key or "").strip()


def get_effective_chat_model(cfg: Config | None = None) -> str:
    cfg = cfg or plugin_config
    return (cfg.nyaturingtest_chat_model or cfg.nyaturingtest_chat_openai_model or "Qwen/Qwen3-32B").strip()


def get_effective_chat_base_url(cfg: Config | None = None) -> str:
    cfg = cfg or plugin_config
    return (cfg.nyaturingtest_chat_base_url or cfg.nyaturingtest_chat_openai_base_url or "").strip()
