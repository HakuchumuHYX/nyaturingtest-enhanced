# nyaturingtest/config.py
from nonebot import get_driver, get_plugin_config
from pydantic import BaseModel

class Config(BaseModel):
    nyaturingtest_chat_openai_api_key: str
    nyaturingtest_chat_openai_model: str = "Qwen/Qwen3-32B"
    nyaturingtest_chat_openai_base_url: str = "https://api.siliconflow.cn/v1" # 确保 Base URL 正确
    nyaturingtest_siliconflow_api_key: str
    nyaturingtest_enabled_groups: list[int] = []

plugin_config: Config = get_plugin_config(Config)
global_config = get_driver().config
