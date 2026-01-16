# nyaturingtest/utils.py
import json
import re
import ssl
import httpx
from nonebot import logger
from .mem import Message

# 全局客户端变量
_GLOBAL_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """获取全局优化的 HTTP 客户端"""
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.set_ciphers("ALL:@SECLEVEL=1")
        _GLOBAL_HTTP_CLIENT = httpx.AsyncClient(
            verify=ssl_context,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )
    return _GLOBAL_HTTP_CLIENT


async def close_http_client():
    """关闭全局 HTTP 客户端"""
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT:
        await _GLOBAL_HTTP_CLIENT.aclose()
        _GLOBAL_HTTP_CLIENT = None
        logger.info("全局 HTTP 客户端已关闭")


def smart_split_text(text: str, max_chars: int = 40) -> list[str]:
    """
    严格断句逻辑：
    只要遇到句号、问号、感叹号等标点，强制进行切分，不合并短句。
    """
    text = text.strip()
    if not text:
        return []

    # 正则：匹配标点符号 [。！？!?~\n]，(?<=...) 为后视断言，保留标点在前半句
    raw_parts = re.split(r'(?<=[。！？!?~\n])\s*', text)

    final_parts = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        final_parts.append(part)

    return final_parts if final_parts else [text]


def extract_and_parse_json(text: str) -> dict | list | None:
    """
    提取并解析 JSON，自动去除 Markdown 代码块和思考过程标签
    """
    if not text:
        return None

    # 1. 强力去除 <think>...</think> 标签及其内容 (支持跨行)
    # flag=re.DOTALL 让 . 可以匹配换行符
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. 去除 Markdown 代码块包裹
    # 匹配 ```json ... ``` 或 ``` ... ```，捕获中间的内容
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # 如果没匹配到成对的 ```，尝试单边清理
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

    # 3. 寻找最外层的 { } 或 [ ]
    try:
        start_idx = text.find("{")
        list_start_idx = text.find("[")

        # 判断是对象还是列表
        if start_idx != -1 and (list_start_idx == -1 or start_idx < list_start_idx):
            # 提取对象
            end_idx = text.rfind("}")
            if end_idx != -1:
                json_str = text[start_idx: end_idx + 1]
                return json.loads(json_str)
        elif list_start_idx != -1:
            # 提取列表
            end_idx = text.rfind("]")
            if end_idx != -1:
                json_str = text[list_start_idx: end_idx + 1]
                return json.loads(json_str)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}\n原文: {text}")
    except Exception as e:
        logger.error(f"JSON 提取未知错误: {e}")

    return None


def estimate_split_count(text: str) -> int:
    """
    估算实际发送的消息条数
    """
    if not text:
        return 0

    raw_parts = re.split(r'(?<=[。！？!?.~\n])\s*', text)
    final_parts = []

    for part in raw_parts:
        part = part.strip()
        if not part: continue
        final_parts.append(part)

    return len(final_parts) if final_parts else 1


def check_relevance(bot_name: str, aliases: list[str], messages: list[Message]) -> bool:
    """
    检查这一批消息中是否有与机器人强相关的内容
    支持检查 bot_name 和 aliases (别名)
    增加忽略大小写的提及判定
    """
    # 合并主名和别名作为所有触发词
    triggers = [bot_name]
    if aliases:
        triggers.extend(aliases)

    # 过滤掉空字符串，防止误触
    triggers = [t for t in triggers if t and t.strip()]

    for msg in messages:
        # 统一转为小写，实现忽略大小写匹配
        content = msg.content.lower()

        for trigger in triggers:
            t = trigger.lower()
            if t in content:
                return True


    return False
