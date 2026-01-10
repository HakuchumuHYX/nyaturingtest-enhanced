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


def extract_and_parse_json(response: str) -> dict | None:
    """
    从 LLM 响应中提取并解析 JSON
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试提取 markdown 代码块
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取最外层 {}
    match = re.search(r"(\{[\s\S]*\})", response)
    if match:
        json_str = match.group(1)
        try:
            #  修复 LLM 常见的 JSON 尾部逗号错误 (例如 {"a":1,} )
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

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
