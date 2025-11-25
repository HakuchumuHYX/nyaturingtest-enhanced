# utils.py
import json
import re
from nonebot import logger  # [修正] 使用 nonebot 的 logger 以保持格式统一
from .mem import Message


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
            # [细节保留] 修复 LLM 常见的 JSON 尾部逗号错误 (例如 {"a":1,} )
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


def estimate_split_count(text: str) -> int:
    """
    估算实际发送的消息条数 (复用 __init__.py 的逻辑)
    """
    if not text:
        return 0
    # 使用与 __init__.py 一致的切分逻辑
    raw_parts = re.split(r'(?<=[。！？!?.~\n])\s*', text)
    final_parts = []
    current_buffer = ""
    for part in raw_parts:
        part = part.strip()
        if not part: continue
        if len(current_buffer) + len(part) < 15:
            current_buffer += part
        else:
            if current_buffer: final_parts.append(current_buffer)
            current_buffer = part
    if current_buffer: final_parts.append(current_buffer)

    # 至少算 1 条
    return len(final_parts) if final_parts else 1


def check_relevance(bot_name: str, messages: list[Message]) -> bool:
    """
    检查这一批消息中是否有与机器人强相关的内容
    """
    for msg in messages:
        content = msg.content
        # 1. 检查名字出现在文本中
        if bot_name in content:
            return True

        # 2. 检查 @ (NoneBot 的 segment 转换为了 " @name ")
        if f"@{bot_name}" in content:
            return True

        # 3. 检查回复 (格式 [回复 name: ...])
        if f"[回复 {bot_name}" in content:
            return True

    return False
