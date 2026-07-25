import json
import re

from nonebot import logger

from ..memory.short_term import Message


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    try:
        return text.encode("utf-8", "ignore").decode("utf-8")
    except (AttributeError, UnicodeError):
        return ""


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


def check_relevance(
    bot_name: str,
    aliases: list[str],
    messages: list[Message],
) -> bool:
    triggers = [bot_name, *(aliases or [])]
    triggers = [
        value.strip().lower()
        for value in triggers
        if value and len(value.strip()) >= 2
    ]
    return any(
        trigger in message.content.lower()
        for message in messages
        for trigger in triggers
    )


def score_message_interest(
    contents,
    bot_name: str = "",
    aliases=None,
    *,
    lo: float = 0.3,
    hi: float = 2.0,
) -> float:
    aliases = aliases or []
    text = " ".join(str(content or "") for content in (contents or []))
    if not text.strip():
        return lo
    score = 1.0
    if "?" in text or "？" in text:
        score += 0.6
    names = [str(bot_name or "").strip()]
    names.extend(
        str(alias).strip()
        for alias in aliases
        if alias and len(str(alias).strip()) >= 2
    )
    if any(name and name in text for name in names):
        score += 0.7
    stripped = text.strip()
    if len(set(stripped)) <= 2 and len(stripped) >= 3:
        score -= 0.6
    if stripped in {"[图片]", "[表情包]"}:
        score -= 0.5
    if len(stripped) >= 15:
        score += 0.2
    return max(lo, min(hi, score))


def should_store_memory(content: str) -> bool:
    if not content or len(content.strip()) < 10:
        return False
    noise_words = {
        "好的", "好", "嗯", "嗯嗯", "哦", "哦哦", "ok", "收到", "了解",
        "明白", "哈哈", "哈哈哈", "233", "666", "厉害", "是的", "对",
        "对的", "是啊", "好吧", "行", "可以", "没问题", "好呀", "好哒",
        "谢谢", "感谢", "辛苦了", "拜拜", "再见", "早", "晚安", "午安",
        "早安", "晚上好",
    }
    return content.strip().lower() not in noise_words


def calculate_dynamic_k(
    interaction_count: int,
    memory_count: int,
    days_since_first: int,
) -> int:
    if memory_count <= 10:
        max_limit = memory_count
    elif memory_count <= 30:
        max_limit = 20
    elif memory_count <= 50:
        max_limit = 30
    else:
        max_limit = 40
    interaction_bonus = min(interaction_count // 50, 6)
    memory_bonus = min(memory_count // 10, 8)
    time_bonus = 4 if days_since_first > 90 else 3 if days_since_first > 30 else 2 if days_since_first > 7 else 0
    return max(5, min(5 + interaction_bonus + memory_bonus + time_bonus, max_limit))
