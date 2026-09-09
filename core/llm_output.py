"""LLM 输出的解析与校验边界。"""

import math
from dataclasses import dataclass

from nonebot import logger

from ..models.emotion import EmotionState
from .text_utils import extract_and_parse_json


@dataclass(frozen=True)
class ParsedFeedback:
    payload: dict | None
    failure_reason: str = ""


def parse_feedback(response: str, current_emotion: EmotionState) -> ParsedFeedback:
    """解析并校验 Feedback 的 new_emotion 边界。"""

    try:
        parsed = extract_and_parse_json(response)
    except Exception:
        return ParsedFeedback(None, "invalid_json")
    if not isinstance(parsed, dict) or not parsed:
        return ParsedFeedback(None, "invalid_payload")
    raw_emotion = parsed.get("new_emotion")
    if not isinstance(raw_emotion, dict):
        return ParsedFeedback(None, "missing_new_emotion")

    specs = {
        "valence": (-1.0, 1.0, current_emotion.valence),
        "arousal": (0.0, 1.0, current_emotion.arousal),
        "dominance": (-1.0, 1.0, current_emotion.dominance),
    }
    normalized = {}
    valid_fields = 0
    for field_name, (minimum, maximum, default) in specs.items():
        if field_name not in raw_emotion:
            normalized[field_name] = default
            continue
        try:
            value = float(raw_emotion[field_name])
        except (TypeError, ValueError):
            return ParsedFeedback(None, f"invalid_new_emotion_{field_name}")
        if not math.isfinite(value):
            return ParsedFeedback(None, f"invalid_new_emotion_{field_name}")
        normalized[field_name] = max(minimum, min(maximum, value))
        valid_fields += 1
    if valid_fields == 0:
        return ParsedFeedback(None, "empty_new_emotion")

    payload = dict(parsed)
    payload["new_emotion"] = normalized
    return ParsedFeedback(payload)


@dataclass(frozen=True)
class ReplyPlan:
    replies: list[dict | str]
    failure_reason: str = ""


def parse_reply(response: str) -> ReplyPlan:
    """解析 Chat 回复列表；裸 list 是模型常见偏差，直接兼容。"""

    try:
        payload = extract_and_parse_json(response)
    except Exception:
        return ReplyPlan([], "invalid_json")
    if isinstance(payload, dict):
        replies = payload.get("reply", [])
    elif isinstance(payload, list):
        replies = payload
        logger.warning("LLM 返回了 List 而非 Object，已自动兼容")
    else:
        return ReplyPlan([], "invalid_payload")
    if not isinstance(replies, list):
        return ReplyPlan([], "invalid_reply_list")
    return ReplyPlan(replies)
