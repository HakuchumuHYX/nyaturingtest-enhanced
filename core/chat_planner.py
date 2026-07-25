from dataclasses import dataclass

from nonebot import logger

from .text_utils import extract_and_parse_json


@dataclass(frozen=True)
class ReplyPlan:
    replies: list[dict | str]
    failure_reason: str = ""


class ChatPlanner:
    """Validate Chat output without performing transport or persistence."""

    @staticmethod
    def parse(response: str) -> ReplyPlan:
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
