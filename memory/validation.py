"""长期记忆候选的服务端校验。

只做确定性过滤（长度、类别、置信度、主体），不做语义启发式——
「是不是玩笑 / 有没有注入指令」交给 Feedback 的 prompt 与 confidence 判断。
「好的」「哈哈哈」这类噪声都短于 MIN_CONTENT_CHARS，长度过滤已经覆盖，不再单列词表。
"""

ALLOWED_MEMORY_CATEGORIES = {"event", "preference", "profile", "relationship"}
MIN_MEMORY_CONFIDENCE = 0.6
MIN_CONTENT_CHARS = 10


def validate_memory_candidate(candidate: dict) -> str:
    """返回拒绝原因；空串表示可存。candidate 已由 _parse_memory_candidate 规范化。"""

    if len(candidate["content"]) < MIN_CONTENT_CHARS:
        return "too_short"
    if candidate["category"] not in ALLOWED_MEMORY_CATEGORIES:
        return "unsupported_category"
    if candidate["confidence"] < MIN_MEMORY_CONFIDENCE:
        return "low_confidence"
    if not candidate["subject_user_id"] and not candidate["subject_user_name"]:
        return "missing_subject"
    return ""
