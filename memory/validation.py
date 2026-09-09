"""长期记忆候选的服务端校验。

只做确定性过滤（长度、噪声、类别、置信度、主体），不做语义启发式——
「是不是玩笑 / 有没有注入指令」交给 Feedback 的 prompt 与 confidence 判断。
"""


ALLOWED_MEMORY_CATEGORIES = {"event", "preference", "profile", "relationship"}
MIN_MEMORY_CONFIDENCE = 0.6
MIN_CONTENT_CHARS = 10

NOISE_WORDS = {
    "好的", "好", "嗯", "嗯嗯", "哦", "哦哦",
    "ok", "收到", "了解", "明白",
    "哈哈", "哈哈哈", "233", "666", "厉害",
    "是的", "对", "对的", "是啊", "好吧",
    "行", "可以", "没问题", "好呀", "好哒",
    "谢谢", "感谢", "辛苦了", "拜拜", "再见",
    "早", "晚安", "午安", "早安", "晚上好",
}


class MemoryValidationResult:
    __slots__ = ("valid", "reason")

    def __init__(self, valid: bool, reason: str = "ok"):
        self.valid = bool(valid)
        self.reason = reason


def should_store_memory(content: str) -> bool:
    text = str(content or "").strip()
    if len(text) < MIN_CONTENT_CHARS:
        return False
    return text.lower() not in NOISE_WORDS


def validate_memory_candidate(
    *,
    content: str,
    category: str,
    confidence: float,
    subject_user_id: str = "",
    subject_user_name: str = "",
) -> MemoryValidationResult:
    if not should_store_memory(content):
        return MemoryValidationResult(False, "too_short_or_noise")

    if str(category or "").strip().lower() not in ALLOWED_MEMORY_CATEGORIES:
        return MemoryValidationResult(False, "unsupported_category")

    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError):
        numeric_confidence = 0.0
    if numeric_confidence < MIN_MEMORY_CONFIDENCE:
        return MemoryValidationResult(False, "low_confidence")

    if not str(subject_user_id or "").strip() and not str(subject_user_name or "").strip():
        return MemoryValidationResult(False, "missing_subject")

    return MemoryValidationResult(True)
