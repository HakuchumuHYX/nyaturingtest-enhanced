class MemoryValidationResult:
    __slots__ = ("valid", "reason")

    def __init__(self, valid: bool, reason: str = "ok"):
        self.valid = bool(valid)
        self.reason = reason


ALLOWED_MEMORY_CATEGORIES = {"event", "preference", "profile", "relationship"}
MIN_MEMORY_CONFIDENCE = 0.6

INSTRUCTION_LIKE_MARKERS = (
    "忽略系统",
    "忽略之前",
    "忽略以上",
    "无视系统",
    "无视规则",
    "覆盖系统",
    "改变规则",
    "更改规则",
    "不要遵守",
    "越狱",
    "只输出 json",
    "输出 json",
    "更改输出格式",
    "执行命令",
    "运行命令",
    "调用工具",
    "system prompt",
    "developer message",
    "ignore previous",
    "ignore all previous",
    "ignore system",
    "output json",
    "only output json",
    "execute command",
    "run command",
)

JOKE_OR_SARCASM_MARKERS = (
    "开玩笑",
    "只是玩笑",
    "不是认真的",
    "别当真",
    "才怪",
    "狗头",
    "/s",
    "just kidding",
    "not serious",
    "sarcasm",
)

AMBIGUOUS_MARKERS = (
    "可能",
    "好像",
    "似乎",
    "大概",
    "也许",
    "听说",
    "据说",
    "不确定",
    "maybe",
    "probably",
    "seems",
    "might",
    "rumor",
    "heard that",
)

NOISE_WORDS = {
    "好的", "好", "嗯", "嗯嗯", "哦", "哦哦",
    "ok", "收到", "了解", "明白",
    "哈哈", "哈哈哈", "233", "666", "厉害",
    "是的", "对", "对的", "是啊", "好吧",
    "行", "可以", "没问题", "好呀", "好哒",
    "谢谢", "感谢", "辛苦了", "拜拜", "再见",
    "早", "晚安", "午安", "早安", "晚上好",
}


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _clean_text(value) -> str:
    return str(value or "").strip()


def validate_memory_candidate(
    *,
    content: str,
    category: str,
    confidence: float,
    subject_user_id: str = "",
    subject_user_name: str = "",
    reason: str = "",
) -> MemoryValidationResult:
    normalized = _clean_text(content)
    if not normalized:
        return MemoryValidationResult(False, "empty")
    if len(normalized) < 10:
        return MemoryValidationResult(False, "too_short")
    if normalized.lower() in NOISE_WORDS:
        return MemoryValidationResult(False, "noise")

    normalized_category = _clean_text(category).lower()
    if normalized_category not in ALLOWED_MEMORY_CATEGORIES:
        return MemoryValidationResult(False, "unsupported_category")

    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError):
        numeric_confidence = 0.0
    if numeric_confidence < MIN_MEMORY_CONFIDENCE:
        return MemoryValidationResult(False, "low_confidence")

    if not _clean_text(subject_user_id) and not _clean_text(subject_user_name):
        return MemoryValidationResult(False, "missing_subject")

    inspect_text = f"{normalized} {_clean_text(reason)}".lower()
    if _contains_any(inspect_text, INSTRUCTION_LIKE_MARKERS):
        return MemoryValidationResult(False, "instruction_like")
    if _contains_any(inspect_text, JOKE_OR_SARCASM_MARKERS):
        return MemoryValidationResult(False, "joke_or_sarcasm")
    if _contains_any(inspect_text, AMBIGUOUS_MARKERS):
        return MemoryValidationResult(False, "ambiguous_or_hedged")

    return MemoryValidationResult(True)
