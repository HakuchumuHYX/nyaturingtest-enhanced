import re


_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?~\n])\s*|(?<!\.)\.(?!\.)(?=\s|$|[\u4e00-\u9fff])\s*")
_SINGLE_TRAILING_PERIOD = re.compile(r"(?<!\.)\.$")


def _normalize_send_part(text: str) -> str:
    part = text.strip()
    if not part:
        return ""
    part = part.rstrip("。")
    part = _SINGLE_TRAILING_PERIOD.sub("", part)
    return part.strip()


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SPLIT_PATTERN.split(text) if part.strip()]
    return parts or [text]


def build_send_parts(text: str, *, max_messages: int = 2, strategy: str = "split_by_sentence") -> list[str]:
    strategy = (strategy or "split_by_sentence").strip().lower()
    if strategy == "single":
        raw_parts = [text.strip()] if text and text.strip() else []
    else:
        raw_parts = _split_text(text)
    parts = [_normalize_send_part(part) for part in raw_parts]
    parts = [part for part in parts if part]
    if max_messages > 0:
        return parts[:max_messages]
    return parts
