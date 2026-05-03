import re


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?~\n])\s*", text) if part.strip()]
    return parts or [text]


def build_send_parts(text: str, *, max_messages: int = 2, strategy: str = "split_by_sentence") -> list[str]:
    strategy = (strategy or "split_by_sentence").strip().lower()
    if strategy == "single":
        parts = [text.strip()] if text and text.strip() else []
    else:
        parts = _split_text(text)
    if max_messages > 0:
        return parts[:max_messages]
    return parts
