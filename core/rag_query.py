import re


_NOISE_QUERIES = {
    "?", "？", "??", "？？", "???", "？？？",
    "。", "！", "!", "...", "…",
    "草", "艹", "笑死", "哈哈", "哈哈哈", "hhh", "www",
    "233", "666", "ok", "OK", "嗯", "嗯嗯", "哦", "好", "好的",
}
_EMOJI_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _active_user_query_names(
    active_user_names: list[str] | None,
    active_users: list[dict] | None,
) -> list[str]:
    result = []
    seen = set()
    seen_names = set()

    for user in active_users or []:
        if not isinstance(user, dict):
            continue
        user_id = str(user.get("user_id") or "").strip()
        user_name = str(user.get("user_name") or "").strip()
        if not user_name:
            continue
        key = f"id:{user_id}" if user_id else f"name:{user_name}"
        if key in seen or user_name in seen_names:
            continue
        seen.add(key)
        seen_names.add(user_name)
        result.append(user_name)

    for user_name in active_user_names or []:
        user_name = str(user_name or "").strip()
        if not user_name:
            continue
        key = f"name:{user_name}"
        if key in seen or user_name in seen_names:
            continue
        seen.add(key)
        seen_names.add(user_name)
        result.append(user_name)

    return result


def is_low_value_rag_query(text: str) -> bool:
    query = str(text or "").strip()
    if not query:
        return True
    if query in _NOISE_QUERIES or query.lower() in _NOISE_QUERIES:
        return True
    if query.startswith("[表情包]"):
        return True
    if len(query) < 4:
        return True
    return bool(_EMOJI_ONLY_RE.fullmatch(query))


def build_chat_rag_queries(
    raw_queries: list[str],
    *,
    chat_summary: str = "",
    active_user_names: list[str] | None = None,
    active_users: list[dict] | None = None,
) -> list[str]:
    effective_queries = [
        str(query or "").strip()
        for query in raw_queries or []
        if not is_low_value_rag_query(str(query or ""))
    ]

    if chat_summary and str(chat_summary).strip():
        effective_queries.append(str(chat_summary).strip())

    active_query_names = _active_user_query_names(active_user_names, active_users)
    effective_queries.extend([f"关于{name}" for name in active_query_names])

    return _dedupe_preserve_order(effective_queries)
