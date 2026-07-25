def sender_display_name(event, user_id: str) -> str:
    sender = getattr(event, "sender", None)
    card = str(getattr(sender, "card", "") or "").strip()
    nickname = str(getattr(sender, "nickname", "") or "").strip()
    return card or nickname or str(user_id)
