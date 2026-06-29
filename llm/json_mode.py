def is_json_mode_unsupported_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "json mode is not supported" in text
        or "response_format" in text and "not supported" in text
    )
