"""Compatibility facade for helpers moved to focused modules.

New production code should import from ``core.http_client``,
``core.text_utils``, ``core.time_context`` or ``presenters`` directly.
"""

from .core.http_client import close_http_client, get_http_client
from .core.text_utils import (
    calculate_dynamic_k,
    check_relevance,
    extract_and_parse_json,
    sanitize_text,
    score_message_interest,
    should_store_memory,
)
from .core.time_context import get_time_description
from .presenters.token_stats_card import render_token_stats_card


__all__ = [
    "calculate_dynamic_k",
    "check_relevance",
    "close_http_client",
    "extract_and_parse_json",
    "get_http_client",
    "get_time_description",
    "render_token_stats_card",
    "sanitize_text",
    "score_message_interest",
    "should_store_memory",
]
