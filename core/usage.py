import asyncio
from collections.abc import Callable

from nonebot import logger

from ..database.token_repository import TokenUsageRepository


def record_token_usage(session_id: str, model_name: str, usage: dict) -> None:
    task = asyncio.create_task(
        TokenUsageRepository.log_token_usage(
            session_id=session_id,
            model_name=model_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0),
            prompt_cache_miss_tokens=usage.get("prompt_cache_miss_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            finish_reason=usage.get("finish_reason", ""),
            provider=usage.get("provider", ""),
        )
    )
    task.add_done_callback(_log_usage_task_error)


def make_usage_recorder(
    session_id: str,
    model_name: str,
    *,
    event_logger: Callable[[dict], None] | None = None,
) -> Callable[[dict], None]:
    def _recorder(usage: dict) -> None:
        if event_logger:
            event_logger(usage)
        record_token_usage(session_id, model_name, usage)

    return _recorder


def _log_usage_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"记录 Token 消耗失败: {exc}")
