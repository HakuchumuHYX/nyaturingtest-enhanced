import asyncio
from collections.abc import Callable

from nonebot import logger

from ..database.token_repository import TokenUsageRepository


_PENDING_USAGE_TASKS: set[asyncio.Task] = set()


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
    _PENDING_USAGE_TASKS.add(task)
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
    _PENDING_USAGE_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"记录 Token 消耗失败: {exc}")


async def drain_usage_tasks(timeout: float | None = None) -> None:
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout

    while _PENDING_USAGE_TASKS:
        tasks = list(_PENDING_USAGE_TASKS)
        wait_timeout = None
        if deadline is not None:
            wait_timeout = max(0.0, deadline - loop.time())
            if wait_timeout <= 0:
                break

        done, pending = await asyncio.wait(tasks, timeout=wait_timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        if pending:
            break

    if _PENDING_USAGE_TASKS:
        pending_count = len(_PENDING_USAGE_TASKS)
        logger.warning(f"等待 Token 消耗记录任务超时，仍有 {pending_count} 个任务未完成")
        pending = list(_PENDING_USAGE_TASKS)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
