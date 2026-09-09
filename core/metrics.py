# 由多个模块合并而来：core/metrics.py, core/usage.py

import json
from dataclasses import dataclass
from nonebot import logger
import asyncio
from collections.abc import Callable
from ..db import TokenUsageRepository


# ======== from core/metrics.py ========
def log_event(event: str, **fields):
    """输出一行结构化 JSON 事件日志。"""

    payload = {"event": event}
    payload.update({key: value for key, value in fields.items() if value is not None})
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


@dataclass
class RuntimeMetrics:
    llm_success: int = 0
    llm_failure: int = 0
    memory_query_count: int = 0
    memory_query_cache_hit: int = 0
    memory_query_singleflight_reused: int = 0
    memory_query_cooldown_rejected: int = 0
    memory_query_total_ms: float = 0.0
    memory_query_rag_calls: int = 0
    memory_query_feedback_calls: int = 0
    memory_query_chat_calls: int = 0


metrics = RuntimeMetrics()

# ======== from core/usage.py ========
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
