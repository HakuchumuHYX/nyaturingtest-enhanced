import asyncio
import json
from dataclasses import dataclass

from nonebot import logger

from .storage.db import log_token_usage


def log_event(event: str, level: str = "INFO", **fields):
    """输出一行结构化事件日志：event k=v k=v，字段按调用顺序。"""

    parts = [event]
    parts.extend(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    logger.log(level, " ".join(parts))


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    # 含空白、等号、引号或为空时加引号；换行顺带转义，保证一事件一行
    if not text or any(ch.isspace() or ch in '="' for ch in text):
        return json.dumps(text, ensure_ascii=False)
    return text


@dataclass
class RuntimeMetrics:
    llm_success: int = 0
    llm_failure: int = 0


metrics = RuntimeMetrics()

_PENDING_USAGE_TASKS: set[asyncio.Task] = set()
USAGE_DRAIN_TIMEOUT_SECONDS = 10.0


def record_token_usage(session_id: str, model_name: str, usage: dict) -> None:
    task =asyncio.create_task(log_token_usage(session_id, model_name, usage))
    _PENDING_USAGE_TASKS.add(task)
    task.add_done_callback(_log_usage_task_error)


def _log_usage_task_error(task: asyncio.Task) -> None:
    _PENDING_USAGE_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"记录 Token 消耗失败: {exc}")


async def drain_usage_tasks() -> None:
    if not _PENDING_USAGE_TASKS:
        return

    tasks = list(_PENDING_USAGE_TASKS)
    done, pending = await asyncio.wait(tasks, timeout=USAGE_DRAIN_TIMEOUT_SECONDS)
    if done:
        await asyncio.gather(*done, return_exceptions=True)
    if pending:
        logger.warning(f"等待 Token 消耗记录任务超时，取消 {len(pending)} 个任务")
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
