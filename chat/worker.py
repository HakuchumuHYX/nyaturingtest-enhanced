"""每群的后台 worker：防抖取一批消息、等图片描述、跑一轮对话、把结果发出去；有未读又没新消息时到点补看。"""

import asyncio
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event

from ..messages.message import ChatMessage
from ..metrics import log_event
from .engagement import seconds_until_look
from .groups import is_shutting_down
from .orchestrator import process_chunk
from .send import dispatch_replies, send_poke, send_reaction
from .session import Session

DEBOUNCE_SECONDS = 2.0
# 一批消息处理前最多等多久图片描述；超时带着占位继续，描述出来后照样补上
CAPTION_WAIT_SECONDS = 15.0


@dataclass(frozen=True)
class InboxBatch:
    # 为空是到点补看：没有新消息，按全部未读看一眼群
    messages: list[ChatMessage]
    bot: Bot
    event: Event


async def next_inbox_batch(session: Session) -> InboxBatch | None:
    """等到静默窗口结束，再一次性取走积压的消息。

    有未读时最多等到该看群的时候：群里安静下来，「距上次看群满 180s」早就成立了，只是没有新消息来触发检查。
    """

    # 启动后还没收到过消息就没有 bot / event，不定时
    timeout = None
    if session.bot is not None and session.event is not None:
        timeout = seconds_until_look(session.state, session.unread(), datetime.now())
    try:
        await asyncio.wait_for(session.new_message_signal.wait(), timeout)
    except asyncio.TimeoutError:
        return InboxBatch(messages=[], bot=session.bot, event=session.event)
    await asyncio.sleep(DEBOUNCE_SECONDS)
    session.new_message_signal.clear()
    async with session.data_lock:
        if session.bot is None or session.event is None or not session.messages_chunk:
            return None
        batch = InboxBatch(
            messages=list(session.messages_chunk),
            bot=session.bot,
            event=session.event,
        )
        session.messages_chunk.clear()
        return batch


async def _process_inbox_batch(session: Session, batch: InboxBatch) -> None:
    current_chunk = await _prepare_messages(session, batch)
    if is_shutting_down():
        return

    generation = session.state.generation

    turn = await process_chunk(session, current_chunk, generation)
    if turn is None:
        return

    if turn.react is not None:
        await send_reaction(session, batch.bot, turn.react, generation)
    if turn.poke is not None:
        await send_poke(session, batch.bot, batch.event, turn.poke, generation)
    await dispatch_replies(
        session=session,
        replies=turn.replies,
        bot=batch.bot,
        event=batch.event,
        generation=generation,
    )


async def _prepare_messages(session: Session, batch: InboxBatch) -> list[ChatMessage]:
    """等这批图片描述、记 message 事件。"""

    current_chunk = batch.messages

    pending = [
        task
        for message in current_chunk
        for task in message.caption_tasks
        if not task.done()
    ]
    if pending:
        _, not_done = await asyncio.wait(pending, timeout=CAPTION_WAIT_SECONDS)
        if not_done:
            logger.warning(
                f"{len(not_done)} 张图 {CAPTION_WAIT_SECONDS:.0f}s 内没看完，先带着占位处理"
            )
    for message in current_chunk:
        log_event(
            "message",
            "DEBUG",
            session_id=session.id,
            user=message.user_name,
            content=message.content,
        )
    return current_chunk


async def run_worker(session: Session):
    """群 worker 的主循环：防抖取一批、跑一轮；单轮异常记日志后继续，不让任务退出。"""

    while True:
        try:
            batch = await next_inbox_batch(session)
            if batch is None:
                continue
            await _process_inbox_batch(session, batch)
        except asyncio.CancelledError:
            logger.info(f"群 {session.id} 的后台任务已取消")
            break
        except Exception as e:
            logger.exception(f"群 worker 异常: {e}")
            await asyncio.sleep(5.0)
