"""每群的后台 worker：防抖取一批消息、等图片描述、跑一轮对话、把结果发出去。"""

import asyncio
from dataclasses import dataclass

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event

from ..messages.message import Message as MMessage
from ..metrics import log_event
from .groups import SELF_SENT_MSG_IDS, GroupState, is_shutting_down
from .orchestrator import process_chunk
from .send import dispatch_replies, send_poke, send_reaction

DEBOUNCE_SECONDS = 2.0
# 一批消息处理前最多等多久图片描述；超时带着占位继续，描述出来后照样补上
CAPTION_WAIT_SECONDS = 15.0


@dataclass(frozen=True)
class InboxBatch:
    messages: list[MMessage]
    bot: Bot
    event: Event


async def next_inbox_batch(state: GroupState) -> InboxBatch | None:
    """等到静默窗口结束，再一次性取走积压的消息。"""

    await state.new_message_signal.wait()
    await asyncio.sleep(DEBOUNCE_SECONDS)
    state.new_message_signal.clear()
    async with state.data_lock:
        if state.bot is None or state.event is None or not state.messages_chunk:
            return None
        batch = InboxBatch(
            messages=list(state.messages_chunk),
            bot=state.bot,
            event=state.event,
        )
        state.messages_chunk.clear()
        return batch


def _filter_local_self_echoes(
    messages: list[MMessage], bot_self_id: str
) -> tuple[list[MMessage], list[MMessage]]:
    """按本地已发送消息 ID 分离自身回显。"""

    filtered, local_echoes = [], []
    for msg in messages:
        if msg.id and msg.user_id == bot_self_id and msg.id in SELF_SENT_MSG_IDS:
            local_echoes.append(msg)
        else:
            filtered.append(msg)
    return filtered, local_echoes


async def _process_inbox_batch(state: GroupState, batch: InboxBatch) -> None:
    bot_self_id = str(batch.bot.self_id)
    current_chunk, local_echoes = _filter_local_self_echoes(batch.messages, bot_self_id)
    if local_echoes:
        logger.debug(f"过滤本机自身回显消息 {len(local_echoes)} 条")
    if not current_chunk:
        return

    if all(message.user_id == bot_self_id for message in current_chunk):
        # 自身账号发出的消息已由 append_self_message 写入记忆，这里直接丢弃
        return

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
            session_id=state.session.id,
            user=message.user_name,
            content=message.content,
        )
    if is_shutting_down():
        return

    await state.load()
    generation = state.session.state.generation

    images = [item for message in current_chunk for item in message.image_inputs]
    try:
        turn = await process_chunk(state.session, current_chunk, images, generation)
    finally:
        for message in current_chunk:
            message.image_inputs.clear()
    if turn is None:
        return

    if turn.react is not None:
        await send_reaction(state, batch.bot, turn.react, generation)
    if turn.poke is not None:
        await send_poke(state, batch.bot, batch.event, turn.poke, generation)
    await dispatch_replies(
        state=state,
        replies=turn.replies,
        bot=batch.bot,
        event=batch.event,
        generation=generation,
    )


async def spawn_state(state: GroupState):
    """Small worker boundary: debounce, invoke one turn, contain failures."""

    while True:
        try:
            batch = await next_inbox_batch(state)
            if batch is None:
                continue
            await _process_inbox_batch(state, batch)
        except asyncio.CancelledError:
            logger.info(f"后台任务被取消: {id(state)}")
            break
        except Exception as e:
            logger.exception(f"Spawn loop error: {e}")
            await asyncio.sleep(5.0)
