"""每群的后台 worker：防抖取一批消息、等图片描述、跑一轮对话、把结果发出去。"""

import asyncio
from dataclasses import dataclass

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event

from ..messages.message import ChatMessage
from ..metrics import log_event
from .groups import SELF_SENT_MSG_IDS, GroupState, is_shutting_down
from .orchestrator import process_chunk
from .send import dispatch_replies, send_poke, send_reaction

DEBOUNCE_SECONDS = 2.0
# 一批消息处理前最多等多久图片描述；超时带着占位继续，描述出来后照样补上
CAPTION_WAIT_SECONDS = 15.0


@dataclass(frozen=True)
class InboxBatch:
    messages: list[ChatMessage]
    bot: Bot
    event: Event


async def next_inbox_batch(group: GroupState) -> InboxBatch | None:
    """等到静默窗口结束，再一次性取走积压的消息。"""

    await group.new_message_signal.wait()
    await asyncio.sleep(DEBOUNCE_SECONDS)
    group.new_message_signal.clear()
    async with group.data_lock:
        if group.bot is None or group.event is None or not group.messages_chunk:
            return None
        batch = InboxBatch(
            messages=list(group.messages_chunk),
            bot=group.bot,
            event=group.event,
        )
        group.messages_chunk.clear()
        return batch


def _filter_local_self_echoes(
    messages: list[ChatMessage], bot_self_id: str
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """按本地已发送消息 ID 分离自身回显。"""

    filtered, local_echoes = [], []
    for msg in messages:
        if msg.id and msg.user_id == bot_self_id and msg.id in SELF_SENT_MSG_IDS:
            local_echoes.append(msg)
        else:
            filtered.append(msg)
    return filtered, local_echoes


async def _process_inbox_batch(group: GroupState, batch: InboxBatch) -> None:
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
            session_id=group.session.id,
            user=message.user_name,
            content=message.content,
        )
    if is_shutting_down():
        return

    await group.load()
    generation = group.session.state.generation

    images = [item for message in current_chunk for item in message.image_inputs]
    try:
        turn = await process_chunk(group.session, current_chunk, images, generation)
    finally:
        for message in current_chunk:
            message.image_inputs.clear()
    if turn is None:
        return

    if turn.react is not None:
        await send_reaction(group, batch.bot, turn.react, generation)
    if turn.poke is not None:
        await send_poke(group, batch.bot, batch.event, turn.poke, generation)
    await dispatch_replies(
        group=group,
        replies=turn.replies,
        bot=batch.bot,
        event=batch.event,
        generation=generation,
    )


async def run_worker(group: GroupState):
    """群 worker 的主循环：防抖取一批、跑一轮；单轮异常记日志后继续，不让任务退出。"""

    while True:
        try:
            batch = await next_inbox_batch(group)
            if batch is None:
                continue
            await _process_inbox_batch(group, batch)
        except asyncio.CancelledError:
            logger.info(f"后台任务被取消: {id(group)}")
            break
        except Exception as e:
            logger.exception(f"群 worker 异常: {e}")
            await asyncio.sleep(5.0)
