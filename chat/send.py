"""回复切分与发送、表情包、贴表情、戳一戳，以及各自的冷却。"""

import asyncio
import random
import re
from datetime import datetime

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from ..messages.faces import Reaction
from ..messages.stickers import Sticker, mark_sent, pick
from ..metrics import log_event
from .engagement import SPEAK_COOLDOWN_SECONDS
from .groups import SELF_SENT_MSG_IDS, GroupState
from .orchestrator import Poke, Reply

MAX_REPLY_MESSAGES = 2
REACT_COOLDOWN_SECONDS = 60
# 戳一戳会给对方弹提醒，比贴表情打扰人，间隔放长些
POKE_COOLDOWN_SECONDS = 120
STICKER_COOLDOWN_SECONDS = 120

_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？!?~\n])\s*|(?<!\.)\.(?!\.)(?=\s|$|[\u4e00-\u9fff])\s*"
)
_SINGLE_TRAILING_PERIOD = re.compile(r"(?<!\.)\.$")


def build_send_parts(text: str, max_messages: int = MAX_REPLY_MESSAGES) -> list[str]:
    """按句切开、去掉句末句号；超过 max_messages 条时，多出的并进最后一条。"""

    parts = []
    for part in _SPLIT_PATTERN.split(text.strip()):
        part = _SINGLE_TRAILING_PERIOD.sub("", part.strip().rstrip("。")).strip()
        if part:
            parts.append(part)
    if len(parts) > max_messages:
        return [*parts[: max_messages - 1], " ".join(parts[max_messages - 1 :])]
    return parts


def _delay_seconds(part: str) -> float:
    return min(1.0 + len(part) * 0.1, 5.0)


async def send_one(
    *,
    group: GroupState,
    bot: Bot,
    event: Event,
    message: Message,
    memory_text: str,
    generation: int,
) -> bool:
    try:
        result = await bot.send(message=message, event=event)
        message_id = ""
        if isinstance(result, dict) and "message_id" in result:
            message_id = str(result["message_id"])
            SELF_SENT_MSG_IDS.append(message_id)
        log_event("reply", session_id=group.session.id, content=memory_text)

        # 同步写入短时记忆，中间没有 await，不需要再拿 session_lock
        if not group.session.stale(generation, "append_self_message"):
            group.session.append_self_message(
                memory_text,
                message_id,
                str(bot.self_id),
            )
        return True
    except ActionFailed as e:
        if getattr(e, "retcode", 0) == 1200 or "120" in str(e):
            logger.warning("风控拦截 (1200), 冷却中...")
            await asyncio.sleep(random.uniform(5.0, 10.0))
        else:
            logger.error(f"发送失败: {e}")
    except Exception as e:
        logger.error(f"发送未知错误: {e}")
    return False


async def _pick_sticker(
    group: GroupState, tag: str, want: str
) -> tuple[Sticker, float | None] | None:
    """挑一张要发的表情包和它的分数；冷却中或标签下没图返回 None。"""

    since_last = (datetime.now() - group.last_sticker_time).total_seconds()
    if since_last < STICKER_COOLDOWN_SECONDS:
        outcome = "cooldown"
    else:
        picked = await pick(tag, want)
        if picked is not None:
            return picked
        outcome = "empty"
    log_event(
        "sticker", session_id=group.session.id, tag=tag, want=want, outcome=outcome
    )
    return None


async def dispatch_replies(
    *,
    group: GroupState,
    replies: list[Reply],
    bot: Bot,
    event: Event,
    generation: int,
) -> int:
    if not replies:
        return 0
    # 发言冷却在发送前等待，而不是在轮次开头跳过：对方秒回时不会因为冷却把对话掐断
    since_last = (datetime.now() - group.session.state.last_speak_time).total_seconds()
    if since_last < SPEAK_COOLDOWN_SECONDS:
        await asyncio.sleep(SPEAK_COOLDOWN_SECONDS - since_last)
    if group.session.stale(generation, "pre_send"):
        return 0

    total = len(replies)
    sent_count = 0
    text_count = 0
    # 一轮最多一张表情包，不占文字的 MAX_REPLY_MESSAGES 名额
    sticker_requested = False
    for reply_index, reply in enumerate(replies):
        # (消息, 写进短期记忆的文本, 表情包和挑图分数)；模型写的文字一律按纯文本发，不解析 CQ 码
        outgoing: list[tuple[Message, str, tuple[Sticker, float | None] | None]] = []
        if text_count < MAX_REPLY_MESSAGES:
            parts = build_send_parts(reply.text, MAX_REPLY_MESSAGES - text_count)
            outgoing = [
                (Message(MessageSegment.text(part)), part, None) for part in parts
            ]
        if reply.sticker is not None and not sticker_requested:
            sticker_requested = True
            picked = await _pick_sticker(group, *reply.sticker)
            if picked is not None:
                sticker = picked[0]
                # sub_type=1 让 QQ 按表情包显示；SnowLuma 容器挂载了同一路径，直接给本地文件
                segment = MessageSegment(
                    "image", {"file": sticker.path.as_uri(), "sub_type": 1}
                )
                outgoing.append((Message(segment), f"[表情包: {sticker.caption}]", picked))

        for part_index, (message, memory_text, picked_sticker) in enumerate(outgoing):
            if group.session.stale(generation, "send_loop"):
                break
            if reply.target_id is not None and reply_index == 0 and part_index == 0:
                message.insert(0, MessageSegment.reply(reply.target_id))

            sent = await send_one(
                group=group,
                bot=bot,
                event=event,
                message=message,
                memory_text=memory_text,
                generation=generation,
            )
            if sent:
                sent_count += 1
                if picked_sticker is None:
                    text_count += 1
                else:
                    group.last_sticker_time = datetime.now()
                    await mark_sent(picked_sticker[0])
            if picked_sticker is not None:
                sticker, score = picked_sticker
                tag, want = reply.sticker
                log_event(
                    "sticker",
                    session_id=group.session.id,
                    tag=tag,
                    want=want,
                    digest=sticker.digest[:8],
                    score=None if score is None else round(score, 3),
                    outcome="ok" if sent else "failed",
                )

            has_more = part_index < len(outgoing) - 1 or reply_index < total - 1
            if has_more:
                await asyncio.sleep(_delay_seconds(memory_text))

    if sent_count:
        group.session.state.last_speak_time = datetime.now()
        group.session.schedule_save()
    return sent_count


async def send_reaction(
    group: GroupState, bot: Bot, react: Reaction, generation: int
) -> None:
    """贴表情不等发言冷却；同一个群两次之间至少隔 REACT_COOLDOWN_SECONDS。"""

    now = datetime.now()
    if (now - group.last_react_time).total_seconds() < REACT_COOLDOWN_SECONDS:
        outcome = "cooldown"
    elif group.session.stale(generation, "react"):
        return
    else:
        group.last_react_time = now
        try:
            await bot.call_api(
                "set_msg_emoji_like",
                message_id=int(react.target_id),
                emoji_id=react.emoji_id,
                set=True,
            )
            outcome = "ok"
        except ActionFailed as e:
            logger.warning(f"贴表情失败: {e}")
            outcome = "failed"
    log_event(
        "react",
        session_id=group.session.id,
        target_id=react.target_id,
        emoji=react.name,
        emoji_id=react.emoji_id,
        outcome=outcome,
    )


async def send_poke(
    group: GroupState, bot: Bot, event: Event, target: Poke, generation: int
) -> None:
    """戳一戳不等发言冷却；同一个群两次之间至少隔 POKE_COOLDOWN_SECONDS。

    自己戳的那条记录不在这里写：SnowLuma 会上报 bot 自己的戳一戳通知，由 handlers 按通知原文写进短期记忆。
    """

    now = datetime.now()
    if target.user_id == str(bot.self_id):
        return
    if (now - group.last_poke_time).total_seconds() < POKE_COOLDOWN_SECONDS:
        outcome = "cooldown"
    elif group.session.stale(generation, "poke"):
        return
    else:
        group.last_poke_time = now
        try:
            await bot.call_api(
                "group_poke", group_id=event.group_id, user_id=int(target.user_id)
            )
            outcome = "ok"
        except ActionFailed as e:
            logger.warning(f"戳一戳失败: {e}")
            outcome = "failed"
    log_event(
        "poke",
        session_id=group.session.id,
        user_id=target.user_id,
        outcome=outcome,
    )


