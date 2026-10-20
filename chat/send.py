"""回复切分与发送、表情包、贴表情、戳一戳，以及各自的冷却。"""

import asyncio
import random
import re
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from ..messages.faces import Reaction
from ..messages.stickers import Sticker, mark_sent, pick
from ..metrics import log_event
from ..prompts import STICKER_TAGS
from .engagement import SPEAK_COOLDOWN_SECONDS
from .groups import SELF_SENT_MSG_IDS, GroupState
from .orchestrator import Poke

MAX_REPLY_MESSAGES = 2
REACT_COOLDOWN_SECONDS = 60
# 戳一戳会给对方弹提醒，比贴表情打扰人，间隔放长些
POKE_COOLDOWN_SECONDS = 120
STICKER_COOLDOWN_SECONDS = 120

_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？!?~\n])\s*|(?<!\.)\.(?!\.)(?=\s|$|[\u4e00-\u9fff])\s*"
)
_SINGLE_TRAILING_PERIOD = re.compile(r"(?<!\.)\.$")


def _normalize_send_part(text: str) -> str:
    part = text.strip()
    if not part:
        return ""
    part = part.rstrip("。")
    part = _SINGLE_TRAILING_PERIOD.sub("", part)
    return part.strip()


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SPLIT_PATTERN.split(text) if part.strip()]
    return parts or [text]


def build_send_parts(text: str, max_messages: int = MAX_REPLY_MESSAGES) -> list[str]:
    parts = [_normalize_send_part(part) for part in _split_text(text)]
    parts = [part for part in parts if part]
    if len(parts) > max_messages:
        return [*parts[: max_messages - 1], " ".join(parts[max_messages - 1 :])]
    return parts


def _response_content(response) -> tuple[str, object | None, tuple[str, str] | None]:
    """(文字, 引用的消息 id, 表情包请求 (tag, want))；词表外的标签当没带。"""

    if isinstance(response, str):
        return response, None, None
    if not isinstance(response, dict):
        return "", None, None
    sticker = response.get("sticker")
    request = None
    if isinstance(sticker, dict):
        tag = str(sticker.get("tag") or "").strip()
        if tag in STICKER_TAGS:
            request = (tag, str(sticker.get("want") or "").strip())
        else:
            logger.debug(f"表情包标签不在词表里，丢弃: {tag}")
    return (
        str(response.get("content") or ""),
        response.get("target_id") or response.get("reply_to"),
        request,
    )


def _delay_seconds(part: str) -> float:
    return min(1.0 + len(part) * 0.1, 5.0)


async def send_one(
    *,
    state: GroupState,
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
        log_event("reply", session_id=state.session.id, content=memory_text)

        # 同步写入短时记忆，中间没有 await，不需要再拿 session_lock
        if not state.session.stale(generation, "append_self_message"):
            state.session.append_self_message(
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


@dataclass(frozen=True)
class _PickedSticker:
    sticker: Sticker
    tag: str
    want: str
    score: float | None


async def _pick_sticker(
    state: GroupState, tag: str, want: str
) -> tuple[Message, str, _PickedSticker] | None:
    """挑一张要发的表情包：(消息, 写进短期记忆的文本, 发送后记日志用的信息)。

    冷却中或标签下没图返回 None。
    """

    since_last = (datetime.now() - state.last_sticker_time).total_seconds()
    if since_last < STICKER_COOLDOWN_SECONDS:
        outcome = "cooldown"
    else:
        picked = await pick(tag, want)
        if picked is not None:
            sticker, score = picked
            # sub_type=1 让 QQ 按表情包显示；SnowLuma 容器挂载了同一路径，直接给本地文件
            segment = MessageSegment(
                "image", {"file": sticker.path.as_uri(), "sub_type": 1}
            )
            return (
                Message(segment),
                f"[表情包: {sticker.caption}]",
                _PickedSticker(sticker, tag, want, score),
            )
        outcome = "empty"
    log_event(
        "sticker", session_id=state.session.id, tag=tag, want=want, outcome=outcome
    )
    return None


async def dispatch_replies(
    *,
    state: GroupState,
    responses: list,
    bot: Bot,
    event: Event,
    generation: int,
) -> int:
    if not responses:
        return 0
    # 发言冷却在发送前等待，而不是在轮次开头跳过：对方秒回时不会因为冷却把对话掐断
    since_last = (datetime.now() - state.session.state.last_speak_time).total_seconds()
    if since_last < SPEAK_COOLDOWN_SECONDS:
        await asyncio.sleep(SPEAK_COOLDOWN_SECONDS - since_last)
    if state.session.stale(generation, "pre_send"):
        return 0

    total = len(responses)
    sent_count = 0
    text_count = 0
    # 一轮最多一张表情包，不占文字的 MAX_REPLY_MESSAGES 名额
    sticker_requested = False
    for response_index, response in enumerate(responses):
        raw_content, reply_id, sticker_request = _response_content(response)
        # (消息, 写进短期记忆的文本, 这条是不是表情包)；模型写的文字一律按纯文本发，不解析 CQ 码
        outgoing: list[tuple[Message, str, _PickedSticker | None]] = []
        if text_count < MAX_REPLY_MESSAGES:
            parts = build_send_parts(raw_content, MAX_REPLY_MESSAGES - text_count)
            outgoing = [
                (Message(MessageSegment.text(part)), part, None) for part in parts
            ]
        if sticker_request is not None and not sticker_requested:
            sticker_requested = True
            picked = await _pick_sticker(state, *sticker_request)
            if picked is not None:
                outgoing.append(picked)

        for part_index, (message, memory_text, picked_sticker) in enumerate(outgoing):
            if state.session.stale(generation, "send_loop"):
                break
            if reply_id and response_index == 0 and part_index == 0:
                try:
                    message.insert(0, MessageSegment.reply(int(reply_id)))
                except ValueError:
                    logger.warning(f"引用ID无效: {reply_id}")

            sent = await send_one(
                state=state,
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
                    state.last_sticker_time = datetime.now()
                    await mark_sent(picked_sticker.sticker)
            if picked_sticker is not None:
                score = picked_sticker.score
                log_event(
                    "sticker",
                    session_id=state.session.id,
                    tag=picked_sticker.tag,
                    want=picked_sticker.want,
                    digest=picked_sticker.sticker.digest[:8],
                    score=None if score is None else round(score, 3),
                    outcome="ok" if sent else "failed",
                )

            has_more = part_index < len(outgoing) - 1 or response_index < total - 1
            if has_more:
                await asyncio.sleep(_delay_seconds(memory_text))

    if sent_count:
        state.session.state.last_speak_time = datetime.now()
        state.session.schedule_save()
    return sent_count


async def send_reaction(
    state: GroupState, bot: Bot, react: Reaction, generation: int
) -> None:
    """贴表情不等发言冷却；同一个群两次之间至少隔 REACT_COOLDOWN_SECONDS。"""

    now = datetime.now()
    if (now - state.last_react_time).total_seconds() < REACT_COOLDOWN_SECONDS:
        outcome = "cooldown"
    elif state.session.stale(generation, "react"):
        return
    else:
        state.last_react_time = now
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
        session_id=state.session.id,
        target_id=react.target_id,
        emoji=react.name,
        emoji_id=react.emoji_id,
        outcome=outcome,
    )


async def send_poke(
    state: GroupState, bot: Bot, event: Event, target: Poke, generation: int
) -> None:
    """戳一戳不等发言冷却；同一个群两次之间至少隔 POKE_COOLDOWN_SECONDS。

    自己戳的那条记录不在这里写：SnowLuma 会上报 bot 自己的戳一戳通知，由 handlers 按通知原文写进短期记忆。
    """

    now = datetime.now()
    if target.user_id == str(bot.self_id):
        return
    if (now - state.last_poke_time).total_seconds() < POKE_COOLDOWN_SECONDS:
        outcome = "cooldown"
    elif state.session.stale(generation, "poke"):
        return
    else:
        state.last_poke_time = now
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
        session_id=state.session.id,
        user_id=target.user_id,
        outcome=outcome,
    )


