"""回复发送、表情包、贴表情、戳一戳。"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from ..messages import photos
from ..messages.faces import Reaction
from ..messages.stickers import Sticker, mark_sent, pick
from ..metrics import error_text, log_event
from ..persona.status import describe_now
from ..prompts import PHOTO_BLOCKED_NOTE, PHOTO_FAILED_NOTE, PRESETS
from ..storage.db import get_photo
from .engagement import SPEAK_COOLDOWN_SECONDS
from .outputs import PhotoWish, Poke, Reply
from .session import Session

# 只管拍不出来的时候：上游挂了时每轮都卡一次超时，等一会儿再试。发得勤不勤她自己看 photos_last_hour
PHOTO_FAILED_COOLDOWN_SECONDS = 600


@dataclass
class _ReadyPhoto:
    """这一项要发的照片：新拍的还没入库，photo_id 为 None。"""

    path: Path
    caption: str
    new: photos.NewPhoto | None = None
    photo_id: int | None = None


def _delay_seconds(part: str) -> float:
    return min(1.0 + len(part) * 0.1, 5.0)


async def send_one(
    *,
    session: Session,
    bot: Bot,
    event: Event,
    message: Message,
    memory_text: str,
    generation: int,
) -> bool:
    try:
        async with session.send_lock:
            result = await bot.send(message=message, event=event)
            message_id = ""
            if isinstance(result, dict) and "message_id" in result:
                message_id = str(result["message_id"])
            log_event("reply", session_id=session.id, content=memory_text)

            # 同步写入短时记忆，中间没有 await，不需要再拿 admin_lock
            if not session.stale(generation, "append_self_message"):
                session.append_self_message(
                    memory_text,
                    message_id,
                    str(bot.self_id),
                )
        return True
    except ActionFailed as e:
        logger.error(f"群 {session.id} 发送失败: {error_text(e)}")
    except Exception as e:
        logger.error(f"群 {session.id} 发送未知错误: {error_text(e)}")
    return False


async def _pick_sticker(
    session: Session, tag: str, want: str
) -> tuple[Sticker, float | None] | None:
    """挑一张要发的表情包和它的分数；标签下没图返回 None。"""

    picked = await pick(tag, want)
    if picked is None:
        log_event(
            "sticker", session_id=session.id, tag=tag, want=want, outcome="empty"
        )
    return picked


def _photo_cooldown(session: Session) -> None:
    session.photo_ready_at = datetime.now() + timedelta(seconds=PHOTO_FAILED_COOLDOWN_SECONDS)


async def _ready_photo(
    session: Session, wish: PhotoWish, generation: int
) -> _ReadyPhoto | None:
    """拿到这一项要发的照片：翻相册或现拍。拿不到返回 None，这一项的文字也不发。"""

    fields = {"session_id": session.id, "kind": wish.kind, "photo_id": wish.photo_id}
    if datetime.now() < session.photo_ready_at:
        log_event("photo", **fields, outcome="cooldown")
        return None

    if wish.photo_id is not None:
        row = await get_photo(session.state.preset, wish.photo_id)
        if row is None or not photos.photo_path(row.digest).is_file():
            log_event("photo", **fields, outcome="failed", error="missing")
            return None
        return _ReadyPhoto(photos.photo_path(row.digest), row.caption, photo_id=row.id)

    preset = session.state.preset
    started_at = time.perf_counter()
    outcome, new, error = await photos.take(
        session.id,
        PRESETS[preset],
        wish.kind,
        wish.describe,
        describe_now(preset, datetime.max),
    )
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    if outcome == "ok" and session.stale(generation, "photo"):
        photos.discard(new)
        outcome = "stale"
    log_event(
        "photo",
        "WARNING" if outcome in ("blocked", "failed", "no_portrait") else "INFO",
        **fields,
        describe=wish.describe,
        latency_ms=latency_ms,
        outcome=outcome,
        error=error or None,
    )
    if outcome == "blocked":
        session.note_photo_problem(PHOTO_BLOCKED_NOTE)
    elif outcome == "failed":
        _photo_cooldown(session)
        session.note_photo_problem(PHOTO_FAILED_NOTE)
    elif outcome == "no_portrait":
        session.note_photo_problem(PHOTO_FAILED_NOTE)
    if outcome != "ok":
        return None
    return _ReadyPhoto(photos.photo_path(new.digest), new.caption, new=new)


async def _after_photo_send(session: Session, photo: _ReadyPhoto, sent: bool) -> None:
    if sent:
        session.photo_sent_times.append(datetime.now())
        if photo.new is not None:
            await photos.save(session.id, session.state.preset, photo.new)
        else:
            await photos.mark_sent(photo.photo_id)
        return
    if photo.new is not None:
        photos.discard(photo.new)
    _photo_cooldown(session)
    session.note_photo_problem(PHOTO_FAILED_NOTE)


async def dispatch_replies(
    *,
    session: Session,
    replies: list[Reply],
    bot: Bot,
    event: Event,
    generation: int,
) -> int:
    if not replies:
        return 0
    # 发言冷却在发送前等：对方秒回时不会因为冷却把对话掐断
    since_last = (datetime.now() - session.state.last_speak_time).total_seconds()
    if since_last < SPEAK_COOLDOWN_SECONDS:
        await asyncio.sleep(SPEAK_COOLDOWN_SECONDS - since_last)
    if session.stale(generation, "pre_send"):
        return 0

    total = len(replies)
    sent_count = 0
    # 一轮最多一张表情包
    sticker_requested = False
    # 一轮最多一张：新拍的每张都要等半分钟，这期间 worker 不看新消息
    photo_requested = False
    for reply_index, reply in enumerate(replies):
        # 先拿图再发字：拿不到图时这一项的文字也不发，不留下「给你看」却没有图
        photo = None
        if reply.photo is not None:
            if photo_requested:
                continue
            photo_requested = True
            photo = await _ready_photo(session, reply.photo, generation)
            if photo is None:
                continue
        # (消息, 写进短期记忆的文本, 表情包和挑图分数, 照片)；reply 的一项就是一条消息，
        # 文字原样按纯文本发，不切分、不解析 CQ 码
        outgoing: list[
            tuple[Message, str, tuple[Sticker, float | None] | None, _ReadyPhoto | None]
        ] = []
        if reply.text:
            outgoing.append((Message(MessageSegment.text(reply.text)), reply.text, None, None))
        if photo is not None:
            # 普通图片，不带 sub_type=1；路径在 SnowLuma 容器挂载的目录里
            outgoing.append(
                (
                    Message(MessageSegment("image", {"file": photo.path.as_uri()})),
                    photos.memory_text(photo.caption),
                    None,
                    photo,
                )
            )
        if reply.sticker is not None and not sticker_requested:
            sticker_requested = True
            picked = await _pick_sticker(session, *reply.sticker)
            if picked is not None:
                sticker = picked[0]
                # sub_type=1 让 QQ 按表情包显示；SnowLuma 容器挂载了同一路径，直接给本地文件
                segment = MessageSegment(
                    "image", {"file": sticker.path.as_uri(), "sub_type": 1}
                )
                outgoing.append((Message(segment), f"[表情包: {sticker.caption}]", picked, None))

        for part_index, (message, memory_text, picked_sticker, part_photo) in enumerate(outgoing):
            if session.stale(generation, "send_loop"):
                if photo is not None and photo.new is not None:
                    photos.discard(photo.new)
                break
            if reply.target_id is not None and part_index == 0:
                message.insert(0, MessageSegment.reply(reply.target_id))

            sent = await send_one(
                session=session,
                bot=bot,
                event=event,
                message=message,
                memory_text=memory_text,
                generation=generation,
            )
            if part_photo is not None:
                await _after_photo_send(session, part_photo, sent)
            if sent:
                sent_count += 1
                if picked_sticker is not None:
                    await mark_sent(picked_sticker[0])
            if picked_sticker is not None:
                sticker, score = picked_sticker
                tag, want = reply.sticker
                log_event(
                    "sticker",
                    session_id=session.id,
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
        session.state.last_speak_time = datetime.now()
        session.schedule_save()
    return sent_count


async def send_reaction(
    session: Session, bot: Bot, react: Reaction, generation: int
) -> None:
    """贴表情不等发言冷却。"""

    if session.stale(generation, "react"):
        return
    try:
        await bot.call_api(
            "set_msg_emoji_like",
            message_id=int(react.target_id),
            emoji_id=react.emoji_id,
            set=True,
        )
        outcome = "ok"
    except ActionFailed as e:
        logger.warning(f"群 {session.id} 贴表情失败: {error_text(e)}")
        outcome = "failed"
    log_event(
        "react",
        session_id=session.id,
        target_id=react.target_id,
        emoji=react.name,
        emoji_id=react.emoji_id,
        outcome=outcome,
    )


async def send_poke(
    session: Session, bot: Bot, event: Event, target: Poke, generation: int
) -> None:
    """戳一戳不等发言冷却。

    自己戳的那条记录不在这里写：SnowLuma 会上报 bot 自己的戳一戳通知，由 handlers 按通知原文写进短期记忆。
    """

    if target.user_id == str(bot.self_id) or session.stale(generation, "poke"):
        return
    try:
        await bot.call_api(
            "group_poke", group_id=event.group_id, user_id=int(target.user_id)
        )
        outcome = "ok"
    except ActionFailed as e:
        logger.warning(f"群 {session.id} 戳一戳失败: {error_text(e)}")
        outcome = "failed"
    log_event(
        "poke",
        session_id=session.id,
        user_id=target.user_id,
        outcome=outcome,
    )


