import asyncio
import hashlib
import random
import re
import traceback
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from ..memory.image import fetch_image_input
from ..memory.short_term import Message as MMessage
from .engagement import SPEAK_COOLDOWN_SECONDS
from .faces import Reaction, ensure_faces, face_name, sendable_face
from .llm import VisionInput, build_turn_calls
from .metrics import log_event
from .orchestrator import ConversationOrchestrator, Poke
from .state_manager import SELF_SENT_MSG_IDS, GroupState, is_shutting_down

DEBOUNCE_SECONDS = 2.0
MAX_REPLY_MESSAGES = 2
REACT_COOLDOWN_SECONDS = 60
# 戳一戳会给对方弹提醒，比贴表情打扰人，间隔放长些
POKE_COOLDOWN_SECONDS = 120

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


def _response_content(response) -> tuple[str, object | None, str]:
    if isinstance(response, str):
        return response, None, ""
    if isinstance(response, dict):
        return (
            str(response.get("content") or ""),
            response.get("target_id") or response.get("reply_to"),
            str(response.get("face") or ""),
        )
    return "", None, ""


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
    for response_index, response in enumerate(responses):
        if sent_count >= MAX_REPLY_MESSAGES:
            break
        raw_content, reply_id, face_text = _response_content(response)
        face = sendable_face(face_text) if face_text else None
        if face_text and face is None:
            logger.debug(f"表情不在目录里，丢弃: {face_text}")
        # (消息, 写进短期记忆的文本, 这条带的表情)；模型写的文字一律按纯文本发，不解析 CQ 码
        outgoing = [
            (Message(MessageSegment.text(part)), part, None)
            for part in build_send_parts(raw_content, MAX_REPLY_MESSAGES - sent_count)
        ]
        if face is not None:
            face_seg = MessageSegment.face(int(face.id))
            face_mark = f"[表情:{face.name}]"
            # 大动画表情和文字混发的显示效果没验证过，单独发一条
            if outgoing and not face.big:
                message, text, _ = outgoing[-1]
                outgoing[-1] = (message + face_seg, text + face_mark, face)
            else:
                outgoing.append((Message(face_seg), face_mark, face))

        for part_index, (message, memory_text, sent_face) in enumerate(outgoing):
            if sent_count >= MAX_REPLY_MESSAGES:
                break
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
                if sent_face is not None:
                    log_event(
                        "face_sent",
                        session_id=state.session.id,
                        name=sent_face.name,
                        big=sent_face.big,
                    )

            has_more = part_index < len(outgoing) - 1 or response_index < total - 1
            if has_more:
                await asyncio.sleep(_delay_seconds(memory_text))

    if sent_count:
        state.session.state.last_speak_time = datetime.now()
        state.session.schedule_save()
    return sent_count


def _is_sticker_segment_data(data: dict) -> bool:
    return str(data.get("sub_type", "")) == "1"


def _face_text(seg_type: str, data: dict) -> str | None:
    """QQ 自带表情转成 [表情:名字]；不是表情返回 None。

    商城表情是带 emoji_id 的 image 段，summary 就是名字（如 [狗头]），不下载、不给视觉模型。
    """

    if seg_type == "face":
        name = face_name(str(data.get("id", "")))
        return f"[表情:{name}]" if name else "[表情]"
    if seg_type == "dice":
        return "[表情:骰子]"
    if seg_type == "rps":
        return "[表情:包剪锤]"
    if seg_type in ("image", "mface") and data.get("emoji_id"):
        summary = str(data.get("summary") or "").strip("[] ")
        if summary:
            return f"[表情:{summary}]"
    return None


def _build_image_ref(
    message_scope: str,
    source: str,
    segment_index: int,
    identifier: str,
) -> str:
    digest = hashlib.sha1(str(identifier or "").encode("utf-8", "ignore")).hexdigest()[
        :12
    ]
    scope_digest = hashlib.sha1(
        str(message_scope or "").encode("utf-8", "ignore")
    ).hexdigest()[:10]
    return f"{scope_digest}:{source}:{segment_index}:{digest}"


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


async def message2BotMessage(
    bot_name: str,
    group_id: int,
    message: Message,
    bot: Bot,
    *,
    message_scope: str = "",
) -> tuple[str, list[VisionInput], dict[str, str]]:
    """把 OneBot 消息转成可读文本，并收集原生图片输入与被 @/被回复的人（QQ 号 -> 群名片）。"""

    mentions: dict[str, str] = {}
    await ensure_faces(bot)

    async def process_segment(
        seg: MessageSegment,
        segment_index: int,
    ) -> tuple[str, list[VisionInput]]:
        if seg.type == "text":
            return (f"{seg.data.get('text', '')}", [])

        face_text = _face_text(seg.type, seg.data)
        if face_text is not None:
            return (face_text, [])

        if seg.type == "image":
            url = seg.data.get("url", "")
            file_unique = seg.data.get("file_unique", "")
            text, vision_input = await fetch_image_input(
                url,
                file_unique,
                is_sticker=_is_sticker_segment_data(seg.data),
                ref_id=_build_image_ref(
                    message_scope, "primary", segment_index, file_unique or url
                ),
                source="primary",
            )
            return (text, [vision_input] if vision_input else [])

        if seg.type == "at":
            target = seg.data.get("qq")
            if not target:
                return ("", [])
            if target == str(bot.self_id):
                return (f" @{bot_name} ", [])
            try:
                user_info = await bot.get_group_member_info(
                    group_id=group_id, user_id=int(target)
                )
                nickname = (
                    user_info.get("card") or user_info.get("nickname") or str(target)
                )
                mentions[str(target)] = nickname
                return (f" @{nickname} ", [])
            except Exception:
                return (f" @{target} ", [])

        if seg.type == "reply":
            reply_id = seg.data.get("id")
            if not reply_id:
                return ("", [])
            try:
                source_msg = await bot.get_msg(message_id=int(reply_id))
                # 优先群名片，与 @ 和发言人名字保持同一套称呼，否则模型会把昵称当成另一个人
                sender_info = source_msg.get("sender", {})
                sender = sender_info.get("card") or sender_info.get("nickname", "未知")
                if sender_info.get("user_id"):
                    mentions[str(sender_info["user_id"])] = sender
                content_data = source_msg.get("message", [])
                source_text = ""
                image_inputs: list[VisionInput] = []

                if isinstance(content_data, str):
                    source_text = content_data
                elif isinstance(content_data, list):
                    for reply_index, segment in enumerate(content_data):
                        msg_type = segment.get("type")
                        data = segment.get("data", {})
                        face_text = _face_text(msg_type, data)
                        if face_text is not None:
                            source_text += face_text
                        elif msg_type == "text":
                            source_text += data.get("text", "")
                        elif msg_type == "image":
                            img_url = data.get("url", "")
                            img_file_unique = data.get("file_unique", "")
                            img_text, vision_input = await fetch_image_input(
                                img_url,
                                img_file_unique,
                                is_sticker=_is_sticker_segment_data(data),
                                ref_id=_build_image_ref(
                                    message_scope,
                                    "referenced",
                                    reply_index,
                                    img_file_unique or img_url,
                                ),
                                source="referenced",
                            )
                            source_text += img_text
                            if vision_input:
                                image_inputs.append(vision_input)

                if len(source_text) > 800:
                    source_text = source_text[:800] + "..."
                return (f' [回复 {sender}: "{source_text}"] ', image_inputs)
            except Exception as e:
                logger.warning(f"获取回复内容失败: {e}")
                return (" [回复] ", [])

        return ("", [])

    tasks = [process_segment(seg, index) for index, seg in enumerate(message)]
    results = await asyncio.gather(*tasks)

    content = "".join(result[0] for result in results).strip()
    image_inputs = [item for result in results for item in result[1]]
    return (content, image_inputs, mentions)


async def _send_reaction(
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


async def _send_poke(
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
    if is_shutting_down():
        return

    async with state.session_lock:
        await state.session.load_session()
        generation = state.session.state.generation
        session_id = str(state.session.id)

    images = [item for message in current_chunk for item in message.image_inputs]
    chat_call, feedback_call = build_turn_calls(session_id, images)
    try:
        turn = await ConversationOrchestrator(state.session).process_chunk(
            current_chunk, chat_call, feedback_call, generation
        )
    finally:
        for message in current_chunk:
            message.image_inputs.clear()
    if turn is None:
        return

    if turn.react is not None:
        await _send_reaction(state, batch.bot, turn.react, generation)
    if turn.poke is not None:
        await _send_poke(state, batch.bot, batch.event, turn.poke, generation)
    await dispatch_replies(
        state=state,
        responses=turn.replies,
        bot=batch.bot,
        event=batch.event,
        generation=generation,
    )


async def spawn_state(state: GroupState):
    """Small worker boundary: debounce, invoke one turn, contain failures."""

    logger.info(f"GroupState 后台任务启动: {id(state)}")
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
            logger.error(f"Spawn loop error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5.0)
