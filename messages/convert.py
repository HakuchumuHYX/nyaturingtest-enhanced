"""把 OneBot 事件读成 ChatMessage：各类消息段的文字标记、被回复的原文、图片占位与后台描述，
以及戳一戳、贴表情、撤回标记的写法。"""

import asyncio
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime

from nonebot import logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    Message,
    MessageSegment,
    PokeNotifyEvent,
)
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.exception import NetworkError

from ..chat.session import Session
from ..llm import VisionInput
from ..metrics import log_event
from ..prompts import truncate_text
from .faces import ensure_faces, face_name
from .image import cached_caption, caption_image, caption_sticker, fetch_image_input
from .message import ChatMessage
from .stickers import collect

CARD_CHARS = 120
FORWARD_PLACEHOLDER = "[合并转发]"
# 群友自己说的话至少这么多字才算说了东西：「？？？」「哈哈哈」「233」这类都短于 4 字
SUBSTANTIVE_MIN_CHARS = 4
_PUNCTUATION_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
# 这几类消息段的文字算群友自己说的话；图片另算（不含表情包）
_OWN_WORD_SEGMENTS = {"text", "json", "file", "forward"}


def is_sticker_segment(data: dict) -> bool:
    return str(data.get("sub_type", "")) == "1"


async def member_name(bot: Bot, group_id: int, user_id: str) -> str | None:
    """群名片，没有就用 QQ 昵称；查不到（退群、接口失败、@全体成员的 all）返回 None。"""

    try:
        info = await bot.get_group_member_info(group_id=group_id, user_id=int(user_id))
    except (ActionFailed, NetworkError, ValueError) as e:
        logger.debug(f"查群 {group_id} 成员 {user_id} 的名片失败: {e}")
        return None
    return info.get("card") or info.get("nickname") or user_id


def _card_text(raw) -> str:
    """QQ 卡片（小程序、分享链接）：prompt 是「[QQ小程序]标题」这类摘要，meta 里补上 prompt 没写到的标题和简介。"""

    try:
        card = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        logger.warning(f"卡片 JSON 解析失败: {e}")
        return "[卡片]"
    if not isinstance(card, dict):
        return "[卡片]"
    prompt = str(card.get("prompt") or "").strip()
    meta = card.get("meta")
    extras: list[str] = []
    for item in meta.values() if isinstance(meta, dict) else []:
        if not isinstance(item, dict):
            continue
        for key in ("title", "desc"):
            value = str(item.get(key) or "").strip()
            if value and value not in prompt and value not in extras:
                extras.append(value)
    text = " ".join([prompt, *extras]).strip()
    return f"[卡片: {truncate_text(text, CARD_CHARS)}]" if text else "[卡片]"


def segment_text(seg_type: str, data: dict) -> str | None:
    """不用调接口就能转成文字的消息段；图片、@、回复交给调用方，返回 None。

    商城表情是带 emoji_id 的 image 段，summary 就是名字（如 [狗头]），不下载、不给视觉模型。
    合并转发在这里只是占位，正文里的由 messages/forward.py 另行读完再改写。
    """

    if seg_type == "text":
        return str(data.get("text", ""))
    if seg_type == "json":
        return _card_text(data.get("data"))
    if seg_type == "record":
        return "[语音]"
    if seg_type == "video":
        return "[视频]"
    if seg_type == "file":
        return f"[文件: {data.get('name') or data.get('file') or '未知'}]"
    if seg_type == "forward":
        return FORWARD_PLACEHOLDER
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


@dataclass(frozen=True)
class CaptionJob:
    """描述缓存没命中的图：消息里先写占位 token，后台看完再换成描述。"""

    token: str
    vision_input: VisionInput
    # 只有表情包带：看图要拼九宫格，收进库也要原始字节
    image_bytes: bytes | None


def _image_kind(vision_input: VisionInput) -> str:
    return "表情包" if vision_input.is_sticker else "图片"


async def fill_caption(
    session: Session, message: ChatMessage, job: CaptionJob, generation: int
) -> None:
    """后台看完一张图，把消息里的占位换成描述；表情包顺带判断收不收。

    会话作废、或占位已经不在（被撤回）就只留缓存；没看成就换回 [图片] / [表情包]。
    收图另起任务：worker 处理这批消息前会等这个任务，入库不该拖住它。
    """

    if job.image_bytes is None:
        caption = await caption_image(job.vision_input, session.id)
    else:
        result = await caption_sticker(
            job.image_bytes, job.vision_input.ref_id, session.id
        )
        caption = result.caption if result else ""
        if result is not None and result.keep:
            session.spawn(collect(job.image_bytes, result, f"group:{session.id}"))
    if session.stale(generation, "image_caption") or job.token not in message.content:
        return
    kind = _image_kind(job.vision_input)
    message.content = message.content.replace(
        job.token, f"[{kind}: {caption}]" if caption else f"[{kind}]"
    )
    session.short_term_memory.mark_dirty(message)
    session.schedule_save()


@dataclass
class _Part:
    """一个消息段读出来的东西：文字、原图、被 @/被回复的人、待看的图。

    own 是这段里算作群友自己说的话：只有文字、卡片、文件、转发和图片（不含表情包）算；
    回复引用、表情、@、语音、视频不算。
    """

    text: str
    images: list[VisionInput] = field(default_factory=list)
    mentions: dict[str, str] = field(default_factory=dict)
    jobs: list[CaptionJob] = field(default_factory=list)
    own: str = ""


async def _image_part(data: dict, label: str, group_id: int, caption: bool) -> _Part:
    text, vision_input, image_bytes = await fetch_image_input(
        data.get("url", ""),
        data.get("file_unique", ""),
        is_sticker=is_sticker_segment(data),
        ref_id=label,
    )
    if vision_input is None:
        return _Part(text)
    if not caption:
        return _Part(text, [vision_input])
    kind = _image_kind(vision_input)
    cached = await cached_caption(vision_input.digest)
    if cached:
        log_event("image_caption", "DEBUG", session_id=str(group_id), outcome="cache_hit")
        return _Part(f"[{kind}: {cached}]", [vision_input])
    token = f"[{kind}#{secrets.token_hex(2)}]"
    sticker_bytes = image_bytes if vision_input.is_sticker else None
    return _Part(token, [vision_input], jobs=[CaptionJob(token, vision_input, sticker_bytes)])


async def _at_part(bot: Bot, data: dict, group_id: int, bot_name: str) -> _Part:
    target = data.get("qq")
    if not target:
        return _Part("")
    if target == str(bot.self_id):
        return _Part(f" @{bot_name} ")
    name = await member_name(bot, group_id, str(target))
    if name is None:
        return _Part(f" @{target} ")
    return _Part(f" @{name} ", mentions={str(target): name})


async def _reply_part(
    bot: Bot, data: dict, message_scope: str, group_id: int, caption: bool
) -> _Part:
    """被回复的原文，引用的图和正文的图一样放进附图。"""

    reply_id = data.get("id")
    if not reply_id:
        return _Part("")
    try:
        source_msg = await bot.get_msg(message_id=int(reply_id))
    except (ActionFailed, NetworkError, ValueError) as e:
        logger.warning(f"获取回复内容失败: {e}")
        return _Part(" [回复] ")

    # 优先群名片，与 @ 和发言人名字保持同一套称呼，否则模型会把昵称当成另一个人
    sender_info = source_msg.get("sender", {})
    sender = sender_info.get("card") or sender_info.get("nickname", "未知")
    part = _Part("")
    if sender_info.get("user_id"):
        part.mentions[str(sender_info["user_id"])] = sender
    content_data = source_msg.get("message", [])
    source_text = ""

    if isinstance(content_data, str):
        source_text = content_data
    elif isinstance(content_data, list):
        for segment in content_data:
            msg_type = segment.get("type")
            seg_data = segment.get("data", {})
            text = segment_text(msg_type, seg_data)
            if text is not None:
                source_text += text
            elif msg_type == "image":
                # 引用的图和原消息是同一份字节，原消息看过就命中描述缓存
                image = await _image_part(
                    seg_data,
                    f"消息 {message_scope} 引用的第 {len(part.images) + 1} 张图",
                    group_id,
                    caption,
                )
                source_text += image.text
                part.images.extend(image.images)
                part.jobs.extend(image.jobs)

    if len(source_text) > 800:
        source_text = source_text[:800] + "..."
    part.text = f' [回复 {sender}: "{source_text}"] '
    return part


async def _segment_part(
    seg: MessageSegment,
    image_number: int,
    *,
    bot: Bot,
    group_id: int,
    bot_name: str,
    message_scope: str,
    caption: bool,
) -> _Part:
    text = segment_text(seg.type, seg.data)
    if text is not None:
        return _Part(text, own=text if seg.type in _OWN_WORD_SEGMENTS else "")
    if seg.type == "image":
        label = f"消息 {message_scope} 的第 {image_number} 张图"
        part = await _image_part(seg.data, label, group_id, caption)
        if not is_sticker_segment(seg.data):
            part.own = part.text
        return part
    if seg.type == "at":
        return await _at_part(bot, seg.data, group_id, bot_name)
    if seg.type == "reply":
        return await _reply_part(bot, seg.data, message_scope, group_id, caption)
    logger.debug(f"未处理的消息段: {seg.type}")
    return _Part("")


def forward_id(message: Message) -> str:
    """合并转发总是单独一条消息；返回转发的 id，不是转发返回空串。"""

    return next(
        (str(seg.data.get("id") or "") for seg in message if seg.type == "forward"),
        "",
    )


def _addresses_bot(bot: Bot, event: GroupMessageEvent) -> bool:
    """@Bot 或回复 Bot 的消息。

    不用 event.to_me：.env 里 NICKNAME=[""] 会让适配器把所有文本开头的消息都判成 to_me。
    """

    self_id = str(bot.self_id)
    if event.reply and str(event.reply.sender.user_id) == self_id:
        return True
    return any(
        seg.type == "at" and str(seg.data.get("qq", "")) == self_id
        for seg in event.original_message
    )


def _is_substantive(parts: list[_Part]) -> bool:
    own = "".join(part.own for part in parts).strip()
    return len(own) >= SUBSTANTIVE_MIN_CHARS and not _PUNCTUATION_ONLY_RE.fullmatch(own)


async def read_message(bot: Bot, event: GroupMessageEvent, session: Session) -> ChatMessage:
    """把一条群消息读成 ChatMessage：可读文本、原图、被 @/被回复的人（QQ 号 -> 群名片），并起后台看图任务。

    图片写在它原本的位置：描述缓存命中就是 [图片: 描述]，没命中先写占位、由 fill_caption 后台补上。
    没设人设的群不看图，只写 [图片]。
    """

    bot_name = session.state.name
    generation = session.state.generation
    message = event.original_message
    await ensure_faces(bot)
    # 附图标签按消息 id 和第几张写，模型靠它把原图和消息对上号
    image_numbers: dict[int, int] = {}
    for index, seg in enumerate(message):
        if seg.type == "image" and segment_text(seg.type, seg.data) is None:
            image_numbers[index] = len(image_numbers) + 1

    parts = await asyncio.gather(
        *(
            _segment_part(
                seg,
                image_numbers.get(index, 0),
                bot=bot,
                group_id=event.group_id,
                bot_name=bot_name,
                message_scope=str(event.message_id),
                caption=bool(bot_name),
            )
            for index, seg in enumerate(message)
        )
    )

    user_id, self_id = str(event.user_id), str(bot.self_id)
    if user_id == self_id:
        user_name = bot_name
    else:
        user_name = (
            str(event.sender.card or "").strip()
            or str(event.sender.nickname or "").strip()
            or user_id
        )
    chat_message = ChatMessage(
        time=datetime.now(),
        user_name=user_name,
        content="".join(part.text for part in parts).strip(),
        id=str(event.message_id),
        user_id=user_id,
        image_inputs=[image for part in parts for image in part.images],
        mentions={uid: name for part in parts for uid, name in part.mentions.items()},
        to_me=user_id != self_id and _addresses_bot(bot, event),
        forward=bool(forward_id(message)),
        substantive=_is_substantive(parts),
    )
    chat_message.caption_tasks = [
        session.spawn(fill_caption(session, chat_message, job, generation))
        for part in parts
        for job in part.jobs
    ]
    return chat_message


async def read_poke(bot: Bot, event: PokeNotifyEvent, bot_name: str) -> ChatMessage:
    """戳一戳通知读成一条消息；被戳的群友记进 mentions，Feedback 才能跟着戳他。"""

    self_id = str(bot.self_id)
    poker_id, target_id = str(event.user_id), str(event.target_id)
    poker = (
        bot_name
        if poker_id == self_id
        else await member_name(bot, event.group_id, poker_id) or poker_id
    )
    target = (
        bot_name
        if target_id == self_id
        else await member_name(bot, event.group_id, target_id) or target_id
    )
    # 动作文字各人自定义（戳了戳、拍了拍、捏了捏…），照通知里的 action / suffix 原样拼
    data = event.model_dump()
    return ChatMessage(
        time=datetime.now(),
        user_name=poker,
        content=f"[戳一戳: {data.get('action') or '戳了戳'} {target}{data.get('suffix') or ''}]",
        user_id=poker_id,
        mentions={} if target_id == self_id else {target_id: target},
        to_me=target_id == self_id,
        poke=True,
    )


async def read_reaction(
    bot: Bot, group_id: int, user_id: str, target: ChatMessage, emoji_id: str
) -> ChatMessage:
    """群友给 bot 的消息贴了表情，读成一条消息。"""

    await ensure_faces(bot)
    emoji = face_name(emoji_id)
    return ChatMessage(
        time=datetime.now(),
        user_name=await member_name(bot, group_id, user_id) or user_id,
        content=f"[给你的消息「{target.content[:20]}」贴了表情:{emoji or '未知'}]",
        user_id=user_id,
    )


def recall_text(by_author: bool) -> str:
    return "[撤回了一条消息]" if by_author else "[一条消息被管理员撤回了]"
