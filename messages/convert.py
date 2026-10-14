"""把 OneBot 消息读成文字：各类消息段的文字标记、被回复的原文、图片占位与后台描述。"""

import asyncio
import json
import secrets
from dataclasses import dataclass

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment

from ..chat.session import Session
from ..llm import VisionInput
from ..metrics import log_event
from ..prompts import truncate_text
from .faces import ensure_faces, face_name
from .image import cached_caption, caption_image, fetch_image_input
from .message import Message as MMessage

CARD_CHARS = 120
FORWARD_PLACEHOLDER = "[合并转发]"


def _is_sticker_segment_data(data: dict) -> bool:
    return str(data.get("sub_type", "")) == "1"


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


def _image_kind(vision_input: VisionInput) -> str:
    return "表情包" if vision_input.is_sticker else "图片"


async def fill_caption(
    session: Session, message: MMessage, job: CaptionJob, generation: int
) -> None:
    """后台看完一张图，把消息里的占位换成描述。

    会话作废、或占位已经不在（被撤回）就只留缓存；没看成就换回 [图片] / [表情包]。
    """

    caption = await caption_image(job.vision_input, session.id)
    if session.stale(generation, "image_caption") or job.token not in message.content:
        return
    kind = _image_kind(job.vision_input)
    message.content = message.content.replace(
        job.token, f"[{kind}: {caption}]" if caption else f"[{kind}]"
    )
    session.runtime.short_term_memory.mark_dirty(message)
    session.schedule_save()


async def message2BotMessage(
    bot_name: str,
    group_id: int,
    message: Message,
    bot: Bot,
    *,
    message_scope: str,
    caption: bool,
) -> tuple[str, list[VisionInput], dict[str, str], list[CaptionJob]]:
    """把 OneBot 消息转成可读文本，并收集原生图片输入、被 @/被回复的人（QQ 号 -> 群名片）和待看的图。

    图片写在它原本的位置：描述缓存命中就是 [图片: 描述]，没命中先写占位、由 fill_caption 后台补上。
    caption 为假（没设人设的群）时不看图，只写 [图片]。
    """

    mentions: dict[str, str] = {}
    caption_jobs: list[CaptionJob] = []
    await ensure_faces(bot)

    async def image_part(data: dict, label: str) -> tuple[str, list[VisionInput]]:
        text, vision_input = await fetch_image_input(
            data.get("url", ""),
            data.get("file_unique", ""),
            is_sticker=_is_sticker_segment_data(data),
            ref_id=label,
        )
        if vision_input is None or not caption:
            return (text, [vision_input] if vision_input else [])
        kind = _image_kind(vision_input)
        cached = await cached_caption(vision_input.digest)
        if cached:
            log_event("image_caption", session_id=str(group_id), outcome="cache_hit")
            return (f"[{kind}: {cached}]", [vision_input])
        token = f"[{kind}#{secrets.token_hex(2)}]"
        caption_jobs.append(CaptionJob(token, vision_input))
        return (token, [vision_input])

    # 附图标签按消息 id 和第几张写，模型靠它把原图和消息对上号
    image_numbers: dict[int, int] = {}
    for index, seg in enumerate(message):
        if seg.type == "image" and segment_text(seg.type, seg.data) is None:
            image_numbers[index] = len(image_numbers) + 1

    async def process_segment(
        seg: MessageSegment,
        segment_index: int,
    ) -> tuple[str, list[VisionInput]]:
        text = segment_text(seg.type, seg.data)
        if text is not None:
            return (text, [])

        if seg.type == "image":
            return await image_part(
                seg.data,
                f"消息 {message_scope} 的第 {image_numbers[segment_index]} 张图",
            )

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
                    for segment in content_data:
                        msg_type = segment.get("type")
                        data = segment.get("data", {})
                        text = segment_text(msg_type, data)
                        if text is not None:
                            source_text += text
                        elif msg_type == "image":
                            # 引用的图和原消息是同一份字节，原消息看过就命中描述缓存
                            img_text, img_inputs = await image_part(
                                data,
                                f"消息 {message_scope} 引用的第 {len(image_inputs) + 1} 张图",
                            )
                            source_text += img_text
                            image_inputs.extend(img_inputs)

                if len(source_text) > 800:
                    source_text = source_text[:800] + "..."
                return (f' [回复 {sender}: "{source_text}"] ', image_inputs)
            except Exception as e:
                logger.warning(f"获取回复内容失败: {e}")
                return (" [回复] ", [])

        logger.debug(f"未处理的消息段: {seg.type}")
        return ("", [])

    tasks = [process_segment(seg, index) for index, seg in enumerate(message)]
    results = await asyncio.gather(*tasks)

    content = "".join(result[0] for result in results).strip()
    image_inputs = [item for result in results for item in result[1]]
    return (content, image_inputs, mentions, caption_jobs)


