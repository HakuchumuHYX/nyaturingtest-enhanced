"""Feedback / Chat 的输出：数据类型和入口校验。模型 JSON 只在这里校验一次，之后按这里的类型用。"""

import math
from dataclasses import dataclass

from nonebot import logger

from ..memory.candidates import clamp01
from ..messages.faces import Reaction, reaction_emoji
from ..messages.message import ChatMessage
from ..messages.photos import PHOTO_KINDS
from ..metrics import log_event
from ..prompts import STICKER_TAGS, SUMMARY_CHARS
from .domain import EmotionState
from .session import Session

FEEDBACK_FIELDS = (
    "corrections",
    "willing",
    "new_emotion",
    "affinity_changes",
    "summary",
    "need_history",
    "focus",
)


@dataclass(frozen=True)
class Poke:
    user_id: str
    user_name: str


@dataclass(frozen=True)
class PhotoWish:
    """Chat 想发的一张照片：要么新拍（kind + describe），要么翻相册（photo_id）。"""

    kind: str = ""
    describe: str = ""
    photo_id: int | None = None


@dataclass(frozen=True)
class Reply:
    """Chat 输出的一项：先发文字，再发照片、表情包。带照片时拿不到图，这一项的文字也不发。"""

    text: str
    target_id: int | None = None
    # (tag, want)，tag 已核对过词表
    sticker: tuple[str, str] | None = None
    photo: PhotoWish | None = None


@dataclass(frozen=True)
class FeedbackOutput:
    """校验过的 Feedback 输出。"""

    emotion: dict[str, float]
    willing: float
    # 只含这段未读里发言的群友
    affinity_changes: dict[str, int]
    summary: str | None
    corrections: list
    need_history: bool
    react: Reaction | None
    poke: Poke | None
    # 角色想接的那条（不会是她自己的）；Chat 回应它、附它的图，聊天对象也按它定
    focus: ChatMessage | None
    # 开口前要上网查的词；没开联网搜索时提示词里没有这一项，总是 None
    web_query: str | None


def _parse_emotion(raw, current: EmotionState) -> tuple[dict[str, float] | None, str]:
    """校验、限幅 new_emotion；缺的轴沿用当前值，一个轴都没有算无效。返回 (心情, 失败原因)。"""

    if not isinstance(raw, dict):
        return None, "missing_new_emotion"
    specs = {
        "valence": (-1.0, 1.0, current.valence),
        "arousal": (0.0, 1.0, current.arousal),
        "dominance": (-1.0, 1.0, current.dominance),
    }
    normalized = {}
    valid_fields = 0
    for field_name, (minimum, maximum, default) in specs.items():
        if field_name not in raw:
            normalized[field_name] = default
            continue
        try:
            value = float(raw[field_name])
        except (TypeError, ValueError):
            return None, f"invalid_new_emotion_{field_name}"
        if not math.isfinite(value):
            return None, f"invalid_new_emotion_{field_name}"
        normalized[field_name] = max(minimum, min(maximum, value))
        valid_fields += 1
    if valid_fields == 0:
        return None, "empty_new_emotion"
    return normalized, ""


def _parse_affinity_changes(raw, speaker_ids: set[str]) -> dict[str, int]:
    """affinity_changes：只收这段未读里发言的群友（不含角色自己）的整数变化量。"""

    if not isinstance(raw, dict):
        return {}
    changes = {}
    for user_id, value in raw.items():
        if str(user_id) not in speaker_ids:
            continue
        try:
            changes[str(user_id)] = int(value)
        except (TypeError, ValueError):
            continue
    return changes


def _parse_reaction(raw, unread: list[ChatMessage], bot_name: str) -> Reaction | None:
    """react：只能贴这段未读里群友的消息，表情要在目录里。

    选中贴表情记录就是跟着贴，贴到它贴的那条上。
    """

    if not isinstance(raw, dict):
        return None
    index = raw.get("index")
    if not isinstance(index, int) or not 0 <= index < len(unread):
        return None
    target = unread[index]
    if target.reacted is not None:
        target = target.reacted.target
    # 群里的bot 的消息在群友眼里是她发的，贴上去就成了给自己贴
    if not target.id or target.user_name == bot_name or target.by_group_bot:
        return None
    emoji = reaction_emoji(str(raw.get("emoji") or ""))
    if emoji is None:
        logger.debug(f"贴表情不在目录里，丢弃: {raw.get('emoji')}")
        return None
    return Reaction(target_id=target.id, emoji_id=emoji[0], name=emoji[1])


def _parse_poke(raw, unread: list[ChatMessage], bot_name: str) -> Poke | None:
    """poke 是 QQ 号：只能戳这段未读里说话的人，或被 @、被回复、被戳的人。"""

    user_id = str(raw or "").strip()
    candidates = {
        msg.user_id: msg.user_name for msg in unread if msg.user_id and not msg.by_group_bot
    }
    for msg in unread:
        candidates.update(msg.mentions)
    name = candidates.get(user_id)
    if not user_id or name is None or name == bot_name:
        return None
    return Poke(user_id=user_id, user_name=name)


def _parse_focus(raw, unread: list[ChatMessage], bot_name: str) -> ChatMessage | None:
    """focus：角色想接的那条在 new_msgs 里的 index，不能是她自己的话。"""

    if not isinstance(raw, int) or not 0 <= raw < len(unread):
        return None
    target = unread[raw]
    return None if target.user_name == bot_name else target


def parse_feedback(
    data: dict | None, unread: list[ChatMessage], session: Session, speaker_ids: set[str]
) -> FeedbackOutput | None:
    """校验 Feedback 输出：心情无效就整份丢弃，其余字段缺了或不对按没给处理。"""

    state = session.state
    emotion, failure_reason = (
        _parse_emotion(data.get("new_emotion"), state.global_emotion)
        if data
        else (None, "invalid_payload")
    )
    if emotion is None:
        log_event(
            "feedback_rejected",
            "WARNING",
            session_id=session.id,
            failure_reason=failure_reason,
        )
        return None
    missing_fields = [name for name in FEEDBACK_FIELDS if name not in data]
    if missing_fields:
        log_event(
            "feedback_fields_missing",
            "WARNING",
            session_id=session.id,
            missing_feedback_fields=missing_fields,
            response_keys=sorted(str(key) for key in data),
        )

    summary = data.get("summary")
    corrections = data.get("corrections")
    web_query = data.get("web_query")
    web_query = web_query.strip() if isinstance(web_query, str) else ""
    return FeedbackOutput(
        emotion=emotion,
        willing=clamp01(data.get("willing"), 0.0),
        affinity_changes=_parse_affinity_changes(data.get("affinity_changes"), speaker_ids),
        summary=None if summary is None else str(summary)[:SUMMARY_CHARS],
        corrections=corrections if isinstance(corrections, list) else [],
        need_history=bool(data.get("need_history")),
        react=_parse_reaction(data.get("react"), unread, state.name),
        poke=_parse_poke(data.get("poke"), unread, state.name),
        focus=_parse_focus(data.get("focus"), unread, state.name),
        web_query=web_query or None,
    )


def _parse_photo(raw, album: list[dict]) -> PhotoWish | None:
    """ref 必须是这轮给过的 album 里的；新拍的 kind 要在白名单内、describe 不能空。"""

    if not isinstance(raw, dict):
        return None
    ref = str(raw.get("ref") or "").strip()
    if ref:
        if any(item["ref"] == ref for item in album):
            return PhotoWish(photo_id=int(ref[1:]))
        logger.debug(f"照片 ref 不在这轮的 album 里，丢弃: {ref}")
        return None
    kind = str(raw.get("kind") or "").strip()
    describe = str(raw.get("describe") or "").strip()
    if kind in PHOTO_KINDS and describe:
        return PhotoWish(kind=kind, describe=describe)
    logger.debug(f"照片字段无效，丢弃: {raw}")
    return None


def parse_reply(data: dict | None, album: list[dict]) -> list[Reply]:
    """校验 Chat 输出：reply 里的每个对象发成一条消息，发几条由模型定；引用 id、表情包标签和照片在这里核对。"""

    items = data.get("reply") if data else None
    replies = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        raw_target = item.get("target_id")
        target_id = None
        if raw_target:
            try:
                target_id = int(raw_target)
            except (TypeError, ValueError):
                logger.warning(f"引用ID无效: {raw_target}")
        sticker = None
        raw_sticker = item.get("sticker")
        if isinstance(raw_sticker, dict):
            tag = str(raw_sticker.get("tag") or "").strip()
            if tag in STICKER_TAGS:
                sticker = (tag, str(raw_sticker.get("want") or "").strip())
            else:
                logger.debug(f"表情包标签不在词表里，丢弃: {tag}")
        photo = _parse_photo(item.get("photo"), album)
        text = str(item.get("content") or "").strip()
        if not text and sticker is None and photo is None:
            continue
        replies.append(Reply(text=text, target_id=target_id, sticker=sticker, photo=photo))
    return replies
