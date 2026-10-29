"""长期记忆候选（Feedback 的更正、分段整理的事实）：规范化、校验、去重落库。

校验只做确定性过滤（长度、类别、置信度、主体），不做语义启发式——
「是不是玩笑 / 有没有注入指令」交给模型的 prompt 与 confidence 判断。
「好的」「哈哈哈」这类噪声都短于 MIN_CONTENT_CHARS，长度过滤已经覆盖，不再单列词表。
"""

from collections.abc import Iterable
from datetime import datetime

from nonebot import logger

from ..chat.session import Session
from ..messages.message import Message
from ..metrics import log_event
from ..storage.db import message_final_id

ALLOWED_MEMORY_CATEGORIES = {"event", "preference", "profile", "relationship"}
MIN_MEMORY_CONFIDENCE = 0.6
MIN_CONTENT_CHARS = 10


def clamp01(value, default: float) -> float:
    """模型给的 0–1 分数；不是数字时取 default。"""

    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def source_messages(item: dict, messages: list[Message]) -> list[Message]:
    """模型给的 source 下标指向的消息；越界和非整数的下标丢掉。"""

    source = item.get("source")
    return [
        messages[index]
        for index in (source if isinstance(source, list) else [])
        if isinstance(index, int) and 0 <= index < len(messages)
    ]


def memory_metadata(
    *,
    category: str,
    date: int,
    confidence: float,
    importance: float,
    sources: list[Message],
    participant_ids: Iterable[str],
    subject: tuple[str, str] = ("", ""),
    speaker: tuple[str, str] = ("", ""),
    is_correction: bool = False,
    replaces: str = "",
) -> dict:
    """add_memories 要的元数据；subject / speaker 是 (QQ 号, 名字)，来源 msg_id 由 sources 算出。"""

    return {
        "category": category,
        "date": date,
        "subject_user_id": subject[0],
        "subject_user_name": subject[1],
        "speaker_user_id": speaker[0],
        "speaker_user_name": speaker[1],
        "confidence": confidence,
        "importance": importance,
        "source_msg_ids": " ".join(dict.fromkeys(message_final_id(m) for m in sources)),
        "participant_ids": " ".join(dict.fromkeys(participant_ids)),
        "is_correction": is_correction,
        "replaces": replaces,
    }


def validate_memory_candidate(content: str, metadata: dict) -> str:
    """返回拒绝原因；空串表示可存。"""

    if not metadata["source_msg_ids"]:
        return "missing_source"
    if len(content) < MIN_CONTENT_CHARS:
        return "too_short"
    if metadata["category"] not in ALLOWED_MEMORY_CATEGORIES:
        return "unsupported_category"
    if metadata["confidence"] < MIN_MEMORY_CONFIDENCE:
        return "low_confidence"
    if not metadata["subject_user_id"] and not metadata["subject_user_name"]:
        return "missing_subject"
    return ""


def _resolve_user(
    user_id: str,
    user_name: str,
    known_users: dict[str, str],
    *,
    fallback_id: str = "",
    speaker_id: str = "",
) -> tuple[str, str]:
    """对齐记忆主体/说话人的 QQ 号与名字，名字一律取上下文里该号的当前群名片。

    「A 说/对 B 怎样」应记成主体 B、说话人 A；模型常把说话人 A 的 id 抄进主体、名字却写 B。
    说话人已经另记了 A（A 的档案也会读到这条），所以这种情况以名字为准把主体落到 B；
    其余情况以 id 为准。上下文里找不到的人只留名字、不挂 id。
    """

    ids_by_name = [uid for uid, name in known_users.items() if name == user_name]
    by_name = ids_by_name[0] if user_name and len(ids_by_name) == 1 else ""
    if by_name and speaker_id and user_id == speaker_id and by_name != speaker_id:
        return by_name, known_users[by_name]
    if user_id in known_users:
        return user_id, known_users[user_id]
    if by_name:
        return by_name, user_name
    if not user_name and fallback_id:
        return fallback_id, known_users.get(fallback_id, "")
    return "", user_name


def _parse_memory_candidate(
    item,
    messages: list[Message],
    known_users: dict[str, str],
    refs: dict[str, str],
    date: int,
) -> tuple[str, str, dict] | None:
    """把 LLM 返回的一条 add/correct 候选规范化成 (action, 正文, 元数据)；ignore、未知 action、非对象返回 None。

    说话人由代码按 source 指向的第一条消息定，不信模型自报；source 缺失时来源为空，由校验拒绝。
    主体没填名字时才默认记到说话人头上；填了别人的名字就不能这样兜底。
    """

    if not isinstance(item, dict):
        return None
    action = str(item.get("action") or "add").strip().lower()
    if action not in ("add", "correct"):
        if action != "ignore":
            logger.debug(f"[Memory] 暂不处理的记忆 action: {action}")
        return None

    sources = source_messages(item, messages)
    speaker_user_id = sources[0].user_id if sources else ""
    speaker_user_name = known_users.get(speaker_user_id, "") if sources else ""
    subject_user_id, subject_user_name = _resolve_user(
        str(item.get("subject_user_id") or "").strip(),
        str(item.get("subject_user_name") or "").strip(),
        known_users,
        fallback_id=speaker_user_id,
        speaker_id=speaker_user_id,
    )
    # 编号只在本轮有效；查不到就当没指认，只写一条更正
    replaces = refs.get(str(item.get("target") or "").strip(), "") if action == "correct" else ""
    metadata = memory_metadata(
        category=str(item.get("category") or "event").strip().lower() or "event",
        date=date,
        confidence=clamp01(item.get("confidence"), 0.7),
        importance=clamp01(item.get("importance"), 0.5),
        sources=sources,
        participant_ids=(uid for uid in (subject_user_id, speaker_user_id) if uid),
        subject=(subject_user_id, subject_user_name),
        speaker=(speaker_user_id, speaker_user_name),
        is_correction=action == "correct",
        replaces=replaces,
    )
    return action, str(item.get("content") or "").strip(), metadata


async def save_memory_candidates(
    session: Session,
    items: list,
    *,
    messages: list[Message],
    known_users: dict[str, str],
    refs: dict[str, str],
    generation: int,
) -> dict[str, int] | None:
    """把模型给出的记忆候选（Feedback 的更正、分段整理的事实）校验、去重后落库。

    messages 是 source 下标指向的消息列表。会话已作废返回 None。
    """

    if session.stale(generation, "long_term_memory"):
        return None

    today = int(datetime.now().strftime("%Y%m%d"))
    skipped_quality = 0
    pending_memories: list[tuple[str, dict]] = []

    for raw_item in items:
        parsed = _parse_memory_candidate(raw_item, messages, known_users, refs, today)
        if parsed is None:
            continue
        action, content, metadata = parsed

        # 质量过滤：长度 + 类别/置信度/主体边界（先过滤，避免为废候选调 embedding）
        reason = validate_memory_candidate(content, metadata)
        if not reason and metadata["speaker_user_name"] == session.state.name:
            # 角色自己说的话只是上下文，不当事实来源
            reason = "bot_source"
        if reason:
            skipped_quality += 1
            log_event(
                "memory_candidate_rejected",
                "DEBUG",
                session_id=session.id,
                action=action,
                category=metadata["category"],
                reason=reason,
                content=content[:30],
            )
            continue
        pending_memories.append((content, metadata))

    store_result = {"added": 0, "skipped_dedup": 0, "corrected": 0}
    if pending_memories:
        store_result = await session.vector_memory.add_memories(
            pending_memories,
            still_current=lambda: session.state.generation == generation,
        )
        if store_result is None:
            session.stale(generation, "long_term_memory_bulk")
            return None

    result = {**store_result, "rejected": skipped_quality}
    if result["added"] or skipped_quality or result["skipped_dedup"]:
        logger.info(
            f"[Memory] 存储结果: 成功 {result['added']}, 质量过滤 {skipped_quality}, "
            f"去重跳过 {result['skipped_dedup']}, 更正替换 {result['corrected']}"
        )
    return result
