"""一轮对话的编排：短时记忆 → 意愿/相关性 → RAG → Feedback → Chat。"""

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from nonebot import logger

from ..llm import VisionInput, chat_client, feedback_client
from ..memory.candidates import clamp01, save_memory_candidates
from ..memory.vector import (
    RAG_FINAL_K,
    RAG_ITEM_CHARS,
    RAG_MEMORY_CHAR_BUDGET,
    RAG_MERGED_CANDIDATE_CAP,
    RAG_PER_QUERY_RECALL_K,
    RetrievalResult,
    build_chat_rag_queries,
)
from ..messages.faces import Reaction, reaction_emoji
from ..messages.message import SHORT_TERM_BUFFER_SIZE, Message
from ..metrics import log_event
from ..prompts import (
    CHAT_SYSTEM_PROMPT,
    FEEDBACK_SYSTEM_PROMPT,
    STICKER_TAGS,
    SUMMARY_CHARS,
    describe_mood,
    describe_relation,
    get_chat_prompt,
    get_feedback_prompt,
    get_time_description,
)
from ..storage.db import get_history_before
from .domain import EmotionState, PersonProfile
from .engagement import (
    RELEVANCE_WILLINGNESS_FLOOR,
    SPEAK_WILLINGNESS_RETAIN_FACTOR,
    chatting_state,
    evaluate_engagement,
)
from .session import Session

CONSOLIDATION_MESSAGE_THRESHOLD = 8
CONSOLIDATION_INTERVAL_SECONDS = 180.0
CONSOLIDATION_MAX_MESSAGES = 60
HISTORY_RECALL_LIMIT = 20
MY_RECENT_REPLIES_LIMIT = 6
GROUP_HEAT_SECONDS = 900
# 略高的温度换取用词和句式的变化；0.9 时会开始编造群友说过的话
CHAT_TEMPERATURE = 0.8
FEEDBACK_TEMPERATURE = 0.1
FEEDBACK_FIELDS = (
    "corrections",
    "willing",
    "new_emotion",
    "affinity_changes",
    "summary",
    "need_history",
)


@dataclass(frozen=True)
class Poke:
    user_id: str
    user_name: str


@dataclass(frozen=True)
class Reply:
    """Chat 输出的一项：先发文字，再发表情包。"""

    text: str
    target_id: int | None = None
    # (tag, want)，tag 已核对过词表
    sticker: tuple[str, str] | None = None


@dataclass(frozen=True)
class FeedbackOutput:
    """校验过的 Feedback 输出。"""

    emotion: dict[str, float]
    willing: float
    # 只含本轮发言的群友
    affinity_changes: dict[str, int]
    summary: str | None
    corrections: list
    need_history: bool
    react: Reaction | None
    poke: Poke | None


@dataclass(frozen=True)
class TurnResult:
    replies: list[Reply]
    react: Reaction | None
    poke: Poke | None


def _user_key(message: Message) -> str:
    return message.user_id or message.user_name


def format_history(messages: list[Message]) -> list[dict]:
    return [
        {"time": m.time.strftime("%H:%M"), "name": m.user_name, "content": m.content}
        for m in messages
    ]


def _group_heat(messages: list[Message], bot_name: str) -> str:
    """群友的消息数、发言人数，加上角色自己的发言量，给 Feedback 判断群热度和自己是不是说多了。

    messages 是窗口内的全部消息。角色连续发的几条算一次开口，接的是开口前最后一个群友。
    """

    others = [m for m in messages if m.user_name != bot_name]
    speakers = {_user_key(m) for m in others}
    heat = f"近15分钟群友 {len(others)} 条、{len(speakers)} 人在说话"

    own_count = 0
    turns = 0
    partner_turns: dict[str, int] = {}
    partner_names: dict[str, str] = {}
    last_other: Message | None = None
    previous_is_own = False
    for m in messages:
        if m.user_name != bot_name:
            last_other = m
            previous_is_own = False
            continue
        own_count += 1
        if not previous_is_own:
            turns += 1
            if last_other is not None:
                key = _user_key(last_other)
                partner_turns[key] = partner_turns.get(key, 0) + 1
                partner_names[key] = last_other.user_name
        previous_is_own = True

    if not own_count:
        return f"{heat}；你一句没说"
    heat += f"；你开口 {turns} 次、共 {own_count} 条"
    if partner_turns:
        top_key = max(partner_turns, key=partner_turns.get)
        if partner_turns[top_key] >= 2:
            heat += f"，其中 {partner_turns[top_key]} 次是在接 {partner_names[top_key]} 的话"
    return heat


def _history_context(session: Session, messages_chunk: list[Message]) -> list[dict]:
    """短时窗口里除本轮新消息以外的历史，避免 Prompt 上下文重复。"""

    chunk_ids = {msg.id for msg in messages_chunk if msg.id}
    history = [
        m
        for m in session.short_term_memory.access()
        if m.id not in chunk_ids and not any(m is chunk_msg for chunk_msg in messages_chunk)
    ]
    return format_history(history)


def _memory_lines(records: list[dict]) -> tuple[list[str], dict[str, str]]:
    """按总字符预算与单条上限把检索结果整理成 prompt 行，返回行与「编号 → 记忆 id」。

    行里带编号和主体：模型能看出一条记忆挂在谁身上，也能用编号指认要更正的那条。
    """

    lines = []
    refs = {}
    remaining = RAG_MEMORY_CHAR_BUDGET
    for item in records:
        if remaining <= 0:
            break
        meta = item["metadata"]
        ref = f"m{len(lines) + 1}"
        tag = "更正|" if meta["is_correction"] else ""
        line = (
            f"【{tag}{ref}|主体:{meta['subject_user_name'] or '无'}|d:{meta['date']}】"
            f"{item['content']}"
        )
        line = line[: min(remaining, RAG_ITEM_CHARS)].rstrip()
        lines.append(line)
        refs[ref] = meta["memory_ref"]
        remaining -= len(line)
    return lines, refs


def _speaker_ids(messages_chunk: list[Message], bot_name: str) -> list[str]:
    """本轮发言的群友，不含角色自己。"""

    return list(
        dict.fromkeys(_user_key(msg) for msg in messages_chunk if msg.user_name != bot_name)
    )


def _related_profiles(
    session: Session, messages_chunk: list[Message], *, for_chat: bool
) -> list[dict]:
    """Feedback 要在当前好感度上打分，给数值；Chat 只需要知道关系，给文字。"""

    state = session.state
    profiles = []
    for uid in _speaker_ids(messages_chunk, state.name):
        affinity = state.profiles.get(uid, PersonProfile(user_id=uid)).level()
        profiles.append(
            {
                "user_id": uid,
                **(
                    {"relation": describe_relation(affinity)}
                    if for_chat
                    else {"affinity": affinity}
                ),
                "summary": state.user_summaries.get(uid, ""),
            }
        )
    return profiles


# ==================== 解析模型输出 ====================


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
    """affinity_changes：只收本轮发言群友（不含角色自己）的整数变化量。"""

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


def _parse_reaction(raw, messages_chunk: list[Message], bot_name: str) -> Reaction | None:
    """react：只能贴这批新消息里群友的消息，表情要在目录里。"""

    if not isinstance(raw, dict):
        return None
    index = raw.get("index")
    if not isinstance(index, int) or not 0 <= index < len(messages_chunk):
        return None
    target = messages_chunk[index]
    if not target.id or target.user_name == bot_name:
        return None
    emoji = reaction_emoji(str(raw.get("emoji") or ""))
    if emoji is None:
        logger.debug(f"贴表情不在目录里，丢弃: {raw.get('emoji')}")
        return None
    return Reaction(target_id=target.id, emoji_id=emoji[0], name=emoji[1])


def _parse_poke(raw, messages_chunk: list[Message], bot_name: str) -> Poke | None:
    """poke 是 QQ 号：只能戳这批里说话的人，或被 @、被回复、被戳的人。"""

    user_id = str(raw or "").strip()
    candidates = {msg.user_id: msg.user_name for msg in messages_chunk if msg.user_id}
    for msg in messages_chunk:
        candidates.update(msg.mentions)
    name = candidates.get(user_id)
    if not user_id or name is None or name == bot_name:
        return None
    return Poke(user_id=user_id, user_name=name)


def parse_feedback(
    data: dict | None, messages_chunk: list[Message], session: Session
) -> FeedbackOutput | None:
    """Feedback 输出的唯一校验入口；心情无效就整份丢弃，其余字段缺了或不对按没给处理。"""

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
    return FeedbackOutput(
        emotion=emotion,
        willing=clamp01(data.get("willing"), 0.0),
        affinity_changes=_parse_affinity_changes(
            data.get("affinity_changes"), set(_speaker_ids(messages_chunk, state.name))
        ),
        summary=None if summary is None else str(summary)[:SUMMARY_CHARS],
        corrections=corrections if isinstance(corrections, list) else [],
        need_history=bool(data.get("need_history")),
        react=_parse_reaction(data.get("react"), messages_chunk, state.name),
        poke=_parse_poke(data.get("poke"), messages_chunk, state.name),
    )


def parse_reply(data: dict | None) -> list[Reply]:
    """Chat 输出的唯一校验入口：reply 里的每个对象一项，引用 id 和表情包标签在这里核对。"""

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
        replies.append(
            Reply(text=str(item.get("content") or ""), target_id=target_id, sticker=sticker)
        )
    return replies


# ==================== 一轮对话 ====================


async def process_chunk(
    session: Session,
    messages_chunk: list[Message],
    images: list[VisionInput],
    generation: int,
) -> TurnResult | None:
    """images 是这批的原图（新发的和被回复的），只给 Chat 看。"""

    try:
        if session.stale(generation, "process_start"):
            return None
        session.record_incoming(messages_chunk)

        state = session.state
        # 没设人设（或预设文件不在）：消息照常记，不判断、不说话
        if not state.name:
            return None
        recent = session.short_term_memory.access()
        self_share = sum(m.user_name == state.name for m in recent) / len(recent)
        engagement = evaluate_engagement(
            state=state,
            messages=messages_chunk,
            self_share=self_share,
            now=datetime.now(),
        )
        is_relevant = engagement.relevant

        if not engagement.engaged:
            if _consolidation_due(session):
                pending_messages = session.short_term_memory.messages_after(
                    state.last_consolidated_time,
                    limit=CONSOLIDATION_MAX_MESSAGES,
                )
                await consolidate_stage(session, pending_messages, generation)
            log_event(
                "speak_decision",
                "DEBUG",
                session_id=session.id,
                engaged=False,
                relevant=is_relevant,
                willingness=round(state.willingness, 2),
                self_share=round(self_share, 2),
            )
            return None

        # 要说话的轮次都重排：不重排时只能按向量分取前 k，闲聊上它分不出相关和无关
        search_result = await search_stage(session, messages_chunk, use_rerank=True)
        if session.stale(generation, "rag_search"):
            return None

        try:
            feedback = await feedback_stage(
                session,
                messages_chunk,
                is_relevant=is_relevant,
                search_result=search_result,
                generation=generation,
            )
        finally:
            session.schedule_save()

        if session.stale(generation, "feedback"):
            return None
        if feedback is not None:
            _advance_consolidation_watermark(session, messages_chunk)

        # 被点名也交给 willing 判断；Feedback 失败时只有被点名才回
        if feedback is None:
            speak = is_relevant
        else:
            speak = feedback.willing >= engagement.speak_threshold
        log_event(
            "speak_decision",
            session_id=session.id,
            engaged=True,
            relevant=is_relevant,
            in_conversation=engagement.in_conversation,
            llm_willing=feedback.willing if feedback else None,
            threshold=engagement.speak_threshold,
            self_share=round(self_share, 2),
            speak=speak,
        )
        react = feedback.react if feedback else None
        poke = feedback.poke if feedback else None
        if not speak:
            return TurnResult(replies=[], react=react, poke=poke)

        replies = await chat_stage(
            session,
            messages_chunk,
            images,
            need_history=feedback is not None and feedback.need_history,
            search_result=search_result,
            generation=generation,
        )
        if session.stale(generation, "chat"):
            return None
        # last_speak_time 在真正发出去之后由 dispatch_replies 更新
        return TurnResult(replies=replies, react=react, poke=poke)
    finally:
        await session.flush_persistence()


async def search_stage(
    session: Session,
    messages_chunk: list[Message],
    *,
    use_rerank: bool,
) -> RetrievalResult:
    started_at = time.perf_counter()
    state = session.state

    # Reranker 使用第一条 query 作为主 query，因此必须最新消息优先。
    raw_queries = [msg.content for msg in reversed(messages_chunk[-3:])]
    queries = build_chat_rag_queries(raw_queries, chat_summary=state.chat_summary)
    rag_stats = {
        "session_id": session.id,
        "query_count": len(queries),
        "queries": queries,
        "use_rerank": use_rerank,
        "skip_reason": "none",
    }

    preset_lines = state.preset_lines
    # 只翻这批发言人和被 @、被回复的人的记忆：拿闲聊原话在全群里撞，翻出来的多是字面巧合
    speaker_ids = {msg.user_id for msg in messages_chunk if msg.user_id}
    user_ids = speaker_ids.union(*(msg.mentions for msg in messages_chunk))
    rag_stats["user_count"] = len(user_ids)

    records = []
    if not queries:
        rag_stats["skip_reason"] = "no_queries"
    elif not user_ids:
        rag_stats["skip_reason"] = "no_users"
    else:
        retrieval = await session.vector_memory.retrieve_with_decay(
            queries,
            k=RAG_FINAL_K,
            user_ids=user_ids,
            use_rerank=use_rerank,
            candidate_k=RAG_PER_QUERY_RECALL_K,
            merged_candidate_cap=RAG_MERGED_CANDIDATE_CAP,
            active_user_ids=speaker_ids,
        )
        records = retrieval.records
        rag_stats.update(retrieval.stats)

    memory_lines, refs = _memory_lines(records)
    rag_stats["injected_count"] = len(preset_lines) + len(memory_lines)
    rag_stats["injected_chars"] = sum(len(line) for line in preset_lines + memory_lines)
    rag_stats["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
    log_event("rag_search", "DEBUG", **rag_stats)
    return RetrievalResult(
        records=records,
        stats=rag_stats,
        preset_lines=preset_lines,
        memory_lines=memory_lines,
        refs=refs,
    )


async def _ask_feedback(
    session: Session,
    messages_chunk: list[Message],
    is_relevant: bool,
    search_result: RetrievalResult,
) -> FeedbackOutput | None:
    """调一次 Feedback 并校验输出；失败返回 None。Feedback 不带图，看消息里的图片描述就够了。"""

    state = session.state
    for uid in _speaker_ids(messages_chunk, state.name):
        state.profiles.setdefault(uid, PersonProfile(user_id=uid))

    now = datetime.now()
    state.global_emotion.settle(now)
    prompt = get_feedback_prompt(
        bot_name=state.name,
        self_life=state.self_life,
        self_in_group=state.self_in_group,
        chat_state_value=chatting_state(state, now).value,
        summary=state.chat_summary,
        recent_msgs=_history_context(session, messages_chunk),
        new_msgs=[
            {"id": msg.user_id, "name": msg.user_name, "content": msg.content}
            for msg in messages_chunk
        ],
        emotion=state.global_emotion.vad(),
        related_profiles=_related_profiles(session, messages_chunk, for_chat=False),
        search_result=search_result.memory_lines,
        is_relevant=is_relevant,
        time_info=get_time_description(now),
        presets=search_result.preset_lines,
        group_notes=state.group_notes,
        new_msg_speakers=[
            {
                "index": index,
                "user_id": msg.user_id,
                "user_name": msg.user_name,
                **({"mentioned": msg.mentions} if msg.mentions else {}),
            }
            for index, msg in enumerate(messages_chunk)
        ],
        group_heat=_group_heat(
            session.short_term_memory.messages_after(
                now - timedelta(seconds=GROUP_HEAT_SECONDS),
                limit=SHORT_TERM_BUFFER_SIZE,
            ),
            state.name,
        ),
    )
    data = await feedback_client.generate_json(
        prompt,
        session_id=session.id,
        temperature=FEEDBACK_TEMPERATURE,
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
    )
    return parse_feedback(data, messages_chunk, session)


def _apply_sediment(
    session: Session,
    feedback: FeedbackOutput,
    messages_chunk: list[Message],
    generation: int,
    refs: dict[str, str],
) -> None:
    """应用 Feedback 的沉淀结果：情绪、画像、摘要、记忆更正。refs 是本轮检索行编号 → 记忆 id。"""

    state = session.state

    # 1. 心情：parse_feedback 已校验范围，apply 再限制每轮步长
    state.global_emotion.apply(feedback.emotion)

    # 2. 互动计数按消息记；好感度只改 Feedback 点名的人。画像已在 _ask_feedback 里建好
    now = datetime.now()
    for msg in messages_chunk:
        if msg.user_name != state.name:
            state.profiles[_user_key(msg)].record_interaction(now)
    for uid, delta in feedback.affinity_changes.items():
        state.profiles[uid].add_affinity(delta)

    # 3. 摘要
    if feedback.summary is not None:
        state.chat_summary = feedback.summary

    # 4. 记忆更正（后台写入）；日常记忆由 memory/episodes.py 在一段聊完后整理
    if feedback.corrections:
        # 上下文里每个 QQ 号的当前群名片：先放被 @/被回复的人，再放发言人（本批在后、以最新为准）
        context = [*session.short_term_memory.access(), *messages_chunk]
        known_users = {uid: name for msg in context for uid, name in msg.mentions.items()}
        known_users.update({msg.user_id: msg.user_name for msg in context if msg.user_id})
        session.spawn(
            save_memory_candidates(
                session,
                feedback.corrections,
                messages=messages_chunk,
                known_users=known_users,
                refs=refs,
                generation=generation,
            )
        )


async def feedback_stage(
    session: Session,
    messages_chunk: list[Message],
    *,
    is_relevant: bool,
    search_result: RetrievalResult,
    generation: int,
) -> FeedbackOutput | None:
    """反馈阶段：分析情绪、记忆更正、更新摘要、给出接话意愿。
    Feedback 失败或会话已作废时返回 None。"""

    feedback = await _ask_feedback(session, messages_chunk, is_relevant, search_result)
    if feedback is None or session.stale(generation, "feedback_sediment"):
        return None
    _apply_sediment(session, feedback, messages_chunk, generation, search_result.refs)

    # 说不说由 willing 按场景门槛决定（见 process_chunk）；规则意愿只向它靠拢一半，
    # 让模型觉得不想说时下一批不必再急着调用 Feedback
    state = session.state
    state.willingness = (state.willingness + feedback.willing) / 2
    if is_relevant and state.willingness < RELEVANCE_WILLINGNESS_FLOOR:
        state.willingness = RELEVANCE_WILLINGNESS_FLOOR
    return feedback


async def consolidate_stage(
    session: Session,
    messages_chunk: list[Message],
    generation: int,
) -> None:
    """常驻记忆固化：分析+沉淀，但不改回复意愿、不做发言决策。"""

    if not messages_chunk:
        return
    session.state.last_consolidation_attempt = datetime.now()
    logger.debug(f"[Session {session.id}] >> 记忆固化 (Consolidate) {len(messages_chunk)} 条")
    search_result = await search_stage(session, messages_chunk, use_rerank=False)
    if session.stale(generation, "consolidation_search"):
        return
    feedback = await _ask_feedback(session, messages_chunk, False, search_result)
    if feedback is None:
        session.schedule_save()
        return
    if session.stale(generation, "consolidation_sediment"):
        return
    _apply_sediment(session, feedback, messages_chunk, generation, search_result.refs)
    _advance_consolidation_watermark(session, messages_chunk)


async def chat_stage(
    session: Session,
    messages_chunk: list[Message],
    images: list[VisionInput],
    *,
    need_history: bool,
    search_result: RetrievalResult,
    generation: int,
) -> list[Reply]:
    state = session.state

    # Feedback 要求回溯时才翻更早的历史，只给 Chat 用
    recalled_history = []
    current_msgs = session.short_term_memory.access()
    if need_history and current_msgs:
        recalled_msgs = await get_history_before(
            session.id, current_msgs[0].time, limit=HISTORY_RECALL_LIMIT
        )
        recalled_history = [
            f"[{m.time:%H:%M}] {m.user_name}: {m.content}" for m in recalled_msgs
        ]
        logger.debug(f"[Session {session.id}] 回溯了 {len(recalled_history)} 条历史消息")

    new_msgs = [
        {"id": msg.id, "name": msg.user_name, "content": msg.content}
        for msg in messages_chunk
    ]
    recent_msgs = _history_context(session, messages_chunk)
    # 自己最近说过的话单独列出来，prompt 要求避免重复句式
    my_recent_replies = [
        m.content for m in session.short_term_memory.access() if m.user_name == state.name
    ][-MY_RECENT_REPLIES_LIMIT:]
    recalled_str = "\n".join(recalled_history) or "无"
    related_profiles = _related_profiles(session, messages_chunk, for_chat=True)
    state.global_emotion.settle()
    prompt = get_chat_prompt(
        bot_name=state.name,
        profile=state.profile,
        bottom_lines=state.bottom_lines,
        self_life=state.self_life,
        self_in_group=state.self_in_group,
        chat_state_value=chatting_state(state, datetime.now()).value,
        summary=state.chat_summary,
        recent_msgs=recent_msgs,
        new_msgs=new_msgs,
        mood=describe_mood(state.global_emotion),
        related_profiles=related_profiles,
        search_result=search_result.memory_lines,
        presets=search_result.preset_lines,
        group_notes=state.group_notes,
        recalled_history=recalled_str,
        my_recent_replies=my_recent_replies,
        time_info=get_time_description(datetime.now()),
    )
    log_event(
        "rag_prompt_budget",
        "DEBUG",
        session_id=session.id,
        chat_prompt_total_chars=len(prompt),
        rag_injected_count=len(search_result.memory_lines),
        rag_injected_chars=sum(len(item) for item in search_result.memory_lines),
        preset_injected_count=len(search_result.preset_lines),
        preset_injected_chars=sum(len(item) for item in search_result.preset_lines),
        group_notes_chars=len(state.group_notes),
        profile_summary_chars=sum(len(item["summary"]) for item in related_profiles),
        history_chars=len(state.chat_summary)
        + sum(len(item["content"]) for item in recent_msgs),
        recent_chars=sum(len(item["content"]) for item in new_msgs),
        recalled_history_chars=len(recalled_str),
        persona_chars=len(state.self_life) + len(state.self_in_group),
    )

    replies = parse_reply(
        await chat_client.generate_json(
            prompt,
            session_id=session.id,
            temperature=CHAT_TEMPERATURE,
            system_prompt=CHAT_SYSTEM_PROMPT,
            images=images,
        )
    )
    if session.stale(generation, "chat_reply"):
        return []

    if replies:
        state.willingness *= SPEAK_WILLINGNESS_RETAIN_FACTOR
        # 这批人就是接下来对话窗口里的聊天对象
        state.conversation_partners = {
            msg.user_id
            for msg in messages_chunk
            if msg.user_id and msg.user_name != state.name
        }
    return replies


def _advance_consolidation_watermark(session: Session, messages_chunk: list[Message]) -> None:
    state = session.state
    latest = max(msg.time for msg in messages_chunk)
    if state.last_consolidated_time is None or latest > state.last_consolidated_time:
        state.last_consolidated_time = latest
    state.messages_since_consolidation = 0
    state.last_consolidation_attempt = datetime.now()
    session.schedule_save()


def _consolidation_due(session: Session) -> bool:
    """攒够消息或距上次尝试足够久，才在不参与时做一次固化。"""

    state = session.state
    if state.messages_since_consolidation >= CONSOLIDATION_MESSAGE_THRESHOLD:
        return True
    return (
        state.messages_since_consolidation > 0
        and (datetime.now() - state.last_consolidation_attempt).total_seconds()
        >= CONSOLIDATION_INTERVAL_SECONDS
    )
