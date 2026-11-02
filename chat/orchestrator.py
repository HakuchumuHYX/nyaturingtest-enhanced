"""一轮对话的编排：短时记忆 → 看不看群（away 时只记下谁在叫她）→ 补看 → RAG → Feedback（沉淀 + 说不说）→ Chat。"""

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from nonebot import logger

from ..config import get_app_settings
from ..llm import chat_client, feedback_client, web_search
from ..memory.candidates import clamp01, save_memory_candidates
from ..memory.vector import (
    RAG_FINAL_K,
    RAG_ITEM_CHARS,
    RAG_MEMORY_CHAR_BUDGET,
    RetrievedMemory,
)
from ..messages.faces import Reaction, reaction_emoji
from ..messages.message import SHORT_CONTEXT_LIMIT, SHORT_TERM_BUFFER_SIZE, ChatMessage
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
    truncate_text,
)
from ..storage.db import get_catch_up, get_history_before
from .domain import EmotionState, PersonProfile
from .engagement import (
    LOOK_MAX_MESSAGES,
    SELF_SHARE_LIMIT,
    LookTrigger,
    calls_me,
    decide_look,
    unread_count,
)
from .session import Session
from .status import apply_said, describe_now, is_away, note_call, recent_diary, today_lines

HISTORY_RECALL_LIMIT = 20
# 补看：被挤出未读的那段里捞回几条找过她的消息；错过的分段小结最多几条、几个字
CATCH_UP_CALLS = 5
CATCH_UP_EPISODES = 8
MISSED_CHARS = 1200
# 被叫到时交给状态模型的每条消息最多几个字
CALL_LINE_CHARS = 100
# Chat / 主动话题读几天日记
CHAT_DIARY_DAYS = 2
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
    "focus",
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


@dataclass(frozen=True)
class MemoryContext:
    """注入 prompt 的记忆行，和行号 → 记忆 id（Feedback 用 correct 指认要替换的那条）。"""

    lines: list[str]
    refs: dict[str, str]


@dataclass(frozen=True)
class TurnResult:
    replies: list[Reply]
    react: Reaction | None
    poke: Poke | None


def _user_key(message: ChatMessage) -> str:
    return message.user_id or message.user_name


def format_history(messages: list[ChatMessage]) -> list[dict]:
    return [
        {"time": m.time.strftime("%H:%M"), "name": m.user_name, "content": m.content}
        for m in messages
    ]


def _group_heat(messages: list[ChatMessage], bot_name: str) -> str:
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
    last_other: ChatMessage | None = None
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


def _read_context(session: Session, unread: list[ChatMessage]) -> list[ChatMessage]:
    """未读之前的 SHORT_CONTEXT_LIMIT 条，作为 recent_msgs。

    不能用「窗口减掉未读」：未读一多，上下文就被挤没了。
    """

    return session.short_term_memory.messages_before(unread[0].time, SHORT_CONTEXT_LIMIT)


def _memory_lines(records: list[RetrievedMemory]) -> MemoryContext:
    """按总字符预算与单条上限把检索结果整理成 prompt 行。

    行里带编号和主体：模型能看出一条记忆挂在谁身上，也能用编号指认要更正的那条。
    """

    lines = []
    refs = {}
    remaining = RAG_MEMORY_CHAR_BUDGET
    for item in records:
        if remaining <= 0:
            break
        row = item.row
        ref = f"m{len(lines) + 1}"
        tag = "更正|" if row.is_correction else ""
        line = f"【{tag}{ref}|主体:{row.subject_user_name or '无'}|d:{row.date}】{row.content}"
        line = line[: min(remaining, RAG_ITEM_CHARS)].rstrip()
        lines.append(line)
        refs[ref] = row.id
        remaining -= len(line)
    return MemoryContext(lines, refs)


def _speaker_ids(unread: list[ChatMessage], bot_name: str) -> list[str]:
    """这段未读里发言的群友，不含角色自己。"""

    return list(
        dict.fromkeys(_user_key(msg) for msg in unread if msg.user_name != bot_name)
    )


def _last_seen_days(session: Session, unread: list[ChatMessage]) -> dict[str, int | None]:
    """这段发言人上次在这个群被她看到说话是几天前，None 是第一次见。

    要在沉淀之前算：沉淀会把 last_interaction_at 刷成现在，Chat 再算就全是 0。
    """

    today = datetime.now().date()
    days = {}
    for uid in _speaker_ids(unread, session.state.name):
        profile = session.state.profiles.get(uid)
        last = profile.last_interaction_at if profile else None
        days[uid] = (today - last.date()).days if last else None
    return days


def _related_profiles(
    session: Session, last_seen: dict[str, int | None], *, for_chat: bool
) -> list[dict]:
    """Feedback 要在当前好感度上打分，给数值；Chat 只需要知道关系，给文字。last_seen 的键就是这段的发言人。"""

    state = session.state
    profiles = []
    for uid, days in last_seen.items():
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
                "last_seen_days": days,
            }
        )
    return profiles


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
    if not target.id or target.user_name == bot_name:
        return None
    emoji = reaction_emoji(str(raw.get("emoji") or ""))
    if emoji is None:
        logger.debug(f"贴表情不在目录里，丢弃: {raw.get('emoji')}")
        return None
    return Reaction(target_id=target.id, emoji_id=emoji[0], name=emoji[1])


def _parse_poke(raw, unread: list[ChatMessage], bot_name: str) -> Poke | None:
    """poke 是 QQ 号：只能戳这段未读里说话的人，或被 @、被回复、被戳的人。"""

    user_id = str(raw or "").strip()
    candidates = {msg.user_id: msg.user_name for msg in unread if msg.user_id}
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
    data: dict | None, unread: list[ChatMessage], session: Session
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
        affinity_changes=_parse_affinity_changes(
            data.get("affinity_changes"), set(_speaker_ids(unread, state.name))
        ),
        summary=None if summary is None else str(summary)[:SUMMARY_CHARS],
        corrections=corrections if isinstance(corrections, list) else [],
        need_history=bool(data.get("need_history")),
        react=_parse_reaction(data.get("react"), unread, state.name),
        poke=_parse_poke(data.get("poke"), unread, state.name),
        focus=_parse_focus(data.get("focus"), unread, state.name),
        web_query=web_query or None,
    )


def parse_reply(data: dict | None) -> list[Reply]:
    """校验 Chat 输出：reply 里的每个对象一项，引用 id 和表情包标签在这里核对。"""

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


async def process_chunk(
    session: Session,
    messages_chunk: list[ChatMessage],
    generation: int,
) -> TurnResult | None:
    """messages_chunk 是这次新进来的一批，到点补看时为空；看不看群按全部未读判断。"""

    try:
        if session.stale(generation, "process_start"):
            return None
        if messages_chunk:
            session.record_incoming(messages_chunk)

        state = session.state
        # 没设人设（或预设文件不在）：消息照常记，不判断、不说话，原图也用不上
        if not state.name:
            for message in messages_chunk:
                message.image_inputs.clear()
            return None
        away = is_away(state.preset)
        if away:
            # 手机放下了：不看群，只把叫她的消息交给状态模型判断拿不拿起手机；不写群名
            note_call(
                state.preset,
                [
                    f"{m.time:%H:%M} {m.user_name}：{truncate_text(m.content, CALL_LINE_CHARS)}"
                    for m in messages_chunk
                    if m.user_name != state.name and calls_me(state, m)
                ],
            )
        unread = session.unread()
        context = _read_context(session, unread) if unread else []
        calls, missed = ([], []) if away else await _catch_up(session, unread)
        trigger = decide_look(state, calls + unread, datetime.now(), away)
        if trigger is None:
            log_event(
                "speak_decision",
                "DEBUG",
                session_id=session.id,
                looked=False,
                away=away,
                unread=unread_count(state, unread),
            )
            return None
        try:
            return await _look(session, calls + unread, context, missed, trigger, generation)
        finally:
            # 看过这段就放掉原图；失败也放：描述还在消息里，留着只会越攒越多
            for message in unread:
                message.image_inputs.clear()
    finally:
        await session.flush_persistence()


async def _catch_up(
    session: Session, unread: list[ChatMessage]
) -> tuple[list[ChatMessage], list[str]]:
    """未读超过 LOOK_MAX_MESSAGES 条时，水位到这些未读之间被挤掉的那段：找过她的消息、那段时间的分段小结。

    睡醒、忙完、群里刷得太快走同一套。找过她的消息接在未读最前面，被点名照常按点名看群。
    醒来时还在聊的那段，前半截可能既没整理成小结、也不在未读里，就略过：真人醒来翻群也是扫一眼。
    """

    watermark = session.state.last_read_time
    if watermark is None or len(unread) < LOOK_MAX_MESSAGES:
        return [], []
    calls, episodes, skipped = await get_catch_up(
        session.id,
        watermark,
        unread[0].time,
        calls_limit=CATCH_UP_CALLS,
        episodes_limit=CATCH_UP_EPISODES,
    )
    if not skipped:
        return [], []
    missed = []
    remaining = MISSED_CHARS
    for episode in episodes:
        if remaining <= 0:
            break
        missed.append(truncate_text(episode, remaining))
        remaining -= len(missed[-1])
    missed.append(f"中间还有 {skipped} 条没细看")
    return calls, missed


@dataclass(frozen=True)
class LookContext:
    """这次看群时 Feedback 和 Chat 共用的背景：上下文、补看、她自己在干嘛、群信息、发言人多久没见。"""

    recent: list[ChatMessage]
    missed: list[str]
    now: str
    group_info: str
    last_seen: dict[str, int | None]


async def _look(
    session: Session,
    unread: list[ChatMessage],
    context: list[ChatMessage],
    missed: list[str],
    trigger: LookTrigger,
    generation: int,
) -> TurnResult | None:
    """看一眼群：检索 → Feedback（沉淀 + willing）→ 说不说 → Chat。

    unread 开头可能是补看捞回的、找过她的消息；context 是未读之前的聊天，按捞回之前的未读取。
    """

    state = session.state
    # 刚回来要按上次看这个群的时间判断；重启后 last_look_time 清零，已读水位还在
    previous_look = max(state.last_look_time, state.last_read_time or datetime.min)
    state.last_look_time = datetime.now()
    # 水位推到这一刻读到的最后一条：看群途中合并转发读完会把那条的时间改成现在（move_to_latest），
    # 按读完时的最大时间推进，就会把它和途中新来的消息一起误标成已读
    read_until = max(m.time for m in unread)
    # 只数说的话：贴表情、戳一戳记录算进来，她动一下手占比就被抬高，群友互贴又把占比拉低
    recent = [m for m in session.short_term_memory.access() if not m.is_action]
    self_share = sum(m.user_name == state.name for m in recent) / max(len(recent), 1)
    is_relevant = trigger is LookTrigger.RELEVANT
    look = LookContext(
        recent=context,
        missed=missed,
        now=describe_now(state.preset, previous_look),
        group_info=await session.refresh_group_info(),
        last_seen=_last_seen_days(session, unread),
    )

    memories = await search_stage(session, unread)
    if session.stale(generation, "rag_search"):
        return None

    feedback = await feedback_stage(
        session,
        unread,
        look,
        read_until=read_until,
        is_relevant=is_relevant,
        memories=memories,
        generation=generation,
    )
    if session.stale(generation, "feedback"):
        return None

    # 被点名也交给 willing 判断；Feedback 失败时只有被点名才回
    if feedback is None:
        speak = is_relevant
    else:
        speak = feedback.willing >= trigger.speak_threshold
        # 频率归代码管：自己说多了就不主动插嘴；被点名、接话、被戳不受限
        if trigger is LookTrigger.RHYTHM and self_share >= SELF_SHARE_LIMIT:
            speak = False
    # 禁言期间平台发不出去：照常看群、沉淀，只是不开口
    muted = session.muted
    speak = speak and not muted
    focus = feedback.focus if feedback else None
    log_event(
        "speak_decision",
        session_id=session.id,
        trigger=trigger.value,
        unread=unread_count(state, unread),
        llm_willing=feedback.willing if feedback else None,
        threshold=trigger.speak_threshold,
        self_share=round(self_share, 2),
        focus=f"{focus.user_name}: {focus.content[:30]}" if focus else None,
        muted=muted,
        speak=speak,
    )
    react = feedback.react if feedback else None
    poke = feedback.poke if feedback else None
    if not speak:
        return TurnResult(replies=[], react=react, poke=poke)

    # 决定开口了才搜：不开口就用不上，也不花钱
    web_results = []
    if feedback is not None and feedback.web_query:
        # 方括号里是图片描述、回复引用这类标记，不是群友问的话
        ori_question = re.sub(r"\[[^\]]*\]", "", focus.content).strip() if focus else ""
        web_results = await web_search(feedback.web_query, ori_question, session_id=session.id)

    replies = await chat_stage(
        session,
        unread,
        look,
        focus,
        need_history=feedback is not None and feedback.need_history,
        memories=memories,
        web_results=web_results,
        generation=generation,
    )
    if session.stale(generation, "chat"):
        return None
    # last_speak_time 在真正发出去之后由 dispatch_replies 更新
    return TurnResult(replies=replies, react=react, poke=poke)


async def search_stage(session: Session, unread: list[ChatMessage]) -> MemoryContext:
    """按这段未读检索。每次看群都可能开口，所以都重排：不重排时只能按向量分取前 k，闲聊上它分不出相关和无关。"""

    started_at = time.perf_counter()
    state = session.state

    # Reranker 使用第一条 query 作为主 query，因此必须最新消息优先；
    # 回复引用、表情、戳一戳这类群友没说什么的不拿去检索，
    # 取最近 3 条说了东西的，免得一段以表情包结尾就没得查
    queries = [msg.content.strip() for msg in reversed(unread) if msg.substantive][:3]
    if state.chat_summary.strip():
        queries.append(state.chat_summary.strip())
    queries = list(dict.fromkeys(queries))
    rag_stats = {
        "session_id": session.id,
        "query_count": len(queries),
        "queries": queries,
        "skip_reason": "none",
    }

    # 只翻这段的发言人和被 @、被回复的人的记忆：拿闲聊原话在全群里撞，翻出来的多是字面巧合
    speaker_ids = {msg.user_id for msg in unread if msg.user_id}
    user_ids = speaker_ids.union(*(msg.mentions for msg in unread))
    rag_stats["user_count"] = len(user_ids)

    records = []
    if not queries:
        rag_stats["skip_reason"] = "no_queries"
    elif not user_ids:
        rag_stats["skip_reason"] = "no_users"
    else:
        retrieval = await session.vector_memory.retrieve(
            queries,
            k=RAG_FINAL_K,
            user_ids=user_ids,
            active_user_ids=speaker_ids,
        )
        records = retrieval.records
        rag_stats.update(retrieval.stats)

    memories = _memory_lines(records)
    injected = state.preset_lines + memories.lines
    rag_stats["injected_count"] = len(injected)
    rag_stats["injected_chars"] = sum(len(line) for line in injected)
    rag_stats["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
    log_event("rag_search", "DEBUG", **rag_stats)
    return memories


async def _ask_feedback(
    session: Session,
    unread: list[ChatMessage],
    look: LookContext,
    is_relevant: bool,
    memories: MemoryContext,
) -> FeedbackOutput | None:
    """调一次 Feedback 并校验输出；失败返回 None。Feedback 不带图，看消息里的图片描述就够了。"""

    state = session.state
    now = datetime.now()
    state.global_emotion.settle(now)
    prompt = get_feedback_prompt(
        bot_name=state.name,
        self_life=state.self_life,
        self_in_group=state.self_in_group,
        summary=state.chat_summary,
        recent_msgs=format_history(look.recent),
        # 带上时间：一段未读可能跨好几分钟，Feedback 要看出最后一句是多久前的
        new_msgs=[
            {
                "time": msg.time.strftime("%H:%M"),
                "id": msg.user_id,
                "name": msg.user_name,
                "content": msg.content,
            }
            for msg in unread
        ],
        emotion=state.global_emotion.vad(),
        related_profiles=_related_profiles(session, look.last_seen, for_chat=False),
        search_result=memories.lines,
        is_relevant=is_relevant,
        time_info=get_time_description(now),
        presets=state.preset_lines,
        group_notes=state.group_notes,
        new_msg_speakers=[
            {
                "index": index,
                "user_id": msg.user_id,
                "user_name": msg.user_name,
                **({"mentioned": msg.mentions} if msg.mentions else {}),
            }
            for index, msg in enumerate(unread)
        ],
        group_heat=_group_heat(
            session.short_term_memory.messages_after(
                now - timedelta(seconds=GROUP_HEAT_SECONDS),
                limit=SHORT_TERM_BUFFER_SIZE,
            ),
            state.name,
        ),
        web_search=bool(get_app_settings().web_search.api_key),
        group_info=look.group_info,
        missed=look.missed,
        now=look.now,
    )
    data = await feedback_client.generate_json(
        prompt,
        session_id=session.id,
        temperature=FEEDBACK_TEMPERATURE,
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
    )
    return parse_feedback(data, unread, session)


def _apply_sediment(
    session: Session,
    feedback: FeedbackOutput,
    unread: list[ChatMessage],
    read_until: datetime,
    generation: int,
    refs: dict[str, str],
) -> None:
    """应用 Feedback 的沉淀结果：情绪、画像、摘要、记忆更正，最后把已读水位推到 read_until。

    refs 是本轮检索行编号 → 记忆 id。
    """

    state = session.state

    # parse_feedback 已校验范围，apply 再限制每轮步长
    state.global_emotion.apply(feedback.emotion)

    # 互动计数按消息记；好感度只改 Feedback 点名的人（parse_feedback 只收这段的发言人）
    now = datetime.now()
    for msg in unread:
        if msg.user_name != state.name:
            uid = _user_key(msg)
            state.profiles.setdefault(uid, PersonProfile(user_id=uid)).record_interaction(now)
    for uid, delta in feedback.affinity_changes.items():
        state.profiles[uid].add_affinity(delta)

    if feedback.summary is not None:
        state.chat_summary = feedback.summary

    if feedback.corrections:
        # 上下文里每个 QQ 号的当前群名片：先放被 @/被回复的人，再放发言人（未读在后、以最新为准）
        context = [*session.short_term_memory.access(), *unread]
        known_users = {uid: name for msg in context for uid, name in msg.mentions.items()}
        known_users.update({msg.user_id: msg.user_name for msg in context if msg.user_id})
        session.spawn(
            save_memory_candidates(
                session,
                feedback.corrections,
                messages=unread,
                known_users=known_users,
                refs=refs,
                generation=generation,
            )
        )

    if state.last_read_time is None or read_until > state.last_read_time:
        state.last_read_time = read_until
    session.schedule_save()


async def feedback_stage(
    session: Session,
    unread: list[ChatMessage],
    look: LookContext,
    *,
    read_until: datetime,
    is_relevant: bool,
    memories: MemoryContext,
    generation: int,
) -> FeedbackOutput | None:
    """反馈阶段：分析情绪、记忆更正、更新摘要、给出接话意愿，并沉淀。
    Feedback 失败或会话已作废时返回 None，不推进水位，下次看群重读。"""

    feedback = await _ask_feedback(session, unread, look, is_relevant, memories)
    if feedback is None or session.stale(generation, "feedback_sediment"):
        return None
    _apply_sediment(session, feedback, unread, read_until, generation, memories.refs)
    return feedback


async def chat_stage(
    session: Session,
    unread: list[ChatMessage],
    look: LookContext,
    focus: ChatMessage | None,
    *,
    need_history: bool,
    memories: MemoryContext,
    web_results: list[str],
    generation: int,
) -> list[Reply]:
    state = session.state

    # Feedback 要求回溯时才翻更早的历史，只给 Chat 用
    recalled_history = []
    if need_history:
        earliest = (look.recent or unread)[0].time
        recalled_msgs = await get_history_before(
            session.id, earliest, limit=HISTORY_RECALL_LIMIT
        )
        recalled_history = [
            f"[{m.time:%H:%M}] {m.user_name}: {m.content}" for m in recalled_msgs
        ]
        logger.debug(f"[Session {session.id}] 回溯了 {len(recalled_history)} 条历史消息")

    new_msgs = [
        {"id": msg.id, "name": msg.user_name, "content": msg.content}
        for msg in unread
    ]
    recent_msgs = format_history(look.recent)
    # 自己最近说过的话单独列出来，prompt 要求避免重复句式
    my_recent_replies = [
        m.content
        for m in session.short_term_memory.access()
        if m.user_name == state.name and not m.is_action
    ][-MY_RECENT_REPLIES_LIMIT:]
    recalled_str = "\n".join(recalled_history) or "无"
    related_profiles = _related_profiles(session, look.last_seen, for_chat=True)
    state.global_emotion.settle()
    prompt = get_chat_prompt(
        bot_name=state.name,
        profile=state.profile,
        bottom_lines=state.bottom_lines,
        self_life=state.self_life,
        self_in_group=state.self_in_group,
        summary=state.chat_summary,
        recent_msgs=recent_msgs,
        new_msgs=new_msgs,
        focus=(
            {"id": focus.id, "name": focus.user_name, "content": focus.content}
            if focus
            else None
        ),
        mood=describe_mood(state.global_emotion),
        related_profiles=related_profiles,
        search_result=memories.lines,
        presets=state.preset_lines,
        group_notes=state.group_notes,
        recalled_history=recalled_str,
        my_recent_replies=my_recent_replies,
        web_results=web_results,
        time_info=get_time_description(datetime.now()),
        diary=recent_diary(state.preset, CHAT_DIARY_DAYS),
        group_info=look.group_info,
        missed=look.missed,
        today=today_lines(state.preset),
        now=look.now,
    )
    log_event(
        "rag_prompt_budget",
        "DEBUG",
        session_id=session.id,
        chat_prompt_total_chars=len(prompt),
        rag_injected_count=len(memories.lines),
        rag_injected_chars=sum(len(item) for item in memories.lines),
        preset_injected_count=len(state.preset_lines),
        preset_injected_chars=sum(len(item) for item in state.preset_lines),
        group_notes_chars=len(state.group_notes),
        profile_summary_chars=sum(len(item["summary"]) for item in related_profiles),
        history_chars=len(state.chat_summary)
        + sum(len(item["content"]) for item in recent_msgs),
        recent_chars=sum(len(item["content"]) for item in new_msgs),
        recalled_history_chars=len(recalled_str),
        web_results_chars=sum(len(item) for item in web_results),
        persona_chars=len(state.self_life) + len(state.self_in_group),
    )

    data = await chat_client.generate_json(
        prompt,
        session_id=session.id,
        temperature=CHAT_TEMPERATURE,
        system_prompt=CHAT_SYSTEM_PROMPT,
        # 只附想接的那条的图：一段未读里可能有好几张，全附上既占内存，也让模型分不清该接哪句
        images=list(focus.image_inputs) if focus else [],
    )
    replies = parse_reply(data)
    if session.stale(generation, "chat_reply"):
        return []

    if replies:
        # 她说出口的新情况就是她接下来真的在做的事，各群共用
        await apply_said(state.preset, data.get("status"))
        # 接下来对话窗口里的聊天对象：接的是谁就是谁；没指认就是这段的发言人。
        # 不加区分地把一段里五六个人都算进来，3 分钟内谁开口都会马上看群、走低门槛
        partners = [focus] if focus else unread
        state.conversation_partners = {
            msg.user_id for msg in partners if msg.user_id and msg.user_name != state.name
        }
    return replies
