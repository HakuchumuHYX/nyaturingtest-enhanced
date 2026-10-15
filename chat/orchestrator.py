import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from nonebot import logger

from ..llm import extract_and_parse_json
from ..memory.candidates import save_memory_candidates
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

LLMCall = Callable[[str], Awaitable[str]]

CONSOLIDATION_MESSAGE_THRESHOLD = 8
CONSOLIDATION_INTERVAL_SECONDS = 180.0
CONSOLIDATION_MAX_MESSAGES = 60
HISTORY_RECALL_LIMIT = 20
MY_RECENT_REPLIES_LIMIT = 6
GROUP_HEAT_SECONDS = 900


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


def _history_without_current_chunk(
    all_messages: list[Message], messages_chunk: list[Message]
) -> list[Message]:
    chunk_ids = {msg.id for msg in messages_chunk if msg.id}
    return [
        m
        for m in all_messages
        if m.id not in chunk_ids and not any(m is chunk_msg for chunk_msg in messages_chunk)
    ]


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


@dataclass(frozen=True)
class Poke:
    user_id: str
    user_name: str


@dataclass(frozen=True)
class FeedbackDecision:
    recalled_history: list[str]
    llm_willing: float
    react: Reaction | None
    poke: Poke | None


@dataclass(frozen=True)
class TurnResult:
    replies: list
    react: Reaction | None
    poke: Poke | None


def _parse_poke(
    response: dict, messages_chunk: list[Message], bot_name: str
) -> Poke | None:
    """Feedback 的 poke 是 QQ 号：只能戳这批里说话的人，或被 @、被回复、被戳的人。"""

    user_id = str(response.get("poke") or "").strip()
    candidates = {msg.user_id: msg.user_name for msg in messages_chunk if msg.user_id}
    for msg in messages_chunk:
        candidates.update(msg.mentions)
    name = candidates.get(user_id)
    if not user_id or name is None or name == bot_name:
        return None
    return Poke(user_id=user_id, user_name=name)


def _parse_affinity_changes(response: dict, speaker_ids: set[str]) -> dict[str, int]:
    """Feedback 的 affinity_changes：只收本轮发言群友（不含角色自己）的整数变化量。"""

    raw = response.get("affinity_changes")
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


def _parse_reaction(
    response: dict, messages_chunk: list[Message], bot_name: str
) -> Reaction | None:
    """Feedback 的 react：只能贴这批新消息里群友的消息，表情要在目录里。"""

    raw = response.get("react")
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


class ConversationOrchestrator:
    """一轮对话的编排：短时记忆 → 意愿/相关性 → RAG → Feedback → Chat。"""

    def __init__(self, session: Session):
        self.session = session

    async def process_chunk(
        self,
        messages_chunk: list[Message],
        chat_call: LLMCall,
        feedback_call: LLMCall,
        generation: int,
    ) -> TurnResult | None:
        session = self.session
        try:
            if session.stale(generation, "process_start"):
                return None
            session.record_incoming(messages_chunk)

            state = session.state
            # 没设人设（或预设文件不在）：消息照常记，不判断、不说话
            if not state.name:
                return None
            recent = session.runtime.short_term_memory.access()
            self_share = sum(m.user_name == state.name for m in recent) / len(recent)
            engagement = evaluate_engagement(
                state=state,
                messages=messages_chunk,
                self_share=self_share,
                now=datetime.now(),
            )
            is_relevant = engagement.relevant

            if not engagement.engaged:
                if self._consolidation_due():
                    pending_messages = session.runtime.short_term_memory.messages_after(
                        state.last_consolidated_time,
                        limit=CONSOLIDATION_MAX_MESSAGES,
                    )
                    await self.consolidate_stage(
                        pending_messages, feedback_call, generation
                    )
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
            search_result = await self.search_stage(messages_chunk, use_rerank=True)
            if session.stale(generation, "rag_search"):
                return None

            try:
                decision = await self.feedback_stage(
                    messages_chunk,
                    feedback_call,
                    is_relevant=is_relevant,
                    search_result=search_result,
                    generation=generation,
                )
            finally:
                session.schedule_save()

            if session.stale(generation, "feedback"):
                return None
            if decision is not None:
                self._advance_consolidation_watermark(messages_chunk)

            # 被点名也交给 willing 判断；Feedback 失败时只有被点名才回
            if decision is None:
                speak = is_relevant
            else:
                speak = decision.llm_willing >= engagement.speak_threshold
            log_event(
                "speak_decision",
                session_id=session.id,
                engaged=True,
                relevant=is_relevant,
                in_conversation=engagement.in_conversation,
                llm_willing=decision.llm_willing if decision else None,
                threshold=engagement.speak_threshold,
                self_share=round(self_share, 2),
                speak=speak,
            )
            react = decision.react if decision else None
            poke = decision.poke if decision else None
            if not speak:
                return TurnResult(replies=[], react=react, poke=poke)

            reply_messages = await self.chat_stage(
                messages_chunk,
                chat_call,
                recalled_history=decision.recalled_history if decision else [],
                search_result=search_result,
                generation=generation,
            )
            if session.stale(generation, "chat"):
                return None
            # last_speak_time 在真正发出去之后由 dispatch_replies 更新
            return TurnResult(replies=reply_messages, react=react, poke=poke)
        finally:
            await session.flush_persistence()

    async def search_stage(
        self,
        messages_chunk: list[Message],
        *,
        use_rerank: bool,
    ) -> RetrievalResult:
        started_at = time.perf_counter()
        state = self.session.state
        vector_memory = self.session.runtime.vector_memory

        # Reranker 使用第一条 query 作为主 query，因此必须最新消息优先。
        raw_queries = [msg.content for msg in reversed(messages_chunk[-3:])]
        queries = build_chat_rag_queries(raw_queries, chat_summary=state.chat_summary)
        rag_stats = {
            "session_id": self.session.id,
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
            retrieval = await vector_memory.retrieve_with_decay(
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

    def _history_context(self, messages_chunk: list[Message]) -> list[dict]:
        """短时窗口里除本轮新消息以外的历史，避免 Prompt 上下文重复。"""

        all_messages = self.session.runtime.short_term_memory.access()
        return format_history(_history_without_current_chunk(all_messages, messages_chunk))

    def _speaker_ids(self, messages_chunk: list[Message]) -> list[str]:
        """本轮发言的群友，不含角色自己。"""

        name = self.session.state.name
        return list(
            dict.fromkeys(_user_key(msg) for msg in messages_chunk if msg.user_name != name)
        )

    def _related_profiles(
        self, messages_chunk: list[Message], *, for_chat: bool
    ) -> list[dict]:
        """Feedback 要在当前好感度上打分，给数值；Chat 只需要知道关系，给文字。"""

        state = self.session.state
        profiles = []
        for uid in self._speaker_ids(messages_chunk):
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

    def _advance_consolidation_watermark(self, messages_chunk: list[Message]) -> None:
        state = self.session.state
        latest = max(msg.time for msg in messages_chunk)
        if state.last_consolidated_time is None or latest > state.last_consolidated_time:
            state.last_consolidated_time = latest
        state.messages_since_consolidation = 0
        state.last_consolidation_attempt = datetime.now()
        self.session.schedule_save()

    async def _run_feedback_llm(
        self,
        messages_chunk: list[Message],
        llm_call: LLMCall,
        is_relevant: bool,
        search_result: RetrievalResult,
    ) -> dict | None:
        """运行 Feedback LLM 并返回校验过的输出；失败返回 None。"""

        state = self.session.state
        for uid in self._speaker_ids(messages_chunk):
            state.profiles.setdefault(uid, PersonProfile(user_id=uid))

        now = datetime.now()
        state.global_emotion.settle(now)
        prompt = get_feedback_prompt(
            bot_name=state.name,
            self_life=state.self_life,
            self_in_group=state.self_in_group,
            chat_state_value=chatting_state(state, now).value,
            summary=state.chat_summary,
            recent_msgs=self._history_context(messages_chunk),
            new_msgs=[
                {"id": msg.user_id, "name": msg.user_name, "content": msg.content}
                for msg in messages_chunk
            ],
            emotion=state.global_emotion.vad(),
            related_profiles=self._related_profiles(messages_chunk, for_chat=False),
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
                self.session.runtime.short_term_memory.messages_after(
                    now - timedelta(seconds=GROUP_HEAT_SECONDS),
                    limit=SHORT_TERM_BUFFER_SIZE,
                ),
                state.name,
            ),
        )

        response, failure_reason = parse_feedback(
            await llm_call(prompt), state.global_emotion
        )
        if response is None:
            log_event(
                "feedback_rejected",
                "WARNING",
                session_id=self.session.id,
                failure_reason=failure_reason,
            )
            return None

        expected_fields = [
            "corrections",
            "willing",
            "new_emotion",
            "affinity_changes",
            "summary",
            "need_history",
        ]
        missing_fields = [name for name in expected_fields if name not in response]
        if missing_fields:
            log_event(
                "feedback_fields_missing",
                "WARNING",
                session_id=self.session.id,
                missing_feedback_fields=missing_fields,
                response_keys=sorted(str(key) for key in response),
            )

        return response

    def _apply_sediment(
        self,
        response: dict,
        messages_chunk: list[Message],
        generation: int,
        refs: dict[str, str],
    ) -> None:
        """应用 Feedback 的沉淀结果：情绪、画像、摘要、记忆更正。refs 是本轮检索行编号 → 记忆 id。"""

        state = self.session.state

        # 1. 心情：parse_feedback 已校验范围，apply 再限制每轮步长
        state.global_emotion.apply(response["new_emotion"])

        # 2. 互动计数按消息记；好感度只改 Feedback 点名的人。画像已在 _run_feedback_llm 里建好
        now = datetime.now()
        for msg in messages_chunk:
            if msg.user_name != state.name:
                state.profiles[_user_key(msg)].record_interaction(now)
        speaker_ids = set(self._speaker_ids(messages_chunk))
        for uid, delta in _parse_affinity_changes(response, speaker_ids).items():
            state.profiles[uid].add_affinity(delta)

        # 3. 摘要
        summary = response.get("summary")
        if summary is not None:
            state.chat_summary = str(summary)[:SUMMARY_CHARS]

        # 4. 记忆更正（后台写入）；日常记忆由 memory/episodes.py 在一段聊完后整理
        corrections = response.get("corrections", [])
        if isinstance(corrections, list) and corrections:
            # 上下文里每个 QQ 号的当前群名片：先放被 @/被回复的人，再放发言人（本批在后、以最新为准）
            context = [*self.session.runtime.short_term_memory.access(), *messages_chunk]
            known_users = {uid: name for msg in context for uid, name in msg.mentions.items()}
            known_users.update({msg.user_id: msg.user_name for msg in context if msg.user_id})
            self.session.spawn(
                save_memory_candidates(
                    self.session,
                    corrections,
                    messages=messages_chunk,
                    known_users=known_users,
                    refs=refs,
                    generation=generation,
                )
            )

    async def _apply_decision(
        self,
        response: dict,
        messages_chunk: list[Message],
        is_relevant: bool,
        generation: int,
    ) -> FeedbackDecision | None:
        """应用 Feedback 的发言决策：历史溯源、意愿、贴表情、戳一戳。会话已作废时返回 None。"""

        recalled_history = []

        # 主动历史溯源 (Historical Recall)
        if response.get("need_history", False):
            current_msgs = self.session.runtime.short_term_memory.access()
            if current_msgs:
                recalled_msgs = await get_history_before(
                    self.session.id,
                    current_msgs[0].time,
                    limit=HISTORY_RECALL_LIMIT,
                )
                recalled_history = [
                    f"[{m.time:%H:%M}] {m.user_name}: {m.content}" for m in recalled_msgs
                ]
                logger.debug(
                    f"[Session {self.session.id}] 回溯了 {len(recalled_history)} 条历史消息"
                )

        if self.session.stale(generation, "feedback_decision"):
            return None

        # 说不说由 llm_willing 按场景门槛决定（见 process_chunk）；规则意愿只向它靠拢一半，
        # 让模型觉得不想说时下一批不必再急着调用 Feedback
        state = self.session.state
        try:
            llm_willing = max(0.0, min(1.0, float(response.get("willing"))))
        except (TypeError, ValueError):
            llm_willing = 0.0
        state.willingness = (state.willingness + llm_willing) / 2
        if is_relevant and state.willingness < RELEVANCE_WILLINGNESS_FLOOR:
            state.willingness = RELEVANCE_WILLINGNESS_FLOOR
        return FeedbackDecision(
            recalled_history=recalled_history,
            llm_willing=llm_willing,
            react=_parse_reaction(response, messages_chunk, state.name),
            poke=_parse_poke(response, messages_chunk, state.name),
        )

    async def feedback_stage(
        self,
        messages_chunk: list[Message],
        llm_call: LLMCall,
        *,
        is_relevant: bool,
        search_result: RetrievalResult,
        generation: int,
    ) -> FeedbackDecision | None:
        """反馈阶段：分析情绪、记忆更正、更新摘要、给出接话意愿。
        Feedback 失败或会话已作废时返回 None。"""

        response = await self._run_feedback_llm(
            messages_chunk, llm_call, is_relevant, search_result
        )
        if response is None or self.session.stale(generation, "feedback_sediment"):
            return None
        self._apply_sediment(response, messages_chunk, generation, search_result.refs)
        decision = await self._apply_decision(
            response, messages_chunk, is_relevant, generation
        )
        return decision

    async def consolidate_stage(
        self,
        messages_chunk: list[Message],
        feedback_call: LLMCall,
        generation: int,
    ) -> None:
        """常驻记忆固化：分析+沉淀，但不改回复意愿、不做发言决策。"""

        if not messages_chunk:
            return
        self.session.state.last_consolidation_attempt = datetime.now()
        logger.debug(
            f"[Session {self.session.id}] >> 记忆固化 (Consolidate) {len(messages_chunk)} 条"
        )
        search_result = await self.search_stage(messages_chunk, use_rerank=False)
        if self.session.stale(generation, "consolidation_search"):
            return
        response = await self._run_feedback_llm(
            messages_chunk, feedback_call, False, search_result
        )
        if response is None:
            self.session.schedule_save()
            return
        if self.session.stale(generation, "consolidation_sediment"):
            return
        self._apply_sediment(response, messages_chunk, generation, search_result.refs)
        self._advance_consolidation_watermark(messages_chunk)

    async def chat_stage(
        self,
        messages_chunk: list[Message],
        llm_call: LLMCall,
        *,
        recalled_history: list[str],
        search_result: RetrievalResult,
        generation: int,
    ) -> list:
        state = self.session.state
        new_msgs = [
            {"id": msg.id, "name": msg.user_name, "content": msg.content}
            for msg in messages_chunk
        ]
        recent_msgs = self._history_context(messages_chunk)
        # 自己最近说过的话单独列出来，prompt 要求避免重复句式
        my_recent_replies = [
            m.content
            for m in self.session.runtime.short_term_memory.access()
            if m.user_name == state.name
        ][-MY_RECENT_REPLIES_LIMIT:]
        recalled_str = "\n".join(recalled_history) or "无"
        related_profiles = self._related_profiles(messages_chunk, for_chat=True)
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
            session_id=self.session.id,
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

        replies = parse_reply(await llm_call(prompt))
        if self.session.stale(generation, "chat_reply"):
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

    def _consolidation_due(self) -> bool:
        """攒够消息或距上次尝试足够久，才在不参与时做一次固化。"""

        state = self.session.state
        if state.messages_since_consolidation >= CONSOLIDATION_MESSAGE_THRESHOLD:
            return True
        return (
            state.messages_since_consolidation > 0
            and (datetime.now() - state.last_consolidation_attempt).total_seconds()
            >= CONSOLIDATION_INTERVAL_SECONDS
        )


def parse_feedback(
    response: str, current_emotion: EmotionState
) -> tuple[dict | None, str]:
    """解析 Feedback 输出并校验、限幅 new_emotion；返回 (payload, 失败原因)。"""

    parsed = extract_and_parse_json(response)
    if not isinstance(parsed, dict) or not parsed:
        return None, "invalid_payload"
    raw_emotion = parsed.get("new_emotion")
    if not isinstance(raw_emotion, dict):
        return None, "missing_new_emotion"

    specs = {
        "valence": (-1.0, 1.0, current_emotion.valence),
        "arousal": (0.0, 1.0, current_emotion.arousal),
        "dominance": (-1.0, 1.0, current_emotion.dominance),
    }
    normalized = {}
    valid_fields = 0
    for field_name, (minimum, maximum, default) in specs.items():
        if field_name not in raw_emotion:
            normalized[field_name] = default
            continue
        try:
            value = float(raw_emotion[field_name])
        except (TypeError, ValueError):
            return None, f"invalid_new_emotion_{field_name}"
        if not math.isfinite(value):
            return None, f"invalid_new_emotion_{field_name}"
        normalized[field_name] = max(minimum, min(maximum, value))
        valid_fields += 1
    if valid_fields == 0:
        return None, "empty_new_emotion"

    return {**parsed, "new_emotion": normalized}, ""


def parse_reply(response: str) -> list:
    """解析 Chat 回复列表；裸 list 是模型常见偏差，直接兼容。"""

    payload = extract_and_parse_json(response)
    if isinstance(payload, list):
        logger.warning("LLM 返回了 List 而非 Object，已自动兼容")
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("reply"), list):
        return payload["reply"]
    return []
