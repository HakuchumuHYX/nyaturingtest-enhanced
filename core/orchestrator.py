import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from nonebot import logger

from ..db import get_history_before, message_final_id
from ..domain import EmotionState, PersonProfile
from ..memory.short_term import Message
from ..memory.validation import validate_memory_candidate
from ..memory.vector import (
    RAG_FINAL_K,
    RAG_ITEM_CHARS,
    RAG_MEMORY_CHAR_BUDGET,
    RAG_MERGED_CANDIDATE_CAP,
    RAG_PER_QUERY_RECALL_K,
    RetrievalResult,
    build_chat_rag_queries,
)
from .engagement import (
    RELEVANCE_WILLINGNESS_FLOOR,
    SPEAK_WILLINGNESS_RETAIN_FACTOR,
    chatting_state,
    evaluate_engagement,
)
from .llm import extract_and_parse_json
from .metrics import log_event
from .prompts import (
    SUMMARY_CHARS,
    get_chat_prompt,
    get_feedback_prompt,
    get_time_description,
)
from .session import Session

LLMCall = Callable[[str], Awaitable[str]]

CONSOLIDATION_MESSAGE_THRESHOLD = 8
CONSOLIDATION_INTERVAL_SECONDS = 180.0
CONSOLIDATION_MAX_MESSAGES = 60
HISTORY_RECALL_LIMIT = 20
MY_RECENT_REPLIES_LIMIT = 6
GROUP_HEAT_SECONDS = 600


def _user_key(message: Message) -> str:
    return message.user_id or message.user_name


def _format_history(messages: list[Message]) -> list[dict]:
    return [
        {"time": m.time.strftime("%H:%M"), "name": m.user_name, "content": m.content}
        for m in messages
    ]


def _group_heat(messages: list[Message], bot_name: str, now: datetime) -> str:
    """近 10 分钟群友的消息数和发言人数，给 Feedback 判断群里聊得热不热。"""

    recent = [
        m
        for m in messages
        if m.user_name != bot_name and (now - m.time).total_seconds() <= GROUP_HEAT_SECONDS
    ]
    speakers = {_user_key(m) for m in recent}
    return f"近10分钟 {len(recent)} 条、{len(speakers)} 人在说话"


def _history_without_current_chunk(
    all_messages: list[Message], messages_chunk: list[Message]
) -> list[Message]:
    chunk_ids = {msg.id for msg in messages_chunk if msg.id}
    return [
        m
        for m in all_messages
        if m.id not in chunk_ids and not any(m is chunk_msg for chunk_msg in messages_chunk)
    ]


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


def _parse_memory_candidate(
    item,
    messages: list[Message],
    known_users: dict[str, str],
    refs: dict[str, str],
) -> dict | None:
    """把 LLM 返回的一条 add/correct 候选规范化；ignore、未知 action、非对象返回 None。

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

    def bounded_float(value, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    source = item.get("source")
    sources = [
        messages[index]
        for index in (source if isinstance(source, list) else [])
        if isinstance(index, int) and 0 <= index < len(messages)
    ]
    speaker_user_id = sources[0].user_id if sources else ""
    speaker_user_name = known_users.get(speaker_user_id, "") if sources else ""
    subject_user_id, subject_user_name = _resolve_user(
        str(item.get("subject_user_id") or "").strip(),
        str(item.get("subject_user_name") or "").strip(),
        known_users,
        fallback_id=speaker_user_id,
        speaker_id=speaker_user_id,
    )
    return {
        "action": action,
        "content": str(item.get("content") or "").strip(),
        "category": str(item.get("category") or "event").strip().lower() or "event",
        "confidence": bounded_float(item.get("confidence", 0.7), 0.7),
        "importance": bounded_float(item.get("importance", 0.5), 0.5),
        "subject_user_id": subject_user_id,
        "subject_user_name": subject_user_name,
        "speaker_user_id": speaker_user_id,
        "speaker_user_name": speaker_user_name,
        "source_msg_ids": " ".join(dict.fromkeys(message_final_id(m) for m in sources)),
        "participant_ids": " ".join(
            dict.fromkeys(uid for uid in (subject_user_id, speaker_user_id) if uid)
        ),
        "is_correction": action == "correct",
        # 编号只在本轮有效；查不到就当没指认，只写一条更正
        "replaces": refs.get(str(item.get("target") or "").strip(), "")
        if action == "correct"
        else "",
    }


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
        candidate = _parse_memory_candidate(raw_item, messages, known_users, refs)
        if candidate is None:
            continue

        # 质量过滤：长度 + 类别/置信度/主体边界（先过滤，避免为废候选调 embedding）
        reason = validate_memory_candidate(candidate)
        if not reason and candidate["speaker_user_name"] == session.state.name:
            # 角色自己说的话只是上下文，不当事实来源
            reason = "bot_source"
        if reason:
            skipped_quality += 1
            log_event(
                "memory_candidate_rejected",
                session_id=session.id,
                action=candidate["action"],
                category=candidate["category"],
                reason=reason,
            )
            logger.debug(f"[Memory] 跳过不可靠记忆({reason}): {candidate['content'][:30]}...")
            continue

        pending_memories.append(
            (
                candidate["content"],
                {
                    "category": candidate["category"],
                    "date": today,
                    "subject_user_id": candidate["subject_user_id"],
                    "subject_user_name": candidate["subject_user_name"],
                    "speaker_user_id": candidate["speaker_user_id"],
                    "speaker_user_name": candidate["speaker_user_name"],
                    "confidence": candidate["confidence"],
                    "importance": candidate["importance"],
                    "source_msg_ids": candidate["source_msg_ids"],
                    "participant_ids": candidate["participant_ids"],
                    "is_correction": candidate["is_correction"],
                    "replaces": candidate["replaces"],
                },
            )
        )

    store_result = {"added": 0, "skipped_dedup": 0, "corrected": 0}
    if pending_memories:
        store_result = await session.runtime.vector_memory.add_memories_with_dedup(
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


@dataclass(frozen=True)
class FeedbackDecision:
    recalled_history: list[str]
    llm_willing: float


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
    ) -> list | None:
        session = self.session
        try:
            if session.stale(generation, "process_start"):
                return None
            session.record_incoming(messages_chunk)

            state = session.state
            recent = session.runtime.short_term_memory.access()
            self_share = sum(m.user_name == state.name for m in recent) / len(recent)
            engagement = evaluate_engagement(
                state=state,
                messages=messages_chunk,
                self_share=self_share,
                now=datetime.now(),
            )
            is_relevant = engagement.relevant
            if is_relevant:
                logger.info("检测到强关联，意愿值提升")

            if not engagement.engaged:
                if self._consolidation_due():
                    pending_messages = session.runtime.short_term_memory.messages_after(
                        state.last_consolidated_time,
                        limit=CONSOLIDATION_MAX_MESSAGES,
                    )
                    await self.consolidate_stage(
                        pending_messages, feedback_call, generation
                    )
                logger.debug(
                    f"未进入参与态 (意愿 {state.willingness:.2f}, 自己发言占比 {self_share:.0%})，跳过响应"
                )
                return None

            # 要说话的轮次都重排：不重排时只能按向量分取前 k，闲聊上它分不出相关和无关
            search_result = await self.search_stage(messages_chunk, use_rerank=True)
            if session.stale(generation, "rag_search"):
                return None

            logger.debug("启用拟人化串行模式: Feedback -> Check -> Chat")
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

            # Feedback 失败时只有被点名才回
            speak = is_relevant or (
                decision is not None
                and decision.llm_willing >= engagement.speak_threshold
            )
            log_event(
                "speak_decision",
                session_id=session.id,
                relevant=is_relevant,
                in_conversation=engagement.in_conversation,
                llm_willing=decision.llm_willing if decision else None,
                threshold=engagement.speak_threshold,
                self_share=round(self_share, 2),
                speak=speak,
            )
            if not speak:
                return None

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
            return reply_messages
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
            "queries_preview": [q[:40] for q in queries[:3]],
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
            logger.debug(f"触发长期记忆检索: {queries[:5]}...")
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
        log_event("rag_search", **rag_stats)
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
        return _format_history(_history_without_current_chunk(all_messages, messages_chunk))

    def _related_profiles(self, messages_chunk: list[Message]) -> list[dict]:
        state = self.session.state
        return [
            {
                "user_id": uid,
                "emotion_tends_to_user": asdict(
                    state.profiles.get(uid, PersonProfile(user_id=uid)).emotion
                ),
                "summary": state.user_summaries.get(uid, ""),
            }
            for uid in dict.fromkeys(_user_key(msg) for msg in messages_chunk)
        ]

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
        for uid in dict.fromkeys(_user_key(msg) for msg in messages_chunk):
            state.profiles.setdefault(uid, PersonProfile(user_id=uid))

        now = datetime.now()
        prompt = get_feedback_prompt(
            bot_name=state.name,
            role=state.role,
            chat_state_value=chatting_state(state, now).value,
            summary=state.chat_summary,
            recent_msgs=self._history_context(messages_chunk),
            new_msgs=[
                {"id": msg.user_id, "name": msg.user_name, "content": msg.content}
                for msg in messages_chunk
            ],
            emotion=asdict(state.global_emotion),
            related_profiles=self._related_profiles(messages_chunk),
            search_result=search_result.memory_lines,
            is_relevant=is_relevant,
            time_info=get_time_description(now),
            presets=search_result.preset_lines,
            group_notes=state.group_notes,
            new_msg_speakers=[
                {"index": index, "user_id": msg.user_id, "user_name": msg.user_name}
                for index, msg in enumerate(messages_chunk)
            ],
            group_heat=_group_heat(
                self.session.runtime.short_term_memory.access(), state.name, now
            ),
        )

        response, failure_reason = parse_feedback(
            await llm_call(prompt), state.global_emotion
        )
        if response is None:
            log_event(
                "feedback_rejected",
                session_id=self.session.id,
                failure_reason=failure_reason,
            )
            return None

        expected_fields = [
            "corrections",
            "willing",
            "new_emotion",
            "emotion_tends",
            "summary",
            "need_history",
        ]
        missing_fields = [name for name in expected_fields if name not in response]
        if missing_fields:
            log_event(
                "feedback_fields_missing",
                session_id=self.session.id,
                missing_feedback_fields=missing_fields,
                response_keys=sorted(str(key) for key in response),
            )

        return response

    def _apply_image_observations(
        self,
        response: dict,
        messages_chunk: list[Message],
    ) -> None:
        """把 Feedback 对图片的一句话观察写回消息文本，作为历史里的图片痕迹。"""

        raw_observations = response.get("image_observations", [])
        if not isinstance(raw_observations, list):
            return
        observations_by_ref = {
            str(item.get("image_ref") or ""): item
            for item in raw_observations
            if isinstance(item, dict) and str(item.get("image_ref") or "")
        }
        if not observations_by_ref:
            return

        for msg in messages_chunk:
            labels = []
            for image_input in msg.image_inputs:
                observation = observations_by_ref.get(image_input.ref_id)
                if not observation:
                    continue
                summary = str(
                    observation.get("summary") or observation.get("description") or ""
                ).strip()
                if not summary:
                    continue
                label = "表情包" if image_input.is_sticker else "图片"
                labels.append(f"[{label}: {summary[:80]}]")
            if not labels:
                continue
            content = msg.content.replace("[表情包]", "").replace("[图片]", "").strip()
            msg.content = f"{content}{''.join(labels)}".strip()
            self.session.runtime.short_term_memory.mark_dirty(msg)

    def _apply_sediment(
        self,
        response: dict,
        messages_chunk: list[Message],
        generation: int,
        refs: dict[str, str],
    ) -> None:
        """应用 Feedback 的沉淀结果：情绪、画像、摘要、记忆更正。refs 是本轮检索行编号 → 记忆 id。"""

        state = self.session.state

        # 1. 情绪：parse_feedback 已校验并限幅
        state.global_emotion = EmotionState(**response["new_emotion"])

        # 2. 用户印象：emotion_tends 与新消息逐条对应
        emo_tends = response.get("emotion_tends", [])
        interaction_updates: list[tuple[str, dict]] = []
        if isinstance(emo_tends, list):
            for msg, raw_delta in zip(messages_chunk, emo_tends):
                if isinstance(raw_delta, (int, float)):
                    delta = {
                        "valence": float(raw_delta),
                        "arousal": abs(float(raw_delta)) * 0.5,
                        "dominance": 0.0,
                    }
                elif isinstance(raw_delta, dict) and raw_delta:
                    delta = raw_delta
                else:
                    continue
                uid = _user_key(msg)
                profile = state.profiles.get(uid)
                if profile is None:
                    continue
                profile.push_interaction(delta)
                interaction_updates.append((uid, delta))

        if interaction_updates:
            self.session.spawn(
                self.session.save_interaction_logs(interaction_updates, generation)
            )

        for profile in state.profiles.values():
            profile.update_emotion_tends()

        # 3. 摘要
        summary = response.get("summary")
        if summary is not None:
            state.chat_summary = str(summary)[:SUMMARY_CHARS]

        # 4. 记忆更正（后台写入）；日常记忆由 core/episodes.py 在一段聊完后整理
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
        is_relevant: bool,
        generation: int,
    ) -> FeedbackDecision | None:
        """应用 Feedback 的发言决策：历史溯源、意愿。会话已作废时返回 None。"""

        recalled_history = []

        # 主动历史溯源 (Historical Recall)
        if response.get("need_history", False):
            logger.info(f"[Session {self.session.id}] 观察者请求翻阅历史记录...")
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
                if recalled_history:
                    logger.info(
                        f"[Session {self.session.id}] 成功回溯了 {len(recalled_history)} 条历史消息"
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
        return FeedbackDecision(recalled_history=recalled_history, llm_willing=llm_willing)

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

        logger.debug(">> 反馈阶段 (Feedback) 开始")
        response = await self._run_feedback_llm(
            messages_chunk, llm_call, is_relevant, search_result
        )
        if response is None or self.session.stale(generation, "feedback_sediment"):
            return None
        self._apply_image_observations(response, messages_chunk)
        self._apply_sediment(response, messages_chunk, generation, search_result.refs)
        decision = await self._apply_decision(response, is_relevant, generation)
        logger.debug(f"<< 反馈结束: 意愿 {self.session.state.willingness:.2f}")
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
        self._apply_image_observations(response, messages_chunk)
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
        logger.debug(">> 对话阶段 (Chat) 开始")
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
        related_profiles = self._related_profiles(messages_chunk)
        # role 里可能带 [对话样本] 后缀（examples 的持久化位置），单独用 examples_text 传
        chat_role = state.role.split("[对话样本]")[0].strip()
        prompt = get_chat_prompt(
            bot_name=state.name,
            role=chat_role,
            chat_state_value=chatting_state(state, datetime.now()).value,
            summary=state.chat_summary,
            recent_msgs=recent_msgs,
            new_msgs=new_msgs,
            emotion=asdict(state.global_emotion),
            related_profiles=related_profiles,
            search_result=search_result.memory_lines,
            examples_text=state.examples,
            presets=search_result.preset_lines,
            group_notes=state.group_notes,
            recalled_history=recalled_str,
            my_recent_replies=my_recent_replies,
            time_info=get_time_description(datetime.now()),
        )
        log_event(
            "rag_prompt_budget",
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
            examples_chars=len(state.examples),
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
