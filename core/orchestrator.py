import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from nonebot import logger
from nonebot.utils import run_sync

from ..db import get_history_before
from ..domain import EmotionState, Impression, PersonProfile, clamp_vad_value
from ..memory.short_term import Message
from ..memory.validation import validate_memory_candidate
from ..memory.vector import (
    RAG_DEBUG_LOG,
    RAG_DEFAULT_EVENT_TTL_DAYS,
    RAG_FINAL_K,
    RAG_MEMORY_CHAR_BUDGET,
    RAG_MERGED_CANDIDATE_CAP,
    RAG_PER_QUERY_RECALL_K,
    RetrievalResult,
    build_chat_rag_queries,
    search_memories,
    where_any,
)
from .engagement import (
    ACTIVE_TO_BUBBLE_THRESHOLD,
    LOW_WILLINGNESS_SKIP_THRESHOLD,
    POST_FEEDBACK_SKIP_THRESHOLD,
    RELEVANCE_WILLINGNESS_FLOOR,
    RERANK_WILLINGNESS_THRESHOLD,
    SPEAK_WILLINGNESS_RETAIN_FACTOR,
    evaluate_engagement,
)
from .llm import extract_and_parse_json
from .metrics import log_event
from .prompts import (
    PromptBudget,
    get_chat_prompt,
    get_feedback_prompt,
    get_time_description,
)
from .session import STALE_GENERATION_WRITE, ChattingState, FeedbackOutcome

CONSOLIDATION_ENABLED = True
CONSOLIDATION_MESSAGE_THRESHOLD = 8
CONSOLIDATION_INTERVAL_SECONDS = 180.0
CONSOLIDATION_MAX_MESSAGES = 60
HISTORY_RECALL_LIMIT = 20


def _history_without_current_chunk(
    all_messages: list[Message], messages_chunk: list[Message]
) -> list[Message]:
    chunk_message_ids = {str(msg.id) for msg in messages_chunk if msg.id}
    return [
        m
        for m in all_messages
        if not (
            (m.id and str(m.id) in chunk_message_ids)
            or any(m is chunk_msg for chunk_msg in messages_chunk)
        )
    ]


@dataclass
class _FeedbackContext:
    response_dict: dict
    existing_related_memories: list[dict]
    allow_memory_supersede: bool
    active_user_ids: set[str]


def _rag_debug_records(records: list[dict]) -> list[dict]:
    debug_items = []
    for item in records:
        content = item.get("content", "")
        meta = item.get("metadata", {}) or {}
        debug_items.append(
            {
                "source": meta.get("source"),
                "type": meta.get("type"),
                "subtype": meta.get("subtype"),
                "retrieval_score": meta.get("retrieval_score"),
                "rerank_score": meta.get("rerank_score"),
                "adjusted_score": meta.get("adjusted_score"),
                "days_ago": meta.get("days_ago"),
                "content_preview": str(content)[:80],
            }
        )
    return debug_items


def _existing_related_memories(
    raw_records: list[dict] | None,
    active_user_ids: set[str],
    *,
    limit: int = 5,
) -> list[dict]:
    related = []
    for item in raw_records or []:
        content = str(item.get("content") or "")
        meta = item.get("metadata", {}) or {}
        if not content or str(meta.get("source") or "memory") != "memory":
            continue
        if str(meta.get("status") or "active") != "active":
            continue
        subject_user_id = str(meta.get("subject_user_id") or "").strip()
        if subject_user_id and subject_user_id not in active_user_ids:
            continue

        memory_ref = str(meta.get("memory_ref") or "").strip()
        if not memory_ref:
            continue

        entry = {
            "content_preview": content[:80],
            "source": str(meta.get("source") or "memory"),
            "type": str(meta.get("type") or "event"),
            "subtype": str(meta.get("subtype") or meta.get("type") or "event"),
            "category": str(meta.get("category") or meta.get("type") or "event"),
            "confidence": meta.get("confidence", 1.0),
            "subject_user_id": subject_user_id,
            "subject_user_name": str(meta.get("subject_user_name") or ""),
            "speaker_user_id": str(meta.get("speaker_user_id") or ""),
            "speaker_user_name": str(meta.get("speaker_user_name") or ""),
        }
        entry["memory_ref"] = memory_ref
        related.append(entry)
        if len(related) >= limit:
            break
    return related


class ConversationOrchestrator:
    """一轮对话的编排：短时记忆 → 意愿/相关性 → RAG → Feedback → Chat。"""

    def __init__(self, session):
        self.session = session

    async def process_chunk(
        self,
        messages_chunk: list[Message],
        chat_llm_func: Callable[[str, bool], Awaitable[str]],
        feedback_llm_func: Callable[[str, bool], Awaitable[str]],
        publish: bool = True,
        expected_generation: int | None = None,
    ) -> list[dict] | None:
        try:
            session = self.session
            state = session.state

            if session.is_generation_stale(expected_generation):
                session._log_stale_generation("process_start", expected_generation)
                return None

            await session.record_incoming(messages_chunk)
            if session.is_generation_stale(expected_generation):
                session._log_stale_generation("short_term_memory", expected_generation)
                return None

            if not publish:
                return None

            now = datetime.now()
            engagement = evaluate_engagement(
                state=state,
                messages=messages_chunk,
                now=now,
            )
            is_relevant = engagement.relevant
            if is_relevant:
                logger.info("检测到强关联，意愿值提升")

            if not engagement.engaged and not is_relevant:
                if CONSOLIDATION_ENABLED and self._consolidation_due():
                    state.last_consolidation_attempt = datetime.now()
                    pending_messages = session.runtime.short_term_memory.messages_after(
                        state.last_consolidated_time,
                        limit=CONSOLIDATION_MAX_MESSAGES,
                    )
                    await self.consolidate_stage(
                        pending_messages,
                        feedback_llm_func,
                        expected_generation=expected_generation,
                    )
                logger.debug(f"未进入参与态 (意愿 {state.willingness:.2f})，跳过响应")
                return None

            if engagement.cooldown_remaining > 0 and not is_relevant:
                logger.debug(
                    f"处于发言冷却期（剩余 {engagement.cooldown_remaining:.1f}s），"
                    "跳过响应"
                )
                return None

            # Reranker 使用第一条 query 作为主 query，因此必须最新消息优先。
            queries = [msg.content for msg in reversed(messages_chunk[-3:])]
            active_user_names = [
                msg.user_name for msg in messages_chunk if msg.user_name
            ]
            active_users = [
                {
                    "user_id": str(msg.user_id or ""),
                    "user_name": msg.user_name,
                }
                for msg in messages_chunk
                if msg.user_name
            ]
            use_rerank_strategy = (
                state.willingness > RERANK_WILLINGNESS_THRESHOLD or is_relevant
            )

            search_result = await self.search_stage(
                queries,
                active_user_names=active_user_names,
                active_users=active_users,
                use_rerank=use_rerank_strategy,
            )
            if session.is_generation_stale(expected_generation):
                session._log_stale_generation("rag_search", expected_generation)
                return None

            logger.debug("启用拟人化串行模式: Feedback -> Check -> Chat")

            try:
                feedback_outcome = await self.feedback_stage(
                    messages_chunk,
                    feedback_llm_func,
                    is_relevant=is_relevant,
                    search_result=search_result,
                    expected_generation=expected_generation,
                )
            finally:
                session._schedule_save_session()

            if session.is_generation_stale(expected_generation):
                session._log_stale_generation("feedback", expected_generation)
                return None
            if feedback_outcome.accepted:
                latest = max((msg.time for msg in messages_chunk), default=None)
                if latest is not None and (
                    state.last_consolidated_time is None
                    or latest > state.last_consolidated_time
                ):
                    state.last_consolidated_time = latest
                state.messages_since_consolidation = 0
                state.last_consolidation_attempt = datetime.now()
                session._schedule_save_session()

            if state.willingness < POST_FEEDBACK_SKIP_THRESHOLD and not is_relevant:
                return None

            reply_messages = await self.chat_stage(
                messages_chunk,
                chat_llm_func,
                recalled_history=feedback_outcome.recalled_history,
                search_result=search_result,
                expected_generation=expected_generation,
            )
            if session.is_generation_stale(expected_generation):
                session._log_stale_generation("chat", expected_generation)
                return None

            if reply_messages:
                state.last_speak_time = datetime.now()

            return reply_messages
        finally:
            await self.session.flush_persistence()

    async def search_stage(
        self,
        queries: list[str],
        active_user_names: list[str] | None = None,
        *,
        active_users: list[dict] | None = None,
        use_rerank: bool = True,
        force_retrieve: bool = False,
    ):
        """
        优化检索阶段
        """
        started_at = time.perf_counter()
        logger.debug(f"检索阶段开始 (Use Rerank: {use_rerank})")
        rag_stats = {
            "session_id": self.session.id,
            "query_count": 0,
            "queries_preview": [],
            "use_rerank": bool(use_rerank),
            "skip_reason": "none",
            "fallback_reason": "none",
            "candidate_count": 0,
            "returned_count": 0,
            "injected_count": 0,
            "injected_chars": 0,
            "elapsed_ms": 0,
            "adjusted_score_min": None,
            "adjusted_score_p50": None,
            "adjusted_score_p90": None,
            "adjusted_score_max": None,
            "other_subject_downweighted_count": 0,
            "legacy_subject_count": 0,
            "scope_counts": {},
        }

        active_scope_user_ids = {
            str(user.get("user_id") or "").strip()
            for user in active_users or []
            if str(user.get("user_id") or "").strip()
        }
        queries = build_chat_rag_queries(
            queries,
            chat_summary=self.session.state.chat_summary,
            active_user_names=active_user_names,
            active_users=active_users,
        )
        rag_stats["query_count"] = len(queries)
        rag_stats["queries_preview"] = [q[:40] for q in queries[:3]]

        should_retrieve = (
            force_retrieve
            or self.session.state.willingness > LOW_WILLINGNESS_SKIP_THRESHOLD
        )

        long_term_memory = []
        raw_results = []
        search_result = RetrievalResult(records=[], stats=rag_stats)
        try:
            if not queries:
                rag_stats["skip_reason"] = "no_queries"
            elif not should_retrieve:
                rag_stats["skip_reason"] = "low_willingness"
            else:
                logger.debug(f"触发长期记忆检索: {queries[:5]}...")

                where_filter = where_any("source", ["preset", "memory"])

                retrieval_result = await search_memories(
                    self.session.runtime.vector_memory,
                    queries,
                    k=RAG_FINAL_K,
                    where=where_filter,
                    use_rerank=use_rerank,
                    candidate_k=RAG_PER_QUERY_RECALL_K,
                    merged_candidate_cap=RAG_MERGED_CANDIDATE_CAP,
                    active_user_ids=active_scope_user_ids,
                )
                raw_results = retrieval_result.records
                retrieval_stats = retrieval_result.stats
                rag_stats.update(
                    {
                        "candidate_count": int(
                            retrieval_stats.get("candidate_count") or 0
                        ),
                        "returned_count": int(
                            retrieval_stats.get("returned_count") or len(raw_results)
                        ),
                        "fallback_reason": str(
                            retrieval_stats.get("fallback_reason") or "none"
                        ),
                        "other_subject_downweighted_count": int(
                            retrieval_stats.get("other_subject_downweighted_count") or 0
                        ),
                        "legacy_subject_count": int(
                            retrieval_stats.get("legacy_subject_count") or 0
                        ),
                        "scope_counts": dict(retrieval_stats.get("scope_counts") or {}),
                        "adjusted_score_min": retrieval_stats.get("adjusted_score_min"),
                        "adjusted_score_p50": retrieval_stats.get("adjusted_score_p50"),
                        "adjusted_score_p90": retrieval_stats.get("adjusted_score_p90"),
                        "adjusted_score_max": retrieval_stats.get("adjusted_score_max"),
                    }
                )

                if raw_results:
                    formatted_results = []
                    total_len = 0
                    max_len = RAG_MEMORY_CHAR_BUDGET

                    for item in raw_results:
                        content = item.get("content", "")
                        meta = item.get("metadata", {})
                        source = meta.get("source", "unknown")
                        date_str = str(meta.get("date", ""))

                        if source == "preset":
                            subtype = str(meta.get("subtype") or "legacy_rule")
                            prefix = f"【设定/{subtype}】"
                        else:
                            prefix = f"【记忆/d:{date_str}】"
                        line = f"{prefix} {content}"

                        remaining = max_len - total_len
                        if remaining <= 0:
                            break
                        if len(line) > remaining:
                            line = line[:remaining].rstrip()
                        if not line:
                            break
                        formatted_results.append(line)
                        total_len += len(line)

                    long_term_memory = formatted_results
                    rag_stats["injected_count"] = len(long_term_memory)
                    rag_stats["injected_chars"] = sum(
                        len(item) for item in long_term_memory
                    )
                    logger.debug(f"搜索结果：命中 {len(long_term_memory)} 条")
        finally:
            rag_stats["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
            if RAG_DEBUG_LOG:
                rag_stats["result_debug"] = _rag_debug_records(raw_results)
            log_event("rag_search", **rag_stats)
            search_result = RetrievalResult(
                records=raw_results,
                stats=rag_stats,
                prompt_lines=long_term_memory,
            )
        return search_result

    async def _run_feedback_llm(
        self,
        messages_chunk: list[Message],
        llm_func: Callable,
        is_relevant: bool = False,
        search_result: RetrievalResult | None = None,
    ) -> tuple[_FeedbackContext | None, str]:
        """运行 Feedback LLM 并返回可复用的分析上下文。"""
        reaction_users = list(
            {msg.user_id if msg.user_id else msg.user_name for msg in messages_chunk}
        )
        related_profiles = [
            self.session.state.profiles.get(uid, PersonProfile(user_id=uid))
            for uid in reaction_users
        ]
        for p in related_profiles:
            if p.user_id not in self.session.state.profiles:
                self.session.state.profiles[p.user_id] = p

        related_profiles_data = [
            {"user_id": p.user_id, "emotion_tends_to_user": asdict(p.emotion)}
            for p in related_profiles
        ]
        search_history = search_result.prompt_lines if search_result else []
        active_user_ids = {
            str(msg.user_id)
            for msg in messages_chunk
            if msg.user_id and str(msg.user_id).strip()
        }
        existing_related_memories = _existing_related_memories(
            search_result.records if search_result else [],
            active_user_ids,
        )
        allow_memory_supersede = any(
            item.get("memory_ref") for item in existing_related_memories
        )

        formatted_msgs = [
            {
                "id": str(msg.user_id or ""),
                "name": msg.user_name,
                "content": msg.content,
            }
            for msg in messages_chunk
        ]
        new_msg_speakers = [
            {
                "index": index,
                "user_id": str(msg.user_id or ""),
                "user_name": msg.user_name,
            }
            for index, msg in enumerate(messages_chunk)
        ]

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        context_record = self.session.runtime.short_term_memory.access()
        all_messages = context_record.messages
        history_msgs = _history_without_current_chunk(all_messages, messages_chunk)
        # 历史消息格式化为结构化 dict
        history_msgs_formatted = [
            {
                "time": m.time.strftime("%H:%M"),
                "name": m.user_name,
                "content": m.content,
            }
            for m in history_msgs
        ]

        # 2. 调用 LLM (使用传入的 feedback_llm_func)
        time_str = get_time_description(datetime.now())
        prompt = get_feedback_prompt(
            self.session.state.name,
            self.session.state.role,
            self.session.state.willingness,
            self.session.state.chatting_state.value,
            context_record.compressed_history,
            history_msgs_formatted,  # 传入格式化后的历史
            formatted_msgs,
            asdict(self.session.state.global_emotion),
            related_profiles_data,
            search_history,
            self.session.state.chat_summary,
            is_relevant=is_relevant,
            time_info=time_str,
            existing_related_memories=existing_related_memories,
            allow_memory_supersede=allow_memory_supersede,
            new_msg_speakers=new_msg_speakers,
            budget=PromptBudget(),
            has_images=any(message.image_inputs for message in messages_chunk),
        )

        try:
            response = await llm_func(prompt, json_mode=True)
        except Exception as e:
            logger.error(f"反馈阶段 LLM 错误，跳过本次处理: {e}")
            return None, "llm_error"

        parsed_feedback = parse_feedback(response, self.session.state.global_emotion)
        response_dict = parsed_feedback.payload
        if response_dict is None:
            log_event(
                "feedback_rejected",
                session_id=self.session.id,
                failure_reason=parsed_feedback.failure_reason,
            )
            return None, parsed_feedback.failure_reason

        expected_feedback_fields = [
            "analyze_result",
            "willing",
            "new_emotion",
            "emotion_tends",
            "summary",
            "need_history",
        ]
        missing_feedback_fields = [
            field for field in expected_feedback_fields if field not in response_dict
        ]
        if missing_feedback_fields:
            log_event(
                "feedback_fields_missing",
                session_id=self.session.id,
                missing_feedback_fields=missing_feedback_fields,
                response_keys=sorted(str(key) for key in response_dict.keys()),
            )

        return (
            _FeedbackContext(
                response_dict=response_dict,
                existing_related_memories=existing_related_memories,
                allow_memory_supersede=allow_memory_supersede,
                active_user_ids=active_user_ids,
            ),
            "",
        )

    def _apply_image_observations(
        self,
        response_dict: dict,
        messages_chunk: list[Message],
    ) -> None:
        """把 Feedback 对图片的一句话观察写回消息文本，作为历史里的图片痕迹。"""

        raw_observations = response_dict.get("image_observations", [])
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
            if not msg.image_inputs:
                continue
            labels = []
            for image_input in msg.image_inputs:
                observation = observations_by_ref.get(str(image_input.ref_id or ""))
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
        ctx: _FeedbackContext,
        messages_chunk: list[Message],
        expected_generation: int | None = None,
    ) -> None:
        """应用 Feedback 的沉淀结果：情绪、画像、摘要、长期记忆。"""
        response_dict = ctx.response_dict

        # 3. 更新情绪
        new_emo = response_dict.get("new_emotion", {})
        if not new_emo:
            logger.warning(
                f"[Session {self.session.id}] Feedback 未返回 new_emotion，跳过情绪更新。response_dict keys: {list(response_dict.keys())}"
            )
        else:
            logger.debug(
                f"[Session {self.session.id}] Feedback 返回 emotion: V={new_emo.get('valence')}, A={new_emo.get('arousal')}, D={new_emo.get('dominance')}"
            )
            self.session.state.global_emotion.valence = clamp_vad_value(
                new_emo.get("valence"),
                -1.0,
                1.0,
                self.session.state.global_emotion.valence,
            )
            self.session.state.global_emotion.arousal = clamp_vad_value(
                new_emo.get("arousal"),
                0.0,
                1.0,
                self.session.state.global_emotion.arousal,
            )
            self.session.state.global_emotion.dominance = clamp_vad_value(
                new_emo.get("dominance"),
                -1.0,
                1.0,
                self.session.state.global_emotion.dominance,
            )

        # 4. 更新用户印象
        emo_tends = response_dict.get("emotion_tends", [])
        interaction_updates: list[tuple[str, dict]] = []
        if isinstance(emo_tends, list):
            for i, msg in enumerate(messages_chunk):
                if i >= len(emo_tends):
                    break
                uid = msg.user_id if msg.user_id else msg.user_name
                raw_delta = emo_tends[i]

                delta = {}
                if isinstance(raw_delta, (int, float)):
                    delta = {
                        "valence": float(raw_delta),
                        "arousal": abs(float(raw_delta)) * 0.5,
                        "dominance": 0.0,
                    }
                elif isinstance(raw_delta, dict):
                    delta = raw_delta

                if uid in self.session.state.profiles and delta:
                    self.session.state.profiles[uid].push_interaction(
                        Impression(timestamp=datetime.now().astimezone(), delta=delta)
                    )
                    interaction_updates.append((uid, delta))

        if interaction_updates:
            self.session._create_safe_task(
                self.session._save_interaction_logs(
                    interaction_updates,
                    expected_generation=expected_generation,
                )
            )

        for p in self.session.state.profiles.values():
            p.update_emotion_tends()
            p.merge_old_interactions()

        # 5. 更新摘要
        summary = response_dict.get("summary")
        if summary is not None:
            prompt_budget = PromptBudget()
            self.session.state.chat_summary = str(summary)[
                : prompt_budget.summary_chars
            ]
        # 同步更新到 Memory，确保下一次 Prompt 使用最新摘要
        self.session.runtime.short_term_memory.update_summary(
            self.session.state.chat_summary
        )

        # 6. 记忆提取
        analyze_result = response_dict.get("analyze_result", [])
        if isinstance(analyze_result, list) and analyze_result:
            unique_user_ids = {
                str(msg.user_id)
                for msg in messages_chunk
                if msg.user_id and str(msg.user_id).strip()
            }
            fallback_uid = list(unique_user_ids)[0] if len(unique_user_ids) == 1 else ""

            self.session._create_safe_task(
                self.save_long_term_memory(
                    analyze_result,
                    default_user_id=fallback_uid,
                    supersede_candidates=ctx.existing_related_memories,
                    expected_generation=expected_generation,
                )
            )

    async def _apply_decision(
        self,
        ctx: _FeedbackContext,
        messages_chunk: list[Message],
        is_relevant: bool,
        expected_generation: int | None = None,
    ) -> list[str]:
        """应用 Feedback 的发言决策结果：历史溯源、意愿、状态。"""
        response_dict = ctx.response_dict
        recalled_history = []

        # 6.5 主动历史溯源 (Historical Recall)
        need_history = response_dict.get("need_history", False)
        if need_history:
            logger.info(f"[Session {self.session.id}] 观察者请求翻阅历史记录...")
            current_msgs = self.session.runtime.short_term_memory.access().messages
            if current_msgs:
                earliest_time = current_msgs[0].time
                # 使用 Repository 查库
                recalled_msgs = await get_history_before(
                    self.session.id,
                    earliest_time,
                    limit=HISTORY_RECALL_LIMIT,
                )

                if recalled_msgs:
                    formatted_history = []
                    for m in recalled_msgs:
                        time_str = m.time.strftime("%H:%M")
                        formatted_history.append(
                            f"[{time_str}] {m.user_name}: {m.content}"
                        )

                    recalled_history = formatted_history
                    logger.info(
                        f"[Session {self.session.id}] 成功回溯了 {len(formatted_history)} 条历史消息"
                    )

        if self.session.is_generation_stale(expected_generation):
            self.session._log_stale_generation("feedback_decision", expected_generation)
            return []

        # 7. 更新意愿值 (带强关联兜底)
        try:
            new_willing = float(
                response_dict.get("willing", self.session.state.willingness)
            )
            self.session.state.willingness = max(0.0, min(1.0, new_willing))
            relevance_floor = RELEVANCE_WILLINGNESS_FLOOR
            if is_relevant and self.session.state.willingness < relevance_floor:
                self.session.state.willingness = relevance_floor
                logger.debug(
                    f"[Session {self.session.id}] 强关联强制提升意愿值至 {relevance_floor:.2f}"
                )
        except:
            pass

        # 8. 状态流转
        random_threshold = random.uniform(0.4, 0.7)
        if self.session.state.willingness < 0.2:
            self.session.state.chatting_state = ChattingState.IDLE
        elif (
            self.session.state.chatting_state == ChattingState.ACTIVE
            and self.session.state.willingness < ACTIVE_TO_BUBBLE_THRESHOLD
        ):
            self.session.state.chatting_state = ChattingState.BUBBLE
        elif self.session.state.willingness > random_threshold:
            if self.session.state.chatting_state == ChattingState.IDLE:
                self.session.state.chatting_state = ChattingState.BUBBLE

        return recalled_history

    async def feedback_stage(
        self,
        messages_chunk: list[Message],
        llm_func: Callable,
        is_relevant: bool = False,
        search_result: RetrievalResult | None = None,
        expected_generation: int | None = None,
    ) -> FeedbackOutcome:
        """
        反馈阶段：分析情绪、提取记忆、更新摘要
        返回：recalled_history (溯源到的历史消息列表)
        """
        logger.debug(">> 反馈阶段 (Feedback) 开始")
        ctx, failure_reason = await self._run_feedback_llm(
            messages_chunk,
            llm_func,
            is_relevant,
            search_result,
        )
        if ctx is None:
            return FeedbackOutcome.rejected(failure_reason)
        if self.session.is_generation_stale(expected_generation):
            self.session._log_stale_generation("feedback_sediment", expected_generation)
            return FeedbackOutcome.rejected("stale_generation")
        self._apply_image_observations(ctx.response_dict, messages_chunk)
        self._apply_sediment(
            ctx, messages_chunk, expected_generation=expected_generation
        )
        recalled_history = await self._apply_decision(
            ctx, messages_chunk, is_relevant, expected_generation
        )
        if self.session.is_generation_stale(expected_generation):
            return FeedbackOutcome.rejected("stale_generation")
        logger.debug(
            f"<< 反馈结束: 意愿 {self.session.state.willingness:.2f}, 状态 {self.session.state.chatting_state}"
        )
        return FeedbackOutcome(
            accepted=True,
            recalled_history=recalled_history,
        )

    async def consolidate_stage(
        self,
        messages_chunk: list[Message],
        feedback_llm_func: Callable,
        expected_generation: int | None = None,
    ) -> FeedbackOutcome:
        """常驻记忆固化：分析+沉淀，但不改回复意愿、不做发言决策。"""
        if not messages_chunk:
            return FeedbackOutcome.rejected("no_messages")
        self.session.state.last_consolidation_attempt = datetime.now()
        logger.debug(
            f"[Session {self.session.id}] >> 记忆固化 (Consolidate) {len(messages_chunk)} 条"
        )
        queries = [m.content for m in reversed(messages_chunk[-3:])]
        active_user_names = [m.user_name for m in messages_chunk if m.user_name]
        active_users = [
            {"user_id": str(m.user_id or ""), "user_name": m.user_name}
            for m in messages_chunk
            if m.user_name
        ]
        search_result = await self.search_stage(
            queries,
            active_user_names=active_user_names,
            active_users=active_users,
            use_rerank=False,
            force_retrieve=True,
        )
        if self.session.is_generation_stale(expected_generation):
            self.session._log_stale_generation(
                "consolidation_search", expected_generation
            )
            return FeedbackOutcome.rejected("stale_generation")
        ctx, failure_reason = await self._run_feedback_llm(
            messages_chunk,
            feedback_llm_func,
            is_relevant=False,
            search_result=search_result,
        )
        if ctx is None:
            self.session._schedule_save_session()
            return FeedbackOutcome.rejected(failure_reason)
        if self.session.is_generation_stale(expected_generation):
            self.session._log_stale_generation(
                "consolidation_sediment", expected_generation
            )
            return FeedbackOutcome.rejected("stale_generation")
        self._apply_image_observations(ctx.response_dict, messages_chunk)
        self._apply_sediment(
            ctx, messages_chunk, expected_generation=expected_generation
        )
        latest = max((m.time for m in messages_chunk), default=None)
        if latest is not None:
            if (
                self.session.state.last_consolidated_time is None
                or latest > self.session.state.last_consolidated_time
            ):
                self.session.state.last_consolidated_time = latest
        self.session.state.messages_since_consolidation = 0
        self.session._schedule_save_session()
        return FeedbackOutcome(accepted=True)

    @staticmethod
    def _parse_memory_candidate(item, default_user_id: str) -> dict | None:
        """把 LLM 返回的一条候选规范化；ignore / 未知 action / 空内容返回 None。"""

        if isinstance(item, str):
            content = item.strip()
            if not content:
                return None
            return {
                "action": "add",
                "content": content,
                "category": "event",
                "confidence": 0.7,
                "importance": 0.5,
                "subject_user_id": default_user_id,
                "subject_user_name": "",
                "speaker_user_id": "",
                "speaker_user_name": "",
                "target_ref": "",
                "reason": "",
            }

        if not isinstance(item, dict):
            return None

        action = str(item.get("action") or "add").strip().lower()
        if action not in {"add", "supersede"}:
            logger.debug(f"[Memory] 暂不处理的记忆 action: {action}")
            return None

        def bounded_float(value, default: float) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return default

        return {
            "action": action,
            "content": str(item.get("content") or "").strip(),
            "category": str(item.get("category") or "event").strip().lower() or "event",
            "confidence": bounded_float(item.get("confidence", 0.7), 0.7),
            "importance": bounded_float(item.get("importance", 0.5), 0.5),
            "subject_user_id": str(item.get("subject_user_id") or "").strip()
            or default_user_id,
            "subject_user_name": str(item.get("subject_user_name") or "").strip(),
            "speaker_user_id": str(item.get("speaker_user_id") or "").strip(),
            "speaker_user_name": str(item.get("speaker_user_name") or "").strip(),
            "target_ref": str(item.get("target_ref") or "").strip(),
            "reason": str(item.get("reason") or ""),
        }

    async def _supersede_target_allowed(self, target_ref: str, candidate: dict) -> bool:
        """确认 supersede 目标存在且可替换，否则记一条拒绝事件。"""

        metadata = await run_sync(
            self.session.runtime.vector_memory.get_metadata_by_id
        )(target_ref)
        if not metadata:
            log_event(
                "rag_action_hallucination",
                session_id=self.session.id,
                action="supersede",
                target_ref=target_ref,
                reason="target_ref_missing_in_vector_store",
            )
            return False

        source = str(metadata.get("source") or candidate.get("source") or "memory")
        memory_type = str(metadata.get("type") or candidate.get("type") or "event")
        subtype = str(
            metadata.get("subtype") or candidate.get("subtype") or memory_type
        )
        category = str(
            metadata.get("category") or candidate.get("category") or memory_type
        )
        allowed_types = {"event", "preference", "profile", "relationship"}
        if (
            source != "memory"
            or subtype == "bot_self"
            or (memory_type not in allowed_types and category not in allowed_types)
        ):
            log_event(
                "rag_action_rejected",
                session_id=self.session.id,
                action="supersede",
                target_ref=target_ref,
                source=source,
                type=memory_type,
                subtype=subtype,
                category=category,
                reason="target_not_supersedable",
            )
            return False
        return True

    async def save_long_term_memory(
        self,
        analyze_result: list,
        default_user_id: str = "",
        supersede_candidates: list[dict] | None = None,
        expected_generation: int | None = None,
    ):
        """后台任务：把 Feedback 提取的候选落进向量库（质量过滤 + 去重）。"""

        try:
            if self.session.is_generation_stale(expected_generation):
                self.session._log_stale_generation(
                    "long_term_memory", expected_generation
                )
                return

            today = int(datetime.now().strftime("%Y%m%d"))
            skipped_quality = 0
            superseded_count = 0
            pending_memories: list[tuple[str, dict]] = []
            allowed_supersede_refs = {
                str(item.get("memory_ref")): item
                for item in supersede_candidates or []
                if isinstance(item, dict) and item.get("memory_ref")
            }

            for raw_item in analyze_result:
                candidate = self._parse_memory_candidate(raw_item, default_user_id)
                if candidate is None:
                    continue

                # 质量过滤：长度/噪声 + 类别/置信度/主体边界（先过滤，避免为废候选查库）
                valid, reason = validate_memory_candidate(
                    content=candidate["content"],
                    category=candidate["category"],
                    confidence=candidate["confidence"],
                    subject_user_id=candidate["subject_user_id"],
                    subject_user_name=candidate["subject_user_name"],
                )
                if not valid:
                    skipped_quality += 1
                    log_event(
                        "memory_candidate_rejected",
                        session_id=self.session.id,
                        action=candidate["action"],
                        category=candidate["category"],
                        reason=reason,
                    )
                    logger.debug(
                        f"[Memory] 跳过不可靠记忆({reason}): {candidate['content'][:30]}..."
                    )
                    continue

                if candidate["action"] == "supersede":
                    target_ref = candidate["target_ref"]
                    if target_ref not in allowed_supersede_refs:
                        log_event(
                            "rag_action_hallucination",
                            session_id=self.session.id,
                            action="supersede",
                            target_ref=target_ref,
                            reason="target_ref_not_in_current_candidates",
                        )
                        continue
                    if not await self._supersede_target_allowed(
                        target_ref, allowed_supersede_refs[target_ref]
                    ):
                        continue

                metadata = {
                    "schema_version": 2,
                    "source": "memory",
                    "type": candidate["category"],
                    "date": today,
                    "subject_user_id": candidate["subject_user_id"],
                    "subject_user_name": candidate["subject_user_name"],
                    "speaker_user_id": candidate["speaker_user_id"],
                    "speaker_user_name": candidate["speaker_user_name"],
                    "status": "active",
                    "category": candidate["category"],
                    "confidence": candidate["confidence"],
                    "importance": candidate["importance"],
                    "ttl_days": RAG_DEFAULT_EVENT_TTL_DAYS,
                }

                if candidate["action"] == "supersede":
                    operation_result = (
                        await self.session._run_sync_if_generation_current(
                            self.session.runtime.vector_memory.supersede_memory,
                            candidate["content"],
                            metadata,
                            candidate["target_ref"],
                            reason=candidate["reason"],
                            expected_generation=expected_generation,
                            stage="long_term_memory_supersede",
                        )
                    )
                    if operation_result is STALE_GENERATION_WRITE:
                        return
                    if not (
                        isinstance(operation_result, dict)
                        and operation_result.get("completed")
                    ):
                        log_event(
                            "rag_action_rejected",
                            session_id=self.session.id,
                            action="supersede",
                            target_ref=candidate["target_ref"],
                            reason="supersede_queued_for_repair",
                            queued_repair=(
                                operation_result.get("queued_repair")
                                if isinstance(operation_result, dict)
                                else 0
                            ),
                        )
                        continue
                    superseded_count += 1
                else:
                    pending_memories.append((candidate["content"], metadata))

            store_result = {"added": 0, "skipped_dedup": 0}
            if pending_memories:
                store_result = await self.session._run_sync_if_generation_current(
                    self.session.runtime.vector_memory.add_memories_with_dedup,
                    pending_memories,
                    expected_generation=expected_generation,
                    stage="long_term_memory_bulk",
                )
                if store_result is STALE_GENERATION_WRITE:
                    return

            saved_count = store_result.get("added", 0)
            skipped_dedup = store_result.get("skipped_dedup", 0)
            if (
                saved_count > 0
                or skipped_quality > 0
                or skipped_dedup > 0
                or superseded_count > 0
            ):
                logger.info(
                    f"[Memory] 存储结果: 成功 {saved_count}, 替换 {superseded_count}, 质量过滤 {skipped_quality}, 去重跳过 {skipped_dedup}"
                )
        except Exception as e:
            logger.error(f"[Async] 保存记忆失败: {e}")

    async def chat_stage(
        self,
        messages_chunk: list[Message],
        llm_func: Callable,
        recalled_history: list[str],
        search_result: RetrievalResult | None = None,
        expected_generation: int | None = None,
    ) -> list[dict]:
        logger.debug(">> 对话阶段 (Chat) 开始")
        search_history = search_result.prompt_lines if search_result else []
        formatted_msgs = [
            {
                "id": str(msg.id or ""),
                "name": msg.user_name,
                "content": msg.content,
            }
            for msg in messages_chunk
        ]

        # 格式化回溯的历史记录
        recalled_str = "\n".join(recalled_history) if recalled_history else "无"

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        context_record = self.session.runtime.short_term_memory.access()
        all_messages = context_record.messages
        history_msgs = _history_without_current_chunk(all_messages, messages_chunk)
        history_msgs_formatted = [
            {
                "time": m.time.strftime("%H:%M"),
                "name": m.user_name,
                "content": m.content,
            }
            for m in history_msgs
        ]

        time_str = get_time_description(datetime.now())
        # 分离 role 和 examples：role 中可能包含 [对话样本] 后缀，需要去除避免重复
        chat_role = (
            self.session.state.role.split("[对话样本]")[0].strip()
            if "[对话样本]" in self.session.state.role
            else self.session.state.role
        )
        reaction_users = list(
            {msg.user_id if msg.user_id else msg.user_name for msg in messages_chunk}
        )
        related_profiles = [
            self.session.state.profiles.get(uid, PersonProfile(user_id=uid))
            for uid in reaction_users
        ]
        related_profiles_data = [
            {"user_id": p.user_id, "emotion_tends_to_user": asdict(p.emotion)}
            for p in related_profiles
        ]
        prompt = get_chat_prompt(
            self.session.state.name,
            chat_role,
            self.session.state.chatting_state.value,
            context_record.compressed_history,
            history_msgs_formatted,  # 传入格式化后的历史
            formatted_msgs,
            asdict(self.session.state.global_emotion),
            related_profiles_data,
            search_history,
            self.session.state.chat_summary,
            examples_text=self.session.state.examples,
            recalled_history=recalled_str,
            time_info=time_str,
            budget=PromptBudget(),
        )
        log_event(
            "rag_prompt_budget",
            session_id=self.session.id,
            chat_prompt_total_chars=len(prompt),
            rag_injected_count=len(search_history),
            rag_injected_chars=sum(len(item) for item in search_history),
            history_chars=len(context_record.compressed_history or "")
            + sum(len(item.get("content", "")) for item in history_msgs_formatted),
            recent_chars=sum(len(item.get("content", "")) for item in formatted_msgs),
            recalled_history_chars=len(recalled_str),
            examples_chars=len(self.session.state.examples or ""),
        )

        try:
            # 使用传入的 chat_llm_func
            response = await llm_func(prompt, json_mode=True)
            replies = parse_reply(response).replies

            if self.session.is_generation_stale(expected_generation):
                self.session._log_stale_generation("chat_reply", expected_generation)
                return []

            if replies:
                retain = SPEAK_WILLINGNESS_RETAIN_FACTOR
                self.session.state.willingness = max(
                    0.0, self.session.state.willingness * retain
                )
                self.session.state.chatting_state = ChattingState.ACTIVE

            return replies

        except Exception as e:
            logger.error(f"对话阶段异常: {e}")
            return []

    def _consolidation_due(self) -> bool:
        """提高插话阈值，防止连击。"""

        state = self.session.state
        if state.messages_since_consolidation >= CONSOLIDATION_MESSAGE_THRESHOLD:
            return True
        return (
            state.messages_since_consolidation > 0
            and (datetime.now() - state.last_consolidation_attempt).total_seconds()
            >= CONSOLIDATION_INTERVAL_SECONDS
        )


@dataclass(frozen=True)
class ParsedFeedback:
    payload: dict | None
    failure_reason: str = ""


def parse_feedback(response: str, current_emotion: EmotionState) -> ParsedFeedback:
    """解析并校验 Feedback 的 new_emotion 边界。"""

    try:
        parsed = extract_and_parse_json(response)
    except Exception:
        return ParsedFeedback(None, "invalid_json")
    if not isinstance(parsed, dict) or not parsed:
        return ParsedFeedback(None, "invalid_payload")
    raw_emotion = parsed.get("new_emotion")
    if not isinstance(raw_emotion, dict):
        return ParsedFeedback(None, "missing_new_emotion")

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
            return ParsedFeedback(None, f"invalid_new_emotion_{field_name}")
        if not math.isfinite(value):
            return ParsedFeedback(None, f"invalid_new_emotion_{field_name}")
        normalized[field_name] = max(minimum, min(maximum, value))
        valid_fields += 1
    if valid_fields == 0:
        return ParsedFeedback(None, "empty_new_emotion")

    payload = dict(parsed)
    payload["new_emotion"] = normalized
    return ParsedFeedback(payload)


@dataclass(frozen=True)
class ReplyPlan:
    replies: list[dict | str]
    failure_reason: str = ""


def parse_reply(response: str) -> ReplyPlan:
    """解析 Chat 回复列表；裸 list 是模型常见偏差，直接兼容。"""

    try:
        payload = extract_and_parse_json(response)
    except Exception:
        return ReplyPlan([], "invalid_json")
    if isinstance(payload, dict):
        replies = payload.get("reply", [])
    elif isinstance(payload, list):
        replies = payload
        logger.warning("LLM 返回了 List 而非 Object，已自动兼容")
    else:
        return ReplyPlan([], "invalid_payload")
    if not isinstance(replies, list):
        return ReplyPlan([], "invalid_reply_list")
    return ReplyPlan(replies)
