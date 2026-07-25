from collections.abc import Awaitable, Callable
from datetime import datetime

from nonebot import logger

from ..config import get_runtime_settings
from ..memory.short_term import Message
from .engagement import EngagementPolicy
from .services import ChatService, FeedbackService, MemoryService


def _is_generation_stale(session, expected_generation: int | None) -> bool:
    checker = getattr(session, "is_generation_stale", None)
    if checker is None:
        return False
    return bool(checker(expected_generation))


def _log_stale_generation(session, stage: str, expected_generation: int | None) -> None:
    logger_func = getattr(session, "_log_stale_generation", None)
    if logger_func is not None:
        logger_func(stage, expected_generation)
    else:
        logger.info(f"[Session {getattr(session, 'id', '-')}] 丢弃过期 turn: {stage}")


class ConversationOrchestrator:
    def __init__(
        self,
        session,
        *,
        memory_service: MemoryService | None = None,
        feedback_service: FeedbackService | None = None,
        chat_service: ChatService | None = None,
        engagement_policy: EngagementPolicy | None = None,
    ):
        self.session = session
        self.memory_service = memory_service or MemoryService(session)
        self.feedback_service = feedback_service or FeedbackService(session)
        self.chat_service = chat_service or ChatService(session)
        self.engagement_policy = engagement_policy or EngagementPolicy()

    async def process_chunk(
        self,
        messages_chunk: list[Message],
        chat_llm_func: Callable[[str, bool], Awaitable[str]],
        feedback_llm_func: Callable[[str, bool], Awaitable[str]],
        publish: bool = True,
        expected_generation: int | None = None,
    ) -> list[dict] | None:
        begin_batch = getattr(self.session, "begin_persistence_batch", None)
        end_batch = getattr(self.session, "end_persistence_batch", None)
        if begin_batch is not None:
            begin_batch()
        try:
            return await self._process_chunk(
                messages_chunk,
                chat_llm_func,
                feedback_llm_func,
                publish=publish,
                expected_generation=expected_generation,
            )
        finally:
            if end_batch is not None:
                await end_batch(flush=True)

    async def _process_chunk(
        self,
        messages_chunk: list[Message],
        chat_llm_func: Callable[[str, bool], Awaitable[str]],
        feedback_llm_func: Callable[[str, bool], Awaitable[str]],
        publish: bool = True,
        expected_generation: int | None = None,
    ) -> list[dict] | None:
        if _is_generation_stale(self.session, expected_generation):
            _log_stale_generation(self.session, "process_start", expected_generation)
            return None

        await self.memory_service.update_short_term(messages_chunk)
        if _is_generation_stale(self.session, expected_generation):
            _log_stale_generation(self.session, "short_term_memory", expected_generation)
            return None

        if not publish:
            return None

        now = datetime.now()
        runtime_settings = get_runtime_settings()
        engagement = self.engagement_policy.evaluate(
            session=self.session,
            messages=messages_chunk,
            settings=runtime_settings,
            now=now,
        )
        is_relevant = engagement.relevant
        if is_relevant:
            logger.info("检测到强关联，意愿值提升")

        if not engagement.engaged and not is_relevant:
            if runtime_settings["consolidation_enabled"] and self._consolidation_due(runtime_settings):
                max_messages = runtime_settings["consolidation_max_messages"]
                self.session._last_consolidation_attempt = datetime.now()
                pending_messages = self.memory_service.unconsolidated_messages(max_messages)
                await self.memory_service.consolidate(
                    pending_messages,
                    feedback_llm_func,
                    expected_generation=expected_generation,
                )
            logger.debug(f"未进入参与态 (意愿 {self.session.willingness:.2f})，跳过响应")
            return None

        if engagement.cooldown_remaining > 0 and not is_relevant:
            logger.debug(
                f"处于发言冷却期（剩余 {engagement.cooldown_remaining:.1f}s），"
                "跳过响应"
            )
            return None

        # Reranker 使用第一条 query 作为主 query，因此必须最新消息优先。
        queries = [msg.content for msg in reversed(messages_chunk[-3:])]
        active_user_names = [msg.user_name for msg in messages_chunk if msg.user_name]
        active_users = [
            {
                "user_id": str(msg.user_id or ""),
                "user_name": msg.user_name,
            }
            for msg in messages_chunk
            if msg.user_name
        ]
        use_rerank_strategy = self.session.willingness > runtime_settings["rerank_willingness_threshold"] or is_relevant

        search_result = await self.memory_service.search(
            queries,
            active_user_names=active_user_names,
            active_users=active_users,
            use_rerank=use_rerank_strategy,
        )
        if _is_generation_stale(self.session, expected_generation):
            _log_stale_generation(self.session, "rag_search", expected_generation)
            return None

        logger.debug("启用拟人化串行模式: Feedback -> Check -> Chat")

        feedback_outcome = await self.feedback_service.process(
            messages_chunk,
            feedback_llm_func,
            is_relevant=is_relevant,
            search_result=search_result,
            expected_generation=expected_generation,
        )
        if _is_generation_stale(self.session, expected_generation):
            _log_stale_generation(self.session, "feedback", expected_generation)
            return None
        if feedback_outcome.accepted:
            latest = max((msg.time for msg in messages_chunk), default=None)
            if latest is not None and (
                self.session.last_consolidated_time is None
                or latest > self.session.last_consolidated_time
            ):
                self.session.last_consolidated_time = latest
            self.session._messages_since_consolidation = 0
            self.session._last_consolidation_attempt = datetime.now()
            schedule_save = getattr(self.session, "_schedule_save_session", None)
            if schedule_save is not None:
                schedule_save()
            else:
                await self.session.save_session(expected_generation=expected_generation)

        if self.session.willingness < runtime_settings["post_feedback_skip_threshold"] and not is_relevant:
            return None

        reply_messages = await self.chat_service.plan_reply(
            messages_chunk,
            chat_llm_func,
            recalled_history=feedback_outcome.recalled_history,
            search_result=search_result,
            expected_generation=expected_generation,
        )
        if _is_generation_stale(self.session, expected_generation):
            _log_stale_generation(self.session, "chat", expected_generation)
            return None

        if reply_messages:
            self.session._last_speak_time = datetime.now()

        return reply_messages

    def _consolidation_due(self, runtime_settings) -> bool:
        pending = getattr(self.session, "_messages_since_consolidation", 0)
        if pending >= runtime_settings["consolidation_message_threshold"]:
            return True
        last = getattr(self.session, "_last_consolidation_attempt", datetime.min)
        interval = runtime_settings["consolidation_interval_seconds"]
        return pending > 0 and interval > 0 and (datetime.now() - last).total_seconds() >= interval
