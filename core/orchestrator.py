from collections.abc import Awaitable, Callable
from datetime import datetime
import random

from nonebot import logger

from ..config import get_runtime_settings
from ..memory.short_term import Message
from ..utils import check_relevance, score_message_interest
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
    ):
        self.session = session
        self.memory_service = memory_service or MemoryService(session)
        self.feedback_service = feedback_service or FeedbackService(session)
        self.chat_service = chat_service or ChatService(session)

    async def process_chunk(
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
        last_decay = getattr(self.session, "_last_decay_time", None) or now
        seconds_since_decay = max(0.0, (now - last_decay).total_seconds())

        last_speak_for_decay = self.session._last_speak_time
        if last_speak_for_decay.tzinfo is not None:
            last_speak_for_decay = last_speak_for_decay.astimezone(None).replace(tzinfo=None)
        idle = (now - last_speak_for_decay).total_seconds() >= runtime_settings["willingness_idle_after_seconds"]
        decay_rate = (
            runtime_settings["willingness_decay_rate_idle"] if idle
            else runtime_settings["willingness_decay_rate_active"]
        )
        decay = (seconds_since_decay / 60.0) * decay_rate
        self.session.willingness = max(0.0, self.session.willingness - decay)
        self.session._last_decay_time = now
        self.session._last_activity_time = now

        is_relevant = check_relevance(
            self.session.name(),
            self.session.aliases(),
            messages_chunk,
        )
        if is_relevant:
            self.session.willingness = max(self.session.willingness, runtime_settings["relevance_willingness_floor"])
            logger.info("检测到强关联，意愿值提升")
        elif self.session.willingness < runtime_settings["passive_willingness_growth_limit"]:
            interest = score_message_interest(
                [msg.content for msg in messages_chunk],
                bot_name=self.session.name(),
                aliases=self.session.aliases(),
                lo=runtime_settings["passive_growth_min_factor"],
                hi=runtime_settings["passive_growth_max_factor"],
            )
            growth = runtime_settings["passive_willingness_growth_per_message"] * interest * len(messages_chunk)
            self.session.willingness = min(1.0, self.session.willingness + growth)
            if interest >= 1.6:
                self.session.willingness = max(
                    self.session.willingness,
                    runtime_settings["interest_topic_willingness_floor"],
                )

        enter_threshold = runtime_settings["willingness_reply_threshold"]
        exit_threshold = runtime_settings["low_willingness_skip_threshold"]
        if is_relevant:
            self.session._engaged = True
        elif self.session._engaged and self.session.willingness < exit_threshold:
            self.session._engaged = False
        elif not self.session._engaged and self.session.willingness >= enter_threshold:
            self.session._engaged = True

        if not self.session._engaged and not is_relevant:
            if runtime_settings["consolidation_enabled"] and self._consolidation_due(runtime_settings):
                max_messages = runtime_settings["consolidation_max_messages"]
                await self.memory_service.consolidate(
                    messages_chunk[-max_messages:],
                    feedback_llm_func,
                    expected_generation=expected_generation,
                )
            logger.debug(f"未进入参与态 (意愿 {self.session.willingness:.2f})，跳过响应")
            return None

        last_speak = self.session._last_speak_time
        if last_speak.tzinfo is not None:
            last_speak = last_speak.astimezone(None).replace(tzinfo=None)

        time_since_speak = (now - last_speak).total_seconds()

        if time_since_speak < 0:
            logger.warning(f"[Session {self.session.id}] 检测到最后发言时间在未来 ({time_since_speak:.1f}s)，忽略冷却限制")
        elif time_since_speak < runtime_settings["speak_cooldown_seconds"] and not is_relevant:
            logger.debug(f"处于发言冷却期 ({time_since_speak:.1f}s)，跳过响应")
            return None

        queries = [msg.content for msg in messages_chunk[-2:]]
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

        recalled_history = await self.feedback_service.process(
            messages_chunk,
            feedback_llm_func,
            is_relevant=is_relevant,
            search_result=search_result,
            expected_generation=expected_generation,
        )
        if _is_generation_stale(self.session, expected_generation):
            _log_stale_generation(self.session, "feedback", expected_generation)
            return None
        latest = max((msg.time for msg in messages_chunk), default=None)
        if latest is not None and (
            self.session.last_consolidated_time is None
            or latest > self.session.last_consolidated_time
        ):
            self.session.last_consolidated_time = latest
        self.session._messages_since_consolidation = 0
        self.session._last_consolidation_attempt = datetime.now()
        await self.session.save_session(expected_generation=expected_generation)

        if self.session.willingness < runtime_settings["post_feedback_skip_threshold"] and not is_relevant:
            return None

        reply_messages = await self.chat_service.plan_reply(
            messages_chunk,
            chat_llm_func,
            recalled_history=recalled_history,
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
