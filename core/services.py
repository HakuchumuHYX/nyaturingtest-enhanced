from collections.abc import Awaitable, Callable
from typing import Any

from nonebot.utils import run_sync

from ..memory.short_term import Message


class RagSearchService:
    def __init__(self, long_term_memory):
        self.long_term_memory = long_term_memory

    @staticmethod
    def normalize_record(record: dict) -> dict:
        content = str(record.get("content") or "")
        meta = dict(record.get("metadata") or {})
        score = meta.get("adjusted_score")
        if score is None:
            score = meta.get("rerank_score")
        if score is None:
            score = meta.get("retrieval_score")
        return {
            "content": content,
            "metadata": meta,
            "score": score,
            "memory_ref": meta.get("memory_ref"),
            "preview": content[:80],
        }

    async def _retrieve(self, queries: list[str], **kwargs: Any) -> list[dict]:
        records = await run_sync(self.long_term_memory.retrieve_with_decay)(queries, **kwargs)
        return [self.normalize_record(record) for record in records]

    async def search_for_chat(self, queries: list[str], **kwargs: Any) -> list[dict]:
        return await self._retrieve(queries, **kwargs)

    async def search_for_user_profile(self, queries: list[str], **kwargs: Any) -> list[dict]:
        return await self._retrieve(queries, **kwargs)

    async def search_for_debug(self, queries: list[str], **kwargs: Any) -> list[dict]:
        return await self._retrieve(queries, **kwargs)


class MemoryService:
    def __init__(self, session):
        self.session = session

    def note_incoming(self, count: int) -> None:
        self.session._messages_since_consolidation = (
            getattr(self.session, "_messages_since_consolidation", 0) + count
        )

    async def update_short_term(self, messages_chunk: list[Message]):
        await self.session.global_memory.update(messages_chunk)
        self.note_incoming(len(messages_chunk))
        schedule_save = getattr(self.session, "_schedule_save_session", None)
        if schedule_save is not None:
            schedule_save()
        else:
            self.session._create_safe_task(self.session.save_session())

    async def search(
        self,
        queries: list[str],
        active_user_names: list[str] | None = None,
        use_rerank: bool = True,
        *,
        active_users: list[dict] | None = None,
    ):
        return await self.session.search_stage(
            queries,
            active_user_names=active_user_names,
            active_users=active_users,
            use_rerank=use_rerank,
        )

    async def save_long_term(self, analyze_result: list, default_user_id: str = ""):
        await self.session.save_long_term_memory(analyze_result, default_user_id=default_user_id)

    async def consolidate(self, messages_chunk, feedback_llm_func, *, expected_generation: int | None = None):
        await self.session.consolidate_stage(
            messages_chunk,
            feedback_llm_func,
            expected_generation=expected_generation,
        )


class FeedbackService:
    def __init__(self, session):
        self.session = session

    async def process(
        self,
        messages_chunk: list[Message],
        llm_func: Callable[[str, bool], Awaitable[str]],
        *,
        is_relevant: bool = False,
        search_result=None,
        expected_generation: int | None = None,
    ) -> list[str]:
        try:
            return await self.session.feedback_stage(
                messages_chunk,
                llm_func,
                is_relevant=is_relevant,
                search_result=search_result,
                expected_generation=expected_generation,
            )
        finally:
            await self.session.save_session(expected_generation=expected_generation)


class ChatService:
    def __init__(self, session):
        self.session = session

    async def plan_reply(
        self,
        messages_chunk: list[Message],
        llm_func: Callable[[str, bool], Awaitable[str]],
        recalled_history: list[str],
        search_result=None,
        expected_generation: int | None = None,
    ) -> list[dict]:
        return await self.session.chat_stage(
            messages_chunk,
            llm_func,
            recalled_history=recalled_history,
            search_result=search_result,
            expected_generation=expected_generation,
        )
