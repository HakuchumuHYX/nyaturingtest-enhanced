from collections.abc import Awaitable, Callable

from ..memory.short_term import Message


class MemoryService:
    def __init__(self, session):
        self.session = session

    async def update_short_term(self, messages_chunk: list[Message]):
        await self.session.global_memory.update(messages_chunk)
        self.session._create_safe_task(self.session.save_session())

    async def search(self, queries: list[str], active_user_names: list[str], use_rerank: bool):
        await self.session.search_stage(
            queries,
            active_user_names=active_user_names,
            use_rerank=use_rerank,
        )

    async def save_long_term(self, analyze_result: list, default_user_id: str = ""):
        await self.session.save_long_term_memory(analyze_result, default_user_id=default_user_id)


class FeedbackService:
    def __init__(self, session):
        self.session = session

    async def process(
        self,
        messages_chunk: list[Message],
        llm_func: Callable[[str, bool], Awaitable[str]],
        *,
        is_relevant: bool = False,
    ) -> list[str]:
        try:
            return await self.session.feedback_stage(
                messages_chunk,
                llm_func,
                is_relevant=is_relevant,
            )
        finally:
            await self.session.save_session()


class ChatService:
    def __init__(self, session):
        self.session = session

    async def plan_reply(
        self,
        messages_chunk: list[Message],
        llm_func: Callable[[str, bool], Awaitable[str]],
        recalled_history: list[str],
    ) -> list[dict]:
        return await self.session.chat_stage(
            messages_chunk,
            llm_func,
            recalled_history=recalled_history,
        )
