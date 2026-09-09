from typing import Any

from nonebot.utils import run_sync

from ..memory.vector import RetrievalResult


class RagSearchService:
    """长期记忆检索入口，供对话编排与命令复用。"""

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

    async def search(self, queries: list[str], **kwargs: Any) -> RetrievalResult:
        result = await run_sync(self.long_term_memory.retrieve_with_decay)(queries, **kwargs)
        records = result.records if isinstance(result, RetrievalResult) else list(result or [])
        stats = result.stats if isinstance(result, RetrievalResult) else {}
        return RetrievalResult(
            records=[self.normalize_record(record) for record in records],
            stats=dict(stats),
        )
