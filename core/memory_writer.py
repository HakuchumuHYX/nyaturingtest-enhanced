from typing import Any


class MemoryWriteService:
    """Narrow write facade over VectorMemory lifecycle operations."""

    def __init__(self, vector_memory):
        self.vector_memory = vector_memory

    def supersede(
        self,
        content: str,
        metadata: dict,
        target_ref: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        return self.vector_memory.supersede_memory(
            content,
            metadata,
            target_ref,
            reason=reason,
        )

    def add_candidates(
        self,
        candidates: list[tuple[str, dict]],
    ) -> dict[str, int]:
        return self.vector_memory.add_memories_with_dedup(candidates)
