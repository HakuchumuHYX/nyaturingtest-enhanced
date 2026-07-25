from dataclasses import dataclass


@dataclass
class RuntimeMetrics:
    llm_success: int = 0
    llm_failure: int = 0
    vlm_success: int = 0
    vlm_failure: int = 0
    db_write_failure: int = 0
    memory_query_count: int = 0
    memory_query_cache_hit: int = 0
    memory_query_singleflight_reused: int = 0
    memory_query_cooldown_rejected: int = 0
    memory_query_total_ms: float = 0.0
    memory_query_rag_calls: int = 0
    memory_query_feedback_calls: int = 0
    memory_query_chat_calls: int = 0


metrics = RuntimeMetrics()
