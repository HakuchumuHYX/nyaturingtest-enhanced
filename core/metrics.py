from dataclasses import dataclass


@dataclass
class RuntimeMetrics:
    llm_success: int = 0
    llm_failure: int = 0
    vlm_success: int = 0
    vlm_failure: int = 0
    db_write_failure: int = 0


metrics = RuntimeMetrics()
