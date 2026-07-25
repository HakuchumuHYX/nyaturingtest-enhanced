from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeedbackOutcome:
    """Validated result of one Feedback pass.

    ``accepted`` is intentionally separate from the recalled-history payload so
    callers never infer success from an empty list.
    """

    accepted: bool
    recalled_history: list[str] = field(default_factory=list)
    state_changed: bool = False
    failure_reason: str = ""

    @classmethod
    def rejected(cls, reason: str) -> "FeedbackOutcome":
        return cls(accepted=False, failure_reason=reason)
