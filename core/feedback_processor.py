import math
from dataclasses import dataclass

from ..models.emotion import EmotionState
from .text_utils import extract_and_parse_json


@dataclass(frozen=True)
class ParsedFeedback:
    payload: dict | None
    failure_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.payload is not None


class FeedbackProcessor:
    """Parse and validate the state-changing Feedback boundary."""

    @staticmethod
    def parse(response: str, current_emotion: EmotionState) -> ParsedFeedback:
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
                return ParsedFeedback(
                    None,
                    f"invalid_new_emotion_{field_name}",
                )
            if not math.isfinite(value):
                return ParsedFeedback(
                    None,
                    f"invalid_new_emotion_{field_name}",
                )
            normalized[field_name] = max(minimum, min(maximum, value))
            valid_fields += 1
        if valid_fields == 0:
            return ParsedFeedback(None, "empty_new_emotion")

        payload = dict(parsed)
        payload["new_emotion"] = normalized
        return ParsedFeedback(payload)
