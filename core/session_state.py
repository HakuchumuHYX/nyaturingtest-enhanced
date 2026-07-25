from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.emotion import EmotionState
from ..models.profile import PersonProfile


class ChattingState(Enum):
    IDLE = 0
    BUBBLE = 1
    ACTIVE = 2

    def __str__(self) -> str:
        return {
            ChattingState.IDLE: "潜水状态",
            ChattingState.BUBBLE: "冒泡状态",
            ChattingState.ACTIVE: "对话状态",
        }[self]


@dataclass
class SessionState:
    """Pure mutable domain state; owns no DB, HTTP or vector resources."""

    name: str = "terminus"
    role: str = "一个男性人类"
    aliases: list[str] = field(default_factory=list)
    examples: str = ""
    profiles: dict[str, PersonProfile] = field(default_factory=dict)
    global_emotion: EmotionState = field(default_factory=EmotionState)
    chat_summary: str = ""
    willingness: float = 0.0
    chatting_state: ChattingState = ChattingState.IDLE
    last_activity_time: datetime = field(default_factory=datetime.now)
    last_decay_time: datetime = field(default_factory=datetime.now)
    last_speak_time: datetime = datetime.min
    active_count: int = 0
    engaged: bool = False
    last_consolidated_time: datetime | None = None
    messages_since_consolidation: int = 0
    last_consolidation_attempt: datetime = datetime.min
    loaded: bool = False
    generation: int = 0
