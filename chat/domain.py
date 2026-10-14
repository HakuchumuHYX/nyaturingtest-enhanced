import math
from dataclasses import dataclass, field
from datetime import datetime

# 心情回落速率（每小时）：几个小时内平复，睡一夜基本回到基线
AROUSAL_BASELINE = 0.3
VALENCE_DECAY_PER_HOUR = 0.3
AROUSAL_DECAY_PER_HOUR = 0.5
DOMINANCE_DECAY_PER_HOUR = 0.25
# Feedback 每轮最多把心情的一个轴推动这么多
MOOD_STEP_LIMIT = 0.3

# 好感度：正值只累积不回落；负值随时间回升，免得「讨厌 → 吵架 → 更讨厌」越滚越僵
AFFINITY_LIMIT = 100
AFFINITY_STEP_LIMIT = 5
NEGATIVE_AFFINITY_HALF_LIFE_HOURS = 72.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _hours_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 3600.0)


@dataclass
class EmotionState:
    """角色在群里当下的心情（VAD）：valence [-1,1]、arousal [0,1]、dominance [-1,1]。默认值就是回落基线。"""

    valence: float = 0.0
    arousal: float = AROUSAL_BASELINE
    dominance: float = 0.0
    updated_at: datetime = field(default_factory=datetime.now)

    def settle(self, now: datetime | None = None) -> None:
        """结算 updated_at 以来的回落。指数回落与结算时机无关，结算后不存盘也不会算错。"""

        now = now or datetime.now()
        hours = _hours_between(self.updated_at, now)
        self.valence *= math.exp(-VALENCE_DECAY_PER_HOUR * hours)
        arousal_keep = math.exp(-AROUSAL_DECAY_PER_HOUR * hours)
        self.arousal = self.arousal * arousal_keep + AROUSAL_BASELINE * (1 - arousal_keep)
        self.dominance *= math.exp(-DOMINANCE_DECAY_PER_HOUR * hours)
        self.updated_at = now

    def apply(self, target: dict) -> None:
        """朝 Feedback 给的目标心情移动，每个轴每轮最多 MOOD_STEP_LIMIT。"""

        self.settle()
        for name, lower in (("valence", -1.0), ("arousal", 0.0), ("dominance", -1.0)):
            current = getattr(self, name)
            step = _clamp(target[name] - current, -MOOD_STEP_LIMIT, MOOD_STEP_LIMIT)
            setattr(self, name, _clamp(current + step, lower, 1.0))

    def vad(self) -> dict:
        return {
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "dominance": round(self.dominance, 2),
        }


@dataclass
class PersonProfile:
    """角色对某个群友的长期好感度与互动计数"""

    user_id: str
    # 库里存浮点：负值每轮只回升一点点，存整数会被四舍五入吃掉；对外一律用 level()
    affinity: float = 0.0
    affinity_updated_at: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    first_interaction_at: datetime | None = None
    last_interaction_at: datetime | None = None
    dirty: bool = field(default=True, repr=False, compare=False)

    def settle(self, now: datetime | None = None) -> None:
        """负好感按半衰期向 0 回升；正好感不动。"""

        now = now or datetime.now()
        if self.affinity < 0:
            hours = _hours_between(self.affinity_updated_at, now)
            self.affinity *= 0.5 ** (hours / NEGATIVE_AFFINITY_HALF_LIFE_HOURS)
        self.affinity_updated_at = now

    def level(self) -> int:
        self.settle()
        return round(self.affinity)

    def add_affinity(self, delta: int) -> None:
        self.settle()
        step = _clamp(delta, -AFFINITY_STEP_LIMIT, AFFINITY_STEP_LIMIT)
        self.affinity = _clamp(self.affinity + step, -AFFINITY_LIMIT, AFFINITY_LIMIT)
        self.dirty = True

    def record_interaction(self, now: datetime) -> None:
        self.interaction_count += 1
        if self.first_interaction_at is None:
            self.first_interaction_at = now
        self.last_interaction_at = now
        self.dirty = True
