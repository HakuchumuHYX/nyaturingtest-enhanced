import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


def clamp_vad_value(value, lower: float, upper: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return max(lower, min(upper, number))


@dataclass
class EmotionState:
    """情感状态（VAD）：valence [-1,1]、arousal [0,1]、dominance [-1,1]。"""

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0


@dataclass
class Impression:
    """记录某次互动带来的印象"""

    timestamp: datetime
    delta: dict


@dataclass
class PersonProfile:
    """对人物的记忆与情感"""

    user_id: str
    emotion: EmotionState = field(default_factory=EmotionState)
    interactions: deque[Impression] = field(default_factory=deque)
    last_update_time: datetime = field(default_factory=lambda: datetime.now().astimezone())
    interaction_count: int = 0
    first_interaction_at: datetime | None = None
    last_interaction_at: datetime | None = None
    dirty: bool = field(default=True, repr=False, compare=False)

    def push_interaction(self, impression: Impression):
        """
        添加交互记录 (O(1) 版本的峰值保持模式)
        """
        # 1. 关键步骤：先结算时间衰减
        self.update_emotion_tends()

        # 2. 获取新消息的情感输入
        new_val = impression.delta.get("valence", 0.0)
        new_aro = impression.delta.get("arousal", 0.0)
        new_dom = impression.delta.get("dominance", 0.0)

        # 3. 应用峰值保持逻辑 (Peak Hold Logic)

        # --- Valence (愉悦度) ---
        if (self.emotion.valence >= 0 and new_val >= 0) or (self.emotion.valence < 0 and new_val < 0):
            if abs(new_val) > abs(self.emotion.valence):
                self.emotion.valence = new_val
        else:
            self.emotion.valence += new_val
        self.emotion.valence = max(-1.0, min(1.0, self.emotion.valence))

        # --- Arousal (唤醒度) ---
        self.emotion.arousal = max(self.emotion.arousal, new_aro)
        self.emotion.arousal = max(0.0, min(1.0, self.emotion.arousal))

        # --- Dominance (支配度) ---
        if (self.emotion.dominance >= 0 and new_dom >= 0) or (self.emotion.dominance < 0 and new_dom < 0):
            if abs(new_dom) > abs(self.emotion.dominance):
                self.emotion.dominance = new_dom
        else:
            self.emotion.dominance += new_dom
        self.emotion.dominance = max(-1.0, min(1.0, self.emotion.dominance))

        # 将新的印象加入队列
        self.interactions.appendleft(impression)
        self.interaction_count += 1
        if self.first_interaction_at is None:
            self.first_interaction_at = impression.timestamp
        self.last_interaction_at = impression.timestamp
        self.dirty = True

    def merge_old_interactions(self):
        """
        仅清理过期的交互记录，不再重新计算情感 (增量更新 - 清理)
        """
        if not self.interactions:
            return

        # 统一使用带时区的时间，防止 TypeError
        now = datetime.now().astimezone()

        while len(self.interactions) > 0:
            last_interaction = self.interactions[-1]

            current_interaction_time = last_interaction.timestamp
            # 确保交互记录的时间也是 aware 的，如果不是则假设为本地时间
            if current_interaction_time.tzinfo is None:
                current_interaction_time = current_interaction_time.astimezone()

            if (now - current_interaction_time).total_seconds() / 3600 > 5:
                self.interactions.pop()
            else:
                break

    def update_emotion_tends(self):
        """
        随时间流逝衰减情感 (增量更新 - 衰减)
        """
        # 统一使用带时区的时间
        now = datetime.now().astimezone()

        if self.last_update_time.tzinfo is None:
            # 如果之前的记录没有时区，强制转换为带时区
            self.last_update_time = self.last_update_time.astimezone()

        # 计算距离上次更新经过了多少小时
        elapsed_hours = (now - self.last_update_time).total_seconds() / 3600.0

        # 更新时间戳
        self.last_update_time = now

        # 如果时间极短，跳过计算节省资源
        if elapsed_hours < 0.001:
            return

        # 对当前情感状态应用时间衰减：valence 正向慢、负向快；arousal 回到 0.3；dominance 回到 0
        old_state = (self.emotion.valence, self.emotion.arousal, self.emotion.dominance)
        valence, arousal, dominance = old_state
        if valence > 0:
            self.emotion.valence = valence * math.exp(-0.05 * elapsed_hours)
        elif valence < 0:
            self.emotion.valence = valence * math.exp(-0.15 * elapsed_hours)
        else:
            self.emotion.valence = 0.0
        arousal_decay = math.exp(-0.2 * elapsed_hours)
        self.emotion.arousal = arousal * arousal_decay + 0.3 * (1 - arousal_decay)
        self.emotion.dominance = dominance * math.exp(-0.03 * elapsed_hours)

        new_state = (self.emotion.valence, self.emotion.arousal, self.emotion.dominance)
        if new_state != old_state:
            self.dirty = True
