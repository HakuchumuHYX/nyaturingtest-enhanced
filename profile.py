from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import math

from .emotion import EmotionState
from .impression import Impression


@dataclass
class PersonProfile:
    """
    对人物的记忆与情感
    """

    user_id: str
    """
    "你叫什么名字？"
    """
    emotion: EmotionState = field(default_factory=EmotionState)
    """
    对你的情感倾向
    """
    interactions: deque[Impression] = field(default_factory=deque)
    """
    交互的记录
    """
    last_update_time: datetime = field(default_factory=datetime.now)
    """
    上次更新情感的时间
    """

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

    def merge_old_interactions(self):
        """
        仅清理过期的交互记录，不再重新计算情感 (增量更新 - 清理)
        """
        if not self.interactions:
            return

        # 【修复】获取当前时间，并根据印象的时间戳类型决定是否带时区
        now = datetime.now()
        first_impression = self.interactions[-1]  # 最老的一条

        # 如果印象里的时间带时区（来自数据库），则 now 也必须带时区
        if first_impression.timestamp.tzinfo is not None:
            now = datetime.now().astimezone()

        while len(self.interactions) > 0:
            last_interaction = self.interactions[-1]

            # 二次防御：防止队列里混合了带时区和不带时区的数据
            current_interaction_time = last_interaction.timestamp
            current_now = now

            # 如果类型不匹配，临时转换 current_now 以适应
            if current_interaction_time.tzinfo is None and current_now.tzinfo is not None:
                current_now = current_now.replace(tzinfo=None)
            elif current_interaction_time.tzinfo is not None and current_now.tzinfo is None:
                current_now = current_now.astimezone()

            if (current_now - current_interaction_time).total_seconds() / 3600 > 5:
                self.interactions.pop()
            else:
                break

    def update_emotion_tends(self):
        """
        随时间流逝衰减情感 (增量更新 - 衰减)
        """
        # 【修复】处理时区不一致问题
        # 如果 last_update_time 来自数据库（带时区），则 now 也要带时区
        if self.last_update_time.tzinfo is not None:
            now = datetime.now().astimezone()
        else:
            now = datetime.now()

        # 计算距离上次更新经过了多少小时
        elapsed_hours = (now - self.last_update_time).total_seconds() / 3600.0

        # 更新时间戳
        self.last_update_time = now

        # 如果时间极短，跳过计算节省资源
        if elapsed_hours < 0.001:
            return

        # 对当前情感状态应用时间衰减
        self.emotion.valence = decay_valence(elapsed_hours, self.emotion.valence)
        self.emotion.arousal = decay_arousal(elapsed_hours, self.emotion.arousal)
        self.emotion.dominance = decay_dominance(elapsed_hours, self.emotion.dominance)


def decay_valence(
        elapsed_hours: float, valence: float, decay_rate_positive: float = 0.05, decay_rate_negative: float = 0.15
) -> float:
    if valence > 0:
        rate = decay_rate_positive
    elif valence < 0:
        rate = decay_rate_negative
    else:
        return 0.0
    return valence * math.exp(-rate * elapsed_hours)


def decay_arousal(elapsed_hours: float, arousal: float, target: float = 0.3, decay_rate: float = 0.2) -> float:
    decay = math.exp(-decay_rate * elapsed_hours)
    return arousal * decay + target * (1 - decay)


def decay_dominance(elapsed_hours: float, dominance: float, target: float = 0.5, decay_rate: float = 0.03) -> float:
    decay = math.exp(-decay_rate * elapsed_hours)
    return dominance * decay + target * (1 - decay)