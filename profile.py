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

    def push_interaction(self, impression: Impression):
        """
        添加交互记录 (O(1) 版本的峰值保持模式)
        """
        # 1. 关键步骤：先结算时间衰减
        # 这会将 self.emotion 更新到“此时此刻”的状态，相当于原逻辑中的 decayed_valence
        self.update_emotion_tends()

        # 2. 获取新消息的情感输入
        new_val = impression.delta.get("valence", 0.0)
        new_aro = impression.delta.get("arousal", 0.0)
        new_dom = impression.delta.get("dominance", 0.0)

        # 3. 应用峰值保持逻辑 (Peak Hold Logic)

        # --- Valence (愉悦度) ---
        # 逻辑：如果新情绪比当前更强烈且同向，则更新为新的；如果反向，则抵消。
        if (self.emotion.valence >= 0 and new_val >= 0) or (self.emotion.valence < 0 and new_val < 0):
            # 同向（都高兴 或 都难过）：取绝对值更大者（峰值保持）
            if abs(new_val) > abs(self.emotion.valence):
                self.emotion.valence = new_val
            # else: 如果当前情绪比新消息更强烈，保持当前情绪不变（不累加）
        else:
            # 反向（高兴时被泼冷水）：互相抵消（算术相加）
            self.emotion.valence += new_val
        # 边界截断 [-1, 1]
        self.emotion.valence = max(-1.0, min(1.0, self.emotion.valence))

        # --- Arousal (唤醒度) ---
        # 逻辑：唤醒度通常只取最大值。如果你已经很兴奋(0.8)，来个平淡的消息(0.1)不会让你平静下来。
        # 只有时间会让你平静。
        self.emotion.arousal = max(self.emotion.arousal, new_aro)
        # 边界截断 [0, 1]
        self.emotion.arousal = max(0.0, min(1.0, self.emotion.arousal))

        # --- Dominance (支配度) ---
        # 逻辑：同 Valence
        if (self.emotion.dominance >= 0 and new_dom >= 0) or (self.emotion.dominance < 0 and new_dom < 0):
            if abs(new_dom) > abs(self.emotion.dominance):
                self.emotion.dominance = new_dom
        else:
            self.emotion.dominance += new_dom
        # 边界截断 [-1, 1]
        self.emotion.dominance = max(-1.0, min(1.0, self.emotion.dominance))

    def merge_old_interactions(self):
        """
        仅清理过期的交互记录，不再重新计算情感 (增量更新 - 清理)
        """
        # 这里的逻辑简化为只负责删除过期数据，节省算力
        # 5小时之前的印象会被移除
        while len(self.interactions) > 0:
            # 检查最右侧（最早）的记录
            last_interaction = self.interactions[-1]
            if (datetime.now() - last_interaction.timestamp).total_seconds() / 3600 > 5:
                self.interactions.pop()
            else:
                # 因为是按时间顺序存的，如果最老的一个没过期，后面的肯定也没过期
                break

    def update_emotion_tends(self):
        """
        随时间流逝衰减情感 (增量更新 - 衰减)
        """
        now = datetime.now()
        # 计算距离上次更新经过了多少小时
        elapsed_hours = (now - self.last_update_time).total_seconds() / 3600.0

        # 更新时间戳
        self.last_update_time = now

        # 如果时间极短，跳过计算节省资源
        if elapsed_hours < 0.001:
            return

        # 对当前情感状态应用时间衰减
        # 直接调用同文件下定义的 decay_xxx 函数
        self.emotion.valence = decay_valence(elapsed_hours, self.emotion.valence)
        self.emotion.arousal = decay_arousal(elapsed_hours, self.emotion.arousal)
        self.emotion.dominance = decay_dominance(elapsed_hours, self.emotion.dominance)


def decay_valence(
    elapsed_hours: float, valence: float, decay_rate_positive: float = 0.05, decay_rate_negative: float = 0.15
) -> float:
    """
    愉悦度随时间衰减，负面情绪恢复更慢。

    参数:
        elapsed_hours (float): 距离事件过去的时间，单位小时
        valence (float): 当前愉悦度，范围 [-1, 1]
        decay_rate_positive (float): 正向情绪衰减速度（越大衰减越快）
        decay_rate_negative (float): 负向情绪衰减速度（越小衰减越慢）

    返回:
        float: 经过衰减后的 valence
    """
    if valence > 0:
        rate = decay_rate_positive
    elif valence < 0:
        rate = decay_rate_negative
    else:
        return 0.0
    return valence * math.exp(-rate * elapsed_hours)


def decay_arousal(elapsed_hours: float, arousal: float, target: float = 0.3, decay_rate: float = 0.2) -> float:
    """
    激活度随时间逐渐恢复到 target 的过程。

    参数:
        elapsed_hours (float): 距离事件过去的时间（单位小时）
        arousal (float): 当前 arousal 值（范围 0.0 ~ 1.0）
        target (float): arousal 的恢复目标值（默认 0.3）
        decay_rate (float): 恢复速度（越大恢复越快）

    返回:
        float: 经过衰减后的 arousal 值
    """
    decay = math.exp(-decay_rate * elapsed_hours)
    return arousal * decay + target * (1 - decay)


def decay_dominance(elapsed_hours: float, dominance: float, target: float = 0.5, decay_rate: float = 0.03) -> float:
    """
    支配度随时间缓慢回归中性

    参数:
        elapsed_hours (float): 距离事件的时间（小时）
        dominance (float): 当前支配度
        target (float): 恢复目标
        decay_rate (float): 趋于中性的速度

    返回:
        float: 衰减后的 dominance 值
    """
    decay = math.exp(-decay_rate * elapsed_hours)
    return dominance * decay + target * (1 - decay)
