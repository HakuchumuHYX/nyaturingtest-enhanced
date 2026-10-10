"""发言意愿与参与判定。

两层决策：
1. 规则层（本模块）决定这批消息「值不值得调用 Feedback 考虑说话」：被点名、聊天对象在接话、
   或意愿值够高且自己最近没说太多。
2. Feedback 的 willing 决定「说不说」，门槛按场景区分（被点名必回、聊天对象接话或被戳 0.45、主动插嘴 0.6）。
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..memory.short_term import Message

# 两次发送之间的最小间隔（在发送前等待，不再跳过整轮）
SPEAK_COOLDOWN_SECONDS = 16.0
# 按真实流逝时间衰减：0.6 在约 20 分钟无人理会后归零
WILLINGNESS_DECAY_PER_MINUTE = 0.03
# 被动增长：每条消息 × 兴趣系数；被动增长最多涨到上限，插嘴不至于满格
PASSIVE_GROWTH_PER_MESSAGE = 0.05
PASSIVE_GROWTH_CAP = 0.7
INTEREST_MIN_FACTOR = 0.3
INTEREST_MAX_FACTOR = 1.8
# 意愿达到该值才调用 Feedback 考虑插嘴
ENGAGE_THRESHOLD = 0.45
# 最近 SHORT_CONTEXT_LIMIT 条消息里自己的发言占比达到该值，就不再主动插嘴（被点名/聊天对象接话不受限）
SELF_SHARE_LIMIT = 0.2
# Feedback willing 的接话门槛：聊天对象在接话 / 主动插嘴；被点名时必回
SPEAK_THRESHOLD_CONVERSATION = 0.45
SPEAK_THRESHOLD_PASSIVE = 0.6
# 被点名（名字/别名、@、回复 Bot）时的意愿下限
RELEVANCE_WILLINGNESS_FLOOR = 0.85
# Bot 说话后的对话窗口：窗口内只有上次接话的那批人（聊天对象）再说话才算对话在继续
CONVERSATION_WINDOW_SECONDS = 180.0
# 说话后保留的意愿比例（避免同一话题连续抢话，但不至于掉出对话）
SPEAK_WILLINGNESS_RETAIN_FACTOR = 0.7
# 重启后的初始意愿
WILLINGNESS_LOAD_VALUE = 0.3


class ChattingState(Enum):
    """展示与 prompt 用的派生状态，不参与决策。"""

    IDLE = 0
    BUBBLE = 1
    ACTIVE = 2

    def __str__(self) -> str:
        return {
            ChattingState.IDLE: "潜水状态",
            ChattingState.BUBBLE: "冒泡状态",
            ChattingState.ACTIVE: "对话状态",
        }[self]


def in_conversation_window(state, now: datetime) -> bool:
    return (now - state.last_speak_time).total_seconds() < CONVERSATION_WINDOW_SECONDS


def chatting_state(state, now: datetime) -> ChattingState:
    if in_conversation_window(state, now):
        return ChattingState.ACTIVE
    if state.willingness >= ENGAGE_THRESHOLD:
        return ChattingState.BUBBLE
    return ChattingState.IDLE


@dataclass(frozen=True)
class EngagementDecision:
    relevant: bool
    in_conversation: bool
    poked: bool
    engaged: bool

    @property
    def speak_threshold(self) -> float:
        if self.relevant:
            return 0.0
        if self.in_conversation or self.poked:
            return SPEAK_THRESHOLD_CONVERSATION
        return SPEAK_THRESHOLD_PASSIVE


def evaluate_engagement(
    *,
    state,
    messages: list[Message],
    self_share: float,
    now: datetime,
) -> EngagementDecision:
    """意愿值衰减/增长，并判断这批消息要不要交给 Feedback。

    self_share 是最近上下文里自己发言的占比，用来限制主动插嘴的话量。
    """

    elapsed_minutes = max(0.0, (now - state.last_decay_time).total_seconds()) / 60.0
    state.willingness = max(
        0.0, state.willingness - elapsed_minutes * WILLINGNESS_DECAY_PER_MINUTE
    )
    state.last_decay_time = now

    relevant = check_relevance(state.name, state.aliases, messages)
    # 只有上次接话的那批人再开口才算对话在继续；Bot 自己说话不会给窗口续命到其他人身上
    conversing = in_conversation_window(state, now) and any(
        message.user_id in state.conversation_partners for message in messages
    )
    if relevant:
        state.willingness = max(state.willingness, RELEVANCE_WILLINGNESS_FLOOR)
    elif state.willingness < PASSIVE_GROWTH_CAP:
        interest = score_message_interest([message.content for message in messages])
        growth = PASSIVE_GROWTH_PER_MESSAGE * interest * len(messages)
        state.willingness = min(PASSIVE_GROWTH_CAP, state.willingness + growth)

    wants_to_chime_in = (
        state.willingness >= ENGAGE_THRESHOLD and self_share < SELF_SHARE_LIMIT
    )
    poked = any(message.poke and message.to_me for message in messages)
    return EngagementDecision(
        relevant=relevant,
        in_conversation=conversing,
        poked=poked,
        engaged=relevant or conversing or poked or wants_to_chime_in,
    )


def check_relevance(
    bot_name: str,
    aliases: list[str],
    messages: list[Message],
) -> bool:
    """@Bot、回复 Bot 的消息，或提到名字/别名（至少 2 个字）。

    戳一戳记录不算：被戳了回不回话交给 Feedback，不必回。
    """

    messages = [message for message in messages if not message.poke]
    if any(message.to_me for message in messages):
        return True
    triggers = [
        value.strip().lower()
        for value in [bot_name, *aliases]
        if value and len(value.strip()) >= 2
    ]
    return any(
        trigger in message.content.lower()
        for message in messages
        for trigger in triggers
    )


def score_message_interest(contents: list[str]) -> float:
    """这批消息有多值得插嘴：提问、有实质内容加分，纯复读/纯表情减分。"""

    text = " ".join(contents).strip()
    if not text:
        return INTEREST_MIN_FACTOR
    score = 1.0
    if "?" in text or "？" in text:
        score += 0.6
    if any(len(content.strip()) >= 15 for content in contents):
        score += 0.2
    if len(set(text)) <= 2 and len(text) >= 3:
        score -= 0.6
    if all(content.strip() in {"[图片]", "[表情包]"} for content in contents):
        score -= 0.5
    return max(INTEREST_MIN_FACTOR, min(INTEREST_MAX_FACTOR, score))
