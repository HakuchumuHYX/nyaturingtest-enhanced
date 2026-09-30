"""发言意愿与参与判定。

意愿值只由规则维护：随时间衰减、随群聊消息增长、被点名或正在对话时有下限。
它决定「值不值得调用 Feedback 考虑说话」；真正说不说由 Feedback 的 willing 与它平均后决定
（见 orchestrator._apply_decision），模型不再直接覆盖意愿值。
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
# 规则意愿与 Feedback willing 平均后达到该值才回复（被点名时不看这个）
SPEAK_THRESHOLD = 0.5
# 被点名（名字/别名、@、回复 Bot）时的意愿下限
RELEVANCE_WILLINGNESS_FLOOR = 0.85
# Bot 说话后的对话窗口：窗口内每批消息都会交给 Feedback 判断要不要接话
CONVERSATION_WINDOW_SECONDS = 180.0
CONVERSATION_WILLINGNESS_FLOOR = 0.55
# 说话后保留的意愿比例（避免同一话题连续抢话，但不至于掉出对话）
SPEAK_WILLINGNESS_RETAIN_FACTOR = 0.7
# 重启后的初始意愿
WILLINGNESS_LOAD_VALUE = 0.3
# 意愿高于该值或处于对话中时启用 Rerank
RERANK_WILLINGNESS_THRESHOLD = 0.68


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


def in_conversation(state, now: datetime) -> bool:
    return (now - state.last_speak_time).total_seconds() < CONVERSATION_WINDOW_SECONDS


def chatting_state(state, now: datetime) -> ChattingState:
    if in_conversation(state, now):
        return ChattingState.ACTIVE
    if state.willingness >= ENGAGE_THRESHOLD:
        return ChattingState.BUBBLE
    return ChattingState.IDLE


@dataclass(frozen=True)
class EngagementDecision:
    relevant: bool
    in_conversation: bool
    engaged: bool


def evaluate_engagement(
    *,
    state,
    messages: list[Message],
    now: datetime,
) -> EngagementDecision:
    """意愿值衰减/增长，并判断这批消息要不要交给 Feedback。"""

    elapsed_minutes = max(0.0, (now - state.last_decay_time).total_seconds()) / 60.0
    state.willingness = max(
        0.0, state.willingness - elapsed_minutes * WILLINGNESS_DECAY_PER_MINUTE
    )
    state.last_decay_time = now

    relevant = check_relevance(state.name, state.aliases, messages)
    conversing = in_conversation(state, now)
    if relevant:
        state.willingness = max(state.willingness, RELEVANCE_WILLINGNESS_FLOOR)
    elif state.willingness < PASSIVE_GROWTH_CAP:
        interest = score_message_interest([message.content for message in messages])
        growth = PASSIVE_GROWTH_PER_MESSAGE * interest * len(messages)
        state.willingness = min(PASSIVE_GROWTH_CAP, state.willingness + growth)
    if conversing:
        state.willingness = max(state.willingness, CONVERSATION_WILLINGNESS_FLOOR)

    return EngagementDecision(
        relevant=relevant,
        in_conversation=conversing,
        engaged=relevant or conversing or state.willingness >= ENGAGE_THRESHOLD,
    )


def check_relevance(
    bot_name: str,
    aliases: list[str],
    messages: list[Message],
) -> bool:
    """@Bot、回复 Bot 的消息，或提到名字/别名（至少 2 个字）。"""

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
