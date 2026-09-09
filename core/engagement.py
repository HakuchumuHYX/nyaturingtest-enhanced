from dataclasses import dataclass
from datetime import datetime

from .text_utils import check_relevance, score_message_interest


# 意愿与参与策略参数
SPEAK_COOLDOWN_SECONDS = 16.0
WILLINGNESS_IDLE_AFTER_SECONDS = 300.0
WILLINGNESS_DECAY_RATE_ACTIVE = 0.04
WILLINGNESS_DECAY_RATE_IDLE = 0.08
RELEVANCE_WILLINGNESS_FLOOR = 0.85
WILLINGNESS_REPLY_THRESHOLD = 0.47
INTEREST_TOPIC_WILLINGNESS_FLOOR = 0.34
SPEAK_WILLINGNESS_RETAIN_FACTOR = 0.35
WILLINGNESS_LOAD_VALUE = 0.1
PASSIVE_GROWTH_MIN_FACTOR = 0.25
PASSIVE_GROWTH_MAX_FACTOR = 1.8
PASSIVE_WILLINGNESS_GROWTH_LIMIT = 0.6
PASSIVE_WILLINGNESS_GROWTH_PER_MESSAGE = 0.026
LOW_WILLINGNESS_SKIP_THRESHOLD = 0.32
POST_FEEDBACK_SKIP_THRESHOLD = 0.38
ACTIVE_TO_BUBBLE_THRESHOLD = 0.50
RERANK_WILLINGNESS_THRESHOLD = 0.68


@dataclass(frozen=True)
class EngagementDecision:
    relevant: bool
    engaged: bool
    cooldown_remaining: float

    @property
    def may_reply(self) -> bool:
        return self.engaged and (self.relevant or self.cooldown_remaining <= 0)


class EngagementPolicy:
    """意愿值衰减/增长、参与态滞回与发言冷却。"""

    def evaluate(
        self,
        *,
        state,
        messages: list,
        now: datetime,
    ) -> EngagementDecision:
        last_decay = state.last_decay_time or now
        elapsed_minutes = max(0.0, (now - last_decay).total_seconds()) / 60.0
        last_speak = self._naive_local(state.last_speak_time)
        idle = (now - last_speak).total_seconds() >= WILLINGNESS_IDLE_AFTER_SECONDS
        decay_rate = WILLINGNESS_DECAY_RATE_IDLE if idle else WILLINGNESS_DECAY_RATE_ACTIVE
        state.willingness = max(0.0, state.willingness - elapsed_minutes * decay_rate)
        state.last_decay_time = now
        state.last_activity_time = now

        relevant = check_relevance(state.name, state.aliases, messages)
        if relevant:
            state.willingness = max(state.willingness, RELEVANCE_WILLINGNESS_FLOOR)
        elif state.willingness < PASSIVE_WILLINGNESS_GROWTH_LIMIT:
            interest = score_message_interest(
                [message.content for message in messages],
                bot_name=state.name,
                aliases=state.aliases,
                lo=PASSIVE_GROWTH_MIN_FACTOR,
                hi=PASSIVE_GROWTH_MAX_FACTOR,
            )
            growth = PASSIVE_WILLINGNESS_GROWTH_PER_MESSAGE * interest * len(messages)
            state.willingness = min(1.0, state.willingness + growth)
            if interest >= 1.6:
                state.willingness = max(state.willingness, INTEREST_TOPIC_WILLINGNESS_FLOOR)

        if relevant:
            state.engaged = True
        elif state.engaged and state.willingness < LOW_WILLINGNESS_SKIP_THRESHOLD:
            state.engaged = False
        elif (
            not state.engaged
            and state.willingness >= WILLINGNESS_REPLY_THRESHOLD
        ):
            state.engaged = True

        since_speak = (now - last_speak).total_seconds()
        cooldown_remaining = (
            0.0
            if since_speak < 0
            else max(0.0, SPEAK_COOLDOWN_SECONDS - since_speak)
        )
        return EngagementDecision(
            relevant=relevant,
            engaged=state.engaged,
            cooldown_remaining=cooldown_remaining,
        )

    @staticmethod
    def _naive_local(value: datetime) -> datetime:
        if value.tzinfo is not None:
            return value.astimezone(None).replace(tzinfo=None)
        return value
