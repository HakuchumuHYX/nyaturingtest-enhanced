from dataclasses import dataclass
from datetime import datetime

from .text_utils import check_relevance, score_message_interest


@dataclass(frozen=True)
class EngagementDecision:
    relevant: bool
    engaged: bool
    cooldown_remaining: float

    @property
    def may_reply(self) -> bool:
        return self.engaged and (self.relevant or self.cooldown_remaining <= 0)


class EngagementPolicy:
    """Own willingness decay/growth, hysteresis and speak cooldown."""

    def evaluate(
        self,
        *,
        session,
        messages: list,
        settings,
        now: datetime,
    ) -> EngagementDecision:
        last_decay = getattr(session, "_last_decay_time", None) or now
        elapsed_minutes = max(
            0.0,
            (now - last_decay).total_seconds(),
        ) / 60.0
        last_speak = self._naive_local(session._last_speak_time)
        idle = (
            now - last_speak
        ).total_seconds() >= settings["willingness_idle_after_seconds"]
        decay_rate = settings[
            "willingness_decay_rate_idle"
            if idle
            else "willingness_decay_rate_active"
        ]
        session.willingness = max(
            0.0,
            session.willingness - elapsed_minutes * decay_rate,
        )
        session._last_decay_time = now
        session._last_activity_time = now

        relevant = check_relevance(
            session.name(),
            session.aliases(),
            messages,
        )
        if relevant:
            session.willingness = max(
                session.willingness,
                settings["relevance_willingness_floor"],
            )
        elif session.willingness < settings["passive_willingness_growth_limit"]:
            interest = score_message_interest(
                [message.content for message in messages],
                bot_name=session.name(),
                aliases=session.aliases(),
                lo=settings["passive_growth_min_factor"],
                hi=settings["passive_growth_max_factor"],
            )
            growth = (
                settings["passive_willingness_growth_per_message"]
                * interest
                * len(messages)
            )
            session.willingness = min(1.0, session.willingness + growth)
            if interest >= 1.6:
                session.willingness = max(
                    session.willingness,
                    settings["interest_topic_willingness_floor"],
                )

        if relevant:
            session._engaged = True
        elif (
            session._engaged
            and session.willingness < settings["low_willingness_skip_threshold"]
        ):
            session._engaged = False
        elif (
            not session._engaged
            and session.willingness >= settings["willingness_reply_threshold"]
        ):
            session._engaged = True

        since_speak = (now - last_speak).total_seconds()
        cooldown_remaining = (
            0.0
            if since_speak < 0
            else max(0.0, settings["speak_cooldown_seconds"] - since_speak)
        )
        return EngagementDecision(
            relevant=relevant,
            engaged=session._engaged,
            cooldown_remaining=cooldown_remaining,
        )

    @staticmethod
    def _naive_local(value: datetime) -> datetime:
        if value.tzinfo is not None:
            return value.astimezone(None).replace(tzinfo=None)
        return value
