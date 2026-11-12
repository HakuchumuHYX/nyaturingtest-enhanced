"""看群时机与说话门槛。

被点名、聊天对象在接话、被戳就马上看，否则按节奏攒着。
被点名也不强制回：两个 Bot 互相回复时，只有 Feedback 看着发言量压低 willing 才停得下来。
她 away（手机放下了）时不自己看群，有人叫她交给 persona/status.py 判断拿不拿起手机。
"""

from datetime import datetime, timedelta
from enum import Enum

from ..messages.message import ChatMessage

# 两次发送之间的最小间隔，在发送前等
SPEAK_COOLDOWN_SECONDS = 16.0
# 看群节奏：未读攒够这么多条，或有未读且距上次看群满这么久，就看一次
LOOK_UNREAD_THRESHOLD = 8
LOOK_INTERVAL_SECONDS = 180.0
# 一次最多读这么多条、这么多字（未读和爬楼每批都是）；放不下的留给爬楼，不丢
LOOK_MAX_MESSAGES = 60
LOOK_MAX_CHARS = 6000
# 发言占比和给 Feedback 的 group_heat 只数这段时间里她的开口
OWN_TURN_WINDOW_SECONDS = 900
# 自己的开口占比（见 self_share）达到该值，就不主动插嘴（被点名/聊天对象接话/被戳不受限）；
# 按开口次数算，0.1 约等于最近 20 条里开口两次
SELF_SHARE_LIMIT = 0.1
# Feedback willing 的开口门槛：被点名 / 聊天对象在接话或被戳 / 按节奏看群时主动插嘴
SPEAK_THRESHOLD_ADDRESSED = 0.3
SPEAK_THRESHOLD_CONVERSATION = 0.45
SPEAK_THRESHOLD_PASSIVE = 0.5
# Bot 说话后的对话窗口：窗口内只有上次接话的那批人（聊天对象）再说话才算对话在继续
CONVERSATION_WINDOW_SECONDS = 180.0


class LookTrigger(Enum):
    RELEVANT = "relevant"
    CONVERSATION = "conversation"
    POKE = "poke"
    RHYTHM = "rhythm"

    @property
    def speak_threshold(self) -> float:
        if self is LookTrigger.RELEVANT:
            return SPEAK_THRESHOLD_ADDRESSED
        if self is LookTrigger.RHYTHM:
            return SPEAK_THRESHOLD_PASSIVE
        return SPEAK_THRESHOLD_CONVERSATION


def in_conversation_window(state, now: datetime) -> bool:
    return (now - state.last_speak_time).total_seconds() < CONVERSATION_WINDOW_SECONDS


def unread_count(state, unread: list[ChatMessage]) -> int:
    """未读里群友的消息条数；她自己刚发出去的不算。直接数，不另记计数器，免得和水位对不上。"""

    return sum(message.user_name != state.name for message in unread)


def fit_count(messages: list[ChatMessage]) -> int:
    """按给定顺序数，合计不超过 LOOK_MAX_CHARS 的有几条；第一条再长也算，不然水位卡在它前面推不动。"""

    total = 0
    for count, message in enumerate(messages):
        total += len(message.content)
        if count and total > LOOK_MAX_CHARS:
            return count
    return len(messages)


def own_turns(messages: list[ChatMessage], bot_name: str) -> list[ChatMessage | None]:
    """她的每次开口接的是谁（开口前最后一个群友，前面没人是 None）；连续发的几条算一次开口。"""

    turns: list[ChatMessage | None] = []
    last_other: ChatMessage | None = None
    previous_is_own = False
    for message in messages:
        if message.user_name != bot_name:
            last_other = message
            previous_is_own = False
            continue
        if not previous_is_own:
            turns.append(last_other)
        previous_is_own = True
    return turns


def self_share(messages: list[ChatMessage], bot_name: str, now: datetime) -> float:
    """她近 OWN_TURN_WINDOW_SECONDS 的开口次数 ÷（开口次数 + 群友条数），messages 是最近说的话。

    数开口不数条数：一次回复由 Chat 决定发几条。她更早的开口不算，慢群里很久以前说的话不拦着她；
    群友不限时间，窗口里只剩三四条消息时占比也不会虚高。
    """

    since = now - timedelta(seconds=OWN_TURN_WINDOW_SECONDS)
    turns = len(own_turns([m for m in messages if m.time >= since], bot_name))
    others = sum(m.user_name != bot_name for m in messages)
    return turns / max(turns + others, 1)


def seconds_until_look(
    state, unread: list[ChatMessage], now: datetime, away: bool
) -> float | None:
    """按节奏还要多久该看群；没有群友的未读、没设人设或 away 时返回 None。worker 用它给到点补看定时。"""

    if away or not state.name or unread_count(state, unread) == 0:
        return None
    elapsed = (now - state.last_look_time).total_seconds()
    return max(0.0, LOOK_INTERVAL_SECONDS - elapsed)


def decide_look(
    state, unread: list[ChatMessage], now: datetime, away: bool
) -> LookTrigger | None:
    """看全部未读决定这次看不看群；返回 None 就继续攒着。

    按全部未读而不是这一批判断：Feedback 失败、水位没动时，那条点名不会因为下一批没点名就漏掉。
    """

    if away or unread_count(state, unread) == 0:
        return None
    if check_relevance(state.name, state.aliases, unread):
        return LookTrigger.RELEVANT
    # 只有上次接话的那批人再开口才算对话在继续；Bot 自己说话不会给窗口续命到其他人身上
    if in_conversation_window(state, now) and any(
        message.user_id in state.conversation_partners for message in unread
    ):
        return LookTrigger.CONVERSATION
    if any(message.poke and message.to_me for message in unread):
        return LookTrigger.POKE
    if (
        unread_count(state, unread) >= LOOK_UNREAD_THRESHOLD
        or seconds_until_look(state, unread, now, away) == 0
    ):
        return LookTrigger.RHYTHM
    return None


def calls_me(state, message: ChatMessage) -> bool:
    """这条是在叫她：点名（同 check_relevance）或者戳她。away 时据此判断有没有人找她。"""

    return check_relevance(state.name, state.aliases, [message]) or (message.poke and message.to_me)


def check_relevance(
    bot_name: str,
    aliases: list[str],
    messages: list[ChatMessage],
) -> bool:
    """群友 @Bot、回复 Bot 的消息，或提到名字/别名（至少 2 个字）。

    角色自己的话不算：未读里有她刚发出去的消息。
    戳一戳、贴表情、群通知记录不算：被戳了回不回话交给 Feedback，不必回；贴表情记录里引的原文带名字也不是在叫 bot。
    合并转发也不算：读完后是别人的聊天记录，里面出现名字不是在叫 bot。
    群里的bot 发的不算：同一个号上的其他程序 @ 这个号、复读带她名字的话，都不是在叫她。
    """

    messages = [
        message
        for message in messages
        if message.user_name != bot_name
        and not message.is_action
        and not message.forward
        and not message.by_group_bot
    ]
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
