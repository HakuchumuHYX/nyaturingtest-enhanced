"""看群时机与说话门槛。

一次 Feedback 就是「看一眼群」：读上次看过之后的全部未读，沉淀和「说不说」用同一次输出。
本模块决定什么时候看：被点名、聊天对象在接话、被戳就马上看，否则按节奏攒着；以及这次开口的门槛。
被点名也不强制回：两个 Bot 互相回复时，只有 Feedback 看着发言量压低 willing 才停得下来。
"""

from datetime import datetime
from enum import Enum

from ..messages.message import ChatMessage

# 两次发送之间的最小间隔（在发送前等待，不再跳过整轮）
SPEAK_COOLDOWN_SECONDS = 16.0
# 看群节奏：未读攒够这么多条，或有未读且距上次看群满这么久，就看一次；一次最多读这么多条
LOOK_UNREAD_THRESHOLD = 8
LOOK_INTERVAL_SECONDS = 180.0
LOOK_MAX_MESSAGES = 60
# 最近 SHORT_CONTEXT_LIMIT 条消息里自己的发言占比达到该值，就不再主动插嘴（被点名/聊天对象接话/被戳不受限）
SELF_SHARE_LIMIT = 0.2
# Feedback willing 的开口门槛：被点名 / 聊天对象在接话或被戳 / 按节奏看群时主动插嘴。
# 主动插嘴原为 0.6：试跑里「接得上但一般」的段落模型常给 0.42～0.55，卡在门槛下；先降到 0.5 跑几天看频率
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


def seconds_until_look(state, unread: list[ChatMessage], now: datetime) -> float | None:
    """按节奏还要多久该看群；没有群友的未读（或没设人设）返回 None。worker 用它给到点补看定时。"""

    if not state.name or unread_count(state, unread) == 0:
        return None
    elapsed = (now - state.last_look_time).total_seconds()
    return max(0.0, LOOK_INTERVAL_SECONDS - elapsed)


def decide_look(state, unread: list[ChatMessage], now: datetime) -> LookTrigger | None:
    """看全部未读决定这次看不看群；返回 None 就继续攒着。

    按全部未读而不是这一批判断：Feedback 失败、水位没动时，那条点名不会因为下一批没点名就漏掉。
    """

    if unread_count(state, unread) == 0:
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
        or seconds_until_look(state, unread, now) == 0
    ):
        return LookTrigger.RHYTHM
    return None


def check_relevance(
    bot_name: str,
    aliases: list[str],
    messages: list[ChatMessage],
) -> bool:
    """群友 @Bot、回复 Bot 的消息，或提到名字/别名（至少 2 个字）。

    角色自己的话不算：未读里有她刚发出去的消息。
    戳一戳记录不算：被戳了回不回话交给 Feedback，不必回。
    合并转发也不算：读完后是别人的聊天记录，里面出现名字不是在叫 bot。
    """

    messages = [
        message
        for message in messages
        if message.user_name != bot_name and not message.poke and not message.forward
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
