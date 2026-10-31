from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SHORT_CONTEXT_LIMIT = 20
SHORT_TERM_BUFFER_SIZE = 200


@dataclass
class ChatMessage:
    time: datetime
    user_name: str
    content: str
    id: str = ""
    user_id: str = ""
    # 原生多模态输入，仅当前进程短期持有；不序列化、不写数据库。
    image_inputs: list[Any] = field(default_factory=list, repr=False, compare=False)
    # 被 @ 和被回复的人（QQ 号 -> 群名片），让没发言的人也能在写记忆时挂上号；不落库
    mentions: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    # 是否 @Bot 或回复了 Bot 的消息；只用于本轮相关性判断，不落库
    to_me: bool = field(default=False, repr=False, compare=False)
    # 戳一戳通知转成的记录；被戳不算点名，只让 bot 看一眼、自己决定理不理。不落库
    poke: bool = field(default=False, repr=False, compare=False)
    # 合并转发：读完后内容是别人的聊天记录，里面出现 bot 的名字不算点名。不落库
    forward: bool = field(default=False, repr=False, compare=False)
    # 后台看图的任务（convert.fill_caption）；处理这批消息前先等它们。不落库
    caption_tasks: list[Any] = field(default_factory=list, repr=False, compare=False)
    # 群友自己说了东西（convert.read_message 按消息段算），只有这种才拿去检索记忆。不落库
    substantive: bool = field(default=False, repr=False, compare=False)
    revision: int = field(default=0, repr=False, compare=False)
    _persistence_id: str = field(default="", repr=False, compare=False)

    def recall(self, content: str) -> None:
        """撤回：内容原地改成撤回标记。图片一并清掉，免得进行中这一轮的图片观察再把内容写回去。"""

        self.content = content
        self.image_inputs = []
        self.substantive = False


class Memory:
    """短时消息窗口；话题摘要只存在 SessionState.chat_summary，这里不再保留副本。"""

    def __init__(self, messages: list[ChatMessage] | None = None):
        # 保留缓冲区，access 只返回最近 SHORT_CONTEXT_LIMIT 条
        self.__messages = deque(messages or [], maxlen=SHORT_TERM_BUFFER_SIZE)
        self.__dirty_messages: dict[int, ChatMessage] = {}

    def clear(self) -> None:
        self.__messages.clear()
        self.__dirty_messages.clear()

    def access(self) -> list[ChatMessage]:
        return list(self.__messages)[-SHORT_CONTEXT_LIMIT:]

    def pending_messages(self) -> list[tuple[ChatMessage, int]]:
        return [
            (message, message.revision) for message in self.__dirty_messages.values()
        ]

    def mark_persisted(self, persisted: list[tuple[ChatMessage, int]]) -> None:
        for message, revision in persisted:
            if message.revision == revision:
                self.__dirty_messages.pop(id(message), None)

    def mark_dirty(self, message: ChatMessage) -> None:
        message.revision += 1
        self.__dirty_messages[id(message)] = message

    def recall(self, msg_id: str, content: str) -> bool:
        """按 msg_id 把缓冲里的一条消息改成撤回标记；不在缓冲里返回 False。"""

        for message in self.__messages:
            if message.id == msg_id:
                message.recall(content)
                self.mark_dirty(message)
                return True
        return False

    def messages_after(self, watermark: datetime | None, limit: int) -> list[ChatMessage]:
        """水位之后的缓冲消息，最多 limit 条（取最新的）。"""

        messages = list(self.__messages)
        if watermark is not None:
            watermark_ts = watermark.timestamp()
            messages = [m for m in messages if m.time.timestamp() > watermark_ts]
        return messages[-limit:]

    def messages_before(self, moment: datetime, limit: int) -> list[ChatMessage]:
        """早于 moment 的缓冲消息，最多 limit 条（取最新的）。"""

        moment_ts = moment.timestamp()
        return [m for m in self.__messages if m.time.timestamp() < moment_ts][-limit:]

    def move_to_latest(self, message: ChatMessage) -> None:
        """内容补全后才算出现（合并转发读完摘要）：时间改成现在、挪到窗口末尾。

        已读水位按时间算，不挪的话占位已经被读过，摘要就永远进不了未读。
        """

        message.time = datetime.now()
        for index, existing in enumerate(self.__messages):
            if existing is message:
                del self.__messages[index]
                self.__messages.append(message)
                break
        self.mark_dirty(message)

    def update(self, message_chunk: list[ChatMessage]) -> None:
        """追加到滚动窗口，按消息 ID 去重。"""

        existing_ids = {msg.id for msg in self.__messages if msg.id}
        for message in message_chunk:
            if message.id and message.id in existing_ids:
                continue
            if message.id:
                existing_ids.add(message.id)
            self.__messages.append(message)
            self.mark_dirty(message)
