from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SHORT_CONTEXT_LIMIT = 20
SHORT_TERM_BUFFER_SIZE = 200


@dataclass
class Message:
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
    revision: int = field(default=0, repr=False, compare=False)
    _persistence_id: str = field(default="", repr=False, compare=False)


class Memory:
    """短时消息窗口；话题摘要只存在 SessionState.chat_summary，这里不再保留副本。"""

    def __init__(self, messages: list[Message] | None = None):
        # 保留缓冲区，access 只返回最近 SHORT_CONTEXT_LIMIT 条
        self.__messages = deque(messages or [], maxlen=SHORT_TERM_BUFFER_SIZE)
        self.__dirty_messages: dict[int, Message] = {}

    def clear(self) -> None:
        self.__messages.clear()
        self.__dirty_messages.clear()

    def access(self) -> list[Message]:
        return list(self.__messages)[-SHORT_CONTEXT_LIMIT:]

    def pending_messages(self) -> list[tuple[Message, int]]:
        return [
            (message, message.revision) for message in self.__dirty_messages.values()
        ]

    def mark_persisted(self, persisted: list[tuple[Message, int]]) -> None:
        for message, revision in persisted:
            if message.revision == revision:
                self.__dirty_messages.pop(id(message), None)

    def mark_dirty(self, message: Message) -> None:
        message.revision += 1
        self.__dirty_messages[id(message)] = message

    def messages_after(self, watermark: datetime | None, limit: int) -> list[Message]:
        """固化水位之后的缓冲消息，最多 limit 条。"""

        messages = list(self.__messages)
        if watermark is not None:
            watermark_ts = watermark.timestamp()
            messages = [m for m in messages if m.time.timestamp() > watermark_ts]
        return messages[-limit:]

    def update(self, message_chunk: list[Message]) -> None:
        """追加到滚动窗口，按消息 ID 去重。"""

        existing_ids = {msg.id for msg in self.__messages if msg.id}
        for message in message_chunk:
            if message.id and message.id in existing_ids:
                continue
            if message.id:
                existing_ids.add(message.id)
            self.__messages.append(message)
            self.mark_dirty(message)
