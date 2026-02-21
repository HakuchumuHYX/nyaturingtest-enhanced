# nyaturingtest/mem.py
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger

from ..llm.client import LLMClient
from ..config import plugin_config


@dataclass
class Message:
    time: datetime
    user_name: str
    content: str
    id: str = ""
    user_id: str = ""

    def to_json(self) -> dict:
        return {
            "time": self.time.isoformat(),
            "user_name": self.user_name,
            "content": self.content,
            "id": self.id,
            "user_id": self.user_id,
        }

    @staticmethod
    def from_json(data: dict) -> "Message":
        return Message(
            time=datetime.fromisoformat(data["time"]),
            user_name=data["user_name"],
            content=data["content"],
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
        )


@dataclass
class MemoryRecord:
    messages: list[Message]
    compressed_history: str


class Memory:
    def __init__(
            self,
            llm_client: LLMClient,
            compressed_message: str | None = None,
            messages: list[Message] | None = None,
            length_limit: int = 10,
    ):
        self.__length_limit = length_limit
        self.__compressed_message = compressed_message or ""
        # 上下文窗口：仅用于对话上下文，保留最近 N 条
        self.__messages = deque(messages, maxlen=length_limit * 5) if messages else deque(maxlen=length_limit * 5)
        # 不再使用内部 LLMClient，因为压缩逻辑已移交给 Session 统一管理
        # self.__llm_client = llm_client 

    def related_users(self) -> list[str]:
        """
        获取相关用户列表 (返回 user_id 优先)
        """
        return list({msg.user_id if msg.user_id else msg.user_name for msg in self.__messages})

    async def clear(self) -> None:
        """
        清除所有记忆
        """
        self.__messages.clear()
        self.__compressed_message = ""
        logger.info("已清除所有记忆")

    def update_summary(self, new_summary: str):
        """
        手动更新记忆摘要
        """
        if new_summary:
            self.__compressed_message = new_summary

    def access(self) -> MemoryRecord:
        return MemoryRecord(
            messages=list(self.__messages)[-self.__length_limit:],
            compressed_history=self.__compressed_message,
        )

    async def update(self, message_chunk: list[Message], after_compress: Callable[[], None] | None = None):
        """
        仅更新上下文窗口，不再触发后台压缩任务
        增加基于 message_id 的去重逻辑
        """
        existing_ids = {msg.id for msg in self.__messages if msg.id}
        
        to_add = []
        for m in message_chunk:
            # 如果消息有ID且已存在，则跳过
            if m.id and str(m.id) in existing_ids:
                continue
            to_add.append(m)
            # 把新加的ID也放入集合，防止本次chunk内部重复（虽然不太可能）
            if m.id:
                existing_ids.add(str(m.id))

        if to_add:
            # 1. 更新上下文窗口 (Rolling Window)
            self.__messages.extend(to_add)
        
        # 为了兼容性，保留 after_compress 参数但暂不使用
        if after_compress:
            pass
