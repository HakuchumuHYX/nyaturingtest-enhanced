# nyaturingtest/mem.py
import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger

from .client import LLMClient


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
        self.__messages = deque(messages, maxlen=length_limit * 5) if messages else deque(maxlen=length_limit * 5)
        self.__llm_client = llm_client
        self.__compress_counter = 0
        self.__compress_task: asyncio.Task | None = None

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
        self.__compress_counter = 0
        if self.__compress_task and not self.__compress_task.done():
            self.__compress_task.cancel()
        logger.info("已清除所有记忆")

    async def __compress_message(self, after_compress: Callable[[], None] | None = None):
        # 对当前队列进行快照，防止异步修改
        current_msgs = list(self.__messages)
        # 只取最近的一部分用于"更新"摘要，防止Token溢出，这里取全部当前缓存
        new_dialogue = [f"{msg.user_name}: {msg.content}" for msg in current_msgs]

        # Prompt 必须包含【旧摘要】和【新对话】，要求 LLM 进行合并更新
        prompt = f"""
请基于[旧记忆]和[新发生的对话]，生成一份新的、合并后的记忆摘要。

[旧记忆摘要]:
{self.__compressed_message or "(无)"}

[新发生的对话]:
{new_dialogue}

要求:
1. 你的任务是"更新记忆"，不要遗忘[旧记忆]中的关键信息（如人物关系、重要事件）。
2. 将[新对话]中的核心信息（话题、观点、发生的事件）合并进去。
3. 丢弃琐碎的闲聊。
4. 输出格式（参考）:
[当前话题: xxx]
[重要信息: xxx]
[参与者印象: UserA(性格/观点), UserB(...)]

请直接输出新的记忆摘要。
"""
        try:
            # 适当调高 temperature 增加灵活性，或者调低保证稳定
            response = await self.__llm_client.generate_response(prompt, model="Pro/Qwen/Qwen2.5-7B-Instruct", temperature=0.3)

            if response:
                self.__compressed_message = response
                logger.info(f"记忆摘要更新成功 (长度: {len(response)})")
                if after_compress:
                    after_compress()
            else:
                logger.warning("压缩消息失败：LLM 返回为空")
        except asyncio.CancelledError:
            logger.info("压缩任务被取消")
            raise
        except Exception as e:
            logger.error(f"压缩消息时发生错误: {e}")

    def access(self) -> MemoryRecord:
        return MemoryRecord(
            messages=list(self.__messages)[-self.__length_limit:],
            compressed_history=self.__compressed_message,
        )

    async def update(self, message_chunk: list[Message], after_compress: Callable[[], None] | None = None):
        self.__messages.extend(message_chunk)

        self.__compress_counter += len(message_chunk)

        # 如果还未达到阈值，直接返回
        if self.__compress_counter < self.__length_limit:
            return

        # 并发控制：如果有正在进行的压缩任务，跳过本次触发，避免频繁取消导致饥饿
        if self.__compress_task and not self.__compress_task.done():
            logger.debug("上一次记忆压缩尚未完成，跳过本次触发")
            return

        self.__compress_counter = 0
        # 启动新任务
        self.__compress_task = asyncio.create_task(self.__compress_message(after_compress=after_compress))
