# nyaturingtest/mem.py
import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger

from .client import LLMClient
from .config import plugin_config

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

        # 待压缩缓冲区：专门存储还未合并进摘要的新消息
        # 如果是从数据库加载的 Session，这里初始化为空，假设旧消息都已处理过（或不追究）
        self._uncompressed_buffer: list[Message] = []

        self.__llm_client = llm_client
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
        self._uncompressed_buffer.clear()
        self.__compressed_message = ""
        if self.__compress_task and not self.__compress_task.done():
            self.__compress_task.cancel()
        logger.info("已清除所有记忆")

    async def __compress_message(self, messages_to_compress: list[Message],
                                 after_compress: Callable[[], None] | None = None):
        if not messages_to_compress:
            return

        # 构造新发生的对话文本，仅包含本次增量部分
        new_dialogue = [f"{msg.user_name}: {msg.content}" for msg in messages_to_compress]

        # Prompt 必须包含【旧摘要】和【新对话】，要求 LLM 进行合并更新
        prompt = f"""
请基于[旧记忆]和[新发生的对话]，生成一份新的、合并后的记忆摘要。

[旧记忆摘要]:
{self.__compressed_message or "(无)"}

[新发生的对话] (这是刚刚发生且尚未记录的内容):
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
            response = await self.__llm_client.generate_response(
                prompt,
                model=plugin_config.nyaturingtest_feedback_openai_model,
                temperature=0.3
            )

            if response:
                self.__compressed_message = response
                logger.info(f"记忆摘要更新成功 (合并了 {len(messages_to_compress)} 条新消息)")
                if after_compress:
                    after_compress()
            else:
                logger.warning("压缩消息失败：LLM 返回为空")
                # 如果失败，理论上这些消息应该放回缓冲区重试，但为了防止死循环，这里暂时丢弃或仅记录日志
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
        # 1. 更新上下文窗口 (Rolling Window)
        self.__messages.extend(message_chunk)

        # 2. 追加到待压缩缓冲区 (Accumulate)
        self._uncompressed_buffer.extend(message_chunk)

        # 如果缓冲区未达到阈值，暂不触发
        if len(self._uncompressed_buffer) < self.__length_limit:
            return

        # 并发控制：如果有正在进行的压缩任务，跳过本次触发
        # 注意：我们不清空 buffer，这样消息会继续积累，直到下一次成功获取任务锁
        if self.__compress_task and not self.__compress_task.done():
            logger.debug(f"上一次记忆压缩尚未完成，缓冲区积压中 ({len(self._uncompressed_buffer)})")
            return

        # 3. 原子性切分：取出当前 buffer 所有内容，清空 buffer，交给异步任务
        # 这样做可以防止任务执行期间新进来的消息被错误地清除或重复处理
        chunk_to_process = list(self._uncompressed_buffer)
        self._uncompressed_buffer.clear()

        # 启动新任务
        self.__compress_task = asyncio.create_task(
            self.__compress_message(chunk_to_process, after_compress=after_compress)
        )
