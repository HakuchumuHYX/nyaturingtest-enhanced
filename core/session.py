# nyaturingtest/session.py
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from enum import Enum
from typing import Any

import httpx
from nonebot import logger
from nonebot.utils import run_sync
from ..config import get_vector_dir
from ..models.emotion import EmotionState
from ..memory.vector import VectorMemory
from ..memory.short_term import Memory, Message
from .engagement import (
    LOW_WILLINGNESS_SKIP_THRESHOLD,
    WILLINGNESS_LOAD_VALUE,
)
from ..prompts.presets import PRESETS, reload_presets
from ..models.profile import PersonProfile
from ..database.message_repository import MessageRepository
from ..database.profile_repository import ProfileRepository
from ..database.session_repository import SessionStateRepository
from ..database.backup_lock import BACKUP_IO_LOCK
from .metrics import log_event
from .persistence import PersistenceCoordinator


# 角色与摘要文本上限、后台任务排空超时
ROLE_MAX_CHARS = 4000
EXAMPLES_MAX_CHARS = 2000
MEMORY_DRAIN_TIMEOUT_SECONDS = 10.0


class ChattingState(Enum):
    IDLE = 0
    BUBBLE = 1
    ACTIVE = 2

    def __str__(self) -> str:
        return {
            ChattingState.IDLE: "潜水状态",
            ChattingState.BUBBLE: "冒泡状态",
            ChattingState.ACTIVE: "对话状态",
        }[self]


@dataclass
class SessionState:
    """纯领域状态：不持有数据库、HTTP 或向量资源。"""

    name: str = "terminus"
    role: str = "一个男性人类"
    aliases: list[str] = field(default_factory=list)
    examples: str = ""
    profiles: dict[str, PersonProfile] = field(default_factory=dict)
    global_emotion: EmotionState = field(default_factory=EmotionState)
    chat_summary: str = ""
    willingness: float = 0.0
    chatting_state: ChattingState = ChattingState.IDLE
    last_activity_time: datetime = field(default_factory=datetime.now)
    last_decay_time: datetime = field(default_factory=datetime.now)
    last_speak_time: datetime = datetime.min
    active_count: int = 0
    engaged: bool = False
    last_consolidated_time: datetime | None = None
    messages_since_consolidation: int = 0
    last_consolidation_attempt: datetime = datetime.min
    loaded: bool = False
    generation: int = 0


@dataclass
class SessionRuntime:
    """一个 Session 持有的 I/O 资源与后台任务协调。"""

    short_term_memory: Any = None
    vector_memory: Any = None
    http_client: Any = None
    owns_http_client: bool = False
    persistence: Any = None
    background_tasks: set[asyncio.Task] = field(default_factory=set)
    save_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class FeedbackOutcome:
    """一次 Feedback 的结构化结果。accepted 与 recalled_history 分离，避免用空列表判断成功。"""

    accepted: bool
    recalled_history: list[str] = field(default_factory=list)
    state_changed: bool = False
    failure_reason: str = ""

    @classmethod
    def rejected(cls, reason: str) -> "FeedbackOutcome":
        return cls(accepted=False, failure_reason=reason)


def _limit_role_text(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[内容过长，已截断]"
    return text


STALE_GENERATION_WRITE = object()


class Session:
    """
    群聊会话
    """

    def __init__(
            self,
            siliconflow_api_key: str,
            id: str = "global",
            name: str = "terminus",
            http_client: httpx.AsyncClient | None = None
    ):
        self.id = id
        if http_client is None:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                timeout=60.0
            )
            owns_http_client = True
        else:
            owns_http_client = False

        self.state = SessionState(name=name)
        self.runtime = SessionRuntime(
            short_term_memory=Memory(),
            vector_memory=VectorMemory(
                api_key=siliconflow_api_key,
                persist_directory=str(get_vector_dir(id)),
            ),
            http_client=http_client,
            owns_http_client=owns_http_client,
        )
        self.runtime.persistence = PersistenceCoordinator(
            self._save_coordinated,
            task_factory=self._create_safe_task,
        )

    def bump_generation(self, reason: str = "") -> int:
        self.state.generation += 1
        log_event("session_generation_bumped",
            session_id=self.id,
            generation=self.state.generation,
            reason=reason,
        )
        return self.state.generation

    def is_generation_stale(self, expected_generation: int | None) -> bool:
        return expected_generation is not None and self.state.generation != expected_generation

    def _log_stale_generation(self, stage: str, expected_generation: int | None) -> None:
        log_event("stale_turn_discarded",
            session_id=self.id,
            stage=stage,
            expected_generation=expected_generation,
            current_generation=self.state.generation,
        )

    async def _run_sync_if_generation_current(
        self,
        func,
        *args,
        expected_generation: int | None = None,
        stage: str,
        **kwargs,
    ):
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation(stage, expected_generation)
            return STALE_GENERATION_WRITE

        def guarded():
            with BACKUP_IO_LOCK:
                if self.is_generation_stale(expected_generation):
                    return STALE_GENERATION_WRITE
                return func(*args, **kwargs)

        result = await run_sync(guarded)()
        if result is STALE_GENERATION_WRITE:
            self._log_stale_generation(stage, expected_generation)
        return result

    async def set_role(self, name: str, role: str):
        self.bump_generation("set_role")
        self.state.role = _limit_role_text(role, ROLE_MAX_CHARS)
        self.state.name = name
        self.state.aliases = []
        self.state.examples = ""
        await self.save_session()

    def role(self) -> str:
        return f"{self.state.name}（{self.state.role}）"

    def name(self) -> str:
        return self.state.name

    def aliases(self) -> list[str]:
        return list(self.state.aliases)

    async def reset(self):
        self.bump_generation("reset")
        self.state.name = "terminus"
        self.state.aliases = []
        self.state.role = "一个男性人类"
        self.state.examples = ""
        await self.runtime.short_term_memory.clear()
        self.runtime.vector_memory.clear()
        self.state.profiles = {}
        self.state.global_emotion = EmotionState()
        self.state.chat_summary = ""
        self.state.chatting_state = ChattingState.IDLE
        self.state.willingness = 0.0
        self.state.active_count = 0
        self.state.last_activity_time = datetime.now()
        self.state.last_decay_time = datetime.now()
        self.state.last_speak_time = datetime.min
        self.state.engaged = False
        self.state.last_consolidated_time = None
        self.state.messages_since_consolidation = 0
        self.state.last_consolidation_attempt = datetime.min
        # 清理数据库中的所有关联数据，并与后台持久化共用同一把锁：
        # 旧 generation 的后台写入要么已在删除前完成，要么拿锁后被跳过。
        async with self.runtime.save_lock:
            await SessionStateRepository.delete_session_data(self.id)
            await self._save_session_locked()
        logger.info(f"[Session {self.id}] 已完全重置（含数据库清理）")

    async def calm_down(self):
        self.bump_generation("calm_down")
        self.state.global_emotion = EmotionState()
        self.state.profiles = {}
        self.state.chatting_state = ChattingState.IDLE
        self.state.willingness = 0.0
        self.state.active_count = 0
        self.state.last_activity_time = datetime.now()
        self.state.engaged = False
        await self.save_session()

    async def reset_emotion(self):
        """仅重置情绪状态（VAD），不影响意愿值、聊天状态、记忆等"""
        self.bump_generation("reset_emotion")
        self.state.global_emotion = EmotionState()
        # 同时重置所有用户画像的情绪
        for profile in self.state.profiles.values():
            profile.emotion = EmotionState()
            profile.mark_dirty()
        logger.info(f"[Session {self.id}] 情绪已初始化 (VAD -> 0, 0, 0)")
        await self.save_session()

    def _create_safe_task(self, coro):
        """创建带异常捕获的后台任务"""
        task = asyncio.create_task(coro)
        task.add_done_callback(self._on_task_done)
        self.runtime.background_tasks.add(task)
        task.add_done_callback(self.runtime.background_tasks.discard)
        return task

    def _schedule_save_session(self, force_index: bool = False):
        coordinator = self._get_persistence_coordinator()
        coordinator.request(force_index=force_index)
        return coordinator

    def _get_persistence_coordinator(self) -> PersistenceCoordinator:
        coordinator = self.runtime.persistence
        if coordinator is None:
            coordinator = PersistenceCoordinator(
                self._save_coordinated,
                task_factory=self._create_safe_task,
            )
            self.runtime.persistence = coordinator
        return coordinator

    def begin_persistence_batch(self) -> None:
        self._get_persistence_coordinator().begin_batch()

    async def end_persistence_batch(self, *, flush: bool = False) -> bool:
        return await self._get_persistence_coordinator().end_batch(flush=flush)

    async def flush_persistence(self) -> bool:
        return await self._get_persistence_coordinator().flush()

    async def _save_coordinated(self, force_index: bool = False) -> bool:
        async with self.runtime.save_lock:
            return await self._save_session_locked(force_index=force_index)

    @staticmethod
    def _on_task_done(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[Session] 后台任务异常: {exc}")

    async def save_session(
        self,
        force_index: bool = False,
        expected_generation: int | None = None,
    ) -> bool:
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("save_session", expected_generation)
            return False
        async with self.runtime.save_lock:
            if self.is_generation_stale(expected_generation):
                self._log_stale_generation("save_session_locked", expected_generation)
                return False
            result = await self._save_session_locked(force_index=force_index)
            if result:
                self.runtime.persistence.mark_current_persisted()
            return result

    async def _save_session_locked(self, force_index: bool = False) -> bool:
        try:
            # 1. 保存基础状态
            await SessionStateRepository.save_session_state(
                self.id,
                {
                    "name": self.state.name,
                    "role": self.state.role,
                    "aliases": self.state.aliases,
                    "valence": self.state.global_emotion.valence,
                    "arousal": self.state.global_emotion.arousal,
                    "dominance": self.state.global_emotion.dominance,
                    "chat_summary": self.state.chat_summary,
                    "last_speak_time": self.state.last_speak_time,
                    "last_consolidated_time": self.state.last_consolidated_time,
                    "chatting_state": self.state.chatting_state.value
                }
            )

            # 2. 更新变化过的画像，避免高频保存重复写全量 profiles。
            dirty_profiles = {
                user_id: profile
                for user_id, profile in self.state.profiles.items()
                if profile.is_dirty
            }
            if dirty_profiles:
                await ProfileRepository.update_user_profiles(self.id, dirty_profiles)
                for profile in dirty_profiles.values():
                    profile.mark_clean()

            # 3. 只同步新增或内容被图片观察丰富过的消息。
            pending_messages = self.runtime.short_term_memory.pending_messages()
            if pending_messages:
                await MessageRepository.sync_messages(
                    self.id,
                    [message for message, _ in pending_messages],
                )
                self.runtime.short_term_memory.mark_persisted(pending_messages)

            logger.debug(f"[Session {self.id}] 数据库保存成功")
            return True
        except Exception as e:
            logger.warning(f"[Session {self.id}] 数据库保存警告: {e}")
            return False

    async def load_session(self):
        if self.state.loaded: return

        # 使用 Repository 加载完整数据
        data = await SessionStateRepository.load_full_session_data(self.id)
        
        if not data:
            logger.info(f"[Session {self.id}] 初始化新会话")
            self.state.loaded = True
            return
            
        session_db = data["session"]
        
        self.state.name = session_db.name
        self.state.role = _limit_role_text(session_db.role, ROLE_MAX_CHARS)
        self.state.aliases = session_db.aliases if session_db.aliases else []
        self.state.chat_summary = session_db.chat_summary
        self.state.global_emotion.valence = session_db.valence
        self.state.global_emotion.arousal = session_db.arousal
        self.state.global_emotion.dominance = session_db.dominance
        
        if session_db.last_speak_time:
            t = session_db.last_speak_time
            if t.tzinfo is not None:
                t = t.astimezone(None).replace(tzinfo=None)
            self.state.last_speak_time = t
        self.state.last_consolidated_time = session_db.last_consolidated_time
        self.state.chatting_state = ChattingState(session_db.chatting_state)

        if "[对话样本]" in self.state.role:
            parts = self.state.role.split("[对话样本]")
            if len(parts) > 1:
                self.state.examples = parts[1].strip()

        self.state.willingness = WILLINGNESS_LOAD_VALUE
        # 重启一致性：低意愿时强制回到潜水态，避免「状态=对话中但意愿=静音」的矛盾
        if self.state.willingness < LOW_WILLINGNESS_SKIP_THRESHOLD:
            self.state.chatting_state = ChattingState.IDLE
            self.state.engaged = False
        self.state.profiles = {}
        
        # 恢复用户画像
        for user_data in data["users"]:
            user_id = user_data["user_id"]
            profile = PersonProfile(user_id=user_id)
            profile.emotion.valence = user_data["valence"]
            profile.emotion.arousal = user_data["arousal"]
            profile.emotion.dominance = user_data["dominance"]
            profile.last_update_time = user_data["last_update_time"]
            profile.interaction_count = int(user_data.get("interaction_count") or 0)
            profile.first_interaction_at = user_data.get("first_interaction_at")
            profile.last_interaction_at = user_data.get("last_interaction_at")

            profile.mark_clean()
            self.state.profiles[user_id] = profile

        # 恢复短时记忆
        # 注意：这里将数据库中的 chat_summary 同步给 Memory，确保摘要不丢失
        self.runtime.short_term_memory = Memory(
            compressed_message=self.state.chat_summary,
            messages=data["messages"],
        )

        self.state.loaded = True
        logger.info(f"[Session {self.id}] 加载完成")

    def presets(self) -> list[str]:
        reload_presets()
        return [
            f"{filename}: {preset.name} {preset.role}"
            for filename, preset in PRESETS.items()
            if not preset.hidden
        ]

    async def load_preset(self, filename: str) -> bool:
        reload_presets()
        if not filename.endswith(".json") and f"{filename}.json" in PRESETS.keys():
            filename = f"{filename}.json"
        if filename not in PRESETS:
            return False

        self.bump_generation("load_preset")
        preset = PRESETS[filename]
        base_role = preset.role
        self.state.name = preset.name
        self.state.aliases = preset.aliases

        if preset.examples:
            ex_lines = []
            for ex in preset.examples:
                u = ex.get("user", "")
                b = ex.get("bot", "")
                if u and b:
                    ex_lines.append(f"用户: {u}\n{preset.name}: {b}")
            self.state.examples = _limit_role_text("\n".join(ex_lines), EXAMPLES_MAX_CHARS)
        else:
            self.state.examples = ""

        if self.state.examples:
            self.state.role = _limit_role_text(
                f"{base_role}\n\n[对话样本]\n{self.state.examples}",
                ROLE_MAX_CHARS,
            )
        else:
            self.state.role = _limit_role_text(base_role, ROLE_MAX_CHARS)

        await run_sync(self.runtime.vector_memory.delete_by_metadata)({"source": "preset"})

        preset_items: list[tuple[str, str]] = []
        preset_items.extend((item, "knowledge") for item in preset.knowledges)
        preset_items.extend((item, "relationship") for item in preset.relationships)
        preset_items.extend((item, "event") for item in preset.events)
        preset_items.extend((item, "bot_self") for item in preset.bot_self)
        to_add = [item for item, _ in preset_items]
        if to_add:
            metadatas = [
                {"source": "preset", "type": "rule", "subtype": subtype}
                for _, subtype in preset_items
            ]
            await run_sync(self.runtime.vector_memory.add_texts)(to_add, metadatas=metadatas)

        await self.save_session()
        return True

    def status(self) -> str:
        recent_messages = self.runtime.short_term_memory.access().messages
        recent_str = "\n".join([f"{m.user_name}: {m.content}" for m in recent_messages]) if recent_messages else "无"
        return f"""
名字：{self.state.name}
设定：{self.state.role}
意愿值：{self.state.willingness:.2f}
状态: {self.state.chatting_state}
情绪：V{self.state.global_emotion.valence:.2f} A{self.state.global_emotion.arousal:.2f} D{self.state.global_emotion.dominance:.2f}
后台任务数: {len(self.runtime.background_tasks)}
摘要：{self.state.chat_summary}
最近消息：
{recent_str}
"""

    async def append_self_message(self, content: str, msg_id: str, bot_user_id: str):
        """
        主动记录 Bot 自己的发言 (防止等待回显导致记忆延迟)
        """
        logger.debug(f"[Session {self.id}] 主动写入自身记忆: {content[:20]}... (ID: {msg_id})")
        msg = Message(
            time=datetime.now(),
            user_name=self.state.name,
            content=content,
            id=msg_id,
            user_id=bot_user_id
        )
        
        await self.runtime.short_term_memory.update([msg])

    async def record_incoming(self, messages_chunk: list[Message]) -> None:
        """写入短时记忆并累计固化窗口。"""

        await self.runtime.short_term_memory.update(messages_chunk)
        self.state.messages_since_consolidation += len(messages_chunk)
        self._schedule_save_session()

    async def update_without_trigger(self, messages_chunk: list[Message]):
        """
        仅更新记忆，不触发 LLM 回复 (用于处理回显)
        """
        if not messages_chunk: return
        logger.debug(f"[Session {self.id}] 处理回显消息 (Count: {len(messages_chunk)})")
        await self.runtime.short_term_memory.update(messages_chunk)
        self._schedule_save_session()

    async def drain_background_tasks(self, timeout: float | None = None):
        if timeout is None:
            timeout = MEMORY_DRAIN_TIMEOUT_SECONDS
        pending = [task for task in self.runtime.background_tasks if not task.done()]
        if not pending:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[Session {self.id}] 等待后台任务超时，取消 {len(pending)} 个任务")
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self):
        try:
            self.runtime.vector_memory.close()
        except Exception as e:
            logger.warning(f"[Session {self.id}] 关闭向量记忆失败: {e}")
        if self.runtime.http_client is not None and self.runtime.owns_http_client:
            try:
                await self.runtime.http_client.aclose()
            except Exception as e:
                logger.warning(f"[Session {self.id}] 关闭 HTTP 客户端失败: {e}")

    async def _save_interaction_log(
        self,
        user_id: str,
        delta: dict,
        expected_generation: int | None = None,
    ):
        await self._save_interaction_logs(
            [(user_id, delta)],
            expected_generation=expected_generation,
        )

    async def _save_interaction_logs(
        self,
        interactions: list[tuple[str, dict]],
        expected_generation: int | None = None,
    ):
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("interaction_log", expected_generation)
            return
        async with self.runtime.save_lock:
            if self.is_generation_stale(expected_generation):
                self._log_stale_generation("interaction_log_locked", expected_generation)
                return
            await ProfileRepository.log_interactions(self.id, interactions)
