import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from nonebot import logger

from ..storage.db import (
    delete_session_data,
    load_full_session_data,
    log_interactions,
    save_session_state,
    sync_messages,
    update_user_profiles,
)
from .domain import EmotionState, PersonProfile
from ..memory.short_term import Memory, Message
from ..memory.vector import VectorMemory
from .engagement import WILLINGNESS_LOAD_VALUE, chatting_state
from .metrics import log_event
from .prompts import PRESETS, reload_presets, truncate_text

# 角色与摘要文本上限、后台任务排空超时、保存去抖
ROLE_MAX_CHARS = 4000
EXAMPLES_MAX_CHARS = 2000
MEMORY_DRAIN_TIMEOUT_SECONDS = 10.0
SAVE_DEBOUNCE_SECONDS = 0.05


@dataclass
class SessionState:
    """纯领域状态：不持有数据库、HTTP 或向量资源。"""

    name: str = "terminus"
    role: str = "一个男性人类"
    aliases: list[str] = field(default_factory=list)
    examples: str = ""
    preset_lines: list[str] = field(default_factory=list)
    profiles: dict[str, PersonProfile] = field(default_factory=dict)
    # 档案与群志只由每日整理任务写库，这里是只读副本，不随 save_session 回写
    user_summaries: dict[str, str] = field(default_factory=dict)
    group_notes: str = ""
    global_emotion: EmotionState = field(default_factory=EmotionState)
    chat_summary: str = ""
    willingness: float = 0.0
    last_decay_time: datetime = field(default_factory=datetime.now)
    last_speak_time: datetime = datetime.min
    # 上次接话时那批消息的发言人；只在进程内保持，重启后对话窗口自然失效
    conversation_partners: set[str] = field(default_factory=set)
    last_consolidated_time: datetime | None = None
    messages_since_consolidation: int = 0
    last_consolidation_attempt: datetime = datetime.min
    loaded: bool = False
    generation: int = 0


@dataclass
class SessionRuntime:
    """一个 Session 持有的 I/O 资源与后台任务。"""

    short_term_memory: Memory
    vector_memory: VectorMemory
    background_tasks: set[asyncio.Task] = field(default_factory=set)
    save_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Session:
    """
    群聊会话
    """

    def __init__(self, id: str):
        self.id = id
        self.state = SessionState()
        self.runtime = SessionRuntime(
            short_term_memory=Memory(),
            vector_memory=VectorMemory(id),
        )
        # 保存请求去抖 + 单飞：频繁的 schedule_save 合并成一次后台写库
        self._save_pending = False
        self._save_task: asyncio.Task | None = None

    def bump_generation(self, reason: str) -> int:
        self.state.generation += 1
        log_event(
            "session_generation_bumped",
            session_id=self.id,
            generation=self.state.generation,
            reason=reason,
        )
        return self.state.generation

    def stale(self, generation: int, stage: str) -> bool:
        """本轮开始后会话被 reset/set_role 等作废过，就丢弃这一步并记一条事件。"""

        if self.state.generation == generation:
            return False
        log_event(
            "stale_turn_discarded",
            session_id=self.id,
            stage=stage,
            expected_generation=generation,
            current_generation=self.state.generation,
        )
        return True

    async def set_role(self, name: str, role: str):
        self.bump_generation("set_role")
        self.state.role = truncate_text(role, ROLE_MAX_CHARS)
        self.state.name = name
        self.state.aliases = []
        self.state.examples = ""
        await self.save_session()

    async def reset(self):
        generation = self.bump_generation("reset")
        self.state = SessionState(loaded=True, generation=generation)
        self.runtime.short_term_memory.clear()
        await self.runtime.vector_memory.clear()
        # 清理数据库中的所有关联数据，并与后台持久化共用同一把锁：
        # 旧 generation 的后台写入要么已在删除前完成，要么拿锁后被跳过。
        async with self.runtime.save_lock:
            await delete_session_data(self.id)
            await self._save_session_locked()
        logger.info(f"[Session {self.id}] 已完全重置（含数据库清理）")

    async def calm_down(self):
        self.bump_generation("calm_down")
        self.state.global_emotion = EmotionState()
        self.state.profiles = {}
        self.state.willingness = 0.0
        # 退出对话窗口
        self.state.last_speak_time = datetime.min
        await self.save_session()

    async def reset_emotion(self):
        """仅重置情绪状态（VAD），不影响意愿值、聊天状态、记忆等"""
        self.bump_generation("reset_emotion")
        self.state.global_emotion = EmotionState()
        # 同时重置所有用户画像的情绪
        for profile in self.state.profiles.values():
            profile.emotion = EmotionState()
            profile.dirty = True
        logger.info(f"[Session {self.id}] 情绪已初始化 (VAD -> 0, 0, 0)")
        await self.save_session()

    def spawn(self, coro) -> asyncio.Task:
        """创建受 drain_background_tasks 管理、异常会记日志的后台任务"""
        task = asyncio.create_task(coro)
        self.runtime.background_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task):
        self.runtime.background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[Session {self.id}] 后台任务异常: {exc}")

    def schedule_save(self) -> None:
        self._save_pending = True
        if self._save_task is None or self._save_task.done():
            self._save_task = self.spawn(self._run_pending_saves())

    async def _run_pending_saves(self) -> None:
        await asyncio.sleep(SAVE_DEBOUNCE_SECONDS)
        while self._save_pending:
            self._save_pending = False
            await self.save_session()

    async def flush_persistence(self) -> None:
        task = self._save_task
        if task is not None and not task.done():
            await asyncio.shield(task)

    async def save_session(self) -> bool:
        async with self.runtime.save_lock:
            return await self._save_session_locked()

    async def _save_session_locked(self) -> bool:
        try:
            # 1. 保存基础状态
            await save_session_state(
                self.id,
                {
                    "name": self.state.name,
                    "role": self.state.role,
                    "aliases": self.state.aliases,
                    "preset_lines": self.state.preset_lines,
                    "valence": self.state.global_emotion.valence,
                    "arousal": self.state.global_emotion.arousal,
                    "dominance": self.state.global_emotion.dominance,
                    "chat_summary": self.state.chat_summary,
                    "last_speak_time": self.state.last_speak_time,
                    "last_consolidated_time": self.state.last_consolidated_time,
                    # 派生值，只为在库里能看到当时的状态；加载时不读
                    "chatting_state": chatting_state(self.state, datetime.now()).value,
                },
            )

            # 2. 更新变化过的画像，避免高频保存重复写全量 profiles。
            dirty_profiles = {
                user_id: profile
                for user_id, profile in self.state.profiles.items()
                if profile.dirty
            }
            if dirty_profiles:
                await update_user_profiles(self.id, dirty_profiles)
                for profile in dirty_profiles.values():
                    profile.dirty = False

            # 3. 只同步新增或内容被图片观察丰富过的消息。
            pending_messages = self.runtime.short_term_memory.pending_messages()
            if pending_messages:
                await sync_messages(
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
        if self.state.loaded:
            return

        data = await load_full_session_data(self.id)

        if not data:
            logger.info(f"[Session {self.id}] 初始化新会话")
            self.state.loaded = True
            return

        session_db = data["session"]

        self.state.name = session_db.name
        self.state.role = truncate_text(session_db.role, ROLE_MAX_CHARS)
        self.state.aliases = session_db.aliases if session_db.aliases else []
        self.state.preset_lines = session_db.preset_lines
        self.state.group_notes = session_db.group_notes
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

        if "[对话样本]" in self.state.role:
            parts = self.state.role.split("[对话样本]")
            if len(parts) > 1:
                self.state.examples = parts[1].strip()

        self.state.willingness = WILLINGNESS_LOAD_VALUE
        self.state.profiles = {}
        self.state.user_summaries = {
            user_data["user_id"]: user_data["summary"]
            for user_data in data["users"]
            if user_data["summary"]
        }

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

            profile.dirty = False
            self.state.profiles[user_id] = profile

        self.runtime.short_term_memory = Memory(messages=data["messages"])

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
            self.state.examples = truncate_text("\n".join(ex_lines), EXAMPLES_MAX_CHARS)
        else:
            self.state.examples = ""

        if self.state.examples:
            self.state.role = truncate_text(
                f"{base_role}\n\n[对话样本]\n{self.state.examples}",
                ROLE_MAX_CHARS,
            )
        else:
            self.state.role = truncate_text(base_role, ROLE_MAX_CHARS)

        # 预设每轮完全相同，排序固定，才能作为不变的前缀参与缓存
        preset_items: list[tuple[str, str]] = []
        preset_items.extend((item, "knowledge") for item in preset.knowledges)
        preset_items.extend((item, "relationship") for item in preset.relationships)
        preset_items.extend((item, "event") for item in preset.events)
        preset_items.extend((item, "bot_self") for item in preset.bot_self)
        self.state.preset_lines = sorted(
            f"【设定/{subtype}】 {item}" for item, subtype in preset_items if item.strip()
        )

        await self.save_session()
        return True

    def status(self) -> str:
        recent_messages = self.runtime.short_term_memory.access()
        recent_str = (
            "\n".join([f"{m.user_name}: {m.content}" for m in recent_messages])
            if recent_messages
            else "无"
        )
        return f"""
名字：{self.state.name}
设定：{self.state.role}
意愿值：{self.state.willingness:.2f}
状态: {chatting_state(self.state, datetime.now())}
情绪：V{self.state.global_emotion.valence:.2f} A{self.state.global_emotion.arousal:.2f} D{self.state.global_emotion.dominance:.2f}
后台任务数: {len(self.runtime.background_tasks)}
摘要：{self.state.chat_summary}
最近消息：
{recent_str}
"""

    def append_self_message(self, content: str, msg_id: str, bot_user_id: str):
        """
        主动记录 Bot 自己的发言 (防止等待回显导致记忆延迟)
        """
        logger.debug(
            f"[Session {self.id}] 主动写入自身记忆: {content[:20]}... (ID: {msg_id})"
        )
        msg = Message(
            time=datetime.now(),
            user_name=self.state.name,
            content=content,
            id=msg_id,
            user_id=bot_user_id,
        )
        self.runtime.short_term_memory.update([msg])

    def record_reaction(self, msg: Message) -> None:
        """群友给 bot 的消息贴了表情：只进短期记忆，不进固化计数，也不触发新一轮对话。

        一条消息常被好几个人贴同一个表情，已有相同记录就跳过，免得挤满上下文窗口。
        """

        memory = self.runtime.short_term_memory
        if any(m.content == msg.content for m in memory.access()):
            return
        memory.update([msg])
        self.schedule_save()

    def record_incoming(self, messages_chunk: list[Message]) -> None:
        """写入短时记忆并累计固化窗口。"""

        self.runtime.short_term_memory.update(messages_chunk)
        self.state.messages_since_consolidation += len(messages_chunk)
        self.schedule_save()

    async def drain_background_tasks(self):
        pending = [task for task in self.runtime.background_tasks if not task.done()]
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=MEMORY_DRAIN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[Session {self.id}] 等待后台任务超时，取消 {len(pending)} 个任务"
            )
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def save_interaction_logs(
        self, interactions: list[tuple[str, dict]], generation: int
    ):
        async with self.runtime.save_lock:
            if self.stale(generation, "interaction_log"):
                return
            await log_interactions(self.id, interactions)
