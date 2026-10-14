import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from nonebot import logger

from ..storage.db import (
    delete_session_data,
    load_full_session_data,
    load_persona_texts,
    save_session_state,
    skip_persona_absence,
    sync_messages,
    update_user_profiles,
)
from .domain import EmotionState, PersonProfile
from ..memory.short_term import Memory, Message
from ..memory.vector import VectorMemory
from .engagement import WILLINGNESS_LOAD_VALUE, chatting_state
from .metrics import log_event
from .prompts import PRESETS, reload_presets

# 后台任务排空超时、保存去抖
MEMORY_DRAIN_TIMEOUT_SECONDS = 10.0
SAVE_DEBOUNCE_SECONDS = 0.05


@dataclass
class SessionState:
    """纯领域状态：不持有数据库、HTTP 或向量资源。"""

    preset: str = ""
    # 以下由 _apply_preset 从预设和每日整理结果填，只读，不随 save_session 回写；
    # name 为空说明没设人设或预设文件不在，这个群不说话
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)
    bottom_lines: list[str] = field(default_factory=list)
    preset_lines: list[str] = field(default_factory=list)
    self_life: str = ""
    self_in_group: str = ""
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
        """本轮开始后会话被 reset/load_preset 等作废过，就丢弃这一步并记一条事件。"""

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

    async def reset(self):
        generation = self.bump_generation("reset")
        # 重置的是本群的数据，不是换人：人设留着，群层随 delete_session_data 清掉
        self.state = SessionState(loaded=True, generation=generation, preset=self.state.preset)
        self.runtime.short_term_memory.clear()
        await self.runtime.vector_memory.clear()
        # 清理数据库中的所有关联数据，并与后台持久化共用同一把锁：
        # 旧 generation 的后台写入要么已在删除前完成，要么拿锁后被跳过。
        async with self.runtime.save_lock:
            await delete_session_data(self.id)
            await self._save_session_locked()
        await self._apply_preset()
        logger.info(f"[Session {self.id}] 已完全重置（含数据库清理）")

    async def calm_down(self):
        self.bump_generation("calm_down")
        self.state.global_emotion = EmotionState()
        self.state.willingness = 0.0
        # 退出对话窗口
        self.state.last_speak_time = datetime.min
        await self.save_session()

    async def reset_emotion(self):
        """仅把心情重置回基线；好感度是长期养成的，不在这里清"""
        self.bump_generation("reset_emotion")
        self.state.global_emotion = EmotionState()
        logger.info(f"[Session {self.id}] 心情已重置")
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
                    "preset": self.state.preset,
                    "valence": self.state.global_emotion.valence,
                    "arousal": self.state.global_emotion.arousal,
                    "dominance": self.state.global_emotion.dominance,
                    "emotion_updated_at": self.state.global_emotion.updated_at,
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

        self.state.preset = session_db.preset
        await self._apply_preset()
        self.state.group_notes = session_db.group_notes
        self.state.chat_summary = session_db.chat_summary
        self.state.global_emotion = EmotionState(
            valence=session_db.valence,
            arousal=session_db.arousal,
            dominance=session_db.dominance,
            updated_at=session_db.emotion_updated_at,
        )

        if session_db.last_speak_time:
            t = session_db.last_speak_time
            if t.tzinfo is not None:
                t = t.astimezone(None).replace(tzinfo=None)
            self.state.last_speak_time = t
        self.state.last_consolidated_time = session_db.last_consolidated_time

        self.state.willingness = WILLINGNESS_LOAD_VALUE
        self.state.profiles = {}
        self.state.user_summaries = {
            user_data["user_id"]: user_data["summary"]
            for user_data in data["users"]
            if user_data["summary"]
        }

        # 恢复用户画像
        for user_data in data["users"]:
            self.state.profiles[user_data["user_id"]] = PersonProfile(
                user_id=user_data["user_id"],
                affinity=user_data["affinity"],
                affinity_updated_at=user_data["affinity_updated_at"],
                interaction_count=user_data["interaction_count"],
                first_interaction_at=user_data["first_interaction_at"],
                last_interaction_at=user_data["last_interaction_at"],
                dirty=False,
            )

        self.runtime.short_term_memory = Memory(messages=data["messages"])

        self.state.loaded = True
        logger.info(f"[Session {self.id}] 加载完成")

    def presets(self) -> list[str]:
        reload_presets()
        return [
            f"{filename}: {preset.name} {preset.seed.split('。')[0]}"
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
        if filename != self.state.preset:
            await skip_persona_absence(self.id, filename)
        self.state.preset = filename
        await self._apply_preset()
        await self.save_session()
        return True

    async def _apply_preset(self) -> None:
        """按预设文件名填名字、别名、静态常识，再读每日整理出的人生线和本群群层。"""

        preset = PRESETS.get(self.state.preset)
        if preset is None:
            # 文件名留着，可能只是 json 暂时写坏了；名字为空就不说话
            if self.state.preset:
                logger.warning(f"[Session {self.id}] 预设 {self.state.preset} 不存在，按没设人设处理")
            self.state.name = ""
            self.state.aliases = []
            self.state.profile = []
            self.state.bottom_lines = []
            self.state.preset_lines = []
            self.state.self_life = ""
            self.state.self_in_group = ""
            return

        self.state.name = preset.name
        self.state.aliases = preset.aliases
        self.state.profile = preset.profile
        self.state.bottom_lines = preset.bottom_lines
        # 预设每轮完全相同，排序固定
        self.state.preset_lines = sorted(
            f"【设定/knowledge】 {item}" for item in preset.knowledges if item.strip()
        )
        life, self.state.self_in_group = await load_persona_texts(self.id, self.state.preset)
        # 新人设第一次整理要等到当晚，这之前拿出发点顶上，不然这一天没有性格
        self.state.self_life = life or preset.seed

    def status(self) -> str:
        self.state.global_emotion.settle()
        recent_messages = self.runtime.short_term_memory.access()
        recent_str = (
            "\n".join([f"{m.user_name}: {m.content}" for m in recent_messages])
            if recent_messages
            else "无"
        )
        return f"""
人设：{self.state.preset or "未设（不说话）"}
意愿值：{self.state.willingness:.2f}
状态: {chatting_state(self.state, datetime.now())}
心情：V{self.state.global_emotion.valence:.2f} A{self.state.global_emotion.arousal:.2f} D{self.state.global_emotion.dominance:.2f}
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

    def recall_message(self, msg_id: str, content: str) -> bool:
        """群友撤回消息：缓冲里那条原地改成撤回标记，随下次保存写回消息库。"""

        if not self.runtime.short_term_memory.rewrite(msg_id, content):
            return False
        self.schedule_save()
        return True

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
