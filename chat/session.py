import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.exception import NetworkError

from ..memory.vector import VectorMemory
from ..messages.message import ChatMessage, Memory
from ..metrics import log_event
from ..prompts import PRESETS, resolve_preset
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
from .engagement import LOOK_MAX_MESSAGES

# 后台任务排空超时、保存去抖
MEMORY_DRAIN_TIMEOUT_SECONDS = 10.0
SAVE_DEBOUNCE_SECONDS = 0.05
# 待处理队列：满了丢最旧的，2s 防抖下几乎不可能积压到上限
QUEUE_MAX_SIZE = 200
# 群名和人数看群时超过这么久就重新拉；有人进群、退群时作废
GROUP_INFO_TTL = timedelta(hours=1)


@dataclass
class SessionState:
    preset: str = ""
    # 以下由 _apply_preset 从预设和每日整理结果填，只读，不随 save_session 回写；
    # name 为空说明没设人设或预设文件不在，这个群不说话
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)
    appearance: str = ""
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
    last_speak_time: datetime = datetime.min
    # 上次接话时那批消息的发言人；只在进程内保持，重启后对话窗口自然失效
    conversation_partners: set[str] = field(default_factory=set)
    # 已读水位：Feedback 读到的最后一条消息时间，之后的都是未读
    last_read_time: datetime | None = None
    # 爬楼水位：这之前的都细看过。被叫起来看一眼时已读水位会跳过去，这个不动，等真回来从这里爬
    climbed_until: datetime | None = None
    last_look_time: datetime = datetime.min
    generation: int = 0


class Session:
    """群聊会话：领域状态 state，加上它持有的 I/O 资源、后台任务，以及 worker 的收件箱和冷却。"""

    def __init__(self, id: str):
        self.id = id
        self.state = SessionState()
        self.short_term_memory = Memory()
        self.vector_memory = VectorMemory(id)
        self.background_tasks: set[asyncio.Task] = set()
        self.save_lock = asyncio.Lock()
        # 保存请求去抖 + 单飞：频繁的 schedule_save 合并成一次后台写库
        self._save_pending = False
        self._save_task: asyncio.Task | None = None

        # 群 worker 任务，由 chat/groups.py 启动和取消
        self.worker: asyncio.Task | None = None
        # 最近一次收到消息时的 bot / event，worker 发送时用；启动后还没收到过就是 None
        self.bot: Bot | None = None
        self.event: Event | None = None
        self.messages_chunk: deque[ChatMessage] = deque(maxlen=QUEUE_MAX_SIZE)
        self.data_lock = asyncio.Lock()
        # 管理命令（load_preset / reset / calm_down）之间互斥；
        # 一轮对话不持锁，靠 generation 判断过期
        self.admin_lock = asyncio.Lock()
        self.new_message_signal = asyncio.Event()
        # 发送到写进短期记忆之间持有：这个号的 message_sent 上报可能比发送返回早到，等它放开再按 id 认是不是她发的
        self.send_lock = asyncio.Lock()
        # 拍照出问题后，到这个时间才再试
        self.photo_ready_at: datetime = datetime.min
        # 这个群发照片的时间，算 photos_last_hour 用，只存内存：没有硬性间隔，重启后重新数无妨
        self.photo_sent_times: list[datetime] = []
        # 她被禁言到什么时候（全员禁言记到解除为止），只存内存：重启后禁言期内发送会失败，失败记日志
        self.muted_until: datetime | None = None
        # 「群名 X；现在 N 人，上限 M」和拉取时间；拉取失败也记时间，空串就是省略
        self.group_info: str = ""
        self.group_info_at: datetime = datetime.min

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
        self.state = SessionState(generation=generation, preset=self.state.preset)
        self.short_term_memory.clear()
        await self.vector_memory.clear()
        # 清理数据库中的所有关联数据，并与后台持久化共用同一把锁：
        # 旧 generation 的后台写入要么已在删除前完成，要么拿锁后被跳过。
        async with self.save_lock:
            await delete_session_data(self.id)
            await self._save_session_locked()
        await self._apply_preset()
        logger.info(f"[Session {self.id}] 已完全重置（含数据库清理）")

    async def calm_down(self):
        """心情回到基线、退出对话窗口；好感度是长期养成的，不在这里清"""
        self.bump_generation("calm_down")
        self.state.global_emotion = EmotionState()
        self.state.last_speak_time = datetime.min
        logger.info(f"[Session {self.id}] 已冷静：心情重置、退出对话")
        await self.save_session()

    def spawn(self, coro) -> asyncio.Task:
        """创建受 drain_background_tasks 管理、异常会记日志的后台任务"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task):
        self.background_tasks.discard(task)
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
        async with self.save_lock:
            return await self._save_session_locked()

    async def _save_session_locked(self) -> bool:
        try:
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
                    "last_read_time": self.state.last_read_time,
                    "climbed_until": self.state.climbed_until,
                },
            )

            # 只写变过的画像
            dirty_profiles = {
                user_id: profile
                for user_id, profile in self.state.profiles.items()
                if profile.dirty
            }
            if dirty_profiles:
                await update_user_profiles(self.id, dirty_profiles)
                for profile in dirty_profiles.values():
                    profile.dirty = False

            # 只写新增的和描述补全过的消息
            pending_messages = self.short_term_memory.pending_messages()
            if pending_messages:
                await sync_messages(
                    self.id,
                    [message for message, _ in pending_messages],
                )
                self.short_term_memory.mark_persisted(pending_messages)

            return True
        except Exception as e:
            logger.warning(f"[Session {self.id}] 数据库保存警告: {e}")
            return False

    async def load_session(self):
        data = await load_full_session_data(self.id)

        if not data:
            logger.info(f"[Session {self.id}] 初始化新会话")
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
            self.state.last_speak_time = session_db.last_speak_time
        self.state.last_read_time = session_db.last_read_time
        self.state.climbed_until = session_db.climbed_until

        self.state.profiles = {}
        self.state.user_summaries = {
            user_data["user_id"]: user_data["summary"]
            for user_data in data["users"]
            if user_data["summary"]
        }

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

        self.short_term_memory = Memory(messages=data["messages"])

        logger.info(f"[Session {self.id}] 加载完成")

    async def load_preset(self, filename: str) -> bool:
        filename = resolve_preset(filename)
        if filename is None:
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
            self.state.appearance = ""
            self.state.bottom_lines = []
            self.state.preset_lines = []
            self.state.self_life = ""
            self.state.self_in_group = ""
            return

        self.state.name = preset.name
        self.state.aliases = preset.aliases
        self.state.profile = preset.profile
        self.state.appearance = preset.appearance
        self.state.bottom_lines = preset.bottom_lines
        # 预设每轮完全相同，排序固定
        self.state.preset_lines = sorted(
            f"【设定/knowledge】 {item}" for item in preset.knowledges if item.strip()
        )
        life, self.state.self_in_group = await load_persona_texts(self.id, self.state.preset)
        # 新人设第一次整理要等到当晚，这之前拿出发点顶上，不然这一天没有性格
        self.state.self_life = life or preset.seed

    def append_self_message(self, content: str, msg_id: str, bot_user_id: str):
        """自己发的消息在发送返回时直接写进短期记忆；之后到的 message_sent 上报按 id 认出来跳过。"""
        msg = ChatMessage(
            time=datetime.now(),
            user_name=self.state.name,
            content=content,
            id=msg_id,
            user_id=bot_user_id,
        )
        self.short_term_memory.update([msg])

    @property
    def muted(self) -> bool:
        return self.muted_until is not None and datetime.now() < self.muted_until

    async def refresh_group_info(self) -> str:
        """群名和人数，看群时调用；超过 GROUP_INFO_TTL 才重新拉。"""

        now = datetime.now()
        if self.bot is None or now - self.group_info_at < GROUP_INFO_TTL:
            return self.group_info
        self.group_info_at = now
        try:
            info = await self.bot.get_group_info(group_id=int(self.id), no_cache=True)
        except (ActionFailed, NetworkError) as e:
            logger.warning(f"[Session {self.id}] 拉取群信息失败: {e}")
            self.group_info = ""
            return ""
        self.group_info = (
            f"群名 {info.get('group_name', '')}；"
            f"现在 {info.get('member_count')} 人，上限 {info.get('max_member_count')}"
        )
        return self.group_info

    def photos_last_hour(self) -> int:
        since = datetime.now() - timedelta(hours=1)
        self.photo_sent_times = [sent for sent in self.photo_sent_times if sent >= since]
        return len(self.photo_sent_times)

    def note_photo_problem(self, content: str) -> None:
        """照片没发成：记一条给下一轮看，她被问起时自己圆，不算说的话，也不叫醒 worker。"""

        self.record_action(
            ChatMessage(
                time=datetime.now(),
                user_name=self.state.name,
                content=content,
                notice=True,
            )
        )

    def record_action(self, record: ChatMessage) -> None:
        """贴表情、群通知这类记录只进短期记忆、不进队列；贴表情并进旧记录的挪到末尾，重新算未读。

        群友的记录算未读，要唤醒 worker：它没有未读时不定时，不叫醒的话群里安静时要等下一个人说话才看得到。
        队列是空的，worker 醒来只会按新的未读重新定时，到点照常看群。
        """

        memory = self.short_term_memory
        if any(m is record for m in memory.access()):
            memory.move_to_latest(record)
        else:
            memory.update([record])
        self.schedule_save()
        if record.user_name != self.state.name:
            self.new_message_signal.set()

    def recall_message(self, msg_id: str, content: str) -> bool:
        """群友撤回消息：缓冲里那条原地改成撤回标记，随下次保存写回消息库。"""

        if not self.short_term_memory.recall(msg_id, content):
            return False
        self.schedule_save()
        return True

    def record_incoming(self, messages_chunk: list[ChatMessage]) -> None:
        """写入短时记忆；水位之后的就是未读。"""

        self.short_term_memory.update(messages_chunk)
        self.schedule_save()

    def unread(self) -> list[ChatMessage]:
        """已读水位之后的消息，最多 LOOK_MAX_MESSAGES 条（取最新的）。"""

        return self.short_term_memory.messages_after(self.state.last_read_time, LOOK_MAX_MESSAGES)

    async def enqueue(self, message: ChatMessage, bot: Bot, event: Event) -> None:
        """放进待处理队列并唤醒 worker。

        同一条已在队列里就什么都不做：转发读得比防抖还快时占位还在队列里，改写过的内容会随那一轮一起处理。
        """

        async with self.data_lock:
            if any(m is message for m in self.messages_chunk):
                return
            self.event = event
            self.bot = bot
            self.messages_chunk.append(message)
            self.new_message_signal.set()

    async def drain_background_tasks(self):
        pending = [task for task in self.background_tasks if not task.done()]
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
