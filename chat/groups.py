import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from nonebot import logger, require
from nonebot.adapters.onebot.v11 import Bot, Event
from tortoise import Tortoise

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from ..llm import close_http_client  # noqa: E402
from ..memory.digest import digest_group_notes, digest_user_profiles  # noqa: E402
from ..memory.persona import maintain_personas  # noqa: E402
from ..memory.vector import close_clients, maintain_memories  # noqa: E402
from ..messages.message import ChatMessage  # noqa: E402
from ..metrics import drain_usage_tasks  # noqa: E402
from ..storage.db import load_enabled_group_ids  # noqa: E402
from .session import Session  # noqa: E402

QUEUE_MAX_SIZE = 200

SELF_SENT_MSG_IDS = deque(maxlen=50)


@dataclass
class GroupState:
    session: Session

    event: Event | None = None
    bot: Bot | None = None

    # 满了丢最旧的：2s 防抖下几乎不可能积压到上限
    messages_chunk: deque[ChatMessage] = field(
        default_factory=lambda: deque(maxlen=QUEUE_MAX_SIZE)
    )

    data_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    new_message_signal: asyncio.Event = field(default_factory=asyncio.Event)
    # 贴表情、戳一戳、表情包的冷却只存内存，重启后重新计时
    last_react_time: datetime = datetime.min
    last_poke_time: datetime = datetime.min
    last_sticker_time: datetime = datetime.min

    async def load(self) -> None:
        """懒加载会话。session_lock 只让加载和管理命令互斥，一轮对话不持锁，靠 generation 判断过期。"""

        async with self.session_lock:
            await self.session.load_session()

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


# 全局状态字典
group_states: dict[int, GroupState] = {}
# 后台任务字典 group_id -> Task
_group_tasks: dict[int, asyncio.Task] = {}
# 运行时启用的群组集合 (内存缓存)
runtime_enabled_groups: set[int] = set()
# Shutdown 标志：设置后所有新的消息处理都会提前退出
_shutting_down = False


def is_shutting_down() -> bool:
    return _shutting_down


async def init_enabled_groups():
    db_ids = await load_enabled_group_ids()

    runtime_enabled_groups.clear()
    runtime_enabled_groups.update(db_ids)
    logger.info(f"已加载 Autochat 启用群组: {runtime_enabled_groups}")


def ensure_group_state(group_id: int):
    """确保群组状态已初始化，并启动后台任务"""
    if group_id not in runtime_enabled_groups:
        return None

    if group_id not in group_states:
        group_states[group_id] = GroupState(
            session=Session(id=f"{group_id}")
        )

    # 任务守护：任务挂了或没启动就重启（run_worker 自己吞异常，done 即退出）
    task = _group_tasks.get(group_id)
    if task is None or task.done():
        _group_tasks.pop(group_id, None)
        # 局部导入：worker 反向依赖本模块的 GroupState
        from .worker import run_worker

        logger.info(f"启动群 {group_id} 的后台任务")
        _group_tasks[group_id] = asyncio.create_task(
            run_worker(group_states[group_id])
        )

    return group_states[group_id]


async def loaded_group_state(group_id: int) -> GroupState | None:
    """启用的群返回已加载会话的 GroupState，未启用返回 None。"""

    group = ensure_group_state(group_id)
    if group is not None:
        await group.load()
    return group


async def remove_group_state(group_id: int):
    """安全移除群组状态并取消后台任务"""
    task = _group_tasks.pop(group_id, None)
    if task is not None and not task.done():
        logger.info(f"正在取消群 {group_id} 的后台任务...")
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.error(f"取消任务时发生错误: {e}")

    if group_id in group_states:
        logger.info(f"移除群 {group_id} 的 GroupState...")
        group = group_states.pop(group_id)
        await group.session.drain_background_tasks()


def _generation(group_id: int) -> int:
    """未加载的群按 0 算：加载不改 generation，只有 reset/load_preset 等会递增。"""

    group = group_states.get(group_id)
    return group.session.state.generation if group else 0


def _generation_guard(group_id: int) -> Callable[[], bool]:
    """记下此刻的 generation，返回「会话还没被 reset/load_preset 作废」的检查。"""

    generation = _generation(group_id)
    return lambda: _generation(group_id) == generation


async def refresh_group_notes(group_id: int, *, min_new: int = 1) -> str | None:
    """把新碎片整理进群志；群志不随 save_session 回写，已加载的群同步内存副本。"""

    still_current = _generation_guard(group_id)
    notes = await digest_group_notes(str(group_id), still_current, min_new=min_new)
    group = group_states.get(group_id)
    if notes is not None and group is not None and still_current():
        group.session.state.group_notes = notes
    return notes


async def maintain_long_term_memory() -> None:
    """长期记忆每日维护：先把新碎片整理进档案与群志，再整理人设成长，最后删过期碎片，
    这样即将过期的碎片一定先被归纳过；最后让已加载的群下次重新读矩阵。"""

    for group_id in list(runtime_enabled_groups):
        session_id = str(group_id)
        still_current = _generation_guard(group_id)
        try:
            summaries = await digest_user_profiles(session_id, still_current)
            await refresh_group_notes(group_id)
        except Exception as e:
            logger.warning(f"群 {group_id} 档案/群志整理失败: {e}")
            continue
        # 档案不随 save_session 回写，已加载的群要同步内存副本
        group = group_states.get(group_id)
        if group is not None and still_current():
            group.session.state.user_summaries.update(summaries)

    await _maintain_personas()

    try:
        await maintain_memories()
    except Exception as e:
        logger.warning(f"长期记忆定时维护失败: {e}")
    for group in group_states.values():
        group.session.vector_memory.drop_cache()


async def _maintain_personas() -> None:
    group_ids = list(runtime_enabled_groups)
    guards = {str(group_id): _generation_guard(group_id) for group_id in group_ids}
    try:
        lives, group_texts = await maintain_personas(group_ids, guards)
    except Exception as e:
        logger.warning(f"人设成长整理失败: {e}")
        return
    # 人生线和群层不随 save_session 回写，已加载的群要同步内存副本
    for group_id in group_ids:
        group = group_states.get(group_id)
        if group is None or not guards[str(group_id)]():
            continue
        state = group.session.state
        if state.preset in lives:
            state.self_life = lives[state.preset]
        if str(group_id) in group_texts:
            state.self_in_group = group_texts[str(group_id)]


def setup_maintenance_job():
    scheduler.add_job(
        maintain_long_term_memory,
        "cron",
        hour=3,
        minute=30,
        id="nyaturingtest_vector_maintenance",
        misfire_grace_time=3600,
        replace_existing=True,
    )


async def cleanup_global_resources():
    """统一的资源清理逻辑 (关机时调用)"""
    global _shutting_down
    _shutting_down = True
    logger.info("正在执行资源清理（已设置 shutdown 标志）...")

    # 1. 先停止所有可能继续使用 Session/Vector/Provider 的群 worker。
    for gid in list(_group_tasks.keys()):
        task = _group_tasks.pop(gid)
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"清理任务 {gid} 异常: {e}")

    # 2. worker 停止后再排空后台写入，并在数据库仍可用时做最终保存。
    for group in group_states.values():
        try:
            await group.session.drain_background_tasks()
        except Exception as e:
            logger.warning(f"排空群会话后台任务失败: {e}")

    save_tasks = []
    for group_id, group in group_states.items():
        if group.session.state.loaded:
            logger.info(f"正在保存群 {group_id} 的会话状态...")
            save_tasks.append(group.session.save_session())

    if save_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*save_tasks, return_exceptions=True),
                timeout=60.0,
            )
            logger.info("会话保存完毕")
        except Exception as e:
            logger.error(f"关机保存错误: {e}")

    await drain_usage_tasks()

    # 3. Provider/usage 都已停止后关闭共享 HTTP，最后关闭数据库。
    await close_http_client()
    await close_clients()

    logger.info("正在关闭数据库连接...")
    await Tortoise.close_connections()
    logger.info("数据库连接已关闭")
