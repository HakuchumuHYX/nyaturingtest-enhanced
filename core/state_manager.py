import asyncio
from collections import deque
from dataclasses import dataclass, field

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event
from openai import AsyncOpenAI
from tortoise import Tortoise

from ..config import EndpointSettings, get_app_settings
from ..db import load_enabled_group_ids
from ..memory.short_term import Message as MMessage
from .llm import LLMClient, close_http_client, get_http_client
from .metrics import drain_usage_tasks
from .session import MEMORY_DRAIN_TIMEOUT_SECONDS, Session


def build_llm_client(settings: EndpointSettings) -> LLMClient:
    return LLMClient(
        provider=settings.provider,
        openai_client=AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            http_client=get_http_client(),
            max_retries=0,
        ),
        timeout=settings.timeout,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )


SELF_SENT_MSG_IDS = deque(maxlen=50)


@dataclass
class GroupState:
    session: Session

    event: Event | None = None
    bot: Bot | None = None

    messages_chunk: list[MMessage] = field(default_factory=list)

    client: LLMClient = field(default_factory=lambda: build_llm_client(get_app_settings().chat))
    feedback_client: LLMClient = field(
        default_factory=lambda: build_llm_client(get_app_settings().feedback)
    )
    data_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    new_message_signal: asyncio.Event = field(default_factory=asyncio.Event)


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
        logger.info(f"初始化群 {group_id} 的 GroupState...")
        group_states[group_id] = GroupState(
            session=Session(
                id=f"{group_id}",
                siliconflow_api_key=get_app_settings().siliconflow_api_key,
                http_client=get_http_client(),
            )
        )

    # 任务守护：任务挂了或没启动就重启（spawn_state 自己吞异常，done 即退出）
    task = _group_tasks.get(group_id)
    if task is None or task.done():
        _group_tasks.pop(group_id, None)
        # 局部导入：logic 反向依赖本模块的 GroupState
        from .logic import spawn_state

        logger.info(f"启动群 {group_id} 的 spawn_state 后台任务...")
        _group_tasks[group_id] = asyncio.create_task(spawn_state(state=group_states[group_id]))

    return group_states[group_id]


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
        state = group_states.pop(group_id)
        await state.session.drain_background_tasks(timeout=MEMORY_DRAIN_TIMEOUT_SECONDS)
        await state.session.close()


async def maintain_vector_memories() -> None:
    """向量记忆定时维护，避开每轮对话的持久化路径。"""

    for group_id, state in list(group_states.items()):
        if not state.session.state.loaded:
            continue
        try:
            await asyncio.to_thread(
                state.session.runtime.vector_memory.cleanup,
                days_retention=90,
            )
        except Exception as e:
            logger.warning(f"群 {group_id} 向量记忆定时维护失败: {e}")


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
    drain_timeout = MEMORY_DRAIN_TIMEOUT_SECONDS
    for state in group_states.values():
        try:
            await state.session.drain_background_tasks(timeout=drain_timeout)
        except Exception as e:
            logger.warning(f"排空群会话后台任务失败: {e}")

    save_tasks = []
    for group_id, state in group_states.items():
        if state.session.state.loaded:
            logger.info(f"正在保存群 {group_id} 的会话状态...")
            save_tasks.append(state.session.save_session(force_index=True))

    if save_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*save_tasks, return_exceptions=True),
                timeout=60.0,
            )
            logger.info("会话保存完毕")
        except Exception as e:
            logger.error(f"关机保存错误: {e}")

    # 3. 所有 worker 和写任务都停止后，才关闭 Session 持有的资源。
    for state in group_states.values():
        try:
            await state.session.close()
        except Exception as e:
            logger.warning(f"关闭群会话资源失败: {e}")

    await drain_usage_tasks(timeout=MEMORY_DRAIN_TIMEOUT_SECONDS)

    # 5. Provider/usage 都已停止后关闭共享 HTTP，最后关闭数据库。
    await close_http_client()

    logger.info("正在关闭数据库连接...")
    await Tortoise.close_connections()
    logger.info("数据库连接已关闭")
