# nyaturingtest/state_manager.py
import asyncio
from collections import deque
from dataclasses import dataclass, field
from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event
from openai import AsyncOpenAI
from tortoise import Tortoise

from .client import LLMClient
from .config import plugin_config
from .mem import Message as MMessage
from .session import Session
from .utils import get_http_client, close_http_client

SELF_SENT_MSG_IDS = deque(maxlen=50)

@dataclass
class GroupState:
    event: Event | None = None
    bot: Bot | None = None
    session: Session = field(
        default_factory=lambda: Session(
            siliconflow_api_key=plugin_config.nyaturingtest_siliconflow_api_key,
            http_client=get_http_client()
        )
    )
    messages_chunk: list[MMessage] = field(default_factory=list)
    client: LLMClient = field(
        default_factory=lambda: LLMClient(
            client=AsyncOpenAI(
                api_key=plugin_config.nyaturingtest_chat_openai_api_key,
                base_url=plugin_config.nyaturingtest_chat_openai_base_url,
                http_client=get_http_client()
            )
        )
    )
    data_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    new_message_signal: asyncio.Event = field(default_factory=asyncio.Event)


# 全局状态字典
group_states: dict[int, GroupState] = {}
# 后台任务集合
_tasks: set[asyncio.Task] = set()


def ensure_group_state(group_id: int):
    """确保群组状态已初始化，并启动后台任务"""
    if group_id not in group_states:
        allowed_groups = plugin_config.nyaturingtest_enabled_groups
        if group_id not in allowed_groups:
            return None

        # 初始化状态
        new_state = GroupState(
            session=Session(
                id=f"{group_id}",
                siliconflow_api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                http_client=get_http_client()
            )
        )
        group_states[group_id] = new_state

        # 延迟导入以避免循环引用
        from .logic import spawn_state

        # 启动后台任务
        task = asyncio.create_task(spawn_state(state=new_state))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        logger.info(f"群 {group_id} 状态已初始化，后台任务已启动")

    return group_states[group_id]


async def cleanup_global_resources():
    """统一的资源清理逻辑 (关机时调用)"""
    logger.info("正在执行资源清理...")

    # 1. 强制保存会话 (需要数据库连接)
    save_tasks = []
    for group_id, state in group_states.items():
        if state.session._loaded:
            logger.info(f"正在保存群 {group_id} 的会话状态...")
            save_tasks.append(state.session.save_session(force_index=True))

    if save_tasks:
        try:
            # 增加超时时间，防止数据较多时保存中断
            await asyncio.wait_for(asyncio.gather(*save_tasks, return_exceptions=True), timeout=60.0)
            logger.info(f"会话保存完毕")
        except Exception as e:
            logger.error(f"关机保存错误: {e}")

    # 2. 取消后台任务
    for task in _tasks:
        if not task.done():
            task.cancel()
    if _tasks:
        await asyncio.gather(*_tasks, return_exceptions=True)

    # 3. 关闭 HTTP 客户端
    await close_http_client()

    # 4. 最后关闭数据库
    logger.info("正在关闭数据库连接...")
    await Tortoise.close_connections()
    logger.info("数据库连接已关闭")
