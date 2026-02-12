# nyaturingtest/state_manager.py
import asyncio
from collections import deque
from dataclasses import dataclass, field
from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Event
from openai import AsyncOpenAI
from tortoise import Tortoise

from .client import LLMClient
from .config import (
    plugin_config,
    get_effective_chat_api_key,
    get_effective_chat_base_url,
)
from .mem import Message as MMessage
from .session import Session
from .utils import get_http_client, close_http_client
from .models import EnabledGroupModel


def _build_chat_llm_client() -> LLMClient:
    provider = (getattr(plugin_config, "nyaturingtest_chat_provider", None) or "openai_compatible").strip().lower()

    openai_client = None
    if provider == "openai_compatible":
        openai_client = AsyncOpenAI(
            api_key=get_effective_chat_api_key(plugin_config),
            base_url=get_effective_chat_base_url(plugin_config),
            http_client=get_http_client(),
        )

    google_key = (getattr(plugin_config, "nyaturingtest_chat_google_api_key", None) or "").strip()
    if not google_key:
        # allow reuse existing key if user puts google key into legacy field
        google_key = get_effective_chat_api_key(plugin_config)

    google_base_url = (
        getattr(plugin_config, "nyaturingtest_chat_google_base_url", None)
        or "https://generativelanguage.googleapis.com/v1beta"
    )

    return LLMClient(
        provider=provider,
        openai_client=openai_client,
        google_api_key=google_key,
        google_base_url=google_base_url,
    )


def _build_feedback_llm_client() -> LLMClient:
    # feedback remains SiliconFlow (unchanged), OpenAI-compatible
    openai_client = AsyncOpenAI(
        api_key=plugin_config.nyaturingtest_siliconflow_api_key,
        base_url="https://api.siliconflow.cn/v1",
        http_client=get_http_client(),
    )
    return LLMClient(provider="openai_compatible", openai_client=openai_client)

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

    client: LLMClient = field(default_factory=_build_chat_llm_client)

    feedback_client: LLMClient = field(default_factory=_build_feedback_llm_client)
    data_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    new_message_signal: asyncio.Event = field(default_factory=asyncio.Event)


# 全局状态字典
group_states: dict[int, GroupState] = {}
# 后台任务字典 group_id -> Task
_group_tasks: dict[int, asyncio.Task] = {}
# 运行时启用的群组集合 (内存缓存)
runtime_enabled_groups: set[int] = set()


async def init_enabled_groups():
    # 1. 从数据库读取
    db_groups = await EnabledGroupModel.all()
    db_ids = {g.group_id for g in db_groups}

    # 2. 读取配置文件 (用于迁移)
    config_ids = set(plugin_config.nyaturingtest_enabled_groups)

    # 3. 如果配置文件里有 DB 里没有的，自动迁移写入 DB
    new_ids = config_ids - db_ids
    if new_ids:
        logger.info(f"检测到配置文件中的新群组，正在迁移至数据库: {new_ids}")
        await EnabledGroupModel.bulk_create([
            EnabledGroupModel(group_id=gid) for gid in new_ids
        ])
        db_ids.update(new_ids)

    # 4. 更新到内存集合
    runtime_enabled_groups.clear()
    runtime_enabled_groups.update(db_ids)
    logger.info(f"已加载 Autochat 启用群组: {runtime_enabled_groups}")


def ensure_group_state(group_id: int):
    """确保群组状态已初始化，并启动后台任务"""
    if group_id not in runtime_enabled_groups:
        return None

    # 1. 状态初始化
    if group_id not in group_states:
        logger.info(f"初始化群 {group_id} 的 GroupState...")
        new_state = GroupState(
            session=Session(
                id=f"{group_id}",
                siliconflow_api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                http_client=get_http_client()
            )
        )
        group_states[group_id] = new_state
    
    # 2. 任务守护 (如果任务挂了或者没启动，重启它)
    if group_id not in _group_tasks or _group_tasks[group_id].done():
        if group_id in _group_tasks:
            # 清理旧的已完成任务记录
            try:
                # 获取异常以防万一
                exc = _group_tasks[group_id].exception()
                if exc:
                    logger.error(f"群 {group_id} 的后台任务曾异常退出: {exc}")
            except Exception:
                pass
            del _group_tasks[group_id]

        from .logic import spawn_state
        
        # 启动新任务
        logger.info(f"启动群 {group_id} 的 spawn_state 后台任务...")
        task = asyncio.create_task(spawn_state(state=group_states[group_id]))
        _group_tasks[group_id] = task

    return group_states[group_id]


async def remove_group_state(group_id: int):
    """安全移除群组状态并取消后台任务"""
    # 1. 取消任务
    if group_id in _group_tasks:
        task = _group_tasks[group_id]
        if not task.done():
            logger.info(f"正在取消群 {group_id} 的后台任务...")
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"取消任务时发生错误: {e}")
        del _group_tasks[group_id]

    # 2. 移除状态
    if group_id in group_states:
        logger.info(f"移除群 {group_id} 的 GroupState...")
        del group_states[group_id]


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

    # 3. 关闭 HTTP 客户端
    await close_http_client()

    # 4. 最后关闭数据库
    logger.info("正在关闭数据库连接...")
    await Tortoise.close_connections()
    logger.info("数据库连接已关闭")
