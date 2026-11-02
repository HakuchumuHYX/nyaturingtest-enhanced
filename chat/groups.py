import asyncio

from nonebot import logger
from tortoise import Tortoise

from ..llm import close_http_client
from ..memory.vector import close_clients
from ..metrics import drain_usage_tasks
from ..storage.db import disable_group, enable_group, load_enabled_group_ids
from .session import Session

# 启用的群：group_id -> Session，在表里就是启用了
sessions: dict[int, Session] = {}
# Shutdown 标志：设置后所有新的消息处理都会提前退出
_shutting_down = False


def is_shutting_down() -> bool:
    return _shutting_down


def get_session(group_id: int) -> Session | None:
    return sessions.get(group_id)


async def _start(group_id: int, preset: str | None = None) -> None:
    """建会话、加载（带了预设就换上）、启动 worker，放进注册表。

    预设在进注册表之前加载：之后才进来的消息一定能看到人设。
    """

    session = Session(id=str(group_id))
    await session.load_session()
    if preset is not None:
        await session.load_preset(preset)
    # 局部导入：worker 引用本模块的 is_shutting_down，顶层导入会成环
    from .worker import run_worker

    session.worker = asyncio.create_task(run_worker(session))
    sessions[group_id] = session
    logger.info(f"群 {group_id} 的会话已加载，后台任务已启动")


async def init_enabled_groups():
    for group_id in await load_enabled_group_ids():
        await _start(group_id)
    logger.info(f"已加载 Autochat 启用群组: {sorted(sessions)}")


async def _stop_worker(session: Session) -> None:
    task = session.worker
    if task is None or task.done():
        return
    logger.info(f"正在取消群 {session.id} 的后台任务...")
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    except Exception as e:
        logger.error(f"取消任务时发生错误: {e}")


async def enable_autochat(group_id: int, preset: str | None = None) -> None:
    """写库，再建好会话并启动 worker；第一次启用由调用方带上预设。"""

    await enable_group(group_id)
    await _start(group_id, preset)


async def disable_autochat(group_id: int) -> None:
    """写库、移出注册表，再取消 worker、排空后台写入。"""

    await disable_group(group_id)
    session = sessions.pop(group_id)
    await _stop_worker(session)
    await session.drain_background_tasks()


async def cleanup_global_resources():
    """统一的资源清理逻辑 (关机时调用)"""
    global _shutting_down
    _shutting_down = True
    logger.info("正在执行资源清理（已设置 shutdown 标志）...")

    # 1. 先停止所有可能继续使用 Session/Vector/Provider 的群 worker。
    for session in list(sessions.values()):
        await _stop_worker(session)

    # 2. worker 停止后再排空后台写入，并在数据库仍可用时做最终保存。
    for session in list(sessions.values()):
        try:
            await session.drain_background_tasks()
        except Exception as e:
            logger.warning(f"排空群会话后台任务失败: {e}")

    save_tasks = []
    for group_id, session in sessions.items():
        logger.info(f"正在保存群 {group_id} 的会话状态...")
        save_tasks.append(session.save_session())

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
