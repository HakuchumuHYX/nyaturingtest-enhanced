# __init__.py
import asyncio

from nonebot import get_driver, logger
from tortoise import Tortoise
from pathlib import Path

from .core.state_manager import cleanup_global_resources, init_enabled_groups
from .database.migrations import SCHEMA_VERSION, ensure_schema_version
from .handlers import commands
from .handlers import memory
from .database.backup import backup_before_schema_upgrade, setup_backup_job

driver = get_driver()

# 使用项目根目录下的 data 目录
PLUGIN_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "nyaturingtest"


@driver.on_startup
async def init_db():
    PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = PLUGIN_DATA_DIR / "nyabot.sqlite"
    if not await asyncio.to_thread(
        backup_before_schema_upgrade,
        db_path,
        SCHEMA_VERSION,
    ):
        raise RuntimeError("数据库升级前备份失败，已中止启动以保护现有数据")

    await Tortoise.init(
        db_url=f'sqlite://{db_path}',
        modules={'models': [f'{__package__}.models.database']},
        use_tz=False,
        _create_db=True,
        _enable_global_fallback=True
    )
    await Tortoise.generate_schemas()
    await ensure_schema_version()
    logger.info(f"数据库已连接: {db_path}")

    # 初始化群组列表
    await init_enabled_groups()

    # 注册定时备份任务
    setup_backup_job()


@driver.on_shutdown
async def cleanup_tasks():
    """生命周期钩子：关机清理"""
    # 委托给 state_manager 处理，确保顺序正确
    await cleanup_global_resources()
