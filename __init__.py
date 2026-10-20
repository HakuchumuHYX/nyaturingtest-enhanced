# __init__.py

from nonebot import get_driver, logger
from nonebot.adapters import Bot
from tortoise import Tortoise

from . import handlers  # 导入即注册所有 matcher
from .chat.groups import (
    cleanup_global_resources,
    init_enabled_groups,
    setup_maintenance_job,
)
from .chat.initiative import setup_initiative_job
from .config import get_data_dir
from .memory.episodes import setup_episode_job
from .messages.image import setup_image_cache_job
from .messages.stickers import load_stickers
from .storage.backup import setup_backup_job

driver = get_driver()


@driver.on_startup
async def init_db():
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "nyabot.sqlite"
    await Tortoise.init(
        db_url=f"sqlite://{db_path}",
        modules={"models": [f"{__package__}.storage.models"]},
        use_tz=False,
        _create_db=True,
        _enable_global_fallback=True,
    )
    await Tortoise.generate_schemas()
    logger.info(f"数据库已连接: {db_path}")

    # 初始化群组列表
    await init_enabled_groups()
    await load_stickers()

    # 注册定时任务：分段整理、主动话题、图片缓存清理、长期记忆维护、备份
    setup_episode_job()
    setup_initiative_job()
    setup_image_cache_job()
    setup_maintenance_job()
    setup_backup_job()


@driver.on_bot_connect
async def log_bot_connect(bot: Bot):
    logger.info(f"Bot {bot.self_id} 已连接")


@driver.on_bot_disconnect
async def log_bot_disconnect(bot: Bot):
    logger.warning(f"Bot {bot.self_id} 断开连接")


@driver.on_shutdown
async def cleanup_tasks():
    """生命周期钩子：关机清理"""
    # 委托给 chat/groups 处理，确保顺序正确
    await cleanup_global_resources()
