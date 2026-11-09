from nonebot import get_driver, logger
from nonebot.adapters import Bot
from tortoise import Tortoise

from . import handlers  # 导入时注册 matcher
from .chat.groups import cleanup_global_resources, init_enabled_groups
from .chat.initiative import setup_initiative_job
from .config import get_data_dir
from .memory.episodes import setup_episode_job
from .memory.maintenance import setup_maintenance_job
from .messages.image import setup_image_cache_job
from .messages.photos import setup_photo_prune_job
from .messages.stickers import load_stickers, setup_sticker_prune_job
from .persona.status import setup_status_job
from .storage.backup import setup_backup_job
from .webconsole_api import setup_console_api
from .webconsole_store import close_store, open_store

driver = get_driver()


@driver.on_startup
async def init_db():
    open_store()
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

    await init_enabled_groups()
    await load_stickers()

    setup_episode_job()
    setup_initiative_job()
    setup_status_job()
    setup_image_cache_job()
    setup_sticker_prune_job()
    setup_photo_prune_job()
    setup_maintenance_job()
    setup_backup_job()
    setup_console_api()


@driver.on_bot_connect
async def log_bot_connect(bot: Bot):
    logger.info(f"Bot {bot.self_id} 已连接")


@driver.on_bot_disconnect
async def log_bot_disconnect(bot: Bot):
    logger.warning(f"Bot {bot.self_id} 断开连接")


@driver.on_shutdown
async def cleanup_tasks():
    await cleanup_global_resources()
    close_store()
