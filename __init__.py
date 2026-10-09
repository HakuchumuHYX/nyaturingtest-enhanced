# __init__.py

from nonebot import get_driver, logger
from tortoise import Tortoise

from . import handlers  # 导入即注册所有 matcher
from .backup import setup_backup_job
from .config import get_data_dir
from .core.episodes import setup_episode_job
from .core.initiative import setup_initiative_job
from .core.state_manager import cleanup_global_resources, init_enabled_groups

driver = get_driver()


@driver.on_startup
async def init_db():
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "nyabot.sqlite"
    await Tortoise.init(
        db_url=f"sqlite://{db_path}",
        modules={"models": [f"{__package__}.models"]},
        use_tz=False,
        _create_db=True,
        _enable_global_fallback=True,
    )
    await Tortoise.generate_schemas()
    logger.info(f"数据库已连接: {db_path}")

    # 初始化群组列表
    await init_enabled_groups()

    # 注册定时备份、主动发起话题检查与分段整理
    setup_backup_job()
    setup_initiative_job()
    setup_episode_job()


@driver.on_shutdown
async def cleanup_tasks():
    """生命周期钩子：关机清理"""
    # 委托给 state_manager 处理，确保顺序正确
    await cleanup_global_resources()
