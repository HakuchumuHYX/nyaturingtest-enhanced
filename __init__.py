# __init__.py
from nonebot import get_driver, logger, require
from tortoise import Tortoise

require("nonebot_plugin_localstore")
import nonebot_plugin_localstore as store

from .state_manager import cleanup_global_resources
from . import matchers

driver = get_driver()


@driver.on_startup
async def init_db():
    import os
    db_path = os.path.join(store.get_plugin_data_dir(), "nyabot.sqlite")

    await Tortoise.init(
        db_url=f'sqlite://{db_path}',
        modules={'models': [f'{__package__}.models']}
    )
    await Tortoise.generate_schemas()
    logger.info(f"数据库已连接: {db_path}")


@driver.on_shutdown
async def cleanup_tasks():
    """生命周期钩子：关机清理"""
    # 委托给 state_manager 处理，确保顺序正确
    await cleanup_global_resources()
