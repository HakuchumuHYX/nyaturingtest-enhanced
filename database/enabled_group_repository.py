from nonebot import logger
from tortoise.transactions import in_transaction

from ..models.database import EnabledGroupModel


class EnabledGroupRepository:
    @staticmethod
    async def load_enabled_group_ids(config_group_ids: set[int] | None = None) -> set[int]:
        """加载启用群组，并把旧配置中的群组迁移到数据库。"""
        db_groups = await EnabledGroupModel.all()
        db_ids = {g.group_id for g in db_groups}
        config_ids = config_group_ids or set()

        new_ids = config_ids - db_ids
        if new_ids:
            logger.info(f"检测到配置文件中的新群组，正在迁移至数据库: {new_ids}")
            await EnabledGroupModel.bulk_create([EnabledGroupModel(group_id=gid) for gid in new_ids])
            db_ids.update(new_ids)

        return db_ids

    @staticmethod
    async def enable_group(group_id: int):
        async with in_transaction():
            await EnabledGroupModel.get_or_create(group_id=group_id)

    @staticmethod
    async def disable_group(group_id: int):
        async with in_transaction():
            await EnabledGroupModel.filter(group_id=group_id).delete()
