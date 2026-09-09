from tortoise.transactions import in_transaction

from ..models.database import EnabledGroupModel


class EnabledGroupRepository:
    @staticmethod
    async def load_enabled_group_ids() -> set[int]:
        """加载启用群组；数据库是唯一来源，由 /autochat enable|disable 维护。"""

        return {g.group_id for g in await EnabledGroupModel.all()}

    @staticmethod
    async def enable_group(group_id: int):
        async with in_transaction():
            await EnabledGroupModel.get_or_create(group_id=group_id)

    @staticmethod
    async def disable_group(group_id: int):
        async with in_transaction():
            await EnabledGroupModel.filter(group_id=group_id).delete()
