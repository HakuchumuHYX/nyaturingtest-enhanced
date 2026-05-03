from datetime import datetime

from nonebot import logger

from ..models.database import InteractionLogModel, SessionModel, UserProfileModel


class ProfileRepository:
    @staticmethod
    async def update_user_profiles(session_id: str, profiles: dict):
        """批量更新用户画像"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return

            for user_id, profile in profiles.items():
                await UserProfileModel.update_or_create(
                    session=session_db,
                    user_id=str(user_id),
                    defaults={
                        "valence": profile.emotion.valence,
                        "arousal": profile.emotion.arousal,
                        "dominance": profile.emotion.dominance,
                    },
                )
        except Exception as e:
            logger.error(f"[Repo] 更新用户画像失败: {e}")

    @staticmethod
    async def log_interaction(session_id: str, user_id: str, delta: dict):
        """记录交互日志"""
        await ProfileRepository.log_interactions(session_id, [(user_id, delta)])

    @staticmethod
    async def log_interactions(session_id: str, interactions: list[tuple[str, dict]]):
        """批量记录交互日志"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db or not interactions:
                return

            user_ids = [str(user_id) for user_id, _ in interactions]
            existing_users = await UserProfileModel.filter(session=session_db, user_id__in=user_ids)
            user_map = {user.user_id: user for user in existing_users}

            missing_ids = [user_id for user_id in user_ids if user_id not in user_map]
            if missing_ids:
                await UserProfileModel.bulk_create(
                    [UserProfileModel(session=session_db, user_id=user_id) for user_id in set(missing_ids)],
                    ignore_conflicts=True,
                )
                existing_users = await UserProfileModel.filter(session=session_db, user_id__in=user_ids)
                user_map = {user.user_id: user for user in existing_users}

            now = datetime.now()
            rows = []
            for user_id, delta in interactions:
                user_db = user_map.get(str(user_id))
                if not user_db:
                    continue
                rows.append(
                    InteractionLogModel(
                        user=user_db,
                        delta_valence=delta.get("valence", 0.0),
                        delta_arousal=delta.get("arousal", 0.0),
                        delta_dominance=delta.get("dominance", 0.0),
                        timestamp=now,
                    )
                )
            if rows:
                await InteractionLogModel.bulk_create(rows)
        except Exception as e:
            logger.error(f"[Repo] 记录交互日志失败: {e}")

    @staticmethod
    async def get_interaction_count(session_id: str, user_id: str) -> int:
        """获取用户交互次数"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return 0

            user_db = await UserProfileModel.get_or_none(
                session=session_db,
                user_id=str(user_id),
            )
            if user_db:
                return await InteractionLogModel.filter(user=user_db).count()
            return 0
        except Exception as e:
            logger.error(f"[Repo] 获取交互统计失败: {e}")
            return 0

    @staticmethod
    async def get_first_interaction_time(session_id: str, user_id: str) -> datetime | None:
        """获取用户首次交互时间"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return None

            user_db = await UserProfileModel.get_or_none(
                session=session_db,
                user_id=str(user_id),
            )
            if not user_db:
                return None

            first_log = await InteractionLogModel.filter(user=user_db).order_by("timestamp").first()
            return first_log.timestamp if first_log else None
        except Exception as e:
            logger.error(f"[Repo] 获取首次交互时间失败: {e}")
            return None
