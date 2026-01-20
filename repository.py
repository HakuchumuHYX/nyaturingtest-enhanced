# nyaturingtest/repository.py
import uuid
from datetime import datetime
from nonebot import logger
from .models import SessionModel, UserProfileModel, GlobalMessageModel, InteractionLogModel
from .mem import Message
from .utils import sanitize_text

class SessionRepository:
    """
    负责 Session 相关的所有数据库操作
    """

    @staticmethod
    async def get_session(session_id: str) -> SessionModel | None:
        return await SessionModel.filter(id=session_id).first()

    @staticmethod
    async def save_session_state(session_id: str, data: dict):
        """保存会话的基础状态"""
        try:
            await SessionModel.update_or_create(
                id=session_id,
                defaults={
                    "name": sanitize_text(data.get("name", "")),
                    "role": sanitize_text(data.get("role", "")),
                    "aliases": data.get("aliases", []),
                    "valence": data.get("valence", 0.5),
                    "arousal": data.get("arousal", 0.5),
                    "dominance": data.get("dominance", 0.5),
                    "chat_summary": sanitize_text(data.get("chat_summary", "")),
                    "last_speak_time": data.get("last_speak_time"),
                    "chatting_state": data.get("chatting_state", 0)
                }
            )
        except Exception as e:
            logger.error(f"[Repo] 保存会话状态失败: {e}")

    @staticmethod
    async def update_user_profiles(session_id: str, profiles: dict):
        """批量更新用户画像"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db: return

            for user_id, profile in profiles.items():
                await UserProfileModel.update_or_create(
                    session=session_db,
                    user_id=str(user_id),
                    defaults={
                        "valence": profile.emotion.valence,
                        "arousal": profile.emotion.arousal,
                        "dominance": profile.emotion.dominance,
                    }
                )
        except Exception as e:
            logger.error(f"[Repo] 更新用户画像失败: {e}")

    @staticmethod
    async def sync_messages(session_id: str, recent_msgs: list[Message]):
        """增量同步消息到数据库"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db: return

            msg_ids = [msg.id for msg in recent_msgs if msg.id]
            existing_ids = set()

            if msg_ids:
                existing_msgs = await GlobalMessageModel.filter(
                    session=session_db,
                    msg_id__in=msg_ids
                ).values_list("msg_id", flat=True)
                existing_ids = set(existing_msgs)

            bulk_msgs = []
            for msg in recent_msgs:
                # 生成确定性 ID
                final_msg_id = msg.id
                if not final_msg_id:
                    unique_str = f"{sanitize_text(msg.content)}_{msg.time.timestamp()}"
                    final_msg_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

                if final_msg_id not in existing_ids:
                    bulk_msgs.append(GlobalMessageModel(
                        session=session_db,
                        user_name=sanitize_text(msg.user_name),
                        user_id=str(msg.user_id) if msg.user_id else "",
                        content=sanitize_text(msg.content),
                        time=msg.time,
                        msg_id=final_msg_id
                    ))
                    existing_ids.add(final_msg_id)

            if bulk_msgs:
                await GlobalMessageModel.bulk_create(bulk_msgs)
                logger.debug(f"[Repo] 同步了 {len(bulk_msgs)} 条新消息")

        except Exception as e:
            logger.error(f"[Repo] 同步消息失败: {e}")

    @staticmethod
    async def log_interaction(session_id: str, user_id: str, delta: dict):
        """记录交互日志"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db: return

            user_db, _ = await UserProfileModel.get_or_create(
                session=session_db,
                user_id=str(user_id)
            )

            await InteractionLogModel.create(
                user=user_db,
                delta_valence=delta.get("valence", 0.0),
                delta_arousal=delta.get("arousal", 0.0),
                delta_dominance=delta.get("dominance", 0.0),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"[Repo] 记录交互日志失败: {e}")

    @staticmethod
    async def load_full_session_data(session_id: str):
        """加载完整的会话数据"""
        session_db = await SessionModel.filter(id=session_id).first()
        if not session_db: return None

        # 加载用户画像
        users_db = await UserProfileModel.filter(session=session_db).prefetch_related("interactions")
        users_data = []
        for user_db in users_db:
            recent_logs = await user_db.interactions.all().order_by("-timestamp").limit(20)
            logs_data = [{
                "timestamp": log.timestamp,
                "delta": {
                    "valence": log.delta_valence,
                    "arousal": log.delta_arousal,
                    "dominance": log.delta_dominance
                }
            } for log in reversed(recent_logs)]
            
            users_data.append({
                "user_id": user_db.user_id,
                "valence": user_db.valence,
                "arousal": user_db.arousal,
                "dominance": user_db.dominance,
                "last_update_time": user_db.last_update_time,
                "recent_logs": logs_data
            })

        # 加载最近消息
        msgs_db = await GlobalMessageModel.filter(session=session_db).order_by("-time").limit(50)
        history_msgs = []
        for msg_db in reversed(msgs_db):
            history_msgs.append(Message(
                time=msg_db.time,
                user_name=msg_db.user_name,
                content=msg_db.content,
                id=msg_db.msg_id,
                user_id=msg_db.user_id if msg_db.user_id else ""
            ))

        return {
            "session": session_db,
            "users": users_data,
            "messages": history_msgs
        }

    @staticmethod
    async def get_history_before(session_id: str, time_point: datetime, limit: int = 20) -> list[Message]:
        """获取指定时间之前的历史消息"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db: return []

            history_rows = await GlobalMessageModel.filter(
                session=session_db,
                time__lt=time_point
            ).order_by("-time").limit(limit)

            recalled_msgs = []
            if history_rows:
                # 数据库查出来是倒序的 (最新的在前面)，需要反转回正序
                rows_sorted = sorted(history_rows, key=lambda x: x.time)
                for m in rows_sorted:
                    recalled_msgs.append(Message(
                        time=m.time,
                        user_name=m.user_name,
                        content=m.content,
                        id=m.msg_id,
                        user_id=m.user_id if m.user_id else ""
                    ))
            return recalled_msgs
        except Exception as e:
            logger.error(f"[Repo] 历史溯源失败: {e}")
            return []

    @staticmethod
    async def get_interaction_count(session_id: str, user_id: str) -> int:
        """获取用户交互次数"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db: return 0

            user_db = await UserProfileModel.get_or_none(
                session=session_db,
                user_id=str(user_id)
            )
            if user_db:
                return await InteractionLogModel.filter(user=user_db).count()
            return 0
        except Exception as e:
            logger.error(f"[Repo] 获取交互统计失败: {e}")
            return 0

    @staticmethod
    async def get_recent_messages_by_user(session_id: str, user_id: str = "", user_name: str = "", limit: int = 10) -> list[str]:
        """获取用户最近的发言内容"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db: return []

            db_msgs = []
            # 优先 ID 查
            if user_id and str(user_id).strip():
                db_msgs = await GlobalMessageModel.filter(
                    session=session_db,
                    user_id=str(user_id)
                ).order_by("-time").limit(limit)
            
            # 兜底 名字 查 (如果没有查到 ID 记录，或者 ID 为空)
            if not db_msgs and user_name:
                db_msgs = await GlobalMessageModel.filter(
                    session=session_db,
                    user_name=user_name
                ).order_by("-time").limit(limit)

            return [m.content for m in reversed(db_msgs)]
        except Exception as e:
            logger.error(f"[Repo] 获取用户历史消息失败: {e}")
            return []
