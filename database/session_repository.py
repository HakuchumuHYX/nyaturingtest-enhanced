from nonebot import logger

from ..memory.short_term import Message
from ..models.database import GlobalMessageModel, InteractionLogModel, SessionModel, UserProfileModel
from ..config import get_runtime_settings
from ..core.text_utils import sanitize_text


class SessionStateRepository:
    @staticmethod
    async def get_session(session_id: str) -> SessionModel | None:
        return await SessionModel.filter(id=session_id).first()

    @staticmethod
    async def delete_session_data(session_id: str):
        """删除会话的所有关联数据（消息、用户画像、交互日志），不删除会话本身"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return

            users = await UserProfileModel.filter(session=session_db).all()
            for user in users:
                deleted_count = await InteractionLogModel.filter(user=user).delete()
                if deleted_count:
                    logger.debug(f"[Repo] 删除用户 {user.user_id} 的 {deleted_count} 条交互日志")

            profile_count = await UserProfileModel.filter(session=session_db).delete()
            logger.debug(f"[Repo] 删除 {profile_count} 个用户画像")

            msg_count = await GlobalMessageModel.filter(session=session_db).delete()
            logger.debug(f"[Repo] 删除 {msg_count} 条聊天消息")

            logger.info(f"[Repo] 会话 {session_id} 数据已完全清除")
        except Exception as e:
            logger.error(f"[Repo] 删除会话数据失败: {e}")
            raise

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
                    "valence": data.get("valence", 0.0),
                    "arousal": data.get("arousal", 0.0),
                    "dominance": data.get("dominance", 0.0),
                    "chat_summary": sanitize_text(data.get("chat_summary", "")),
                    "last_speak_time": data.get("last_speak_time"),
                    "last_consolidated_time": data.get("last_consolidated_time"),
                    "chatting_state": data.get("chatting_state", 0),
                },
            )
        except Exception as e:
            logger.error(f"[Repo] 保存会话状态失败: {e}")
            raise

    @staticmethod
    async def load_full_session_data(session_id: str):
        """加载完整的会话数据"""
        session_db = await SessionModel.filter(id=session_id).first()
        if not session_db:
            return None

        users_db = await UserProfileModel.filter(session=session_db)
        users_data = []
        for user_db in users_db:
            users_data.append(
                {
                    "user_id": user_db.user_id,
                    "valence": user_db.valence,
                    "arousal": user_db.arousal,
                    "dominance": user_db.dominance,
                    "last_update_time": user_db.last_update_time,
                    "interaction_count": user_db.interaction_count,
                    "first_interaction_at": user_db.first_interaction_at,
                    "last_interaction_at": user_db.last_interaction_at,
                }
            )

        buffer_limit = get_runtime_settings()["short_term_buffer_size"]
        msgs_db = await GlobalMessageModel.filter(session=session_db).order_by("-time").limit(buffer_limit)
        history_msgs = []
        for msg_db in reversed(msgs_db):
            history_msgs.append(
                Message(
                    time=msg_db.time,
                    user_name=msg_db.user_name,
                    content=msg_db.content,
                    id=msg_db.msg_id,
                    user_id=msg_db.user_id if msg_db.user_id else "",
                )
            )

        return {
            "session": session_db,
            "users": users_data,
            "messages": history_msgs,
            "last_consolidated_time": session_db.last_consolidated_time,
        }
