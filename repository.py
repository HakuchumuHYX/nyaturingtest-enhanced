# nyaturingtest/repository.py
import uuid
from datetime import datetime, timedelta
from tortoise.functions import Sum
from nonebot import logger
from .models import SessionModel, UserProfileModel, GlobalMessageModel, InteractionLogModel, TokenUsageModel
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
    async def delete_session_data(session_id: str):
        """删除会话的所有关联数据（消息、用户画像、交互日志），不删除会话本身"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return

            # 1. 删除交互日志（必须先删，因为依赖 UserProfile 外键）
            users = await UserProfileModel.filter(session=session_db).all()
            for user in users:
                deleted_count = await InteractionLogModel.filter(user=user).delete()
                if deleted_count:
                    logger.debug(f"[Repo] 删除用户 {user.user_id} 的 {deleted_count} 条交互日志")

            # 2. 删除用户画像
            profile_count = await UserProfileModel.filter(session=session_db).delete()
            logger.debug(f"[Repo] 删除 {profile_count} 个用户画像")

            # 3. 删除聊天消息
            msg_count = await GlobalMessageModel.filter(session=session_db).delete()
            logger.debug(f"[Repo] 删除 {msg_count} 条聊天消息")

            logger.info(f"[Repo] 会话 {session_id} 数据已完全清除")
        except Exception as e:
            logger.error(f"[Repo] 删除会话数据失败: {e}")

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

    @staticmethod
    async def log_token_usage(session_id: str, model_name: str, prompt_tokens: int, completion_tokens: int):
        """记录 Token 消耗"""
        try:
            await TokenUsageModel.create(
                session_id=session_id,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        except Exception as e:
            logger.error(f"[Repo] 记录 Token 消耗失败: {e}")

    @staticmethod
    async def get_first_interaction_time(session_id: str, user_id: str) -> datetime | None:
        """获取用户首次交互时间"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return None
            
            user_db = await UserProfileModel.get_or_none(
                session=session_db,
                user_id=str(user_id)
            )
            if not user_db:
                return None
            
            first_log = await InteractionLogModel.filter(user=user_db).order_by("timestamp").first()
            return first_log.timestamp if first_log else None
        except Exception as e:
            logger.error(f"[Repo] 获取首次交互时间失败: {e}")
            return None

    @staticmethod
    async def get_token_stats(group_id: str | int, model_names: list[str] | None = None) -> dict:
        """
        获取统计数据
        
        Args:
            group_id: 群组 ID
            model_names: 要统计的模型列表，为 None 则统计所有模型
        
        Returns:
            {
                "1d_local": [{"model": "...", "total": 123}, ...],
                "1d_global": [...],
                "7d_local": [...],
                "7d_global": [...],
                "all_global": [...]
            }
        """
        result = {
            "1d_local": [],
            "1d_global": [],
            "7d_local": [],
            "7d_global": [],
            "all_global": []
        }
        group_id_str = str(group_id)
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        seven_days_ago = now - timedelta(days=7)

        async def _query(filter_kwargs):
            # 如果指定了模型列表，添加过滤条件
            if model_names:
                filter_kwargs["model_name__in"] = model_names
            
            # 按模型分组统计
            stats = await TokenUsageModel.filter(**filter_kwargs) \
                .annotate(
                    total_prompt=Sum("prompt_tokens"),
                    total_completion=Sum("completion_tokens")
                ) \
                .group_by("model_name") \
                .values("model_name", "total_prompt", "total_completion")
            
            return [
                {
                    "model": s["model_name"], 
                    "prompt": s["total_prompt"] or 0,
                    "completion": s["total_completion"] or 0,
                    "total": (s["total_prompt"] or 0) + (s["total_completion"] or 0)
                } 
                for s in stats
            ]

        try:
            # 1. 一天内本群
            result["1d_local"] = await _query({"session_id": group_id_str, "timestamp__gte": one_day_ago})
            # 2. 一天内全局
            result["1d_global"] = await _query({"timestamp__gte": one_day_ago})
            
            # 3. 七天内本群
            result["7d_local"] = await _query({"session_id": group_id_str, "timestamp__gte": seven_days_ago})
            # 4. 七天内全局
            result["7d_global"] = await _query({"timestamp__gte": seven_days_ago})

            # 5. 所有时间所有消耗
            result["all_global"] = await _query({})
            
        except Exception as e:
            logger.error(f"[Repo] 查询 Token 统计失败: {e}")

        return result
