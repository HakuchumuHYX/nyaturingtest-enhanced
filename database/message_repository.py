import uuid
from datetime import datetime

from nonebot import logger

from ..memory.short_term import Message
from ..models.database import GlobalMessageModel, SessionModel
from ..core.text_utils import sanitize_text


class MessageRepository:
    @staticmethod
    def _field_changed(field: str, existing, value) -> bool:
        if field == "time" and isinstance(existing, datetime) and isinstance(value, datetime):
            try:
                return abs(existing.timestamp() - value.timestamp()) > 0.000001
            except (OSError, ValueError):
                pass
        return existing != value

    @staticmethod
    def _message_final_id(msg: Message) -> str:
        cached_id = str(getattr(msg, "_persistence_id", "") or "")
        if cached_id:
            return cached_id
        final_msg_id = str(msg.id or "")
        if not final_msg_id:
            unique_str = "_".join([
                sanitize_text(msg.content),
                str(msg.time.timestamp()),
                str(msg.user_id or ""),
                sanitize_text(msg.user_name),
            ])
            final_msg_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        try:
            msg._persistence_id = final_msg_id
        except (AttributeError, TypeError):
            pass
        return final_msg_id

    @staticmethod
    async def sync_messages(session_id: str, recent_msgs: list[Message]):
        """增量同步消息到数据库"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                raise RuntimeError(f"session not found: {session_id}")

            final_msg_ids = [MessageRepository._message_final_id(msg) for msg in recent_msgs]
            existing_by_id: dict[str, dict] = {}

            if final_msg_ids:
                existing_rows = await GlobalMessageModel.filter(
                    session=session_db,
                    msg_id__in=final_msg_ids,
                ).values(
                    "msg_id",
                    "user_name",
                    "user_id",
                    "content",
                    "time",
                )
                existing_by_id = {
                    str(row["msg_id"]): row
                    for row in existing_rows
                }

            bulk_msgs = []
            updates: list[tuple[str, dict]] = []
            for msg in recent_msgs:
                final_msg_id = MessageRepository._message_final_id(msg)
                values = {
                    "user_name": sanitize_text(msg.user_name),
                    "user_id": str(msg.user_id) if msg.user_id else "",
                    "content": sanitize_text(msg.content),
                    "time": msg.time,
                }
                existing = existing_by_id.get(final_msg_id)

                if existing is None:
                    bulk_msgs.append(
                        GlobalMessageModel(
                            session=session_db,
                            **values,
                            msg_id=final_msg_id,
                        )
                    )
                    existing_by_id[final_msg_id] = {
                        "msg_id": final_msg_id,
                        **values,
                    }
                    continue

                if any(
                    MessageRepository._field_changed(
                        field,
                        existing.get(field),
                        value,
                    )
                    for field, value in values.items()
                ):
                    updates.append((final_msg_id, values))
                    existing.update(values)

            if bulk_msgs:
                await GlobalMessageModel.bulk_create(bulk_msgs)
                logger.debug(f"[Repo] 同步了 {len(bulk_msgs)} 条新消息")
            for final_msg_id, values in updates:
                await GlobalMessageModel.filter(
                    session=session_db,
                    msg_id=final_msg_id,
                ).update(**values)
            if updates:
                logger.debug(f"[Repo] 更新了 {len(updates)} 条已丰富消息")

        except Exception as e:
            logger.error(f"[Repo] 同步消息失败: {e}")
            raise

    @staticmethod
    async def get_history_before(session_id: str, time_point: datetime, limit: int = 20) -> list[Message]:
        """获取指定时间之前的历史消息"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return []

            history_rows = await GlobalMessageModel.filter(
                session=session_db,
                time__lt=time_point,
            ).order_by("-time").limit(limit)

            recalled_msgs = []
            if history_rows:
                rows_sorted = sorted(history_rows, key=lambda x: x.time)
                for m in rows_sorted:
                    recalled_msgs.append(
                        Message(
                            time=m.time,
                            user_name=m.user_name,
                            content=m.content,
                            id=m.msg_id,
                            user_id=m.user_id if m.user_id else "",
                        )
                    )
            return recalled_msgs
        except Exception as e:
            logger.error(f"[Repo] 历史溯源失败: {e}")
            return []

    @staticmethod
    async def get_recent_messages_by_user(
        session_id: str,
        user_id: str = "",
        user_name: str = "",
        limit: int = 10,
    ) -> list[str]:
        """获取用户最近的发言内容"""
        try:
            session_db = await SessionModel.get_or_none(id=session_id)
            if not session_db:
                return []

            db_msgs = []
            if user_id and str(user_id).strip():
                db_msgs = await GlobalMessageModel.filter(
                    session=session_db,
                    user_id=str(user_id),
                ).order_by("-time").limit(limit)

            if not db_msgs and user_name:
                db_msgs = await GlobalMessageModel.filter(
                    session=session_db,
                    user_name=user_name,
                ).order_by("-time").limit(limit)

            return [m.content for m in reversed(db_msgs)]
        except Exception as e:
            logger.error(f"[Repo] 获取用户历史消息失败: {e}")
            return []
