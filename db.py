import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta

from nonebot import logger
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from .memory.short_term import SHORT_TERM_BUFFER_SIZE, Message
from .models import (
    DailyTokenUsageModel,
    EnabledGroupModel,
    GlobalMessageModel,
    InteractionLogModel,
    SessionModel,
    TokenUsageModel,
    UserProfileModel,
)
from .token_stats import TOKEN_FIELDS, merge_token_stats_by_model


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    try:
        return text.encode("utf-8", "ignore").decode("utf-8")
    except (AttributeError, UnicodeError):
        return ""


async def load_enabled_group_ids() -> set[int]:
    """加载启用群组；数据库是唯一来源，由 /autochat enable|disable 维护。"""

    return {g.group_id for g in await EnabledGroupModel.all()}


async def enable_group(group_id: int):
    async with in_transaction():
        await EnabledGroupModel.get_or_create(group_id=group_id)


async def disable_group(group_id: int):
    async with in_transaction():
        await EnabledGroupModel.filter(group_id=group_id).delete()


async def delete_session_data(session_id: str):
    """删除会话的所有关联数据（消息、用户画像、交互日志），不删除会话本身"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        return

    users = await UserProfileModel.filter(session=session_db).all()
    for user in users:
        deleted_count = await InteractionLogModel.filter(user=user).delete()
        if deleted_count:
            logger.debug(
                f"[Repo] 删除用户 {user.user_id} 的 {deleted_count} 条交互日志"
            )

    profile_count = await UserProfileModel.filter(session=session_db).delete()
    msg_count = await GlobalMessageModel.filter(session=session_db).delete()
    logger.info(
        f"[Repo] 会话 {session_id} 数据已清除: "
        f"profiles={profile_count}, messages={msg_count}"
    )


async def save_session_state(session_id: str, data: dict):
    """保存会话的基础状态"""

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


async def load_full_session_data(session_id: str):
    """加载完整的会话数据"""

    session_db = await SessionModel.filter(id=session_id).first()
    if not session_db:
        return None

    users_data = [
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
        for user_db in await UserProfileModel.filter(session=session_db)
    ]

    msgs_db = (
        await GlobalMessageModel.filter(session=session_db)
        .order_by("-time")
        .limit(SHORT_TERM_BUFFER_SIZE)
    )
    history_msgs = [
        Message(
            time=msg_db.time,
            user_name=msg_db.user_name,
            content=msg_db.content,
            id=msg_db.msg_id,
            user_id=msg_db.user_id or "",
        )
        for msg_db in reversed(msgs_db)
    ]

    return {
        "session": session_db,
        "users": users_data,
        "messages": history_msgs,
        "last_consolidated_time": session_db.last_consolidated_time,
    }


def _field_changed(field: str, existing, value) -> bool:
    if (
        field == "time"
        and isinstance(existing, datetime)
        and isinstance(value, datetime)
    ):
        return abs(existing.timestamp() - value.timestamp()) > 0.000001
    return existing != value


def _message_final_id(msg: Message) -> str:
    if msg._persistence_id:
        return msg._persistence_id
    final_msg_id = str(msg.id or "")
    if not final_msg_id:
        unique_str = "_".join(
            [
                sanitize_text(msg.content),
                str(msg.time.timestamp()),
                str(msg.user_id or ""),
                sanitize_text(msg.user_name),
            ]
        )
        final_msg_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
    msg._persistence_id = final_msg_id
    return final_msg_id


async def sync_messages(session_id: str, recent_msgs: list[Message]):
    """增量同步消息到数据库"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        raise RuntimeError(f"session not found: {session_id}")

    existing_by_id: dict[str, dict] = {}
    final_msg_ids = [_message_final_id(msg) for msg in recent_msgs]
    if final_msg_ids:
        existing_rows = await GlobalMessageModel.filter(
            session=session_db,
            msg_id__in=final_msg_ids,
        ).values("msg_id", "user_name", "user_id", "content", "time")
        existing_by_id = {str(row["msg_id"]): row for row in existing_rows}

    bulk_msgs = []
    updates: list[tuple[str, dict]] = []
    for msg in recent_msgs:
        final_msg_id = _message_final_id(msg)
        values = {
            "user_name": sanitize_text(msg.user_name),
            "user_id": str(msg.user_id) if msg.user_id else "",
            "content": sanitize_text(msg.content),
            "time": msg.time,
        }
        existing = existing_by_id.get(final_msg_id)

        if existing is None:
            bulk_msgs.append(
                GlobalMessageModel(session=session_db, **values, msg_id=final_msg_id)
            )
            existing_by_id[final_msg_id] = {"msg_id": final_msg_id, **values}
            continue

        if any(
            _field_changed(field, existing.get(field), value)
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


async def get_history_before(
    session_id: str, time_point: datetime, limit: int = 20
) -> list[Message]:
    """获取指定时间之前的历史消息"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        return []

    history_rows = (
        await GlobalMessageModel.filter(
            session=session_db,
            time__lt=time_point,
        )
        .order_by("-time")
        .limit(limit)
    )

    return [
        Message(
            time=m.time,
            user_name=m.user_name,
            content=m.content,
            id=m.msg_id,
            user_id=m.user_id or "",
        )
        for m in sorted(history_rows, key=lambda x: x.time)
    ]


async def get_recent_messages_by_user(
    session_id: str,
    user_id: str = "",
    user_name: str = "",
    limit: int = 10,
) -> list[str]:
    """获取用户最近的发言内容"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        return []

    db_msgs = []
    if user_id and str(user_id).strip():
        db_msgs = (
            await GlobalMessageModel.filter(
                session=session_db,
                user_id=str(user_id),
            )
            .order_by("-time")
            .limit(limit)
        )

    if not db_msgs and user_name:
        db_msgs = (
            await GlobalMessageModel.filter(
                session=session_db,
                user_name=user_name,
            )
            .order_by("-time")
            .limit(limit)
        )

    return [m.content for m in reversed(db_msgs)]


async def update_user_profiles(session_id: str, profiles: dict):
    """批量更新用户画像"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        raise RuntimeError(f"session not found: {session_id}")

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


async def log_interactions(session_id: str, interactions: list[tuple[str, dict]]):
    """批量记录交互日志"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db or not interactions:
        return

    user_ids = [str(user_id) for user_id, _ in interactions]
    existing_users = await UserProfileModel.filter(
        session=session_db, user_id__in=user_ids
    )
    user_map = {user.user_id: user for user in existing_users}

    missing_ids = [user_id for user_id in user_ids if user_id not in user_map]
    if missing_ids:
        await UserProfileModel.bulk_create(
            [
                UserProfileModel(session=session_db, user_id=user_id)
                for user_id in set(missing_ids)
            ],
            ignore_conflicts=True,
        )
        existing_users = await UserProfileModel.filter(
            session=session_db, user_id__in=user_ids
        )
        user_map = {user.user_id: user for user in existing_users}

    now = datetime.now()
    rows = []
    increments: dict[int, int] = {}
    for user_id, delta in interactions:
        user_db = user_map.get(str(user_id))
        if not user_db:
            continue
        increments[user_db.id] = increments.get(user_db.id, 0) + 1
        rows.append(
            InteractionLogModel(
                user=user_db,
                delta_valence=delta.get("valence", 0.0),
                delta_arousal=delta.get("arousal", 0.0),
                delta_dominance=delta.get("dominance", 0.0),
                timestamp=now,
            )
        )
    if not rows:
        return

    await InteractionLogModel.bulk_create(rows)
    conn = Tortoise.get_connection("default")
    await conn.execute_many(
        """
        UPDATE nyabot_user_profiles
        SET interaction_count = interaction_count + ?,
            first_interaction_at = COALESCE(first_interaction_at, ?),
            last_interaction_at = ?
        WHERE id = ?
        """,
        [[count, now, now, user_pk] for user_pk, count in increments.items()],
    )


async def log_token_usage(
    session_id: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
    reasoning_tokens: int = 0,
    finish_reason: str = "",
    provider: str = "",
):
    await log_token_usages(
        [
            {
                "session_id": session_id,
                "model_name": model_name,
                "provider": provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
                "reasoning_tokens": reasoning_tokens,
                "finish_reason": finish_reason,
            }
        ]
    )


async def log_token_usages(rows: list[dict]):
    """In one transaction, append raw usage and atomically increment daily totals."""

    if not rows:
        return
    now = datetime.now().astimezone()
    day = now.date().isoformat()
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {field: 0 for field in TOKEN_FIELDS} | {"request_count": 0}
    )
    for row in rows:
        key = (
            str(row.get("session_id") or ""),
            str(row.get("model_name") or ""),
            str(row.get("provider") or ""),
        )
        for field in TOKEN_FIELDS:
            grouped[key][field] += int(row.get(field, 0) or 0)
        grouped[key]["request_count"] += 1

    try:
        async with in_transaction("default") as conn:
            await TokenUsageModel.bulk_create(
                [
                    TokenUsageModel(
                        session_id=row.get("session_id", ""),
                        model_name=row.get("model_name", ""),
                        provider=row.get("provider", ""),
                        prompt_tokens=row.get("prompt_tokens", 0),
                        completion_tokens=row.get("completion_tokens", 0),
                        prompt_cache_hit_tokens=row.get("prompt_cache_hit_tokens", 0),
                        prompt_cache_miss_tokens=row.get("prompt_cache_miss_tokens", 0),
                        reasoning_tokens=row.get("reasoning_tokens", 0),
                        finish_reason=row.get("finish_reason", ""),
                    )
                    for row in rows
                ],
                using_db=conn,
            )
            await conn.execute_many(
                """
                INSERT INTO nyabot_daily_token_usage (
                    day, session_id, model_name, provider,
                    prompt_tokens, completion_tokens,
                    prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                    reasoning_tokens, request_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day, session_id, model_name, provider)
                DO UPDATE SET
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    prompt_cache_hit_tokens = prompt_cache_hit_tokens + excluded.prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens = prompt_cache_miss_tokens + excluded.prompt_cache_miss_tokens,
                    reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                    request_count = request_count + excluded.request_count
                """,
                [
                    [
                        day,
                        session_id,
                        model_name,
                        provider,
                        totals["prompt_tokens"],
                        totals["completion_tokens"],
                        totals["prompt_cache_hit_tokens"],
                        totals["prompt_cache_miss_tokens"],
                        totals["reasoning_tokens"],
                        totals["request_count"],
                    ]
                    for (session_id, model_name, provider), totals in grouped.items()
                ],
            )
    except Exception as e:
        logger.error(f"[Repo] 记录 Token 消耗失败: {e}")


async def get_token_stats(
    group_id: str | int,
    model_names: list[str] | None = None,
) -> dict:
    """Read all five views from one compact daily-aggregate query."""

    result = {
        "1d_local": [],
        "1d_global": [],
        "7d_local": [],
        "7d_global": [],
        "all_global": [],
    }
    group_id_str = str(group_id)
    today = date.today()
    seven_day_cutoff = today - timedelta(days=6)
    try:
        query = DailyTokenUsageModel.all()
        if model_names:
            query = query.filter(model_name__in=model_names)
        rows = await query.values(
            "day",
            "session_id",
            "model_name",
            "provider",
            *TOKEN_FIELDS,
        )

        buckets = {
            name: defaultdict(lambda: {field: 0 for field in TOKEN_FIELDS})
            for name in result
        }
        for row in rows:
            row_day = row["day"]
            if isinstance(row_day, str):
                row_day = date.fromisoformat(row_day)
            key = (str(row.get("model_name") or ""), str(row.get("provider") or ""))
            is_local = str(row.get("session_id") or "") == group_id_str
            targets = ["all_global"]
            if row_day >= seven_day_cutoff:
                targets.append("7d_global")
                if is_local:
                    targets.append("7d_local")
            if row_day >= today:
                targets.append("1d_global")
                if is_local:
                    targets.append("1d_local")
            for bucket in targets:
                for field in TOKEN_FIELDS:
                    buckets[bucket][key][field] += int(row.get(field, 0) or 0)

        for name in result:
            result[name] = merge_token_stats_by_model(buckets[name])
    except Exception as e:
        logger.error(f"[Repo] 查询 Token 统计失败: {e}")
    return result
