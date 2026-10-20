import unicodedata
import uuid
from datetime import date, datetime, timedelta

from nonebot import logger
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from ..messages.message import SHORT_TERM_BUFFER_SIZE, Message
from .models import (
    DailyTokenUsageModel,
    EnabledGroupModel,
    GlobalMessageModel,
    PersonaGroupModel,
    PersonaLifeModel,
    SessionModel,
    StickerModel,
    TokenUsageModel,
    UserProfileModel,
)


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
    """删除会话的所有关联数据（消息、用户画像与档案、群志、群层），不删除会话本身；人生线各群共享，不动"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        return

    profile_count = await UserProfileModel.filter(session=session_db).delete()
    await SessionModel.filter(id=session_id).update(
        group_notes="", notes_summarized_until=None, episodes_until=None
    )
    msg_count = await GlobalMessageModel.filter(session=session_db).delete()
    persona_count = await PersonaGroupModel.filter(session=session_db).delete()
    logger.info(
        f"[Repo] 会话 {session_id} 数据已清除: "
        f"profiles={profile_count}, messages={msg_count}, persona_groups={persona_count}"
    )


async def save_session_state(session_id: str, data: dict):
    """保存会话的基础状态"""

    await SessionModel.update_or_create(
        id=session_id,
        defaults={
            "preset": data["preset"],
            "valence": data.get("valence", 0.0),
            "arousal": data.get("arousal", 0.0),
            "dominance": data.get("dominance", 0.0),
            "emotion_updated_at": data["emotion_updated_at"],
            "chat_summary": sanitize_text(data.get("chat_summary", "")),
            "last_speak_time": data.get("last_speak_time"),
            "last_consolidated_time": data.get("last_consolidated_time"),
            "chatting_state": data.get("chatting_state", 0),
        },
    )


async def load_persona_texts(session_id: str, preset: str) -> tuple[str, str]:
    """（人生线, 本群群层）；还没整理过的是空串。"""

    life = await PersonaLifeModel.get_or_none(preset=preset)
    group = await PersonaGroupModel.get_or_none(session_id=session_id, preset=preset)
    return (life.text if life else "", group.text if group else "")


async def skip_persona_absence(session_id: str, preset: str) -> None:
    """切回以前用过的预设：换走期间说话的是别的人设，那段聊天不是 ta 的经历，整理进度直接跳到昨天。"""

    yesterday = date.today() - timedelta(days=1)
    await PersonaGroupModel.filter(
        session_id=session_id, preset=preset, digested_until__lt=yesterday
    ).update(digested_until=yesterday)


async def load_full_session_data(session_id: str):
    """加载完整的会话数据"""

    session_db = await SessionModel.filter(id=session_id).first()
    if not session_db:
        return None

    users_data = [
        {
            "user_id": user_db.user_id,
            "affinity": user_db.affinity,
            "affinity_updated_at": user_db.affinity_updated_at,
            "interaction_count": user_db.interaction_count,
            "first_interaction_at": user_db.first_interaction_at,
            "last_interaction_at": user_db.last_interaction_at,
            "summary": user_db.summary,
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
    }


def _field_changed(field: str, existing, value) -> bool:
    if (
        field == "time"
        and isinstance(existing, datetime)
        and isinstance(value, datetime)
    ):
        return abs(existing.timestamp() - value.timestamp()) > 0.000001
    return existing != value


def message_final_id(msg: Message) -> str:
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
    final_msg_ids = [message_final_id(msg) for msg in recent_msgs]
    if final_msg_ids:
        existing_rows = await GlobalMessageModel.filter(
            session=session_db,
            msg_id__in=final_msg_ids,
        ).values("msg_id", "user_name", "user_id", "content", "time")
        existing_by_id = {str(row["msg_id"]): row for row in existing_rows}

    bulk_msgs = []
    updates: list[tuple[str, dict]] = []
    for msg in recent_msgs:
        final_msg_id = message_final_id(msg)
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
    for final_msg_id, values in updates:
        await GlobalMessageModel.filter(
            session=session_db,
            msg_id=final_msg_id,
        ).update(**values)


async def recall_message(session_id: str, msg_id: str, content: str) -> int:
    """消息库里那条改成撤回标记；还没落库时改不到，由缓冲里改写过的那条落库时带上。"""

    return await GlobalMessageModel.filter(session_id=session_id, msg_id=msg_id).update(
        content=content
    )


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


async def get_latest_user_names(session_id: str) -> dict[str, str]:
    """本群每个 QQ 号最近一条消息的发送者名字，即当前群名片。"""

    # SQLite 对 MAX() 聚合会让同一行的裸列取到最大值那一行的值
    rows = await Tortoise.get_connection("default").execute_query_dict(
        """
        SELECT user_id, user_name, MAX(time) AS latest
        FROM nyabot_global_messages
        WHERE session_id = ? AND user_id != ''
        GROUP BY user_id
        """,
        [session_id],
    )
    return {row["user_id"]: row["user_name"] for row in rows}


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
                "affinity": profile.affinity,
                "affinity_updated_at": profile.affinity_updated_at,
                "interaction_count": profile.interaction_count,
                "first_interaction_at": profile.first_interaction_at,
                "last_interaction_at": profile.last_interaction_at,
            },
        )


TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "reasoning_tokens",
)


async def load_stickers() -> list[StickerModel]:
    """没拉黑的表情包，启动时整表载入内存。"""

    return await StickerModel.filter(blocked=False)


async def sticker_known(digest: str) -> bool:
    """库里有这张，或者它被拉黑过。"""

    return await StickerModel.exists(digest=digest)


async def add_sticker(
    *,
    digest: str,
    ext: str,
    reaction: str,
    subject: str,
    tags: list[str],
    source: str,
    embedding: bytes | None,
    embedding_model: str,
) -> StickerModel:
    return await StickerModel.create(
        digest=digest,
        ext=ext,
        reaction=reaction,
        subject=subject,
        tags=tags,
        source=source,
        embedding=embedding,
        embedding_model=embedding_model,
    )


async def save_sticker_embedding(
    digest: str, embedding: bytes, embedding_model: str
) -> None:
    await StickerModel.filter(digest=digest).update(
        embedding=embedding, embedding_model=embedding_model
    )


async def mark_sticker_sent(digest: str, sent_at: datetime) -> None:
    await StickerModel.filter(digest=digest).update(last_sent_at=sent_at)


async def block_sticker(digest: str) -> None:
    """拉黑：库里有就清掉描述和向量，没有就插一条空行占住这个 digest。"""

    await StickerModel.update_or_create(
        digest=digest,
        defaults={
            "blocked": True,
            "ext": "",
            "reaction": "",
            "subject": "",
            "tags": [],
            "embedding": None,
            "embedding_model": "",
        },
    )


def _model_key(name: str) -> str:
    """同一个模型在不同上游的写法不同：大小写、带不带厂商前缀（deepseek-ai/DeepSeek-V4-Flash）。"""

    return unicodedata.normalize("NFKC", name).strip().casefold().rpartition("/")[2]


async def log_token_usage(session_id: str, model_name: str, usage: dict):
    """同一事务里追加一条明细，并原子累加当日汇总。"""

    day = datetime.now().astimezone().date().isoformat()
    tokens = [int(usage.get(field) or 0) for field in TOKEN_FIELDS]
    try:
        async with in_transaction("default") as conn:
            await TokenUsageModel.create(
                session_id=session_id,
                model_name=model_name,
                finish_reason=usage.get("finish_reason", ""),
                **dict(zip(TOKEN_FIELDS, tokens)),
                using_db=conn,
            )
            await conn.execute_query(
                """
                INSERT INTO nyabot_daily_token_usage (
                    day, session_id, model_name,
                    prompt_tokens, completion_tokens,
                    prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                    reasoning_tokens, request_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(day, session_id, model_name)
                DO UPDATE SET
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    prompt_cache_hit_tokens = prompt_cache_hit_tokens + excluded.prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens = prompt_cache_miss_tokens + excluded.prompt_cache_miss_tokens,
                    reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                    request_count = request_count + excluded.request_count
                """,
                [day, session_id, model_name, *tokens],
            )
    except Exception as e:
        logger.error(f"[Repo] 记录 Token 消耗失败: {e}")


async def get_token_stats(group_id: str | int, model_names: list[str]) -> dict:
    """今日 / 近七日 / 累计用量，只算 model_names 里的模型。
    同一模型的不同写法并成一条，用 model_names 里的名字显示。"""

    names = {_model_key(name): name for name in model_names}
    views: dict[str, dict[str, dict]] = {
        view: {} for view in ("1d_local", "1d_global", "7d_local", "7d_global", "all_global")
    }
    group_id_str = str(group_id)
    today = date.today()
    seven_day_cutoff = today - timedelta(days=6)
    try:
        rows = await DailyTokenUsageModel.all().values(
            "day",
            "session_id",
            "model_name",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        )
        for row in rows:
            name = names.get(_model_key(row["model_name"]))
            if name is None:
                continue
            row_day = row["day"]
            if isinstance(row_day, str):
                row_day = date.fromisoformat(row_day)
            is_local = row["session_id"] == group_id_str
            targets = ["all_global"]
            if row_day >= seven_day_cutoff:
                targets.append("7d_global")
                if is_local:
                    targets.append("7d_local")
            if row_day >= today:
                targets.append("1d_global")
                if is_local:
                    targets.append("1d_local")
            for view in targets:
                entry = views[view].setdefault(
                    name, {"model": name, "prompt": 0, "completion": 0, "reasoning": 0}
                )
                entry["prompt"] += row["prompt_tokens"]
                entry["completion"] += row["completion_tokens"]
                entry["reasoning"] += row["reasoning_tokens"]
    except Exception as e:
        logger.error(f"[Repo] 查询 Token 统计失败: {e}")
    return {
        view: [
            {**entry, "total": entry["prompt"] + entry["completion"]}
            for entry in entries.values()
        ]
        for view, entries in views.items()
    }
