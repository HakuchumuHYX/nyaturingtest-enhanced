import unicodedata
import uuid
from datetime import date, datetime, timedelta

from nonebot import logger
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from ..messages.message import SHORT_TERM_BUFFER_SIZE, ChatMessage
from .models import (
    DailyTokenUsageModel,
    EnabledGroupModel,
    GlobalMessageModel,
    MemoryModel,
    PersonaGroupModel,
    PersonaHistoryModel,
    PersonaLifeModel,
    PersonaStatusModel,
    SessionModel,
    StickerModel,
    ThreadModel,
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
    """启用的群，由 /autochat enable|disable 维护。"""

    return {g.group_id for g in await EnabledGroupModel.all()}


async def enable_group(group_id: int):
    async with in_transaction():
        await EnabledGroupModel.get_or_create(group_id=group_id)


async def disable_group(group_id: int):
    async with in_transaction():
        await EnabledGroupModel.filter(group_id=group_id).delete()


async def delete_session_data(session_id: str):
    """删除会话的所有关联数据（消息、用户画像与档案、群志、群层、悬着的事），不删除会话本身；人生线各群共享，不动"""

    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        return

    profile_count = await UserProfileModel.filter(session=session_db).delete()
    await SessionModel.filter(id=session_id).update(
        group_notes="", notes_summarized_until=None, episodes_until=None
    )
    msg_count = await GlobalMessageModel.filter(session=session_db).delete()
    persona_count = await PersonaGroupModel.filter(session=session_db).delete()
    thread_count = await ThreadModel.filter(session=session_db).delete()
    logger.info(
        f"[Repo] 会话 {session_id} 数据已清除: "
        f"profiles={profile_count}, messages={msg_count}, persona_groups={persona_count}, "
        f"threads={thread_count}"
    )


async def save_session_state(session_id: str, data: dict):
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
            "last_read_time": data.get("last_read_time"),
            "climbed_until": data.get("climbed_until"),
        },
    )


async def load_persona_texts(session_id: str, preset: str) -> tuple[str, str]:
    """（人生线, 本群群层）；还没整理过的是空串。"""

    life = await PersonaLifeModel.get_or_none(preset=preset)
    group = await PersonaGroupModel.get_or_none(session_id=session_id, preset=preset)
    return (life.text if life else "", group.text if group else "")


def _parse_times(items: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [
        {**item, **{key: datetime.fromisoformat(item[key]) for key in keys if item.get(key)}}
        for item in items
    ]


def _dump_times(items: list[dict]) -> list[dict]:
    return [
        {key: value.isoformat() if isinstance(value, datetime) else value for key, value in item.items()}
        for item in items
    ]


async def load_persona_status(preset: str) -> dict | None:
    """她的时间线；时间字段读成 datetime。还没定过状态返回 None。"""

    row = await PersonaStatusModel.get_or_none(preset=preset)
    if row is None:
        return None
    return {
        "timeline": _parse_times(row.timeline, ("start",)),
        "until": row.until,
        "nights": _parse_times(row.nights, ("sleep", "wake")),
        "diary": row.diary,
        "diary_until": row.diary_until,
    }


async def save_persona_status(
    preset: str, timeline: list[dict], until: datetime, nights: list[dict]
) -> None:
    """只写状态这几列，日记由每日整理写，不碰：整行 save 会把读出来那一刻的旧日记写回去。"""

    values = {"timeline": _dump_times(timeline), "until": until, "nights": _dump_times(nights)}
    if not await PersonaStatusModel.filter(preset=preset).update(**values):
        await PersonaStatusModel.create(preset=preset, **values)


async def save_persona_diary(preset: str, diary: list[dict], diary_until: date) -> None:
    await PersonaStatusModel.filter(preset=preset).update(diary=diary, diary_until=diary_until)


async def load_life_text(preset: str) -> str:
    life = await PersonaLifeModel.get_or_none(preset=preset)
    return life.text if life else ""


async def skip_persona_absence(session_id: str, preset: str) -> None:
    """切回以前用过的预设：换走期间说话的是别的人设，那段聊天不是 ta 的经历，整理进度直接跳到昨天。"""

    yesterday = date.today() - timedelta(days=1)
    await PersonaGroupModel.filter(
        session_id=session_id, preset=preset, digested_until__lt=yesterday
    ).update(digested_until=yesterday)


async def load_full_session_data(session_id: str):
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
        ChatMessage(
            time=msg_db.time,
            user_name=msg_db.user_name,
            content=msg_db.content,
            id=msg_db.msg_id,
            user_id=msg_db.user_id or "",
            to_me=msg_db.to_me,
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


def message_final_id(msg: ChatMessage) -> str:
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


async def sync_messages(session_id: str, recent_msgs: list[ChatMessage]):
    session_db = await SessionModel.get_or_none(id=session_id)
    if not session_db:
        raise RuntimeError(f"session not found: {session_id}")

    existing_by_id: dict[str, dict] = {}
    final_msg_ids = [message_final_id(msg) for msg in recent_msgs]
    if final_msg_ids:
        existing_rows = await GlobalMessageModel.filter(
            session=session_db,
            msg_id__in=final_msg_ids,
        ).values("msg_id", "user_name", "user_id", "content", "time", "to_me")
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
            "to_me": msg.to_me,
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


async def message_sender_name(session_id: str, msg_id: str) -> str | None:
    """消息库里这条的发言人名字；不在库里返回 None。"""

    row = await GlobalMessageModel.get_or_none(session_id=session_id, msg_id=msg_id)
    return row.user_name if row else None


def _row_message(row: GlobalMessageModel) -> ChatMessage:
    return ChatMessage(
        time=row.time,
        user_name=row.user_name,
        content=row.content,
        id=row.msg_id,
        user_id=row.user_id or "",
        to_me=row.to_me,
    )


async def get_messages_between(
    session_id: str, after: datetime, before: datetime, limit: int
) -> list[ChatMessage]:
    """after 之后、before 之前最早的 limit 条，从旧到新。"""

    rows = (
        await GlobalMessageModel.filter(session_id=session_id, time__gt=after, time__lt=before)
        .order_by("time")
        .limit(limit)
    )
    return [_row_message(row) for row in rows]


async def get_history_before(
    session_id: str, time_point: datetime, limit: int = 20
) -> list[ChatMessage]:
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

    return [_row_message(m) for m in sorted(history_rows, key=lambda x: x.time)]


async def get_catch_up(
    session_id: str,
    after: datetime,
    before: datetime,
    *,
    calls_after: datetime,
    calls_limit: int,
) -> tuple[list[ChatMessage], list[str], int]:
    """after 到 before 之间被挤出未读的那段：calls_after 之后找过她的消息（最近几条）、after 之后写成的分段小结、这段一共多少条。"""

    rows = GlobalMessageModel.filter(session_id=session_id, time__gt=after, time__lt=before)
    calls = await (
        GlobalMessageModel.filter(
            session_id=session_id, time__gt=calls_after, time__lt=before, to_me=True
        )
        .order_by("-time")
        .limit(calls_limit)
    )
    episodes = (
        await MemoryModel.filter(session_id=session_id, category="episode", created_at__gt=after)
        .order_by("created_at")
        .values_list("content", flat=True)
    )
    return (
        [_row_message(m) for m in reversed(calls)],
        list(episodes),
        await rows.count(),
    )


async def get_recent_messages_by_user(
    session_id: str,
    user_id: str = "",
    user_name: str = "",
    limit: int = 10,
) -> list[str]:
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
    seen_at: datetime,
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
        seen_count=1,
        last_seen_at=seen_at,
    )


async def save_sticker_embedding(
    digest: str, embedding: bytes, embedding_model: str
) -> None:
    await StickerModel.filter(digest=digest).update(
        embedding=embedding, embedding_model=embedding_model
    )


async def mark_sticker_sent(digest: str, sent_at: datetime) -> None:
    await StickerModel.filter(digest=digest).update(last_sent_at=sent_at)


async def mark_sticker_seen(digest: str, seen_count: int, seen_at: datetime) -> None:
    await StickerModel.filter(digest=digest).update(
        seen_count=seen_count, last_seen_at=seen_at
    )


async def delete_stickers(digests: list[str]) -> None:
    """删行不拉黑：清理掉的图以后再有人发还能重新收。"""

    await StickerModel.filter(digest__in=digests).delete()


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


async def token_usage_by_session(since: datetime) -> list[dict]:
    """各会话从 since 起的 Token 和请求数；按明细表算滑动窗口，日汇总表只有自然日。"""

    return await Tortoise.get_connection("default").execute_query_dict(
        """
        SELECT session_id, SUM(prompt_tokens + completion_tokens) AS total, COUNT(*) AS requests
        FROM nyabot_token_usage
        WHERE timestamp >= ?
        GROUP BY session_id
        """,
        [since.isoformat(" ")],
    )


async def token_usage_by_model(session_id: str, since: datetime, model_names: list[str]) -> list[dict]:
    """这个会话从 since 起按模型分的 Token，同一模型的不同写法并成一条。"""

    rows = await Tortoise.get_connection("default").execute_query_dict(
        """
        SELECT model_name, SUM(prompt_tokens) AS prompt, SUM(completion_tokens) AS completion,
               SUM(reasoning_tokens) AS reasoning, COUNT(*) AS requests
        FROM nyabot_token_usage
        WHERE session_id = ? AND timestamp >= ?
        GROUP BY model_name
        """,
        [session_id, since.isoformat(" ")],
    )
    names = {_model_key(name): name for name in model_names}
    merged: dict[str, dict] = {}
    for row in rows:
        key = _model_key(row["model_name"])
        entry = merged.setdefault(
            key,
            {"model": names.get(key, row["model_name"]), "prompt": 0, "completion": 0, "reasoning": 0, "requests": 0},
        )
        for field in ("prompt", "completion", "reasoning", "requests"):
            entry[field] += row[field]
    return sorted(merged.values(), key=lambda e: e["prompt"] + e["completion"], reverse=True)


async def message_buckets(since: datetime) -> list[dict]:
    """从 since（当天零点）起各群每 30 分钟的消息数；bucket 是当天第几个 30 分钟。"""

    return await Tortoise.get_connection("default").execute_query_dict(
        """
        SELECT session_id,
               CAST(strftime('%H', time) AS INTEGER) * 2 + CAST(strftime('%M', time) AS INTEGER) / 30 AS bucket,
               COUNT(*) AS count
        FROM nyabot_global_messages
        WHERE time >= ?
        GROUP BY session_id, bucket
        """,
        [since.isoformat(" ")],
    )


async def self_speak_times(session_id: str, name: str, since: datetime) -> list[datetime]:
    """她从 since 起开口的时刻。她名下还有戳一戳、贴表情的记录，那些不是说的话；表情包算开口。"""

    return await (
        GlobalMessageModel.filter(session_id=session_id, user_name=name, time__gte=since)
        .exclude(content__startswith="[戳一戳:")
        .exclude(content__startswith="[给")
        .order_by("time")
        .values_list("time", flat=True)
    )


async def load_life(preset: str) -> dict:
    """人生线和还没整理进去的自我陈述；还没整理过时文本是空串。"""

    life = await PersonaLifeModel.get_or_none(preset=preset)
    if life is None:
        return {"text": "", "last_digest": None, "pending_facts": []}
    return {
        "text": life.text,
        "last_digest": life.last_digest.isoformat() if life.last_digest else None,
        "pending_facts": life.pending_facts,
    }


async def load_digest_day(day: date) -> dict:
    """这一天的整理内容：人生线和各群的改动说明、日记、观察日记。日记和观察日记只留最近几天，再早就没有了。"""

    rows = await PersonaHistoryModel.filter(date=day).order_by("id").values("scope", "key", "changes")
    prefix = f"{day:%m-%d} "
    return {
        "life": [
            {"preset": row["key"], "changes": row["changes"]}
            for row in rows
            if row["scope"] == "life"
        ],
        "groups": [
            {"session_id": row["key"].partition(":")[0], "changes": row["changes"]}
            for row in rows
            if row["scope"] == "group"
        ],
        "diary": [
            {"preset": status.preset, "text": entry["text"]}
            for status in await PersonaStatusModel.all()
            for entry in status.diary
            if entry["day"] == day.isoformat()
        ],
        "journals": [
            {"session_id": layer.session_id, "text": entry.removeprefix(prefix)}
            for layer in await PersonaGroupModel.all()
            for entry in layer.journal
            if entry.startswith(prefix)
        ],
    }


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
