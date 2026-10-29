"""每日整理：把记忆碎片归纳进用户档案和群志。

用户档案读主体或说话人是此人的事实；群志读全部新碎片，大头是分段整理写的 episode。

碎片只是原料，到期就删；对一个人、一个群的长期认知靠这里原地重写来承载。
水位是已整理到的最新碎片 created_at，每条碎片只整理一次。

新碎片只有一块时，直接「旧文本 + 碎片 → 新文本」；多块时（冷启动、积压多的人）
先逐块独立提炼要点，再「旧文本 + 各段要点 → 新文本」合并一次。
试跑对比过逐块依次累加：模型只添不删，越写越像流水账，「近期」也被最后一块主导。
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta

from nonebot import logger

from ..llm import feedback_client
from ..prompts import (
    fragments_material,
    get_group_notes_prompt,
    get_group_points_prompt,
    get_profile_points_prompt,
    get_profile_prompt,
    points_material,
)
from ..storage.db import get_latest_user_names
from ..storage.models import MemoryModel, SessionModel, UserProfileModel

PROFILE_MIN_NEW_MEMORIES = 10
# 手动查看群志时新碎片攒到这么多才重新整理：群志只在旧稿上小改，几条新碎片通常改不动什么
GROUP_NOTES_REFRESH_MIN_NEW = 10
# 用户档案的整理间隔；群志有新碎片就整理，这一两天的事才进得了群志
DIGEST_INTERVAL = timedelta(days=7)
PROFILE_CHUNK_CHARS = 8000
GROUP_NOTES_CHUNK_CHARS = 12000
DIGEST_TEMPERATURE = 0.2
POINTS_CONCURRENCY = 4

# 夜间整理和手动查看可能撞上，同一群排队；后到的进来时水位已推进，不会重复整理
_group_notes_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

_MEMORY_FIELDS = (
    "subject_user_id",
    "subject_user_name",
    "speaker_user_id",
    "speaker_user_name",
    "content",
    "category",
    "date",
    "created_at",
    "is_correction",
)


def _format_date(date: int) -> str:
    day = str(date)
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def _memory_line(row: dict) -> str:
    tag = "|说到别人" if row.get("said_about_others") else ""
    tag += "|更正" if row["is_correction"] else ""
    return f"[{_format_date(row['date'])}|{row['category']}{tag}] {row['content']}"


def _span(chunk: list[dict]) -> str:
    dates = [row["date"] for row in chunk]
    return f"{_format_date(min(dates))}～{_format_date(max(dates))}"


def _chunks(rows: list[dict], limit: int) -> list[list[dict]]:
    """按 created_at 升序切块，单块约 limit 字，不切断单条碎片。"""

    chunks: list[list[dict]] = []
    chunk: list[dict] = []
    size = 0
    for row in rows:
        if chunk and size >= limit:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(row)
        size += len(_memory_line(row))
    if chunk:
        chunks.append(chunk)
    return chunks


def _chunk_lines(chunk: list[dict]) -> str:
    return "\n".join(_memory_line(row) for row in chunk)


def _due(
    new_count: int,
    until: datetime | None,
    *,
    min_new: int | None = None,
    urgent: bool = False,
) -> bool:
    """有新碎片且（有更正 / 从没整理过 / 攒够 min_new 条 / 水位已满 7 天）。"""

    if new_count == 0:
        return False
    if urgent or until is None or (min_new is not None and new_count >= min_new):
        return True
    return datetime.now() - until >= DIGEST_INTERVAL


async def _generate(session_id: str, prompt: str, key: str) -> str | None:
    data = await feedback_client.generate_json(
        prompt, session_id=session_id, temperature=DIGEST_TEMPERATURE
    )
    if data is None or not isinstance(data.get(key), str):
        logger.warning(f"[Digest] 群 {session_id} 整理输出无效，跳过，下次再试")
        return None
    return data[key].strip()


async def _condense(
    session_id: str,
    chunks: list[list[dict]],
    *,
    rewrite_prompt: Callable[[str], str],
    points_prompt: Callable[[str, str], str],
    key: str,
) -> str | None:
    """一块直接重写；多块先并行提炼各段要点再合并。任一步失败返回 None，水位不动。

    rewrite_prompt(材料)、points_prompt(时间跨度, 碎片行) 由档案和群志各自给出模板。
    """

    if len(chunks) == 1:
        return await _generate(
            session_id, rewrite_prompt(fragments_material(_chunk_lines(chunks[0]))), key
        )

    limit = asyncio.Semaphore(POINTS_CONCURRENCY)

    async def points_of(chunk: list[dict]) -> str | None:
        async with limit:
            prompt = points_prompt(_span(chunk), _chunk_lines(chunk))
            return await _generate(session_id, prompt, "points")

    points = await asyncio.gather(*(points_of(chunk) for chunk in chunks))
    if any(item is None for item in points):
        return None
    segments = [(_span(chunk), item) for chunk, item in zip(chunks, points)]
    return await _generate(session_id, rewrite_prompt(points_material(segments)), key)


async def profile_rows_by_user(session_id: str) -> dict[str, list[dict]]:
    """每个人的档案材料：主体是此人的碎片，加上此人说到别人的碎片（标 said_about_others），按时间排。

    「A 说/对 B 怎样」主体记 B、说话人记 A；只按主体归属的话，A 的档案会缺掉他怎么对待别人。
    """

    rows_by_user: dict[str, list[dict]] = defaultdict(list)
    for row in (
        await MemoryModel.filter(session_id=session_id)
        .order_by("created_at")
        .values(*_MEMORY_FIELDS)
    ):
        subject, speaker = row["subject_user_id"], row["speaker_user_id"]
        if subject:
            rows_by_user[subject].append(row)
        if speaker and speaker != subject:
            rows_by_user[speaker].append({**row, "said_about_others": True})
    return rows_by_user


def profile_name(user_id: str, rows: list[dict], latest_names: dict[str, str]) -> str:
    """称呼以消息记录里的当前群名片为准：碎片里的名字可能过时，也可能被抽取标错。"""

    if latest_names.get(user_id):
        return latest_names[user_id]
    for row in reversed(rows):
        name = row[
            "speaker_user_name" if row.get("said_about_others") else "subject_user_name"
        ]
        if name:
            return name
    return user_id


async def digest_user_profiles(
    session_id: str, still_current: Callable[[], bool]
) -> dict[str, str]:
    """整理到期的用户档案，返回 {user_id: 新档案}。

    到期条件：新碎片 ≥10 条，或有新碎片且有更正 / 从没整理过 / 水位已满 7 天。
    """

    profiles = {
        profile.user_id: profile
        for profile in await UserProfileModel.filter(session_id=session_id)
    }
    rows_by_user = await profile_rows_by_user(session_id)
    latest_names = await get_latest_user_names(session_id)

    updated: dict[str, str] = {}
    calls = 0
    for user_id, rows in rows_by_user.items():
        profile = profiles.get(user_id)
        until = profile.summarized_until if profile else None
        new_rows = [row for row in rows if until is None or row["created_at"] > until]
        urgent = any(row["is_correction"] for row in new_rows)
        if not _due(len(new_rows), until, min_new=PROFILE_MIN_NEW_MEMORIES, urgent=urgent):
            continue

        name = profile_name(user_id, rows, latest_names)
        old_summary = profile.summary if profile else ""
        chunks = _chunks(new_rows, PROFILE_CHUNK_CHARS)
        calls += 1 if len(chunks) == 1 else len(chunks) + 1
        summary = await _condense(
            session_id,
            chunks,
            rewrite_prompt=lambda material: get_profile_prompt(
                name, user_id, old_summary, material
            ),
            points_prompt=lambda span, lines: get_profile_points_prompt(
                name, user_id, span, lines
            ),
            key="summary",
        )
        if summary is None:
            continue
        if not still_current():
            return {}
        await UserProfileModel.update_or_create(
            session_id=session_id,
            user_id=user_id,
            defaults={"summary": summary, "summarized_until": new_rows[-1]["created_at"]},
        )
        updated[user_id] = summary

    logger.info(
        f"[Digest] 群 {session_id} 用户档案：更新 {len(updated)} 人，调用 {calls} 次"
    )
    return updated


def _new_group_fragments(session_id: str, until: datetime | None):
    query = MemoryModel.filter(session_id=session_id)
    return query if until is None else query.filter(created_at__gt=until)


async def count_new_group_fragments(session_id: str) -> int:
    """还没整理进群志的碎片数。"""

    session_db = await SessionModel.get_or_none(id=session_id)
    until = session_db.notes_summarized_until if session_db else None
    return await _new_group_fragments(session_id, until).count()


async def digest_group_notes(
    session_id: str, still_current: Callable[[], bool], *, min_new: int = 1
) -> str | None:
    """新碎片不少于 min_new 条就合进群志，返回新群志；不到期或失败返回 None。"""

    async with _group_notes_locks[session_id]:
        session_db = await SessionModel.get_or_none(id=session_id)
        if session_db is None:
            return None
        until = session_db.notes_summarized_until
        rows = (
            await _new_group_fragments(session_id, until)
            .order_by("created_at")
            .values(*_MEMORY_FIELDS)
        )
        if not _due(len(rows), until, min_new=min_new):
            return None

        chunks = _chunks(rows, GROUP_NOTES_CHUNK_CHARS)
        notes = await _condense(
            session_id,
            chunks,
            rewrite_prompt=lambda material: get_group_notes_prompt(
                session_db.group_notes, material
            ),
            points_prompt=get_group_points_prompt,
            key="group_notes",
        )
        calls = 1 if len(chunks) == 1 else len(chunks) + 1
        logger.info(
            f"[Digest] 群 {session_id} 群志：调用 {calls} 次，"
            f"{'失败，下次再试' if notes is None else f'{len(notes)} 字'}"
        )
        if notes is None or not still_current():
            return None
        await SessionModel.filter(id=session_id).update(
            group_notes=notes, notes_summarized_until=rows[-1]["created_at"]
        )
        return notes
