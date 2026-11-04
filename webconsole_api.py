"""给 WebConsole 拉当前状态的只读接口，挂在 NoneBot 自带的 FastAPI 上。

历史不在这里，见 webconsole_store.py。这里必须只读：心情和好感度的 settle() 会改状态，
都要在副本上算，否则看控制台会影响她。这组路径 nginx 不转发，但里面有聊天内容，同机其他进程也连得上，所以要 token。
"""

import secrets
import time
from dataclasses import replace
from datetime import date, datetime, timedelta

import nonebot
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from nonebot import logger

from .chat.engagement import unread_count
from .chat.groups import sessions
from .chat.session import Session
from .chat.status import get_status
from .config import get_app_settings, get_token_stats_model_names
from .llm import background_client, chat_client, feedback_client
from .memory.threads import KINDS
from .prompts import describe_mood
from .storage.db import (
    get_latest_user_names,
    load_digest_day,
    load_life,
    message_buckets,
    self_speak_times,
    token_usage_by_model,
    token_usage_by_session,
)
from .storage.models import PersonaGroupModel, SessionModel, ThreadModel

ROUTE_PREFIX = "/nyaturingtest/api"
GROUP_NAMES_TTL_SECONDS = 600
TOKEN_WINDOWS = (24, 168, 720)
DAY_BUCKETS = 48
AFFINITY_TOP = 5
DIARY_SHOWN = 7

_started_at_ms = int(time.time() * 1000)
_group_names: dict[str, str] = {}
_group_names_at = 0.0


def _ms(value: datetime | None) -> int | None:
    # datetime.min 是「还没有过」，timestamp() 算不了它
    if value is None or value == datetime.min:
        return None
    return int(value.timestamp() * 1000)


def _day(value: date | None) -> str | None:
    return value.isoformat() if value else None


async def _refresh_group_names() -> None:
    """一次取全部群名。Session.group_info 是拼好的文本，重启后还要等第一次看群才有，所以不用它。"""

    global _group_names, _group_names_at
    bots = list(nonebot.get_bots().values())
    if not bots or time.monotonic() - _group_names_at < GROUP_NAMES_TTL_SECONDS:
        return
    _group_names_at = time.monotonic()
    try:
        groups = await bots[0].get_group_list()
    except Exception as e:
        logger.warning(f"拉取群名失败，先用旧的: {type(e).__name__}: {e}")
        return
    _group_names = {str(group["group_id"]): group["group_name"] for group in groups}


def _llm_status() -> list[dict]:
    return [
        {
            "name": client.name,
            "model": client.settings.model,
            "last_error_type": client.provider_status.last_error_type,
            "last_error_message": client.provider_status.last_error_message,
            "circuit_remaining_seconds": client.provider_status.circuit_remaining_seconds,
        }
        for client in (chat_client, feedback_client, background_client)
    ]


def _presets() -> dict[str, str]:
    """现在启用的预设：文件名 -> 角色名。"""

    return {
        session.state.preset: session.state.name
        for session in sessions.values()
        if session.state.name
    }


def _persona_now(preset: str, name: str) -> dict:
    status = get_status(preset)
    current = status.current if status else None
    if current is None:
        return {"preset": preset, "name": name, "mode": None}
    nights = status.nights
    return {
        "preset": preset,
        "name": name,
        "mode": current["mode"],
        "doing": current["doing"],
        "sleep": bool(current.get("sleep")),
        "source": current["source"],
        "since_ms": _ms(current["start"]),
        "until_ms": _ms(status.until),
        "calls": len(status.calls),
        "last_night": {"sleep_ms": _ms(nights[-1]["sleep"]), "wake_ms": _ms(nights[-1].get("wake"))}
        if nights
        else None,
    }


def _group_card(group_id: int, session: Session, now: datetime) -> dict:
    state = session.state
    emotion = replace(state.global_emotion)
    emotion.settle(now)
    messages = session.short_term_memory.access()
    return {
        "id": str(group_id),
        "preset": state.preset,
        "name": state.name,
        "worker_alive": session.worker is not None and not session.worker.done(),
        "queue": len(session.messages_chunk),
        "background_tasks": len(session.background_tasks),
        "unread": unread_count(state, session.unread()),
        "last_look_ms": _ms(state.last_look_time),
        "last_speak_ms": _ms(state.last_speak_time),
        "last_message_ms": _ms(messages[-1].time) if messages else None,
        "muted_until_ms": _ms(session.muted_until) if session.muted else None,
        "emotion": emotion.vad(),
        "mood": describe_mood(emotion),
        "summary": state.chat_summary,
    }


def _timeline(preset: str, name: str) -> dict:
    status = get_status(preset)
    if status is None:
        return {"preset": preset, "name": name, "timeline": [], "nights": [], "diary": []}
    return {
        "preset": preset,
        "name": name,
        "timeline": [
            {
                "start_ms": _ms(segment["start"]),
                "end_ms": _ms(
                    status.timeline[index + 1]["start"]
                    if index + 1 < len(status.timeline)
                    else status.until
                ),
                "mode": segment["mode"],
                "doing": segment["doing"],
                "sleep": bool(segment.get("sleep")),
                "source": segment["source"],
            }
            for index, segment in enumerate(status.timeline)
        ],
        "nights": [
            {"sleep_ms": _ms(night["sleep"]), "wake_ms": _ms(night.get("wake"))}
            for night in status.nights
        ],
        "diary": status.diary[-DIARY_SHOWN:],
        "diary_until": _day(status.diary_until),
    }


async def _overview() -> dict:
    now = datetime.now()
    return {
        "now_ms": _ms(now),
        "started_at_ms": _started_at_ms,
        "bots": list(nonebot.get_bots()),
        "llm": _llm_status(),
        "personas": [_persona_now(preset, name) for preset, name in _presets().items()],
        "groups": [_group_card(group_id, session, now) for group_id, session in sessions.items()],
    }


async def _activity() -> dict:
    await _refresh_group_names()
    now = datetime.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    groups: dict[str, dict] = {
        str(group_id): {
            "id": str(group_id),
            "name": _group_names.get(str(group_id)),
            "buckets": [0] * DAY_BUCKETS,
            "messages": 0,
            "speaks": [
                _ms(moment)
                for moment in (
                    await self_speak_times(str(group_id), session.state.name, day_start)
                    if session.state.name
                    else []
                )
            ],
        }
        for group_id, session in sessions.items()
    }
    for row in await message_buckets(day_start):
        group = groups.get(row["session_id"])
        if group is not None:
            group["buckets"][row["bucket"]] = row["count"]
            group["messages"] += row["count"]
    return {
        "now_ms": _ms(now),
        "day_start_ms": _ms(day_start),
        "personas": [
            {**_timeline(preset, name), "life": await load_life(preset)}
            for preset, name in _presets().items()
        ],
        "groups": list(groups.values()),
        "tokens": await token_usage_by_session(now - timedelta(hours=24)),
    }


def _enabled_session(group_id: int) -> Session:
    session = sessions.get(group_id)
    if session is None:
        raise HTTPException(404, "这个群没有启用 autochat")
    return session


def _affinity_top(session: Session, names: dict[str, str], now: datetime) -> list[dict]:
    profiles = []
    for profile in session.state.profiles.values():
        settled = replace(profile)
        settled.settle(now)
        profiles.append(settled)
    profiles.sort(key=lambda profile: profile.affinity, reverse=True)
    return [
        {
            "user_id": profile.user_id,
            "name": names.get(profile.user_id, profile.user_id),
            "affinity": round(profile.affinity),
            "interactions": profile.interaction_count,
            "last_ms": _ms(profile.last_interaction_at),
        }
        for profile in profiles[:AFFINITY_TOP]
    ]


async def _group_detail(group_id: int) -> dict:
    session = _enabled_session(group_id)
    state = session.state
    session_id = session.id
    now = datetime.now()
    # 刚启用的群第一次保存前还没有这一行
    session_db = await SessionModel.get_or_none(id=session_id)
    layer = await PersonaGroupModel.get_or_none(session_id=session_id, preset=state.preset)
    threads = await ThreadModel.filter(session_id=session_id, forget_at__gt=now).order_by("ask_from")
    names = await get_latest_user_names(session_id)
    return {
        "id": session_id,
        "group_name": _group_names.get(session_id),
        "summary": state.chat_summary,
        "messages": [
            {
                "time_ms": _ms(message.time),
                "user_name": message.user_name,
                "content": message.content,
                "self": message.user_name == state.name,
            }
            for message in session.short_term_memory.access()
        ],
        "threads": [
            {
                "kind": KINDS[thread.kind],
                "who": thread.user_name,
                "content": thread.content,
                "due": thread.due,
                "ask_from_ms": _ms(thread.ask_from),
                "forget_at_ms": _ms(thread.forget_at),
            }
            for thread in threads
        ],
        "affinity": _affinity_top(session, names, now),
        "persona_group": layer.text if layer else "",
        "journal": layer.journal if layer else [],
        "digested_until": _day(layer.digested_until) if layer else None,
        "notes": session_db.group_notes if session_db else "",
        "notes_until_ms": _ms(session_db.notes_summarized_until) if session_db else None,
        "episodes_until_ms": _ms(session_db.episodes_until) if session_db else None,
    }


def setup_console_api() -> None:
    """启动时才挂：其他入口只是 import 这个包，不走启动流程，不该多出接口。"""

    token = get_app_settings().console_token
    if not token:
        return
    expected = f"Bearer {token}".encode()

    async def authorize(authorization: str = Header("")):
        if not secrets.compare_digest(authorization.encode(), expected):
            raise HTTPException(401)

    router = APIRouter(prefix=ROUTE_PREFIX, dependencies=[Depends(authorize)])

    @router.get("/overview")
    async def overview():
        return await _overview()

    @router.get("/activity")
    async def activity():
        return await _activity()

    @router.get("/groups/{group_id}")
    async def group_detail(group_id: int):
        return await _group_detail(group_id)

    @router.get("/groups/{group_id}/tokens")
    async def group_tokens(group_id: int, hours: int = Query(24)):
        if hours not in TOKEN_WINDOWS:
            raise HTTPException(400, f"hours 只能是 {TOKEN_WINDOWS}")
        since = datetime.now() - timedelta(hours=hours)
        return {
            "hours": hours,
            "models": await token_usage_by_model(str(group_id), since, get_token_stats_model_names()),
        }

    @router.get("/digest")
    async def digest(day: date):
        return await load_digest_day(day)

    nonebot.get_app().include_router(router)
