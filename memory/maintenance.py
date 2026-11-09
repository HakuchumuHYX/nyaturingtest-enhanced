"""长期记忆每日维护：03:30 把新碎片整理进档案、群志和人设成长，再删过期碎片。"""

from collections.abc import Callable

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from ..chat.groups import sessions  # noqa: E402
from ..persona.growth import maintain_personas  # noqa: E402
from ..persona.status import reload_diaries  # noqa: E402
from ..webconsole_store import now_ms, record_maintenance  # noqa: E402
from .digest import digest_group_notes, digest_user_profiles  # noqa: E402
from .threads import drop_forgotten  # noqa: E402
from .vector import maintain_memories  # noqa: E402


def _generation_guard(group_id: int) -> Callable[[], bool]:
    """记下此刻的会话和 generation，返回「群还启用着、会话没被 reset/load_preset 作废」的检查。"""

    session = sessions.get(group_id)
    if session is None:
        return lambda: False
    generation = session.state.generation
    return lambda: (
        sessions.get(group_id) is session and session.state.generation == generation
    )


async def refresh_group_notes(group_id: int, *, min_new: int = 1) -> tuple[str | None, bool]:
    """把新碎片整理进群志，返回（新群志, 整理是否失败）；群志不随 save_session 回写，这里同步内存副本。"""

    still_current = _generation_guard(group_id)
    notes, failed = await digest_group_notes(str(group_id), still_current, min_new=min_new)
    session = sessions.get(group_id)
    if notes is not None and session is not None and still_current():
        session.state.group_notes = notes
    return notes, failed


async def maintain_long_term_memory() -> None:
    """长期记忆每日维护：先把新碎片整理进档案与群志，再整理人设成长，最后删过期碎片和忘掉的悬着的事，
    这样即将过期的碎片一定先被归纳过；最后让所有群下次重新读矩阵。
    每一步的结果收进 detail，结束时写一行给控制台，有任何失败就是 partial。"""

    started_at_ms = now_ms()
    detail: dict = {
        "groups": {},
        "personas": {},
        "memories": None,
        "threads_forgotten": None,
        "errors": [],
    }
    for group_id in list(sessions):
        session_id = str(group_id)
        still_current = _generation_guard(group_id)
        result = detail["groups"][session_id] = {
            "profiles": 0,
            "profiles_failed": 0,
            "notes": False,
            "notes_failed": False,
        }
        try:
            summaries, result["profiles_failed"] = await digest_user_profiles(
                session_id, sessions[group_id].state.name, still_current
            )
            result["profiles"] = len(summaries)
            notes, result["notes_failed"] = await refresh_group_notes(group_id)
            result["notes"] = notes is not None
        except Exception as e:
            logger.warning(f"群 {group_id} 档案/群志整理失败: {e}")
            detail["errors"].append(f"群 {group_id} 档案/群志: {type(e).__name__}: {e}")
            continue
        # 档案不随 save_session 回写，要同步内存副本
        session = sessions.get(group_id)
        if session is not None and still_current():
            session.state.user_summaries.update(summaries)

    await _maintain_personas(detail)

    try:
        detail["memories"] = await maintain_memories()
    except Exception as e:
        logger.warning(f"长期记忆定时维护失败: {e}")
        detail["errors"].append(f"过期清理: {type(e).__name__}: {e}")
    try:
        detail["threads_forgotten"] = await drop_forgotten()
        logger.info(f"忘掉了 {detail['threads_forgotten']} 件悬着的事")
    except Exception as e:
        logger.warning(f"清理悬着的事失败: {e}")
        detail["errors"].append(f"悬着的事: {type(e).__name__}: {e}")
    for session in sessions.values():
        session.vector_memory.drop_cache()
    record_maintenance(started_at_ms, "partial" if _has_failure(detail) else "ok", detail)


def _has_failure(detail: dict) -> bool:
    return (
        bool(detail["errors"])
        or any(g["profiles_failed"] or g["notes_failed"] for g in detail["groups"].values())
        or any(
            p["life_failed"] or p["groups_blocked"] or p["diary_failed"]
            for p in detail["personas"].values()
        )
        or bool(detail["memories"] and detail["memories"]["pending"])
    )


async def _maintain_personas(detail: dict) -> None:
    group_ids = list(sessions)
    guards = {str(group_id): _generation_guard(group_id) for group_id in group_ids}
    try:
        lives, group_texts, detail["personas"] = await maintain_personas(group_ids, guards)
    except Exception as e:
        logger.warning(f"人设成长整理失败: {e}")
        detail["errors"].append(f"人设成长: {type(e).__name__}: {e}")
        return
    # 日记只由每日整理写，时间线那边的内存副本要同步
    await reload_diaries()
    # 人生线和群层不随 save_session 回写，要同步内存副本
    for group_id in group_ids:
        session = sessions.get(group_id)
        if session is None or not guards[str(group_id)]():
            continue
        state = session.state
        if state.preset in lives:
            state.self_life = lives[state.preset]
        if str(group_id) in group_texts:
            state.self_in_group = group_texts[str(group_id)]


def setup_maintenance_job():
    scheduler.add_job(
        maintain_long_term_memory,
        "cron",
        hour=3,
        minute=30,
        id="nyaturingtest_vector_maintenance",
        misfire_grace_time=3600,
        replace_existing=True,
    )
