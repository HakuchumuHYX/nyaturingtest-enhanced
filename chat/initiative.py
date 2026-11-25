"""群里安静久了，在合适的时段主动开个话题。

要不要发起全由代码判断（时间窗、安静时长、频率、随机），都满足后才调一次角色模型写内容，模型也可以放弃。
她 away（手机不在手上）或被禁言时不发起。
"""

import random
from datetime import datetime, time, timedelta

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from ..llm import chat_client  # noqa: E402
from ..memory.threads import close_threads, open_threads, thread_items  # noqa: E402
from ..messages.photos import enabled as photos_enabled  # noqa: E402
from ..metrics import log_event  # noqa: E402
from ..persona.status import (  # noqa: E402
    current_weather,
    describe_now,
    is_away,
    recent_diary,
    recent_feed,
    recent_plan,
    today_lines,
)
from ..prompts import (  # noqa: E402
    CHAT_SYSTEM_PROMPT,
    describe_mood,
    describe_relation,
    get_initiative_prompt,
    get_time_description,
    is_rest_day,
)
from .domain import PersonProfile  # noqa: E402
from .groups import is_shutting_down, sessions  # noqa: E402
from .orchestrator import CHAT_DIARY_DAYS, MY_RECENT_REPLIES_LIMIT, format_history  # noqa: E402
from .outputs import parse_reply_item  # noqa: E402
from .send import dispatch_replies  # noqa: E402
from .session import Session  # noqa: E402

SCAN_INTERVAL_MINUTES = 10
# 群里最后一条消息距今至少这么久才算冷场
SILENCE_MINUTES = 120
# 每群每天最多尝试几次、两次之间至少隔多久；模型放弃也算一次，免得冷场期间反复调模型。只记在内存，重启清零
DAILY_LIMIT = 2
MIN_GAP = timedelta(hours=4)
# 条件都满足后每次扫描的触发概率：避免一到点就开口，显得机械
TRIGGER_PROBABILITY = 0.3
# 允许发起的时段：工作日只在午休和下班后，休息日白天到晚上；凌晨一律不发
WORKDAY_WINDOWS = ((time(12, 0), time(13, 30)), (time(18, 30), time(23, 0)))
RESTDAY_WINDOWS = ((time(10, 0), time(23, 0)),)
# 写话题的素材：最近活跃群友的档案、最近写入的记忆碎片（群志每晚才整理，这一两天的事只在碎片里）
MEMBER_PROFILE_LIMIT = 8
RECENT_MEMORY_LIMIT = 40

_attempts: dict[int, list[datetime]] = {}


def _in_window(now: datetime) -> bool:
    is_rest, _ = is_rest_day(now.date())
    windows = RESTDAY_WINDOWS if is_rest else WORKDAY_WINDOWS
    return any(start <= now.time() < end for start, end in windows)


def _silence_minutes(session: Session, now: datetime) -> int | None:
    """最后一条是群友的消息时返回安静了多少分钟；没有消息或最后一条是自己说的返回 None，
    这样发起后没人理就一直等到有人说话，不会对着空气连说两次。"""

    messages = session.short_term_memory.access()
    if not messages or messages[-1].user_name == session.state.name:
        return None
    return int((now - messages[-1].time).total_seconds() // 60)


def _quota_left(group_id: int, now: datetime) -> bool:
    today = [t for t in _attempts.get(group_id, []) if t.date() == now.date()]
    _attempts[group_id] = today
    return len(today) < DAILY_LIMIT and (not today or now - today[-1] >= MIN_GAP)


async def scan_groups() -> None:
    now = datetime.now()
    if is_shutting_down() or not _in_window(now):
        return
    for group_id, session in list(sessions.items()):
        # 本次启动后群里说过话才有 bot/event 可用来发送；循环途中被停用的群不再发言
        if (
            sessions.get(group_id) is not session
            or session.bot is None
            or session.event is None
            or not session.state.name  # 没设人设不说话
            or is_away(session.state.preset)
            or session.muted
        ):
            continue
        silence = _silence_minutes(session, now)
        if silence is None or silence < SILENCE_MINUTES or not _quota_left(group_id, now):
            continue
        if random.random() >= TRIGGER_PROBABILITY:
            continue
        _attempts[group_id].append(now)
        try:
            await _initiate(session, silence)
        except Exception as e:
            logger.opt(exception=e).warning(f"群 {group_id} 主动发起话题失败")


async def _initiate(session: Session, silence: int) -> None:
    state = session.state
    generation = state.generation
    now = datetime.now()
    today_date = now.date()

    messages = session.short_term_memory.access()
    threads = await open_threads(session.id, None, now)
    # 近期群名片优先，悬着的事里的当事人补齐（久未冒泡时不在短期记忆里）
    names = {row.user_id: row.user_name for row in threads if row.user_id and row.user_name}
    names.update({m.user_id: m.user_name for m in messages if m.user_id})
    # 最近说过话的群友，越近越靠前
    members = list(
        dict.fromkeys(
            m.user_id
            for m in reversed(messages)
            if m.user_id and m.user_name != state.name
        )
    )
    thread_uids = list(dict.fromkeys(row.user_id for row in threads if row.user_id))
    thread_uid_set = set(thread_uids)
    candidate_uids = list(dict.fromkeys([*thread_uids, *members]))
    member_profiles = []
    for uid in candidate_uids:
        if uid not in names:
            continue
        summary = state.user_summaries.get(uid, "")
        if uid not in thread_uid_set and not summary:
            continue
        profile = state.profiles.get(uid, PersonProfile(user_id=uid))
        last = profile.last_interaction_at
        member_profiles.append(
            {
                "name": names[uid],
                "relation": describe_relation(profile.level()),
                "summary": summary,
                "last_seen_days": (today_date - last.date()).days if last else None,
            }
        )
        if len(member_profiles) >= max(MEMBER_PROFILE_LIMIT, len(thread_uid_set)):
            break

    rows = await session.vector_memory.recent(RECENT_MEMORY_LIMIT)
    recent_memories = [
        f"【主体:{row.subject_user_name or '无'}|d:{row.date}】{row.content}"
        for row in reversed(rows)
    ]
    my_recent_replies = [
        m.content for m in messages if m.user_name == state.name and not m.is_action
    ][-MY_RECENT_REPLIES_LIMIT:]

    state.global_emotion.settle(now)
    photos_on = photos_enabled()
    prompt = get_initiative_prompt(
        bot_name=state.name,
        profile=state.profile,
        appearance=state.appearance,
        bottom_lines=state.bottom_lines,
        self_life=state.self_life,
        self_in_group=state.self_in_group,
        presets=state.preset_lines,
        group_notes=state.group_notes,
        summary=state.chat_summary,
        recent_msgs=format_history(messages),
        my_recent_replies=my_recent_replies,
        member_profiles=member_profiles,
        recent_memories=recent_memories,
        threads=thread_items(threads),
        mood=describe_mood(state.global_emotion),
        silence_minutes=silence,
        time_info=get_time_description(now),
        diary=recent_diary(state.preset, CHAT_DIARY_DAYS),
        plan=recent_plan(state.preset),
        group_info=await session.refresh_group_info(),
        today=today_lines(state.preset),
        now=describe_now(state.preset, max(state.last_look_time, state.last_read_time or datetime.min)),
        weather=await current_weather(state.preset),
        feed=recent_feed(state.preset),
        photos=photos_on,
        photos_last_hour=session.photos_last_hour(),
        photo_ready=now >= session.photo_ready_at,
    )
    parsed = await chat_client.generate_json(
        prompt,
        session_id=session.id,
        system_prompt=CHAT_SYSTEM_PROMPT,
    )
    reply = parse_reply_item(parsed, [], allow_photo=photos_on)
    closed = 0

    if session.stale(generation, "initiative"):
        outcome = "stale"
    elif reply is None:
        outcome = "skipped_by_llm"
    elif (_silence_minutes(session, datetime.now()) or 0) < SILENCE_MINUTES:
        outcome = "spoken_meanwhile"
    else:
        sent = await dispatch_replies(
            session=session,
            replies=[reply],
            bot=session.bot,
            event=session.event,
            generation=generation,
        )
        outcome = "sent" if sent else "send_failed"
        if sent:
            closed_refs = parsed.get("threads") if isinstance(parsed, dict) else None
            closed = await close_threads(threads, closed_refs)
            if closed:
                log_event("threads_closed", session_id=session.id, by="initiative", count=closed)
            ref_to_uid = {
                f"t{index + 1}": row.user_id
                for index, row in enumerate(threads)
                if row.user_id
            }
            thread_partners = {
                ref_to_uid[ref]
                for ref in (closed_refs if isinstance(closed_refs, list) else [])
                if isinstance(ref, str) and ref in ref_to_uid
            }
            # 最近活跃的人和本次关心到的当事人接话时按「聊天对象在接话」处理，不 @ 也接得上
            state.conversation_partners = set(members) | thread_partners

    log_event(
        "initiative",
        session_id=session.id,
        silence_minutes=silence,
        outcome=outcome,
        content=(reply.text[:60] if reply else ""),
        has_sticker=bool(reply and reply.sticker),
        has_photo=bool(reply and reply.photo),
        threads_closed=closed,
    )


def setup_initiative_job() -> None:
    scheduler.add_job(
        scan_groups,
        "interval",
        minutes=SCAN_INTERVAL_MINUTES,
        id="nyaturingtest_initiative",
        replace_existing=True,
    )
    logger.info(f"已注册主动发起话题检查: 每 {SCAN_INTERVAL_MINUTES} 分钟")
