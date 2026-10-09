"""群里安静久了，在合适的时段主动开个话题。

要不要发起全由代码判断（时间窗、安静时长、频率、随机），都满足后才调一次角色模型写内容，模型也可以放弃。
"""

import random
from dataclasses import asdict
from datetime import datetime, time, timedelta

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from .llm import build_turn_calls, extract_and_parse_json  # noqa: E402
from .logic import dispatch_replies  # noqa: E402
from .metrics import log_event  # noqa: E402
from .orchestrator import MY_RECENT_REPLIES_LIMIT, _format_history  # noqa: E402
from .prompts import get_initiative_prompt, get_time_description, is_rest_day  # noqa: E402
from .state_manager import (  # noqa: E402
    GroupState,
    group_states,
    is_shutting_down,
    runtime_enabled_groups,
)

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
# 写话题的素材：最近活跃群友的档案、最近写入的记忆碎片（群志每周才整理，这一两天的事只在碎片里）
MEMBER_PROFILE_LIMIT = 8
RECENT_MEMORY_LIMIT = 40

_attempts: dict[int, list[datetime]] = {}


def _in_window(now: datetime) -> bool:
    is_rest, _ = is_rest_day(now.date())
    windows = RESTDAY_WINDOWS if is_rest else WORKDAY_WINDOWS
    return any(start <= now.time() < end for start, end in windows)


def _silence_minutes(state: GroupState, now: datetime) -> int | None:
    """最后一条是群友的消息时返回安静了多少分钟；没有消息或最后一条是自己说的返回 None，
    这样发起后没人理就一直等到有人说话，不会对着空气连说两次。"""

    messages = state.session.runtime.short_term_memory.access()
    if not messages or messages[-1].user_name == state.session.state.name:
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
    for group_id in list(runtime_enabled_groups):
        state = group_states.get(group_id)
        # 本次启动后群里说过话才有 bot/event 可用来发送
        if (
            state is None
            or state.bot is None
            or state.event is None
            or not state.session.state.loaded
        ):
            continue
        silence = _silence_minutes(state, now)
        if silence is None or silence < SILENCE_MINUTES or not _quota_left(group_id, now):
            continue
        if random.random() >= TRIGGER_PROBABILITY:
            continue
        _attempts[group_id].append(now)
        try:
            await _initiate(state, silence)
        except Exception as e:
            logger.warning(f"群 {group_id} 主动发起话题失败: {e}")


async def _initiate(state: GroupState, silence: int) -> None:
    session = state.session
    async with state.session_lock:
        await session.load_session()
        generation = session.state.generation
    session_state = session.state

    messages = session.runtime.short_term_memory.access()
    names = {m.user_id: m.user_name for m in messages if m.user_id}
    # 最近说过话的群友，越近越靠前
    members = list(
        dict.fromkeys(
            m.user_id
            for m in reversed(messages)
            if m.user_id and m.user_name != session_state.name
        )
    )
    member_profiles = [
        {"name": names[uid], "summary": session_state.user_summaries[uid]}
        for uid in members
        if session_state.user_summaries.get(uid)
    ][:MEMBER_PROFILE_LIMIT]
    rows = await session.runtime.vector_memory.recent(RECENT_MEMORY_LIMIT)
    recent_memories = [
        f"【主体:{row.subject_user_name or '无'}|d:{row.date}】{row.content}"
        for row in reversed(rows)
    ]
    my_recent_replies = [
        m.content for m in messages if m.user_name == session_state.name
    ][-MY_RECENT_REPLIES_LIMIT:]

    prompt = get_initiative_prompt(
        bot_name=session_state.name,
        role=session_state.role.split("[对话样本]")[0].strip(),
        examples_text=session_state.examples,
        presets=session_state.preset_lines,
        group_notes=session_state.group_notes,
        summary=session_state.chat_summary,
        recent_msgs=_format_history(messages),
        my_recent_replies=my_recent_replies,
        member_profiles=member_profiles,
        recent_memories=recent_memories,
        emotion=asdict(session_state.global_emotion),
        silence_minutes=silence,
        time_info=get_time_description(datetime.now()),
    )
    chat_call, _ = build_turn_calls(str(session.id), [])
    parsed = extract_and_parse_json(await chat_call(prompt))
    content = str(parsed.get("content") or "").strip() if isinstance(parsed, dict) else ""

    if session.stale(generation, "initiative"):
        outcome = "stale"
    elif not content:
        outcome = "skipped_by_llm"
    elif (_silence_minutes(state, datetime.now()) or 0) < SILENCE_MINUTES:
        outcome = "spoken_meanwhile"
    else:
        sent = await dispatch_replies(
            state=state,
            responses=[{"content": content}],
            bot=state.bot,
            event=state.event,
            generation=generation,
        )
        outcome = "sent" if sent else "send_failed"
        if sent:
            # 最近活跃的人接话时按「聊天对象在接话」处理，不 @ 也接得上
            session_state.conversation_partners = set(members)

    log_event(
        "initiative",
        session_id=session.id,
        silence_minutes=silence,
        outcome=outcome,
        content=content[:60],
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
