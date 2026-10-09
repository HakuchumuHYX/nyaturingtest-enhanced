"""分段整理：一段群聊聊完后，读整段原始消息，记成几条小结（episode）和少量长期事实。

以前碎片由 Feedback 每轮顺手抽，一轮只看约 8 条新消息，看不出哪件事以后还重要，
于是每轮都挑点什么记下来，一件事被拆成好几条流水账。现在 Feedback 只管更正，日常记忆在这里记。
"""

from datetime import datetime, timedelta

from nonebot import logger, require

require("nonebot_plugin_apscheduler")
from nonebot_plugin_apscheduler import scheduler  # noqa: E402  必须在 require 之后导入

from ..db import get_latest_user_names, message_final_id  # noqa: E402
from ..memory.short_term import Message  # noqa: E402
from ..models import GlobalMessageModel, MemoryModel, SessionModel  # noqa: E402
from .llm import extract_and_parse_json, feedback_client  # noqa: E402
from .metrics import log_event  # noqa: E402
from .orchestrator import save_memory_candidates  # noqa: E402
from .prompts import get_episode_prompt, truncate_text  # noqa: E402
from .state_manager import (  # noqa: E402
    GroupState,
    group_states,
    is_shutting_down,
    runtime_enabled_groups,
)

SCAN_INTERVAL_MINUTES = 5
# 最后一条消息距今这么久，算一段聊完了
SEGMENT_IDLE = timedelta(minutes=15)
# 群友消息不到这么多的小段先不整理、并进下一段：试跑里两三句的小段只写得出零碎插话
SEGMENT_MIN_MEMBER_MESSAGES = 5
# 一批最多这么多条：主群很少停，攒够就切，上一条小结会带给下一批接着写
SEGMENT_MAX_MESSAGES = 80
# 最早一条挂了这么久，不够一段也整理，免得一直悬着
SEGMENT_MAX_WAIT = timedelta(hours=6)
MESSAGE_CHARS = 300
MEMBER_PROFILE_LIMIT = 8
EPISODE_MIN_CHARS = 10
EPISODE_CONFIDENCE = 0.8
EPISODE_TEMPERATURE = 0.2


def _trigger(rows: list[GlobalMessageModel], bot_name: str, now: datetime) -> str | None:
    """这批待整理的消息该不该整理、因为什么；还不够一段返回 None。"""

    if not rows:
        return None
    if len(rows) >= SEGMENT_MAX_MESSAGES:
        return "full"
    members = sum(row.user_name != bot_name for row in rows)
    if now - rows[-1].time >= SEGMENT_IDLE and members >= SEGMENT_MIN_MEMBER_MESSAGES:
        return "idle"
    if now - rows[0].time >= SEGMENT_MAX_WAIT:
        return "stale"
    return None


def _episode_memories(
    items, messages: list[Message], bot_name: str, date: int
) -> list[tuple[str, dict]]:
    """把模型给出的小结整理成写库元数据；参与者和来源都按 source 指向的消息由代码算。"""

    memories = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        source = item.get("source")
        sources = [
            messages[index]
            for index in (source if isinstance(source, list) else [])
            if isinstance(index, int) and 0 <= index < len(messages)
        ]
        if len(content) < EPISODE_MIN_CHARS or not sources:
            continue
        try:
            importance = max(0.0, min(1.0, float(item.get("importance", 0.2))))
        except (TypeError, ValueError):
            importance = 0.2
        participants = (m.user_id for m in sources if m.user_id and m.user_name != bot_name)
        memories.append(
            (
                content,
                {
                    "category": "episode",
                    "date": date,
                    "subject_user_id": "",
                    "subject_user_name": "",
                    "speaker_user_id": "",
                    "speaker_user_name": "",
                    "confidence": EPISODE_CONFIDENCE,
                    "importance": importance,
                    "source_msg_ids": " ".join(
                        dict.fromkeys(message_final_id(m) for m in sources)
                    ),
                    "participant_ids": " ".join(dict.fromkeys(participants)),
                    "is_correction": False,
                    "replaces": "",
                },
            )
        )
    return memories


async def _digest_batch(
    state: GroupState, rows: list[GlobalMessageModel], trigger: str
) -> bool:
    """整理一批并推进水位；模型输出无效或会话已作废返回 False，水位不动，下轮重试。"""

    session = state.session
    session_state = session.state
    generation = session_state.generation
    bot_name = session_state.name
    messages = [
        Message(
            time=row.time,
            user_name=row.user_name,
            content=row.content,
            id=row.msg_id,
            user_id=row.user_id,
        )
        for row in rows
    ]
    members = [m for m in messages if m.user_name != bot_name]

    episodes: list[tuple[str, dict]] = []
    facts_result = {"added": 0, "rejected": 0}
    # 只有角色自己说话的一批没什么可记，直接推进水位
    if members:
        latest_names = await get_latest_user_names(session.id)
        speakers = dict.fromkeys(m.user_id for m in members if m.user_id)
        member_profiles = [
            {
                "user_id": uid,
                "name": latest_names.get(uid, ""),
                "summary": session_state.user_summaries[uid],
            }
            for uid in speakers
            if session_state.user_summaries.get(uid)
        ][:MEMBER_PROFILE_LIMIT]
        previous = (
            await MemoryModel.filter(session_id=session.id, category="episode")
            .order_by("-created_at")
            .first()
        )
        prompt = get_episode_prompt(
            bot_name=bot_name,
            group_notes=session_state.group_notes,
            member_profiles=member_profiles,
            previous_episode=previous.content if previous else "",
            messages=[
                {
                    "index": index,
                    "time": f"{m.time:%m-%d %H:%M}",
                    "user_id": m.user_id,
                    "name": m.user_name,
                    "content": truncate_text(m.content, MESSAGE_CHARS),
                }
                for index, m in enumerate(messages)
            ],
        )
        parsed = extract_and_parse_json(
            await feedback_client.generate(
                prompt, session_id=session.id, temperature=EPISODE_TEMPERATURE
            )
        )
        if not isinstance(parsed, dict):
            logger.warning(f"[Episode] 群 {session.id} 整理输出无效，下轮重试")
            return False

        facts = parsed.get("facts")
        if isinstance(facts, list) and facts:
            facts_result = await save_memory_candidates(
                session,
                facts,
                messages=messages,
                known_users=latest_names,
                refs={},
                generation=generation,
            )
            if facts_result is None:
                return False
        date = int(rows[-1].time.strftime("%Y%m%d"))
        episodes = _episode_memories(parsed.get("episodes"), messages, bot_name, date)
        if episodes:
            stored = await session.runtime.vector_memory.add_memories_with_dedup(
                episodes,
                still_current=lambda: session_state.generation == generation,
            )
            if stored is None:
                return False

    if session.stale(generation, "episode_watermark"):
        return False
    await SessionModel.filter(id=session.id).update(episodes_until=rows[-1].time)
    log_event(
        "episode_digest",
        session_id=session.id,
        trigger=trigger,
        messages=len(rows),
        episodes=len(episodes),
        facts_added=facts_result["added"],
        facts_rejected=facts_result["rejected"],
    )
    return True


async def _digest_pending(state: GroupState) -> None:
    """水位之后的消息按批整理，直到剩下的还不够一段。"""

    session_id = state.session.id
    while not is_shutting_down():
        until = (await SessionModel.get(id=session_id)).episodes_until
        query = GlobalMessageModel.filter(session_id=session_id)
        if until is not None:
            query = query.filter(time__gt=until)
        rows = await query.order_by("time").limit(SEGMENT_MAX_MESSAGES)
        trigger = _trigger(rows, state.session.state.name, datetime.now())
        if trigger is None or not await _digest_batch(state, rows, trigger):
            return


async def scan_groups() -> None:
    for group_id in list(runtime_enabled_groups):
        state = group_states.get(group_id)
        # 用已加载的 vector_memory 写入，矩阵缓存才一致；没加载说明重启后还没人说话，也就没有新消息
        if is_shutting_down() or state is None or not state.session.state.loaded:
            continue
        try:
            await _digest_pending(state)
        except Exception as e:
            logger.warning(f"群 {group_id} 分段整理失败: {e}")


def setup_episode_job() -> None:
    scheduler.add_job(
        scan_groups,
        "interval",
        minutes=SCAN_INTERVAL_MINUTES,
        id="nyaturingtest_episodes",
        replace_existing=True,
    )
    logger.info(f"已注册分段整理: 每 {SCAN_INTERVAL_MINUTES} 分钟")
