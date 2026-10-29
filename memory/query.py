import time
from dataclasses import dataclass
from datetime import datetime

from nonebot import logger

from ..chat.groups import GroupState
from ..llm import chat_client
from ..prompts import describe_relation, get_impression_prompt
from ..storage.db import get_recent_messages_by_user
from .vector import (
    RAG_ITEM_CHARS,
    RAG_MEMORY_CHAR_BUDGET,
    RAG_MERGED_CANDIDATE_CAP,
)

MEMORY_QUERY_USER_COOLDOWN_SECONDS = 30.0
MEMORY_QUERY_GROUP_COOLDOWN_SECONDS = 3.0
IMPRESSION_TEMPERATURE = 0.8

_last_user_query: dict[tuple[str, str], float] = {}
_last_group_query: dict[str, float] = {}


def acquire_query_slot(group_id: str, user_id: str) -> float:
    """冷却中返回还需等待的秒数；否则记下本次查询并返回 0。"""

    now = time.monotonic()
    retry_after = max(
        MEMORY_QUERY_USER_COOLDOWN_SECONDS
        - (now - _last_user_query.get((group_id, user_id), float("-inf"))),
        MEMORY_QUERY_GROUP_COOLDOWN_SECONDS
        - (now - _last_group_query.get(group_id, float("-inf"))),
    )
    if retry_after > 0:
        return retry_after
    _last_user_query[(group_id, user_id)] = now
    _last_group_query[group_id] = now
    return 0.0


@dataclass
class Impression:
    """bot 手账里关于某个群友的一页，交给 cards/impression.py 渲染。"""

    target_id: str
    target_name: str
    bot_name: str
    headline: str
    impression: str
    memories: list[str]
    recent: str
    tags: list[str]
    affinity: int
    relation: str
    interactions: int
    first_interaction_at: datetime | None


def impression_text(page: Impression) -> str:
    """卡片渲染失败时的文字版。"""

    lines = [
        f"=== {page.target_name} 的印象档案 ===",
        "",
        f"「{page.headline}」",
        page.impression,
    ]
    if page.memories:
        lines += ["", "记得的事："] + [f"・{item}" for item in page.memories]
    if page.recent:
        lines += ["", f"最近：{page.recent}"]
    lines += [
        "",
        f"标签: {' '.join(page.tags) or '未知'}",
        "------------------",
        f"记忆深度: {page.interactions} | 好感度: {page.affinity}（{page.relation}）",
    ]
    return "\n".join(lines)


async def query_memory_profile(
    group: GroupState, *, target_id: str, target_name: str, sender_id: str
) -> Impression | str:
    """生成 Bot 对目标用户的印象（group 由调用方加载好）；没有印象或生成失败时返回要直接回复的文字。"""

    session = group.session
    profile = session.state.profiles.get(target_id)
    summary = session.state.user_summaries.get(target_id, "")
    bot_name = session.state.name
    self_life = session.state.self_life
    affinity = profile.level() if profile else 0

    if not bot_name:
        return "本群还没设人设，先 set_preset 吧。"

    relation = describe_relation(affinity)
    interactions = profile.interaction_count if profile else 0
    first_interaction_at = profile.first_interaction_at if profile else None

    target_records, unscoped_records = await _retrieve(
        session.vector_memory,
        target_id=target_id,
        target_name=target_name,
        interactions=interactions,
        first_interaction_at=first_interaction_at,
    )
    recent = await get_recent_messages_by_user(
        session.id, user_id=target_id, user_name=target_name, limit=10
    ) or ["(暂无最近发言记录)"]

    if interactions == 0 and not summary and not target_records and not unscoped_records:
        if target_id == sender_id:
            return "我对你还没有形成具体的印象呢，多和我聊聊天吧！"
        return f"我的记忆中暂时没有关于 {target_name} 的印象。"

    target_text = "\n".join(f"- {item}" for item in target_records)
    unscoped_text = "\n".join(f"- {item}" for item in unscoped_records)
    prompt = get_impression_prompt(
        bot_name=bot_name,
        self_life=self_life,
        target_name=target_name,
        affinity=affinity,
        relation=relation,
        interactions=interactions,
        summary=summary,
        target_text=target_text,
        unscoped_text=unscoped_text,
        recent=recent,
    )
    result = await chat_client.generate_json(
        prompt, session_id=session.id, temperature=IMPRESSION_TEMPERATURE
    )
    if not result or not result.get("headline") or not result.get("impression"):
        logger.warning("印象生成 JSON 解析失败")
        return "大脑处理过载，记忆读取失败，请稍后再试。"
    memories = [str(item).strip() for item in result.get("memories") or []]
    tags = [str(item).strip().lstrip("#") for item in result.get("tags") or []]
    return Impression(
        target_id=target_id,
        target_name=target_name,
        bot_name=bot_name,
        headline=str(result["headline"]).strip(),
        impression=str(result["impression"]).strip(),
        memories=[item for item in memories if item][:3],
        recent=str(result.get("recent") or "").strip(),
        tags=[item for item in tags if item][:5],
        affinity=affinity,
        relation=relation,
        interactions=interactions,
        first_interaction_at=first_interaction_at,
    )


async def _retrieve(
    vector_memory,
    *,
    target_id: str,
    target_name: str,
    interactions: int,
    first_interaction_at: datetime | None,
) -> tuple[list[str], list[str]]:
    """返回 (和目标有关的记忆, 不挂任何人的背景记忆)，按 RAG 字符预算截断。"""

    memory_count = await vector_memory.count_by_user(target_id)
    if first_interaction_at and first_interaction_at.tzinfo is not None:
        first_interaction_at = first_interaction_at.replace(tzinfo=None)
    days_since_first = (
        (datetime.now() - first_interaction_at).days if first_interaction_at else 0
    )
    result = await vector_memory.retrieve_with_decay(
        [
            f"关于{target_name}的记忆",
            f"我对{target_name}的看法",
            f"{target_name}做过的事",
            f"{target_name}的性格特点",
        ],
        k=calculate_dynamic_k(interactions, memory_count, days_since_first),
        user_ids={target_id, ""},
        use_rerank=True,
        merged_candidate_cap=RAG_MERGED_CANDIDATE_CAP,
        active_user_ids={target_id},
    )

    target_records, unscoped_records = [], []
    seen = set()
    remaining = RAG_MEMORY_CHAR_BUDGET
    for record in result.records:
        content = record.row.content
        if content in seen:
            continue
        seen.add(content)
        content = content[: min(RAG_ITEM_CHARS, remaining)]
        remaining -= len(content)
        if target_id in record.row.participant_ids.split():
            target_records.append(content)
        else:
            unscoped_records.append(content)
        if remaining <= 0:
            break
    return target_records, unscoped_records


def calculate_dynamic_k(
    interaction_count: int,
    memory_count: int,
    days_since_first: int,
) -> int:
    if memory_count <= 10:
        max_limit = memory_count
    elif memory_count <= 30:
        max_limit = 20
    elif memory_count <= 50:
        max_limit = 30
    else:
        max_limit = 40
    interaction_bonus = min(interaction_count // 50, 6)
    memory_bonus = min(memory_count // 10, 8)
    time_bonus = (
        4
        if days_since_first > 90
        else 3
        if days_since_first > 30
        else 2
        if days_since_first > 7
        else 0
    )
    return max(5, min(5 + interaction_bonus + memory_bonus + time_bonus, max_limit))
