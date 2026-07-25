import json
import time

from nonebot import logger, on_command
from nonebot.adapters.onebot.v11 import Bot, Event, GroupMessageEvent, Message
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER

from ..config import get_runtime_settings
from ..core.logic import llm_response
from ..core.memory_profile_query import (
    MemoryProfileQuery,
    MemoryProfileQueryService,
)
from ..core.memory_query_control import (
    MemoryQueryCoordinator,
    MemoryQueryCooldownError,
)
from ..core.metrics import metrics
from ..core.services import RagSearchService
from ..core.state_manager import ensure_group_state
from ..memory.vector import where_any


async def is_group_message(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


query_memory = on_command(
    "查询记忆",
    aliases={"memory"},
    rule=is_group_message,
    priority=5,
    block=True,
)
rag_debug = on_command(
    "rag_debug",
    aliases={"记忆诊断"},
    rule=is_group_message,
    permission=SUPERUSER,
    priority=0,
    block=True,
)

_runtime = get_runtime_settings()
_MEMORY_QUERY_COORDINATOR = MemoryQueryCoordinator[str](
    user_cooldown_seconds=_runtime.get(
        "memory_query_user_cooldown_seconds",
        30.0,
    ),
    group_cooldown_seconds=_runtime.get(
        "memory_query_group_cooldown_seconds",
        3.0,
    ),
)


def _format_rag_debug_score(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_rag_debug_record(index: int, record: dict) -> str:
    metadata = dict(record.get("metadata") or {})
    preview = str(record.get("preview") or "").replace("\n", " ")[:80]
    subject = metadata.get("subject_user_id") or metadata.get("user_id") or "-"
    return (
        f"{index}. ref={record.get('memory_ref') or '-'} "
        f"source={metadata.get('source') or '-'} "
        f"type={metadata.get('type') or '-'} "
        f"subtype={metadata.get('subtype') or '-'} "
        f"user_id={metadata.get('user_id') or '-'} "
        f"subject={subject} "
        f"speaker={metadata.get('speaker_user_id') or '-'} "
        f"scope={metadata.get('scope') or '-'} "
        f"score={_format_rag_debug_score(record.get('score'))} "
        f"adjusted_score={_format_rag_debug_score(metadata.get('adjusted_score'))} "
        f"retrieval_score={_format_rag_debug_score(metadata.get('retrieval_score'))} "
        f"rerank_score={_format_rag_debug_score(metadata.get('rerank_score'))}\n"
        f"   preview={preview}"
    )


@rag_debug.handle()
async def handle_rag_debug(
    event: GroupMessageEvent,
    args: Message = CommandArg(),
):
    query = args.extract_plain_text().strip()
    if not query:
        await rag_debug.finish("用法: rag_debug <query>")
        return
    state = ensure_group_state(event.group_id)
    if not state:
        await rag_debug.finish("本群尚未启用 AI 功能。")
        return

    runtime = get_runtime_settings()
    where_filter = where_any("source", ["preset", "memory"])
    async with state.session_lock:
        await state.session.load_session()
        memory = getattr(state.session, "long_term_memory", None)
    if memory is None:
        await rag_debug.finish("长期记忆库不可用。")
        return

    result = await RagSearchService(memory).search_for_debug(
        [query],
        k=runtime["rag_final_k"],
        where=where_filter,
        use_rerank=True,
        candidate_k=runtime["rag_per_query_recall_k"],
        merged_candidate_cap=runtime["rag_merged_candidate_cap"],
    )
    lines = [
        "RAG debug",
        f"query: {query}",
        f"where: {json.dumps(where_filter, ensure_ascii=False, sort_keys=True)}",
        f'candidate_count: {result.stats.get("candidate_count", 0)}',
        f'returned_count: {result.stats.get("returned_count", len(result.records))}',
        f'fallback_reason: {result.stats.get("fallback_reason") or "none"}',
        "score_fields: adjusted_score, retrieval_score, rerank_score",
        "top_records:",
    ]
    records = result.records[:5]
    if records:
        lines.extend(
            _format_rag_debug_record(index, record)
            for index, record in enumerate(records, start=1)
        )
    else:
        lines.append("(none)")
    await rag_debug.finish("\n".join(lines))


def _query_target_id(event: GroupMessageEvent, args: Message) -> str:
    for segment in args:
        if segment.type == "at":
            target_id = str(segment.data.get("qq", ""))
            if target_id:
                return target_id
    return str(event.user_id)


async def _target_display_name(
    bot: Bot,
    event: GroupMessageEvent,
    target_id: str,
) -> str:
    sender_id = str(event.user_id)
    if target_id == sender_id:
        return event.sender.card or event.sender.nickname or sender_id
    try:
        info = await bot.get_group_member_info(
            group_id=event.group_id,
            user_id=int(target_id),
        )
        return info.get("card") or info.get("nickname") or target_id
    except Exception:
        return target_id


@query_memory.handle()
async def handle_query_memory(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
):
    target_id = _query_target_id(event, args)
    target_name = await _target_display_name(bot, event, target_id)
    state = ensure_group_state(event.group_id)
    if not state:
        await query_memory.finish("本群尚未启用 AI 功能。")
        return

    memory = getattr(state.session, "long_term_memory", None)
    vector_version = int(getattr(memory, "version", 0) or 0)
    generation = int(getattr(state.session, "generation", 0) or 0)
    key = (str(event.group_id), target_id, vector_version, generation)
    started_at = time.perf_counter()
    metrics.memory_query_count += 1
    await query_memory.send("正在回溯记忆深处...")

    service = MemoryProfileQueryService(
        state=state,
        llm_response=llm_response,
    )
    try:
        message = await _MEMORY_QUERY_COORDINATOR.run(
            key=key,
            group_id=str(event.group_id),
            user_id=str(event.user_id),
            factory=lambda: service.execute(
                MemoryProfileQuery(
                    target_id=target_id,
                    target_name=target_name,
                    sender_id=str(event.user_id),
                )
            ),
        )
        metrics.memory_query_singleflight_reused = (
            _MEMORY_QUERY_COORDINATOR.stats.singleflight_reused
        )
    except MemoryQueryCooldownError as e:
        metrics.memory_query_cooldown_rejected += 1
        await query_memory.finish(
            f"记忆回溯正在冷却，请约 {max(1, int(e.retry_after + 0.5))} 秒后再试。"
        )
    except Exception as e:
        logger.error(f"查询记忆失败: {e}")
        await query_memory.finish("大脑处理过载，记忆读取失败，请稍后再试。")
    else:
        await query_memory.finish(message)
    finally:
        metrics.memory_query_total_ms += (
            time.perf_counter() - started_at
        ) * 1000
