# 由多个模块合并而来：handlers/command_meta.py, handlers/commands.py, handlers/memory.py

from dataclasses import dataclass
from datetime import datetime
from nonebot import on_command, on_message, logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    PrivateMessageEvent,
    Message,
    Event,
    MessageSegment
)
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.matcher import Matcher
from .config import (
    get_config_load_status,
    get_reasoning_effort,
    get_token_stats_model_names,
)
from .core.state_manager import (
    ensure_group_state,
    remove_group_state,
    SELF_SENT_MSG_IDS,
    runtime_enabled_groups,
    group_states,
    is_shutting_down
)
from .core.logic import QUEUE_MAX_SIZE, message2BotMessage
from .core.metrics import log_event, metrics
from .memory.short_term import Message as MMessage
from .db import EnabledGroupRepository
from .db import TokenUsageRepository
from .backup import backup_task
import json
import time
from nonebot import logger, on_command
from nonebot.adapters.onebot.v11 import Bot, Event, GroupMessageEvent, Message
from .core.logic import llm_response
from .core.memory_query import (
    MemoryProfileQuery,
    MemoryProfileQueryService,
    MemoryQueryCooldownError,
    MemoryQueryCoordinator,
)
from .core.metrics import metrics
from .memory.vector import RAG_FINAL_K, RAG_MERGED_CANDIDATE_CAP, RAG_PER_QUERY_RECALL_K
from .core.state_manager import ensure_group_state
from .memory.vector import search_memories, where_any


# ======== from handlers/command_meta.py ========
@dataclass(frozen=True)
class CommandMeta:
    command: str
    description: str
    private_usage: str = ""


COMMANDS: tuple[CommandMeta, ...] = (
    CommandMeta("autochat <enable/disable>", "在本群启用或禁用 Autochat"),
    CommandMeta("status", "查看 Bot 状态、provider 错误和基础 metrics", "status <群号>"),
    CommandMeta("role", "查看当前角色", "role <群号>"),
    CommandMeta("set_role <角色名> <角色设定>", "设置角色，设定可包含空格", "set_role <群号> <角色名> <角色设定>"),
    CommandMeta("presets", "查看可用预设", "presets <群号>"),
    CommandMeta("set_preset <文件名>", "加载预设", "set_preset <群号> <文件名>"),
    CommandMeta("rag_debug <query>", "诊断 RAG 记忆检索"),
    CommandMeta("calm", "冷静并重置短期状态", "calm <群号>"),
    CommandMeta("reset_emotion", "仅重置 VAD 情绪", "reset_emotion <群号>"),
    CommandMeta("reset confirm", "先备份再完全重置本群", "reset <群号> confirm"),
    CommandMeta("token统计", "查看全部模型 Token 与 DeepSeek cache 统计"),
    CommandMeta("backup_data", "手动触发数据备份", "backup_data"),
    CommandMeta("help", "显示帮助", "help"),
)


def render_group_help() -> str:
    lines = ["可用命令:"]
    for item in COMMANDS:
        if item.command == "help":
            continue
        lines.append(f"- {item.command} - {item.description}")
    return "\n".join(lines)


def render_private_help() -> str:
    lines = ["可用命令(私聊需加群号):"]
    for item in COMMANDS:
        usage = item.private_usage or item.command
        lines.append(f"- {usage} - {item.description}")
    return "\n".join(lines)

# ======== from handlers/commands.py ========
# nyaturingtest/matchers.py



# ==================== 辅助规则 ====================


async def is_group_message(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


async def is_private_message(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


def _is_priority_message(message: Message, bot_self_id: str, bot_name: str, rendered_text: str) -> bool:
    for seg in message:
        if seg.type == "at" and str(seg.data.get("qq", "")) == bot_self_id:
            return True
        if seg.type == "reply":
            return True
    return f"@{bot_name}" in rendered_text or bot_self_id in rendered_text


async def _parse_group_id_or_finish(matcher: type[Matcher], raw: str) -> int:
    raw_group_id = raw.strip()
    try:
        return int(raw_group_id)
    except ValueError:
        await matcher.finish("群号必须是数字")
        raise


def sender_display_name(event, user_id: str) -> str:
    card = str(event.sender.card or "").strip()
    nickname = str(event.sender.nickname or "").strip()
    return card or nickname or str(user_id)


async def reset_session_with_backup(state, backup) -> bool:
    """作废进行中的 turn，备份运行数据，然后重置该群会话。

    备份刻意放在 session_lock 之外：它可能长时间打包整个数据目录。
    """

    async with state.session_lock:
        await state.session.load_session()
        state.session.bump_generation("reset_requested")

    if not await backup():
        return False

    async with state.session_lock:
        await state.session.reset()
    return True


async def describe_status(state) -> str:
    """拼装会话与运行时诊断视图。"""

    async with state.session_lock:
        await state.session.load_session()
        status = state.session.status()
    lines = [
        "",
        "Provider:",
        f"- Chat reasoning_effort: {get_reasoning_effort('chat') or '未指定（由上游决定）'}",
        f"- Feedback reasoning_effort: {get_reasoning_effort('feedback') or '未指定（由上游决定）'}",
        f"- Queue length: {len(state.messages_chunk)}",
        f"- Metrics: llm={metrics.llm_success}/{metrics.llm_failure}",
    ]
    for name, client in (
        ("Chat", state.client),
        ("Feedback", state.feedback_client),
    ):
        provider_status = client.provider_status
        if provider_status.last_error_type:
            lines.append(
                f"- {name} last_error={provider_status.last_error_type} "
                "circuit_remaining="
                f"{provider_status.circuit_remaining_seconds}s"
            )
    config_status = get_config_load_status()
    if not config_status.ok or config_status.source != "file":
        lines.append(
            f"- Config: source={config_status.source} "
            f"ok={config_status.ok} error={config_status.error_type}"
        )
    return status + "\n".join(lines)


# ==================== 命令注册 ====================

help_cmd = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="help", aliases={"帮助"}, priority=0, block=True
)
help_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="help", aliases={"帮助"}, priority=0, block=True
)
list_groups_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="list_groups", aliases={"群组列表"}, priority=0, block=True
)
manual_backup_cmd = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="backup_data", aliases={"备份数据"}, priority=0, block=True
)
manual_backup_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="backup_data", aliases={"备份数据"}, priority=0, block=True
)
manage_cmd = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="autochat", priority=1, block=True
)
token_stats = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="token统计", aliases={"autochat token统计"}, priority=1, block=True
)
auto_chat = on_message(rule=is_group_message, priority=99, block=False)


def _dual_command(cmd: str, aliases: set[str], handler):
    """注册群聊与私聊两个 matcher，共用 handler(matcher, group_id, args_text)。

    群聊直接用 event.group_id；私聊先取第一个参数作为群号。
    """

    group = on_command(rule=is_group_message, permission=SUPERUSER, cmd=cmd, aliases=aliases, priority=0, block=True)
    private = on_command(rule=is_private_message, permission=SUPERUSER, cmd=cmd, aliases=aliases, priority=0, block=True)

    @group.handle()
    async def _group_entry(event: GroupMessageEvent, args: Message = CommandArg()):
        await handler(group, event.group_id, args.extract_plain_text().strip())

    @private.handle()
    async def _private_entry(args: Message = CommandArg()):
        parts = args.extract_plain_text().strip().split(" ", 1)
        if not parts[0]:
            await private.finish("请提供<群号>")
        group_id = await _parse_group_id_or_finish(private, parts[0])
        await handler(private, group_id, parts[1].strip() if len(parts) > 1 else "")

    return group, private


async def _group_state_or_finish(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    return state


# ==================== 处理逻辑 ====================


async def _do_get_presets(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        presets = state.session.presets()
    msg = "可选的预设:\n" + "".join(f"- {preset}\n" for preset in presets)
    await matcher.finish(msg + "使用方法: set_presets <预设名称>\n")


async def _do_set_presets(matcher: type[Matcher], group_id: int, args: str):
    if not args:
        await matcher.finish("请提供<预设文件名>")
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        loaded = await state.session.load_preset(filename=args)
    await matcher.finish(f"预设已加载: {args}" if loaded else f"不存在的预设: {args}")


async def _do_set_role(matcher: type[Matcher], group_id: int, args: str):
    parts = args.split(" ", 1)
    if len(parts) != 2:
        await matcher.finish("请提供<角色名> <角色设定>")
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        await state.session.set_role(name=parts[0], role=parts[1])
    await matcher.finish(f"角色已设为: {parts[0]}\n设定: {parts[1]}")


async def _do_get_role(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        role = state.session.role()
    await matcher.finish(f"当前角色: {role}")


async def _do_calm_down(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        await state.session.calm_down()
    await matcher.finish("已老实")


async def _do_reset_emotion(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        await state.session.reset_emotion()
    await matcher.finish("情绪已初始化 (VAD -> 0, 0, 0)")


async def _do_reset(matcher: type[Matcher], group_id: int, args: str):
    if args.lower() != "confirm":
        await matcher.finish("危险操作：将清空本群会话、记忆和画像。确认执行请发送：reset confirm")
    state = await _group_state_or_finish(matcher, group_id)
    await matcher.send("即将重置，会先执行一次数据备份...")
    if not await reset_session_with_backup(state, backup_task):
        await matcher.finish("备份失败，已中止重置；会话、记忆和画像均未清除。")
        return
    await matcher.finish("已重置会话")


async def _do_status(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    await matcher.finish(await describe_status(state))


get_presets, get_presets_pm = _dual_command("presets", {"preset"}, _do_get_presets)
set_presets, set_presets_pm = _dual_command("set_preset", {"set_presets"}, _do_set_presets)
get_role, get_role_pm = _dual_command("role", {"当前角色"}, _do_get_role)
set_role, set_role_pm = _dual_command("set_role", {"设置角色"}, _do_set_role)
calm_down, calm_down_pm = _dual_command("calm", {"冷静"}, _do_calm_down)
reset_emotion, reset_emotion_pm = _dual_command("reset_emotion", {"重置情绪"}, _do_reset_emotion)
reset, reset_pm = _dual_command("reset", {"重置"}, _do_reset)
get_status, get_status_pm = _dual_command("status", {"状态"}, _do_status)


@help_cmd.handle()
async def handle_help():
    await help_cmd.finish(render_group_help())


@help_pm.handle()
async def handle_help_pm():
    await help_pm.finish(render_private_help())


async def _do_manual_backup(matcher: type[Matcher]):
    await matcher.send("开始手动备份 NyaTuringTest 数据，请稍候...")
    if await backup_task():
        await matcher.finish("备份完成！")
    await matcher.finish("备份失败，请检查日志和数据目录。")


@manual_backup_cmd.handle()
async def handle_manual_backup():
    await _do_manual_backup(manual_backup_cmd)


@manual_backup_pm.handle()
async def handle_manual_backup_pm():
    await _do_manual_backup(manual_backup_pm)


@list_groups_pm.handle()
async def handle_list_groups_pm():
    allowed_groups = runtime_enabled_groups
    if not allowed_groups:
        await list_groups_pm.finish("没有启用的群组")
    msg = "启用的群组:\n"
    for group_id in allowed_groups:
        msg += f"- {group_id}\n"
    await list_groups_pm.finish(msg)


@manual_backup_cmd.handle()
async def handle_manual_backup():
    await manual_backup_cmd.send("开始手动备份 NyaTuringTest 数据，请稍候...")
    if await backup_task():
        await manual_backup_cmd.finish("备份完成！")
    await manual_backup_cmd.finish("备份失败，请检查日志和数据目录。")


@manual_backup_pm.handle()
async def handle_manual_backup_pm():
    await manual_backup_pm.send("开始手动备份 NyaTuringTest 数据，请稍候...")
    if await backup_task():
        await manual_backup_pm.finish("备份完成！")
    await manual_backup_pm.finish("备份失败，请检查日志和数据目录。")


@auto_chat.handle()
async def handle_auto_chat(bot: Bot, event: GroupMessageEvent):
    group_id = event.group_id
    state = ensure_group_state(group_id)
    if not state:
        return

    async with state.session_lock:
        await state.session.load_session()
        bot_name = state.session.name()
        recent_context_messages = state.session.runtime.short_term_memory.access_context(limit=4).messages
        conversation_context = "\n".join(
            f"{msg.user_name}: {msg.content}"
            for msg in recent_context_messages
        )[-600:]

    # Shutdown 检查：避免在关机时进入耗时的 VLM 处理
    if is_shutting_down():
        return

    raw_message_text = event.original_message.extract_plain_text()
    pre_queue_priority = _is_priority_message(
        event.original_message,
        str(bot.self_id),
        bot_name,
        raw_message_text,
    )
    async with state.data_lock:
        max_size = QUEUE_MAX_SIZE
        if len(state.messages_chunk) >= max_size and not pre_queue_priority:
            logger.warning(f"群 {group_id} 消息队列已满，转换前丢弃低优先级消息")
            log_event(
                "queue_drop",
                group_id=group_id,
                decision="drop_pre_conversion",
                queue_len=len(state.messages_chunk),
            )
            return

    message_content, image_inputs = await message2BotMessage(
        bot_name=bot_name,
        group_id=group_id,
        message=event.original_message,
        bot=bot,
        message_scope=str(event.message_id),
    )
    if not message_content:
        return

    user_id = str(event.user_id)
    msg_id = str(event.message_id)
    self_id = str(bot.self_id)
    nickname = ""

    if user_id == self_id:
        if msg_id in SELF_SENT_MSG_IDS:
            logger.debug(f"检测到自身回显 (Echo): {msg_id}")
        else:
            logger.debug(f"检测到非本机发送的自身消息 (可能是其他插件或端): {msg_id}")

        nickname = bot_name

    if not nickname:
        nickname = sender_display_name(event, user_id)

    async with state.data_lock:
        max_size = QUEUE_MAX_SIZE
        if len(state.messages_chunk) >= max_size:
            is_priority = pre_queue_priority or _is_priority_message(
                event.original_message,
                str(bot.self_id),
                bot_name,
                message_content,
            )
            if is_priority:
                state.messages_chunk.pop(0)
            else:
                logger.warning(f"群 {group_id} 消息队列已满，丢弃低优先级消息")
                log_event("queue_drop", group_id=group_id, decision="drop_low_priority", queue_len=len(state.messages_chunk))
                return
        state.event = event
        state.bot = bot
        state.messages_chunk.append(
            MMessage(
                time=datetime.now(),
                user_name=nickname,
                content=message_content,
                id=msg_id,
                user_id=user_id,
                image_inputs=image_inputs,
            )
        )
        state.new_message_signal.set()


@manage_cmd.handle()
async def handle_manage_autochat(event: GroupMessageEvent, args: Message = CommandArg()):
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id

    if arg == "enable":
        if group_id in runtime_enabled_groups:
            await manage_cmd.finish("本群 Autochat 已处于启用状态")

        await EnabledGroupRepository.enable_group(group_id)
        # 更新内存
        runtime_enabled_groups.add(group_id)
        # 立即初始化状态
        ensure_group_state(group_id)

        await manage_cmd.finish("Autochat 已在本群启用 (已保存至数据库)")

    elif arg == "disable":
        if group_id not in runtime_enabled_groups:
            await manage_cmd.finish("本群 Autochat 未启用")

        await EnabledGroupRepository.disable_group(group_id)
        # 更新内存
        runtime_enabled_groups.discard(group_id)

        # 安全清理运行时状态和后台任务
        await remove_group_state(group_id)

        await manage_cmd.finish("Autochat 已在本群禁用")

    else:
        await manage_cmd.finish("指令格式错误。请使用: autochat enable 或 autochat disable")


@token_stats.handle()
async def handle_token_stats(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    from .token_stats import render_token_stats_card
    from nonebot.adapters.onebot.v11 import MessageSegment
    from nonebot.exception import FinishedException
    
    group_id = event.group_id
    arg = args.extract_plain_text().strip().lower()
    token_stats_scope_all = arg in {"all", "全部", "历史", "history", "historical"}
    stats_model_names = get_token_stats_model_names()
    if token_stats_scope_all:
        stats_model_names = None
    stats = await TokenUsageRepository.get_token_stats(
        group_id,
        model_names=stats_model_names,
    )
    scope_label = "全部历史模型" if token_stats_scope_all else "当前模型"
    
    try:
        img_bytes = await render_token_stats_card(
            stats=stats,
            scope_label=scope_label,
        )
        
        # 发送图片消息
        await token_stats.finish(MessageSegment.image(img_bytes))
    except FinishedException:
        # FinishedException 是 NoneBot 的流程控制异常，必须重新抛出
        raise
    except Exception as e:
        logger.error(f"渲染 Token 统计图片失败: {e}")
        # 降级：发送文本消息
        text_msg = f"Token 统计（{scope_label}）\n\n"
        text_msg += f"24h本群: {stats.get('1d_local', [])}\n"
        text_msg += f"24h全局: {stats.get('1d_global', [])}\n"
        await token_stats.finish(text_msg)

# ======== from handlers/memory.py ========

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

MEMORY_QUERY_USER_COOLDOWN_SECONDS = 30.0
MEMORY_QUERY_GROUP_COOLDOWN_SECONDS = 3.0

_MEMORY_QUERY_COORDINATOR = MemoryQueryCoordinator[str](
    user_cooldown_seconds=MEMORY_QUERY_USER_COOLDOWN_SECONDS,
    group_cooldown_seconds=MEMORY_QUERY_GROUP_COOLDOWN_SECONDS,
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
    content = str(record.get("content") or "").replace("\n", " ")
    return (
        f"{index}. ref={metadata.get('memory_ref') or '-'} "
        f"source={metadata.get('source') or '-'} "
        f"type={metadata.get('type') or '-'} "
        f"subtype={metadata.get('subtype') or '-'} "
        f"subject={metadata.get('subject_user_id') or '-'} "
        f"speaker={metadata.get('speaker_user_id') or '-'} "
        f"scope={metadata.get('scope') or '-'} "
        f"adjusted_score={_format_rag_debug_score(metadata.get('adjusted_score'))} "
        f"retrieval_score={_format_rag_debug_score(metadata.get('retrieval_score'))} "
        f"rerank_score={_format_rag_debug_score(metadata.get('rerank_score'))}\n"
        f"   preview={content[:80]}"
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

    where_filter = where_any("source", ["preset", "memory"])
    async with state.session_lock:
        await state.session.load_session()
        memory = state.session.runtime.vector_memory
    if memory is None:
        await rag_debug.finish("长期记忆库不可用。")
        return

    result = await search_memories(memory, 
        [query],
        k=RAG_FINAL_K,
        where=where_filter,
        use_rerank=True,
        candidate_k=RAG_PER_QUERY_RECALL_K,
        merged_candidate_cap=RAG_MERGED_CANDIDATE_CAP,
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

    memory = state.session.runtime.vector_memory
    vector_version = int(memory.version or 0)
    generation = int(state.session.state.generation or 0)
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
