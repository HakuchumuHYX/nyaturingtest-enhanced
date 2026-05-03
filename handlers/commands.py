# nyaturingtest/matchers.py
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

from ..config import (
    plugin_config,
    get_chat_thinking_settings,
    get_feedback_thinking_settings,
    get_runtime_settings,
    get_token_stats_model_names,
)
from ..core.state_manager import (
    ensure_group_state,
    remove_group_state,
    SELF_SENT_MSG_IDS,
    runtime_enabled_groups,
    group_states,
    is_shutting_down
)
from ..core.logic import message2BotMessage
from ..core.metrics import metrics
from ..core.structured_log import log_event
from ..memory.short_term import Message as MMessage
from ..database.enabled_group_repository import EnabledGroupRepository
from ..database.token_repository import TokenUsageRepository
from ..database.backup import backup_task
from .command_meta import render_group_help, render_private_help


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


# ==================== 命令定义 ====================

help_cmd = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="help", aliases={"帮助"}, priority=0, block=True
)
help_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="help", aliases={"帮助"}, priority=0, block=True
)

get_status = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="status", aliases={"状态"}, priority=0, block=True
)
get_status_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="status", aliases={"状态"}, priority=0, block=True
)

auto_chat = on_message(rule=is_group_message, priority=99, block=False)

set_role = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="set_role", aliases={"设置角色"}, priority=0, block=True
)
set_role_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="set_role", aliases={"设置角色"}, priority=0, block=True
)

get_role = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="role", aliases={"当前角色"}, priority=0, block=True
)
get_role_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="role", aliases={"当前角色"}, priority=0, block=True
)

calm_down = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="calm", aliases={"冷静"}, priority=0, block=True
)
calm_down_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="calm", aliases={"冷静"}, priority=0, block=True
)

reset_emotion = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="reset_emotion", aliases={"重置情绪"}, priority=0, block=True
)
reset_emotion_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="reset_emotion", aliases={"重置情绪"}, priority=0, block=True
)

reset = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="reset", aliases={"重置"}, priority=0, block=True
)
reset_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="reset", aliases={"重置"}, priority=0, block=True
)

get_presets = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="presets", aliases={"preset"}, priority=0, block=True
)
get_presets_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="presets", aliases={"preset"}, priority=0, block=True
)

set_presets = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="set_preset", aliases={"set_presets"}, priority=0, block=True
)
set_presets_pm = on_command(
    rule=is_private_message, permission=SUPERUSER, cmd="set_preset", aliases={"set_presets"}, priority=0, block=True
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

# ==================== 处理逻辑 ====================


@get_presets.handle()
async def handle_get_presets(event: GroupMessageEvent):
    await do_get_presets(get_presets, event.group_id)


@get_presets_pm.handle()
async def handle_get_presets_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await get_presets_pm.finish("请提供<qq群号>")
    group_id = await _parse_group_id_or_finish(get_presets_pm, arg)
    await do_get_presets(get_presets_pm, group_id)


async def do_get_presets(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        presets = state.session.presets()
    msg = "可选的预设:\n"
    for preset in presets:
        msg += f"- {preset}\n"
    msg += "使用方法: set_presets <预设名称>\n"
    await matcher.finish(msg)


@set_presets.handle()
async def handle_set_presets(event: GroupMessageEvent, args: Message = CommandArg()):
    file = args.extract_plain_text().strip()
    if file == "":
        await set_presets.finish("请提供<预设文件名>")
    await do_set_presets(set_presets, event.group_id, file)


@set_presets_pm.handle()
async def handle_set_presets_pm(args: Message = CommandArg()):
    preset_args = args.extract_plain_text().strip().split(" ", 1)
    if len(preset_args) != 2:
        await set_presets_pm.finish("请提供<qq群号> <预设文件名>")
    group_id = await _parse_group_id_or_finish(set_presets_pm, preset_args[0])
    await do_set_presets(set_presets_pm, group_id, preset_args[1])


async def do_set_presets(matcher: type[Matcher], group_id: int, file: str):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        if await state.session.load_preset(filename=file):
            await matcher.finish(f"预设已加载: {file}")
        else:
            await matcher.finish(f"不存在的预设: {file}")


@help_cmd.handle()
async def handle_help():
    await help_cmd.finish(render_group_help())


@help_pm.handle()
async def handle_help_pm():
    await help_pm.finish(render_private_help())


@set_role.handle()
async def handle_set_role(event: GroupMessageEvent, args: Message = CommandArg()):
    role_args = args.extract_plain_text().strip().split(" ", 1)
    if len(role_args) != 2:
        await set_role.finish("请提供<角色名> <角色设定>")
    await do_set_role(set_role, event.group_id, role_args[0], role_args[1])


@set_role_pm.handle()
async def handle_set_role_pm(args: Message = CommandArg()):
    role_args = args.extract_plain_text().strip().split(" ", 2)
    if len(role_args) != 3:
        await set_role_pm.finish("请提供<群号> <角色名> <角色设定>")
    group_id = await _parse_group_id_or_finish(set_role_pm, role_args[0])
    await do_set_role(set_role_pm, group_id, role_args[1], role_args[2])


async def do_set_role(matcher: type[Matcher], group_id: int, name: str, role: str):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        await state.session.set_role(name=name, role=role)
    await matcher.finish(f"角色已设为: {name}\n设定: {role}")


@get_role.handle()
async def handle_get_role(event: GroupMessageEvent):
    await do_get_role(get_role, event.group_id)


@get_role_pm.handle()
async def handle_get_role_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await get_role_pm.finish("请提供<群号>")
    group_id = await _parse_group_id_or_finish(get_role_pm, arg)
    await do_get_role(get_role_pm, group_id)


async def do_get_role(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        role = state.session.role()
    await matcher.finish(f"当前角色: {role}")


@calm_down.handle()
async def handle_calm_down(event: GroupMessageEvent):
    await do_calm_down(calm_down, event.group_id)


@calm_down_pm.handle()
async def handle_calm_down_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await calm_down_pm.finish("请提供<群号>")
    group_id = await _parse_group_id_or_finish(calm_down_pm, arg)
    await do_calm_down(calm_down_pm, group_id)


async def do_calm_down(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        await state.session.calm_down()
    await matcher.finish("已老实")


@reset_emotion.handle()
async def handle_reset_emotion(event: GroupMessageEvent):
    await do_reset_emotion(reset_emotion, event.group_id)


@reset_emotion_pm.handle()
async def handle_reset_emotion_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await reset_emotion_pm.finish("请提供<qq群号>")
    group_id = await _parse_group_id_or_finish(reset_emotion_pm, arg)
    await do_reset_emotion(reset_emotion_pm, group_id)


async def do_reset_emotion(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        await state.session.reset_emotion()
    await matcher.finish("情绪已初始化 (VAD -> 0, 0, 0)")


@reset.handle()
async def handle_reset(event: GroupMessageEvent, args: Message = CommandArg()):
    arg = args.extract_plain_text().strip().lower()
    if arg != "confirm":
        await reset.finish("危险操作：将清空本群会话、记忆和画像。确认执行请发送：reset confirm")
    await do_reset(reset, event.group_id)


@reset_pm.handle()
async def handle_reset_pm(args: Message = CommandArg()):
    parts = args.extract_plain_text().strip().split(" ", 1)
    if len(parts) != 2 or parts[1].lower() != "confirm":
        await reset_pm.finish("危险操作：请提供<群号> confirm")
    group_id = await _parse_group_id_or_finish(reset_pm, parts[0])
    await do_reset(reset_pm, group_id)


async def do_reset(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    await matcher.send("即将重置，会先执行一次数据备份...")
    await backup_task()
    async with state.session_lock:
        await state.session.load_session()
        await state.session.reset()
    await matcher.finish("已重置会话")


@get_status.handle()
async def handle_status(event: GroupMessageEvent):
    await do_status(get_status, event.group_id)


@get_status_pm.handle()
async def handle_status_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await get_status_pm.finish("请提供<群号>")
    group_id = await _parse_group_id_or_finish(get_status_pm, arg)
    await do_status(get_status_pm, group_id)


async def do_status(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    async with state.session_lock:
        await state.session.load_session()
        status_msg = state.session.status()
    chat_thinking = get_chat_thinking_settings()
    feedback_thinking = get_feedback_thinking_settings()
    chat_provider_status = getattr(state.client, "provider_status", None)
    feedback_provider_status = getattr(state.feedback_client, "provider_status", None)
    provider_lines = [
        "",
        "Provider:",
        f"- Chat thinking: {'on' if chat_thinking.get('enabled') else 'off'} {chat_thinking.get('reasoning_effort', '')}".strip(),
        f"- Feedback thinking: {'on' if feedback_thinking.get('enabled') else 'off'}",
        f"- Queue length: {len(state.messages_chunk)}",
        f"- Metrics: llm={metrics.llm_success}/{metrics.llm_failure}, vlm={metrics.vlm_success}/{metrics.vlm_failure}, db_write_failure={metrics.db_write_failure}",
    ]
    for name, provider_status in (("Chat", chat_provider_status), ("Feedback", feedback_provider_status)):
        if provider_status and provider_status.last_error_type:
            provider_lines.append(
                f"- {name} last_error={provider_status.last_error_type} circuit_remaining={provider_status.circuit_remaining_seconds}s"
            )
    status_msg += "\n".join(provider_lines)
    await matcher.finish(status_msg)


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
    await backup_task()
    await manual_backup_cmd.finish("备份完成！")


@manual_backup_pm.handle()
async def handle_manual_backup_pm():
    await manual_backup_pm.send("开始手动备份 NyaTuringTest 数据，请稍候...")
    await backup_task()
    await manual_backup_pm.finish("备份完成！")


@auto_chat.handle()
async def handle_auto_chat(bot: Bot, event: GroupMessageEvent):
    group_id = event.group_id
    state = ensure_group_state(group_id)
    if not state:
        return

    async with state.session_lock:
        await state.session.load_session()
        bot_name = state.session.name()

    # Shutdown 检查：避免在关机时进入耗时的 VLM 处理
    if is_shutting_down():
        return

    message_content = await message2BotMessage(
        bot_name=bot_name,
        group_id=group_id,
        message=event.original_message,
        bot=bot,
        resolve_images=plugin_config.get("vlm", {}).get("enabled", True),
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
        if is_shutting_down():
            return
        try:
            user_info = await bot.get_group_member_info(group_id=group_id, user_id=int(user_id))
            nickname = user_info.get("card") or user_info.get("nickname") or str(user_id)
        except Exception:
            nickname = str(user_id)

    async with state.data_lock:
        max_size = get_runtime_settings()["queue_max_size"]
        if len(state.messages_chunk) >= max_size:
            is_priority = _is_priority_message(event.original_message, str(bot.self_id), bot_name, message_content)
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
                user_id=user_id
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
async def handle_token_stats(bot: Bot, event: GroupMessageEvent):
    from ..config import plugin_config
    from ..utils import render_token_stats_card
    from nonebot.adapters.onebot.v11 import MessageSegment
    from nonebot.exception import FinishedException
    
    group_id = event.group_id
    
    stats = await TokenUsageRepository.get_token_stats(group_id, model_names=get_token_stats_model_names())
    
    # 获取水印配置
    watermark = plugin_config.get("token_stats", {}).get("watermark", "Generated by HakuBot")
    
    # 渲染图片
    try:
        img_bytes = await render_token_stats_card(
            stats=stats,
            watermark=watermark
        )
        
        # 发送图片消息
        await token_stats.finish(MessageSegment.image(img_bytes))
    except FinishedException:
        # FinishedException 是 NoneBot 的流程控制异常，必须重新抛出
        raise
    except Exception as e:
        logger.error(f"渲染 Token 统计图片失败: {e}")
        # 降级：发送文本消息
        text_msg = "Token 统计（当前模型）\n\n"
        text_msg += f"24h本群: {stats.get('1d_local', [])}\n"
        text_msg += f"24h全局: {stats.get('1d_global', [])}\n"
        await token_stats.finish(text_msg)
