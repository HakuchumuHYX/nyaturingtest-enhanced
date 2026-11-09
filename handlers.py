import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from nonebot import get_bot, logger, on, on_command, on_message, on_notice
from nonebot.adapters.onebot.v11 import (
    Adapter,
    Bot,
    Event,
    GroupBanNoticeEvent,
    GroupDecreaseNoticeEvent,
    GroupIncreaseNoticeEvent,
    GroupMessageEvent,
    GroupRecallNoticeEvent,
    Message,
    MessageSegment,
    NoticeEvent,
    PokeNotifyEvent,
    PrivateMessageEvent,
)
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.exception import NetworkError
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from tortoise.expressions import Q

from .cards.impression import render_impression_card
from .cards.notes import render_group_notes_card
from .cards.persona import Change as PersonaChange
from .cards.persona import render_persona_card
from .cards.token_stats import render_token_stats_card
from .chat.engagement import unread_count
from .chat.groups import (
    disable_autochat,
    enable_autochat,
    get_session,
    is_shutting_down,
    sessions,
)
from .chat.session import Session
from .config import get_app_settings, get_token_stats_model_names
from .llm import background_client, chat_client, feedback_client
from .memory.digest import GROUP_NOTES_REFRESH_MIN_NEW, count_new_group_fragments
from .memory.maintenance import refresh_group_notes
from .memory.query import acquire_query_slot, impression_text, query_memory_profile
from .memory.vector import RAG_FINAL_K, RetrievedMemory
from .messages.convert import (
    forward_id,
    member_name,
    read_group_notice,
    read_message,
    read_poke,
    read_reaction,
    recall_text,
)
from .messages.forward import digest_forward
from .messages.image import fetch_image_input
from .messages.photos import remove_by_message as remove_photo_by_message
from .messages.stickers import block as block_sticker
from .metrics import log_event
from .persona.status import recent_diary, redecide, status_text
from .prompts import PRESETS, reload_presets, resolve_preset
from .storage.backup import backup_task
from .storage.db import get_token_stats, message_content, recall_message
from .storage.models import PersonaHistoryModel, SessionModel

# persona 命令列出的最近改动条数（人生线和本群合计）、最近几天的日记
PERSONA_HISTORY_LIMIT = 4
PERSONA_DIARY_DAYS = 3


@dataclass(frozen=True)
class CommandMeta:
    command: str
    description: str
    private_usage: str = ""


COMMANDS: tuple[CommandMeta, ...] = (
    CommandMeta("autochat <enable/disable>", "在本群启用或禁用 Autochat"),
    CommandMeta(
        "status", "查看她现在在做什么、今天的时间线、Bot 状态和 provider 错误", "status <群号>"
    ),
    CommandMeta("presets", "查看可用预设", "presets <群号>"),
    CommandMeta("set_preset <文件名>", "加载预设", "set_preset <群号> <文件名>"),
    CommandMeta("persona", "查看人生线、本群的她和最近的改动", "persona <群号>"),
    CommandMeta("rag_debug <query>", "诊断 RAG 记忆检索"),
    CommandMeta("group_notes", "查看本群群志原文", "group_notes <群号>"),
    CommandMeta("calm", "重置心情、退出当前对话，并重新定一次她在做什么", "calm <群号>"),
    CommandMeta("reset confirm", "先备份再完全重置本群", "reset <群号> confirm"),
    CommandMeta("token统计", "查看当前 chat / feedback / background 模型的 Token 用量"),
    CommandMeta("stickerdel", "回复一张图：从表情包库删掉并拉黑，以后只描述不收"),
    CommandMeta("photodel", "回复她发的照片：从相册删掉，各群都翻不到了"),
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


async def is_group_message(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


async def is_private_message(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


def _is_group_or_private(event: Event) -> bool:
    return isinstance(event, (GroupMessageEvent, PrivateMessageEvent))


async def _group_name(group_id: int) -> str:
    try:
        group_info = await get_bot().get_group_info(group_id=group_id)
    except (ActionFailed, NetworkError, ValueError) as e:
        logger.debug(f"查群 {group_id} 的群名失败: {e}")
        return f"群 {group_id}"
    return group_info.get("group_name") or f"群 {group_id}"


async def _session_or_finish(matcher: type[Matcher], group_id: int):
    session = get_session(group_id)
    if session is None:
        await matcher.finish("本群 Autochat 未启用，请先使用 autochat enable")
    return session


def _preset_list() -> str:
    reload_presets()
    return "可选的预设:\n" + "".join(
        f"- {filename}: {preset.name} {preset.seed.split('。')[0]}\n"
        for filename, preset in PRESETS.items()
        if not preset.hidden
    )


def _dual_command(cmd: str, aliases: set[str], handler):
    """注册一个群聊/私聊通用的 matcher，调用 handler(matcher, group_id, args_text)。

    群聊直接用 event.group_id；私聊把第一个参数当群号。
    两个 matcher 会触发 nonebot 的「Duplicated prefix rule」告警，所以合成一个。
    """

    matcher = on_command(
        rule=_is_group_or_private,
        permission=SUPERUSER,
        cmd=cmd,
        aliases=aliases,
        priority=0,
        block=True,
    )

    @matcher.handle()
    async def _entry(event: Event, args: Message = CommandArg()):
        if isinstance(event, GroupMessageEvent):
            await handler(matcher, event.group_id, args.extract_plain_text().strip())
            return
        parts = args.extract_plain_text().strip().split(" ", 1)
        if not parts[0]:
            await matcher.finish("请提供<群号>")
        try:
            group_id = int(parts[0])
        except ValueError:
            await matcher.finish("群号必须是数字")
        await handler(matcher, group_id, parts[1].strip() if len(parts) > 1 else "")

    return matcher


async def _take_in(
    bot: Bot, event: GroupMessageEvent, session: Session, *, group_bot: bool = False
) -> None:
    """读一条群消息放进队列；群友的和群里的bot 的走同一条路。"""

    generation = session.state.generation
    message = await read_message(bot, event, session, group_bot=group_bot)
    if not message.content:
        return
    # 合并转发：先以占位进队列，后台读完改写成摘要再放回来（没设人设的群不读）
    if message.forward and session.state.name:
        session.spawn(
            digest_forward(
                session, bot, event, message, forward_id(event.original_message), generation
            )
        )

    await session.enqueue(message, bot, event)


auto_chat = on_message(rule=is_group_message, priority=99, block=False)


@auto_chat.handle()
async def handle_auto_chat(bot: Bot, event: GroupMessageEvent):
    session = get_session(event.group_id)
    # 关机途中不再接新消息
    if session is None or is_shutting_down():
        return
    await _take_in(bot, event, session)


class GroupMessageSentEvent(GroupMessageEvent):
    """这个号自己发出的群消息。

    平台以 post_type=message_sent 上报，适配器没有这个模型，会落到通用 Event、读不出消息段。
    get_type() 是 message_sent，on_message / on_command 收不到它，原有命令不受影响。
    """

    post_type: Literal["message_sent"]


Adapter.add_custom_model(GroupMessageSentEvent)


async def is_group_message_sent(event: Event) -> bool:
    return isinstance(event, GroupMessageSentEvent)


group_message_sent = on("message_sent", rule=is_group_message_sent, priority=99, block=False)


@group_message_sent.handle()
async def handle_group_message_sent(bot: Bot, event: GroupMessageSentEvent):
    """她自己发的已经在短期记忆里，跳过；其余是同一个号上其他程序发的，记在群里的bot 名下。"""

    session = get_session(event.group_id)
    if session is None or is_shutting_down():
        return
    # 上报可能比发送接口返回还早：等这个群正在进行的发送写完短期记忆再认
    async with session.send_lock:
        pass
    msg_id = str(event.message_id)
    if any(m.id == msg_id for m in session.short_term_memory.access()):
        return
    await _take_in(bot, event, session, group_bot=True)


async def is_emoji_like_added(event: Event) -> bool:
    # OneBot 适配器没有这个 notice 的模型，落到通用 NoticeEvent，额外字段在 model_dump 里
    return (
        isinstance(event, NoticeEvent)
        and event.notice_type == "group_msg_emoji_like"
        and event.model_dump().get("sub_type") == "add"
    )


emoji_like = on_notice(rule=is_emoji_like_added, priority=99, block=False)


@emoji_like.handle()
async def handle_emoji_like(bot: Bot, event: NoticeEvent):
    """群里的贴表情都记，包括她自己贴的（SnowLuma 会上报）和群友之间互贴的。"""

    data = event.model_dump()
    if is_shutting_down() or not data.get("likes"):
        return
    session = get_session(int(data["group_id"]))
    if session is None:
        return

    message_id = str(data["message_id"])
    # 被贴的那条要还在最近的上下文里，太旧的不管
    target = next(
        (m for m in session.short_term_memory.access() if m.id == message_id),
        None,
    )
    if target is None:
        return

    like = data["likes"][0]
    session.record_action(
        await read_reaction(
            bot,
            session,
            target,
            str(data["user_id"]),
            str(like["emoji_id"]),
            int(like.get("count") or 1),
        )
    )


async def is_group_poke(event: Event) -> bool:
    return isinstance(event, PokeNotifyEvent) and event.group_id is not None


group_poke = on_notice(rule=is_group_poke, priority=99, block=False)


@group_poke.handle()
async def handle_group_poke(bot: Bot, event: PokeNotifyEvent):
    """群里所有戳一戳都当成一条消息进队列：群友凑热闹戳某人时，bot 要不要跟着戳由 Feedback 判断。"""

    if is_shutting_down():
        return
    session = get_session(event.group_id)
    if session is None:
        return

    message = await read_poke(bot, event, session.state.name)
    if message.user_id == str(bot.self_id):
        # bot 自己戳的：直接进短期记忆，不触发新一轮；用读出来的这条，poke 标记才在
        session.record_incoming([message])
        return
    await session.enqueue(message, bot, event)


async def is_group_change(event: Event) -> bool:
    return isinstance(
        event, (GroupIncreaseNoticeEvent, GroupDecreaseNoticeEvent, GroupBanNoticeEvent)
    )


group_change = on_notice(rule=is_group_change, priority=99, block=False)


@group_change.handle()
async def handle_group_change(
    bot: Bot, event: GroupIncreaseNoticeEvent | GroupDecreaseNoticeEvent | GroupBanNoticeEvent
):
    """进群、退群、禁言：读成记录进短期记忆，和贴表情一样不进队列。

    她被禁言就记到会话上，期间不开口；有人进出群，群人数下次看群时重新拉。
    """

    if is_shutting_down():
        return
    session = get_session(event.group_id)
    if session is None:
        return

    if isinstance(event, GroupBanNoticeEvent):
        if str(event.user_id) in (str(bot.self_id), "0"):
            if event.sub_type == "ban":
                session.muted_until = (
                    datetime.max
                    if event.user_id == 0
                    else datetime.now() + timedelta(seconds=event.duration)
                )
            else:
                session.muted_until = None
    else:
        session.group_info_at = datetime.min

    message = await read_group_notice(bot, event, session)
    if message is None:
        logger.warning(f"群 {event.group_id} 的通知 {event.notice_type}/{event.sub_type} 涉及她自己，只记日志")
        return
    session.record_action(message)
    log_event(
        "group_notice",
        session_id=session.id,
        user=message.user_name,
        content=message.content,
    )


async def is_group_recall(event: Event) -> bool:
    return isinstance(event, GroupRecallNoticeEvent)


group_recall = on_notice(rule=is_group_recall, priority=99, block=False)


@group_recall.handle()
async def handle_group_recall(event: GroupRecallNoticeEvent):
    """撤回：原消息内容原地改成撤回标记，不删，bot 之前对它的回复在上下文里才接得上；不触发新一轮。

    队列、短期缓冲、消息库三处都改，哪处没有就是不在那里。已经整理进记忆碎片、档案、群志的不追删。
    """

    if is_shutting_down():
        return
    session = get_session(event.group_id)
    if session is None:
        return

    msg_id = str(event.message_id)
    by_author = event.operator_id == event.user_id
    content = recall_text(by_author)
    async with session.data_lock:
        queued = [m for m in session.messages_chunk if m.id == msg_id]
        for message in queued:
            message.recall(content)
    in_buffer = session.recall_message(msg_id, content)
    await recall_message(session.id, msg_id, content)
    log_event(
        "recall",
        "DEBUG",
        session_id=session.id,
        by_author=by_author,
        queued=bool(queued),
        in_buffer=in_buffer,
    )


help_cmd = on_command(
    rule=_is_group_or_private,
    permission=SUPERUSER,
    cmd="help",
    aliases={"帮助"},
    priority=0,
    block=True,
)


@help_cmd.handle()
async def handle_help(event: Event):
    await help_cmd.finish(
        render_group_help()
        if isinstance(event, GroupMessageEvent)
        else render_private_help()
    )


list_groups_pm = on_command(
    rule=is_private_message,
    permission=SUPERUSER,
    cmd="list_groups",
    aliases={"群组列表"},
    priority=0,
    block=True,
)


@list_groups_pm.handle()
async def handle_list_groups_pm():
    if not sessions:
        await list_groups_pm.finish("没有启用的群组")
    msg = "启用的群组:\n" + "".join(
        f"- {group_id}\n" for group_id in sessions
    )
    await list_groups_pm.finish(msg)


manual_backup_cmd = on_command(
    rule=_is_group_or_private,
    permission=SUPERUSER,
    cmd="backup_data",
    aliases={"备份数据"},
    priority=0,
    block=True,
)


@manual_backup_cmd.handle()
async def handle_manual_backup():
    await manual_backup_cmd.send("开始手动备份 NyaTuringTest 数据，请稍候...")
    if await backup_task():
        await manual_backup_cmd.finish("备份完成！")
    await manual_backup_cmd.finish("备份失败，请检查日志和数据目录。")


manage_cmd = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="autochat", priority=1, block=True
)


@manage_cmd.handle()
async def handle_manage_autochat(
    event: GroupMessageEvent, args: Message = CommandArg()
):
    parts = args.extract_plain_text().split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    # 预设名是文件名，保留大小写
    preset_arg = parts[1].strip() if len(parts) > 1 else ""
    group_id = event.group_id

    if sub == "enable":
        if group_id in sessions:
            hint = "\n换人设请用 set_preset <文件名>" if preset_arg else ""
            await manage_cmd.finish(f"本群 Autochat 已处于启用状态{hint}")
        # 第一次启用必须选人设：没设过的群启用后会照常记消息、跑整理，却不说话
        if preset_arg:
            filename = resolve_preset(preset_arg)
            if filename is None:
                await manage_cmd.finish(f"不存在的预设: {preset_arg}\n{_preset_list()}")
            await enable_autochat(group_id, filename)
            await manage_cmd.finish(f"Autochat 已在本群启用，人设：{filename}")
        session_db = await SessionModel.get_or_none(id=str(group_id))
        if session_db is None or not session_db.preset:
            await manage_cmd.finish(
                "第一次启用要先选人设。\n"
                f"{_preset_list()}"
                "用法: autochat enable <预设>"
            )
        await enable_autochat(group_id)
        await manage_cmd.finish(f"Autochat 已在本群启用，人设：{session_db.preset}")

    elif sub == "disable":
        if group_id not in sessions:
            await manage_cmd.finish("本群 Autochat 未启用")
        await disable_autochat(group_id)
        await manage_cmd.finish("Autochat 已在本群禁用")

    else:
        await manage_cmd.finish(
            "指令格式错误。请使用: autochat enable [预设] 或 autochat disable"
        )


async def _do_status(matcher: type[Matcher], group_id: int, _args: str):
    session = await _session_or_finish(matcher, group_id)
    state = session.state
    state.global_emotion.settle()
    emotion = state.global_emotion
    recent = "\n".join(
        f"{m.user_name}: {m.content}" for m in session.short_term_memory.access()
    )
    last_look = "（本次启动还没看过）"
    if state.last_look_time != datetime.min:
        last_look = f"{state.last_look_time:%H:%M:%S}"
    settings = get_app_settings()
    muted = f"，禁言到 {session.muted_until:%m-%d %H:%M}" if session.muted else ""
    lines = [
        "",
        f"人设：{state.preset or '未设（不说话）'}{muted}",
        *status_text(state.preset),
        f"未读：{unread_count(state, session.unread())} 条，上次看群 {last_look}",
        f"心情：V{emotion.valence:.2f} A{emotion.arousal:.2f} D{emotion.dominance:.2f}",
        f"后台任务数: {len(session.background_tasks)}",
        f"摘要：{state.chat_summary}",
        "最近消息：",
        recent or "无",
        "",
        "Provider:",
        f"- Chat reasoning_effort: {settings.chat.reasoning_effort or '未指定（由上游决定）'}",
        f"- Feedback reasoning_effort: {settings.feedback.reasoning_effort or '未指定（由上游决定）'}",
        f"- Background reasoning_effort: {settings.background.reasoning_effort or '未指定（由上游决定）'}",
        f"- Queue length: {len(session.messages_chunk)}",
    ]
    for name, client in (
        ("Chat", chat_client),
        ("Feedback", feedback_client),
        ("Background", background_client),
    ):
        provider_status = client.provider_status
        if provider_status.last_error_type:
            lines.append(
                f"- {name} last_error={provider_status.last_error_type} "
                "circuit_remaining="
                f"{provider_status.circuit_remaining_seconds}s"
            )
    await matcher.finish("\n".join(lines))


get_status = _dual_command("status", {"状态"}, _do_status)


async def _do_get_presets(matcher: type[Matcher], group_id: int, _args: str):
    await _session_or_finish(matcher, group_id)
    await matcher.finish(_preset_list() + "使用方法: set_presets <预设名称>\n")


get_presets = _dual_command("presets", {"preset"}, _do_get_presets)


async def _do_set_presets(matcher: type[Matcher], group_id: int, args: str):
    if not args:
        await matcher.finish("请提供<预设文件名>")
    session = await _session_or_finish(matcher, group_id)
    async with session.admin_lock:
        loaded = await session.load_preset(filename=args)
    await matcher.finish(f"预设已加载: {args}" if loaded else f"不存在的预设: {args}")


set_presets = _dual_command("set_preset", {"set_presets"}, _do_set_presets)


async def _do_persona(matcher: type[Matcher], group_id: int, _args: str):
    session = await _session_or_finish(matcher, group_id)
    state = session.state
    preset, name = state.preset, state.name
    self_life, self_in_group = state.self_life, state.self_in_group
    if not name:
        await matcher.finish("本群还没设人设（或预设文件不在），先 set_preset")

    # 「无」是整理过但没改，不算改动，不上尺子也不进流水
    history = [
        PersonaChange(row.date, row.scope, row.changes)
        for row in await PersonaHistoryModel.filter(
            Q(scope="life", key=preset) | Q(scope="group", key=f"{group_id}:{preset}")
        ).order_by("-date", "-id")
        if row.changes and row.changes != "无"
    ]
    recent = history[:PERSONA_HISTORY_LIMIT]
    try:
        image = await render_persona_card(
            name=name,
            as_of=PRESETS[preset].as_of,
            life=self_life,
            group_text=self_in_group,
            group_name=await _group_name(group_id),
            changes=recent,
            marks=history,
            diary=recent_diary(preset, PERSONA_DIARY_DAYS),
        )
    except Exception as e:
        logger.error(f"渲染人设卡片失败: {e}")
        changes = "\n".join(
            f"- {change.day:%m-%d} {'人生线' if change.scope == 'life' else '本群'}：{change.text}"
            for change in recent
        )
        diary = "\n".join(recent_diary(preset, PERSONA_DIARY_DAYS))
        await matcher.finish(
            f"人设：{preset}（{name}）\n\n"
            f"[人生线]\n{self_life}\n\n"
            f"[最近的日记]\n{diary or '还没有'}\n\n"
            f"[在这个群]\n{self_in_group or '还没整理过（每天 03:30 整理）'}\n\n"
            f"[最近的改动]\n{changes or '无'}"
        )
    await matcher.finish(MessageSegment.image(image))


get_persona = _dual_command("persona", {"人设"}, _do_persona)


async def _do_group_notes(matcher: type[Matcher], group_id: int, _args: str):
    await _session_or_finish(matcher, group_id)
    pending = await count_new_group_fragments(str(group_id))
    if pending >= GROUP_NOTES_REFRESH_MIN_NEW:
        await matcher.send(f"有 {pending} 条新记忆还没整理进群志，正在整理，稍等一下")
        try:
            await refresh_group_notes(group_id, min_new=GROUP_NOTES_REFRESH_MIN_NEW)
        except Exception as e:
            # 整理失败照常发旧群志，水位没动，下次查看或夜里再整理
            logger.warning(f"群 {group_id} 手动整理群志失败: {e}")
    # 读库而不是内存副本：要带上整理水位做日期
    session_db = await SessionModel.get_or_none(id=str(group_id))
    if session_db is None or not session_db.group_notes:
        await matcher.finish("本群还没有群志（每天 03:30 整理）")
    try:
        image = await render_group_notes_card(
            notes=session_db.group_notes,
            group_name=await _group_name(group_id),
            updated_at=session_db.notes_summarized_until,
        )
    except Exception as e:
        logger.error(f"渲染群志卡片失败: {e}")
        await matcher.finish(f"群志：\n{session_db.group_notes}")
    await matcher.finish(MessageSegment.image(image))


get_group_notes = _dual_command("group_notes", {"群志", "查看群志"}, _do_group_notes)


async def _do_calm_down(matcher: type[Matcher], group_id: int, _args: str):
    session = await _session_or_finish(matcher, group_id)
    async with session.admin_lock:
        await session.calm_down()
    # 状态按人存，重新定一次各群一起生效
    if session.state.name:
        session.spawn(redecide(session.state.preset))
    await matcher.finish("已老实")


calm_down = _dual_command("calm", {"冷静"}, _do_calm_down)


async def _do_reset(matcher: type[Matcher], group_id: int, args: str):
    if args.lower() != "confirm":
        await matcher.finish(
            "危险操作：将清空本群会话、记忆和画像。确认执行请发送：reset confirm"
        )
    session = await _session_or_finish(matcher, group_id)
    await matcher.send("即将重置，会先执行一次数据备份...")
    # 先作废进行中的 turn 再备份；备份放在 admin_lock 之外：它可能长时间打包整个数据目录
    async with session.admin_lock:
        session.bump_generation("reset_requested")
    if not await backup_task():
        await matcher.finish("备份失败，已中止重置；会话、记忆和画像均未清除。")
    async with session.admin_lock:
        await session.reset()
    await matcher.finish("已重置会话")


reset = _dual_command("reset", {"重置"}, _do_reset)


token_stats = on_command(
    rule=is_group_message,
    permission=SUPERUSER,
    cmd="token统计",
    aliases={"autochat token统计"},
    priority=1,
    block=True,
)


@token_stats.handle()
async def handle_token_stats(event: GroupMessageEvent):
    stats = await get_token_stats(event.group_id, get_token_stats_model_names())
    try:
        image = await render_token_stats_card(stats)
    except Exception as e:
        logger.error(f"渲染 Token 统计卡片失败: {e}")
        lines = [
            f"{label} {item['model']}: {item['total']:,}"
            for label, view in (("今日本群", "1d_local"), ("今日全部", "1d_global"))
            for item in stats[view]
        ]
        await token_stats.finish("Token 统计\n" + ("\n".join(lines) or "今日暂无"))
    await token_stats.finish(MessageSegment.image(image))


sticker_del = on_command(
    rule=is_group_message,
    permission=SUPERUSER,
    cmd="stickerdel",
    priority=1,
    block=True,
)


@sticker_del.handle()
async def handle_sticker_del(event: GroupMessageEvent):
    """回复的消息里每张图都按内容哈希删掉、拉黑；库里没有的也先拉黑，以后见到只描述不收。"""

    images = []
    if event.reply:
        images = [seg for seg in event.reply.message if seg.type == "image"]
    if not images:
        await sticker_del.finish("用法：回复一条带图的消息，发 stickerdel")
    removed = blocked = failed = 0
    for seg in images:
        _, _, image_bytes = await fetch_image_input(
            seg.data.get("url", ""),
            seg.data.get("file_unique", ""),
            is_sticker=True,
            ref_id="stickerdel",
        )
        if image_bytes is None:
            failed += 1
        elif await block_sticker(hashlib.sha1(image_bytes).hexdigest()):
            removed += 1
        else:
            blocked += 1
    parts = [
        f"{label} {count} 张"
        for label, count in (
            ("已从库里删掉并拉黑", removed),
            ("库里没有、已拉黑", blocked),
            ("下载失败", failed),
        )
        if count
    ]
    await sticker_del.finish("，".join(parts))


photo_del = on_command(
    rule=is_group_message,
    permission=SUPERUSER,
    cmd="photodel",
    priority=1,
    block=True,
)


@photo_del.handle()
async def handle_photo_del(event: GroupMessageEvent):
    if not event.reply:
        await photo_del.finish("用法：回复她发的照片，发 photodel")
    session_id = str(event.group_id)
    msg_id = str(event.reply.message_id)
    session = get_session(event.group_id)
    # 刚发的可能还没落库
    buffered = (
        next((m for m in session.short_term_memory.access() if m.id == msg_id), None)
        if session
        else None
    )
    content = buffered.content if buffered else await message_content(session_id, msg_id)
    removed = await remove_photo_by_message(session_id, content)
    await photo_del.finish("已从相册删掉" if removed else "相册里没有这张")


rag_debug = on_command(
    "rag_debug",
    aliases={"记忆诊断"},
    rule=is_group_message,
    permission=SUPERUSER,
    priority=0,
    block=True,
)


def _format_rag_debug_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _format_rag_debug_record(index: int, record: RetrievedMemory) -> str:
    row = record.row
    content = row.content.replace("\n", " ")
    return (
        f"{index}. ref={row.id} "
        f"category={row.category} "
        f"subject={row.subject_user_id or '-'} "
        f"speaker={row.speaker_user_id or '-'} "
        f"scope={record.scope} "
        f"adjusted_score={_format_rag_debug_score(record.adjusted_score)} "
        f"retrieval_score={_format_rag_debug_score(record.retrieval_score)} "
        f"rerank_score={_format_rag_debug_score(record.rerank_score)}\n"
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
    session = await _session_or_finish(rag_debug, event.group_id)

    result = await session.vector_memory.retrieve([query], k=RAG_FINAL_K)
    lines = [
        "RAG debug",
        f"query: {query}",
        f"candidate_count: {result.stats.get('candidate_count', 0)}",
        f"returned_count: {result.stats.get('returned_count', 0)}",
        f"fallback_reason: {result.stats.get('fallback_reason') or 'none'}",
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


query_memory = on_command(
    "查询记忆",
    aliases={"memory"},
    rule=is_group_message,
    priority=5,
    block=True,
)


@query_memory.handle()
async def handle_query_memory(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
):
    target_id = str(event.user_id)
    for segment in args:
        if segment.type == "at" and str(segment.data.get("qq", "")):
            target_id = str(segment.data["qq"])
            break
    session = get_session(event.group_id)
    if session is None:
        await query_memory.finish("本群尚未启用 AI 功能。")
    target_name = (
        await member_name(bot, event.group_id, target_id, session.state.name) or target_id
    )

    retry_after = acquire_query_slot(str(event.group_id), str(event.user_id))
    if retry_after > 0:
        await query_memory.finish(
            f"记忆回溯正在冷却，请约 {max(1, int(retry_after + 0.5))} 秒后再试。"
        )
    await query_memory.send("正在回溯记忆深处...")

    try:
        page = await query_memory_profile(
            session,
            target_id=target_id,
            target_name=target_name,
            sender_id=str(event.user_id),
        )
    except Exception as e:
        logger.error(f"查询记忆失败: {e}")
        await query_memory.finish("大脑处理过载，记忆读取失败，请稍后再试。")
    if isinstance(page, str):
        await query_memory.finish(page)
    try:
        image = await render_impression_card(page)
    except Exception as e:
        logger.error(f"渲染印象卡片失败: {e}")
        await query_memory.finish(impression_text(page))
    await query_memory.finish(MessageSegment.image(image))
