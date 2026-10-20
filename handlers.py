import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime

from nonebot import get_bot, logger, on_command, on_message, on_notice
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    GroupMessageEvent,
    GroupRecallNoticeEvent,
    Message,
    MessageSegment,
    NoticeEvent,
    PokeNotifyEvent,
    PrivateMessageEvent,
)
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from tortoise.expressions import Q

from .cards.impression import render_impression_card
from .cards.notes import render_group_notes_card
from .cards.persona import Change as PersonaChange
from .cards.persona import render_persona_card
from .cards.token_stats import render_token_stats_card
from .chat.groups import (
    ensure_group_state,
    is_shutting_down,
    refresh_group_notes,
    remove_group_state,
    runtime_enabled_groups,
)
from .config import get_app_settings, get_token_stats_model_names
from .llm import chat_client, feedback_client
from .memory.digest import GROUP_NOTES_REFRESH_MIN_NEW, count_new_group_fragments
from .memory.query import acquire_query_slot, impression_text, query_memory_profile
from .memory.vector import (
    RAG_FINAL_K,
    RAG_MERGED_CANDIDATE_CAP,
    RAG_PER_QUERY_RECALL_K,
)
from .messages.convert import fill_caption, message2BotMessage
from .messages.faces import ensure_faces, face_name
from .messages.forward import digest_forward
from .messages.image import fetch_image_input
from .messages.message import Message as MMessage
from .messages.stickers import block as block_sticker
from .metrics import log_event
from .metrics import metrics
from .prompts import PRESETS
from .storage.backup import backup_task
from .storage.db import disable_group, enable_group, get_token_stats, recall_message
from .storage.models import PersonaHistoryModel, SessionModel

# persona 命令列出的最近改动条数（人生线和本群合计）
PERSONA_HISTORY_LIMIT = 4


@dataclass(frozen=True)
class CommandMeta:
    command: str
    description: str
    private_usage: str = ""


COMMANDS: tuple[CommandMeta, ...] = (
    CommandMeta("autochat <enable/disable>", "在本群启用或禁用 Autochat"),
    CommandMeta(
        "status", "查看 Bot 状态、provider 错误和基础 metrics", "status <群号>"
    ),
    CommandMeta("presets", "查看可用预设", "presets <群号>"),
    CommandMeta("set_preset <文件名>", "加载预设", "set_preset <群号> <文件名>"),
    CommandMeta("persona", "查看人生线、本群的她和最近的改动", "persona <群号>"),
    CommandMeta("rag_debug <query>", "诊断 RAG 记忆检索"),
    CommandMeta("group_notes", "查看本群群志原文", "group_notes <群号>"),
    CommandMeta("calm", "冷静并重置短期状态", "calm <群号>"),
    CommandMeta("reset_emotion", "仅重置心情（不动好感度）", "reset_emotion <群号>"),
    CommandMeta("reset confirm", "先备份再完全重置本群", "reset <群号> confirm"),
    CommandMeta("token统计", "查看当前 chat / feedback 模型的 Token 用量"),
    CommandMeta("stickerdel", "回复一张图：从表情包库删掉并拉黑，以后只描述不收"),
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


# ==================== 辅助规则 ====================


async def is_group_message(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


async def is_private_message(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


def _addresses_bot(bot: Bot, event: GroupMessageEvent) -> bool:
    """@Bot 或回复 Bot 的消息。

    不用 event.to_me：.env 里 NICKNAME=[""] 会让适配器把所有文本开头的消息都判成 to_me。
    """

    self_id = str(bot.self_id)
    if event.reply and str(event.reply.sender.user_id) == self_id:
        return True
    return any(
        seg.type == "at" and str(seg.data.get("qq", "")) == self_id
        for seg in event.original_message
    )


async def _parse_group_id_or_finish(matcher: type[Matcher], raw: str) -> int:
    raw_group_id = raw.strip()
    try:
        return int(raw_group_id)
    except ValueError:
        await matcher.finish("群号必须是数字")
        raise


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
    settings = get_app_settings()
    lines = [
        "",
        "Provider:",
        f"- Chat reasoning_effort: {settings.chat.reasoning_effort or '未指定（由上游决定）'}",
        f"- Feedback reasoning_effort: {settings.feedback.reasoning_effort or '未指定（由上游决定）'}",
        f"- Queue length: {len(state.messages_chunk)}",
        f"- Metrics: llm={metrics.llm_success}/{metrics.llm_failure}",
    ]
    for name, client in (("Chat", chat_client), ("Feedback", feedback_client)):
        provider_status = client.provider_status
        if provider_status.last_error_type:
            lines.append(
                f"- {name} last_error={provider_status.last_error_type} "
                "circuit_remaining="
                f"{provider_status.circuit_remaining_seconds}s"
            )
    return status + "\n".join(lines)


# ==================== 命令注册 ====================


def _is_group_or_private(event: Event) -> bool:
    return isinstance(event, (GroupMessageEvent, PrivateMessageEvent))


help_cmd = on_command(
    rule=_is_group_or_private,
    permission=SUPERUSER,
    cmd="help",
    aliases={"帮助"},
    priority=0,
    block=True,
)
list_groups_pm = on_command(
    rule=is_private_message,
    permission=SUPERUSER,
    cmd="list_groups",
    aliases={"群组列表"},
    priority=0,
    block=True,
)
manual_backup_cmd = on_command(
    rule=_is_group_or_private,
    permission=SUPERUSER,
    cmd="backup_data",
    aliases={"备份数据"},
    priority=0,
    block=True,
)
manage_cmd = on_command(
    rule=is_group_message, permission=SUPERUSER, cmd="autochat", priority=1, block=True
)
token_stats = on_command(
    rule=is_group_message,
    permission=SUPERUSER,
    cmd="token统计",
    aliases={"autochat token统计"},
    priority=1,
    block=True,
)
sticker_del = on_command(
    rule=is_group_message,
    permission=SUPERUSER,
    cmd="stickerdel",
    priority=1,
    block=True,
)
auto_chat = on_message(rule=is_group_message, priority=99, block=False)


async def is_emoji_like_added(event: Event) -> bool:
    # OneBot 适配器没有这个 notice 的模型，落到通用 NoticeEvent，额外字段在 model_dump 里
    return (
        isinstance(event, NoticeEvent)
        and event.notice_type == "group_msg_emoji_like"
        and event.model_dump().get("sub_type") == "add"
    )


emoji_like = on_notice(rule=is_emoji_like_added, priority=99, block=False)


async def is_group_poke(event: Event) -> bool:
    return isinstance(event, PokeNotifyEvent) and event.group_id is not None


group_poke = on_notice(rule=is_group_poke, priority=99, block=False)


async def is_group_recall(event: Event) -> bool:
    return isinstance(event, GroupRecallNoticeEvent)


group_recall = on_notice(rule=is_group_recall, priority=99, block=False)


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
        group_id = await _parse_group_id_or_finish(matcher, parts[0])
        await handler(matcher, group_id, parts[1].strip() if len(parts) > 1 else "")

    return matcher


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


async def _do_persona(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    async with state.session_lock:
        await state.session.load_session()
        session_state = state.session.state
        preset, name = session_state.preset, session_state.name
        self_life, self_in_group = session_state.self_life, session_state.self_in_group
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
        )
    except Exception as e:
        logger.error(f"渲染人设卡片失败: {e}")
        changes = "\n".join(
            f"- {change.day:%m-%d} {'人生线' if change.scope == 'life' else '本群'}：{change.text}"
            for change in recent
        )
        await matcher.finish(
            f"人设：{preset}（{name}）\n\n"
            f"[人生线]\n{self_life}\n\n"
            f"[在这个群]\n{self_in_group or '还没整理过（每天 03:30 整理）'}\n\n"
            f"[最近的改动]\n{changes or '无'}"
        )
    await matcher.finish(MessageSegment.image(image))


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
    await matcher.finish("心情已重置")


async def _do_reset(matcher: type[Matcher], group_id: int, args: str):
    if args.lower() != "confirm":
        await matcher.finish(
            "危险操作：将清空本群会话、记忆和画像。确认执行请发送：reset confirm"
        )
    state = await _group_state_or_finish(matcher, group_id)
    await matcher.send("即将重置，会先执行一次数据备份...")
    if not await reset_session_with_backup(state, backup_task):
        await matcher.finish("备份失败，已中止重置；会话、记忆和画像均未清除。")
        return
    await matcher.finish("已重置会话")


async def _group_name(group_id: int) -> str:
    try:
        group_info = await get_bot().get_group_info(group_id=group_id)
        return group_info.get("group_name") or f"群 {group_id}"
    except Exception:
        return f"群 {group_id}"


async def _do_group_notes(matcher: type[Matcher], group_id: int, _args: str):
    await _group_state_or_finish(matcher, group_id)
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


async def _do_status(matcher: type[Matcher], group_id: int, _args: str):
    state = await _group_state_or_finish(matcher, group_id)
    await matcher.finish(await describe_status(state))


get_presets = _dual_command("presets", {"preset"}, _do_get_presets)
set_presets = _dual_command("set_preset", {"set_presets"}, _do_set_presets)
get_persona = _dual_command("persona", {"人设"}, _do_persona)
calm_down = _dual_command("calm", {"冷静"}, _do_calm_down)
reset_emotion = _dual_command("reset_emotion", {"重置情绪"}, _do_reset_emotion)
reset = _dual_command("reset", {"重置"}, _do_reset)
get_status = _dual_command("status", {"状态"}, _do_status)
get_group_notes = _dual_command("group_notes", {"群志", "查看群志"}, _do_group_notes)


@help_cmd.handle()
async def handle_help(event: Event):
    await help_cmd.finish(
        render_group_help()
        if isinstance(event, GroupMessageEvent)
        else render_private_help()
    )


async def _do_manual_backup(matcher: type[Matcher]):
    await matcher.send("开始手动备份 NyaTuringTest 数据，请稍候...")
    if await backup_task():
        await matcher.finish("备份完成！")
    await matcher.finish("备份失败，请检查日志和数据目录。")


@manual_backup_cmd.handle()
async def handle_manual_backup():
    await _do_manual_backup(manual_backup_cmd)


@list_groups_pm.handle()
async def handle_list_groups_pm():
    if not runtime_enabled_groups:
        await list_groups_pm.finish("没有启用的群组")
    msg = "启用的群组:\n" + "".join(
        f"- {group_id}\n" for group_id in runtime_enabled_groups
    )
    await list_groups_pm.finish(msg)


@auto_chat.handle()
async def handle_auto_chat(bot: Bot, event: GroupMessageEvent):
    group_id = event.group_id
    state = ensure_group_state(group_id)
    if not state:
        return

    async with state.session_lock:
        await state.session.load_session()
        bot_name = state.session.state.name
        generation = state.session.state.generation

    # Shutdown 检查：避免在关机时进入耗时的消息处理
    if is_shutting_down():
        return

    message_content, image_inputs, mentions, caption_jobs = await message2BotMessage(
        bot_name=bot_name,
        group_id=group_id,
        message=event.original_message,
        bot=bot,
        message_scope=str(event.message_id),
        caption=bool(bot_name),
    )
    if not message_content:
        return

    user_id = str(event.user_id)
    if user_id == str(bot.self_id):
        nickname = bot_name
    else:
        nickname = (
            str(event.sender.card or "").strip()
            or str(event.sender.nickname or "").strip()
            or user_id
        )

    message = MMessage(
        time=datetime.now(),
        user_name=nickname,
        content=message_content,
        id=str(event.message_id),
        user_id=user_id,
        image_inputs=image_inputs,
        mentions=mentions,
        to_me=user_id != str(bot.self_id) and _addresses_bot(bot, event),
    )
    message.caption_tasks = [
        state.session.spawn(fill_caption(state.session, message, job, generation))
        for job in caption_jobs
    ]
    # 合并转发总是单独一条：先以占位进队列，后台读完改写成摘要再放回来（没设人设的群不读）
    forward_id = next(
        (
            str(seg.data.get("id") or "")
            for seg in event.original_message
            if seg.type == "forward"
        ),
        "",
    )
    if forward_id:
        message.forward = True
        if bot_name:
            state.session.spawn(
                digest_forward(state, bot, event, message, forward_id, generation)
            )

    async with state.data_lock:
        state.event = event
        state.bot = bot
        state.messages_chunk.append(message)
        state.new_message_signal.set()


@emoji_like.handle()
async def handle_emoji_like(bot: Bot, event: NoticeEvent):
    data = event.model_dump()
    user_id = str(data["user_id"])
    # 自己贴的表情也会上报
    if user_id == str(bot.self_id) or is_shutting_down():
        return
    state = ensure_group_state(int(data["group_id"]))
    if not state:
        return

    async with state.session_lock:
        await state.session.load_session()
    session = state.session
    message_id = str(data["message_id"])
    # 只关心 bot 自己的消息；群友互相贴的、太旧不在短期记忆里的都不管
    target = next(
        (
            m
            for m in session.runtime.short_term_memory.access()
            if m.id == message_id and m.user_name == session.state.name
        ),
        None,
    )
    if target is None or not data.get("likes"):
        return

    await ensure_faces(bot)
    emoji = face_name(str(data["likes"][0]["emoji_id"]))
    try:
        info = await bot.get_group_member_info(
            group_id=int(data["group_id"]), user_id=int(user_id)
        )
        nickname = info.get("card") or info.get("nickname") or user_id
    except Exception:
        nickname = user_id
    session.record_reaction(
        MMessage(
            time=datetime.now(),
            user_name=nickname,
            content=f"[给你的消息「{target.content[:20]}」贴了表情:{emoji or '未知'}]",
            user_id=user_id,
        )
    )


@group_poke.handle()
async def handle_group_poke(bot: Bot, event: PokeNotifyEvent):
    """群里所有戳一戳都当成一条消息进队列：群友凑热闹戳某人时，bot 要不要跟着戳由 Feedback 判断。"""

    if is_shutting_down():
        return
    state = ensure_group_state(event.group_id)
    if not state:
        return
    async with state.session_lock:
        await state.session.load_session()
        bot_name = state.session.state.name

    self_id = str(bot.self_id)
    poker_id, target_id = str(event.user_id), str(event.target_id)

    async def display_name(user_id: str) -> str:
        if user_id == self_id:
            return bot_name
        try:
            info = await bot.get_group_member_info(
                group_id=event.group_id, user_id=int(user_id)
            )
            return info.get("card") or info.get("nickname") or user_id
        except Exception:
            return user_id

    poker, target = await asyncio.gather(display_name(poker_id), display_name(target_id))
    # 动作文字各人自定义（戳了戳、拍了拍、捏了捏…），照通知里的 action / suffix 原样拼
    data = event.model_dump()
    content = f"[戳一戳: {data.get('action') or '戳了戳'} {target}{data.get('suffix') or ''}]"
    if poker_id == self_id:
        # bot 自己戳的：和自己发的消息一样直接进短期记忆，不触发新一轮
        state.session.append_self_message(content, "", self_id)
        return

    async with state.data_lock:
        state.event = event
        state.bot = bot
        state.messages_chunk.append(
            MMessage(
                time=datetime.now(),
                user_name=poker,
                content=content,
                user_id=poker_id,
                # 被戳的人记进 mentions，Feedback 才能跟着戳他
                mentions={} if target_id == self_id else {target_id: target},
                to_me=target_id == self_id,
                poke=True,
            )
        )
        state.new_message_signal.set()


@group_recall.handle()
async def handle_group_recall(event: GroupRecallNoticeEvent):
    """撤回：原消息内容原地改成撤回标记，不删，bot 之前对它的回复在上下文里才接得上；不触发新一轮。

    队列、短期缓冲、消息库三处都改，哪处没有就是不在那里。已经整理进记忆碎片、档案、群志的不追删。
    """

    if is_shutting_down():
        return
    state = ensure_group_state(event.group_id)
    if not state:
        return
    async with state.session_lock:
        await state.session.load_session()

    msg_id = str(event.message_id)
    by_author = event.operator_id == event.user_id
    content = "[撤回了一条消息]" if by_author else "[一条消息被管理员撤回了]"
    async with state.data_lock:
        queued = [m for m in state.messages_chunk if m.id == msg_id]
        for message in queued:
            message.content = content
            message.image_inputs = []
    in_buffer = state.session.recall_message(msg_id, content)
    await recall_message(state.session.id, msg_id, content)
    log_event(
        "recall",
        "DEBUG",
        session_id=state.session.id,
        by_author=by_author,
        queued=bool(queued),
        in_buffer=in_buffer,
    )


@manage_cmd.handle()
async def handle_manage_autochat(
    event: GroupMessageEvent, args: Message = CommandArg()
):
    arg = args.extract_plain_text().strip().lower()
    group_id = event.group_id

    if arg == "enable":
        if group_id in runtime_enabled_groups:
            await manage_cmd.finish("本群 Autochat 已处于启用状态")

        await enable_group(group_id)
        # 更新内存
        runtime_enabled_groups.add(group_id)
        # 立即初始化状态
        ensure_group_state(group_id)

        await manage_cmd.finish("Autochat 已在本群启用 (已保存至数据库)")

    elif arg == "disable":
        if group_id not in runtime_enabled_groups:
            await manage_cmd.finish("本群 Autochat 未启用")

        await disable_group(group_id)
        # 更新内存
        runtime_enabled_groups.discard(group_id)

        # 安全清理运行时状态和后台任务
        await remove_group_state(group_id)

        await manage_cmd.finish("Autochat 已在本群禁用")

    else:
        await manage_cmd.finish(
            "指令格式错误。请使用: autochat enable 或 autochat disable"
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
        f"category={metadata.get('category') or '-'} "
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

    async with state.session_lock:
        await state.session.load_session()
    result = await state.session.runtime.vector_memory.retrieve_with_decay(
        [query],
        k=RAG_FINAL_K,
        use_rerank=True,
        candidate_k=RAG_PER_QUERY_RECALL_K,
        merged_candidate_cap=RAG_MERGED_CANDIDATE_CAP,
    )
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
    target_id = str(event.user_id)
    for segment in args:
        if segment.type == "at" and str(segment.data.get("qq", "")):
            target_id = str(segment.data["qq"])
            break
    target_name = await _target_display_name(bot, event, target_id)
    state = ensure_group_state(event.group_id)
    if not state:
        await query_memory.finish("本群尚未启用 AI 功能。")
        return

    retry_after = acquire_query_slot(str(event.group_id), str(event.user_id))
    if retry_after > 0:
        await query_memory.finish(
            f"记忆回溯正在冷却，请约 {max(1, int(retry_after + 0.5))} 秒后再试。"
        )
    await query_memory.send("正在回溯记忆深处...")

    try:
        page = await query_memory_profile(
            state,
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
