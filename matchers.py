# nyaturingtest/matchers.py
from datetime import datetime
from nonebot import on_command, on_message, logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    PrivateMessageEvent,
    Message,
    Event
)
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.matcher import Matcher

from .config import plugin_config
from .state_manager import ensure_group_state
from .logic import message2BotMessage
from .mem import Message as MMessage

# ==================== 辅助规则 ====================


async def is_group_message(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


async def is_private_message(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)

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

auto_chat = on_message(rule=is_group_message, priority=1, block=False)

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

# ==================== 处理逻辑 ====================


@get_presets.handle()
async def handle_get_presets(event: GroupMessageEvent):
    await do_get_presets(get_presets, event.group_id)


@get_presets_pm.handle()
async def handle_get_presets_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await get_presets_pm.finish("请提供<qq群号>")
    group_id = int(arg)
    await do_get_presets(get_presets_pm, group_id)


async def do_get_presets(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        return
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
    await do_set_presets(set_presets_pm, int(preset_args[0]), preset_args[1])


async def do_set_presets(matcher: type[Matcher], group_id: int, file: str):
    state = ensure_group_state(group_id)
    if not state:
        return
    async with state.session_lock:
        await state.session.load_session()
        if await state.session.load_preset(filename=file):
            await matcher.finish(f"预设已加载: {file}")
        else:
            await matcher.finish(f"不存在的预设: {file}")


@help_cmd.handle()
async def handle_help():
    help_message = """
可用命令:
1. set_role <角色名> <角色设定> - 设置角色
2. role - 获取当前角色
3. calm - 冷静
4. reset - 重置会话
5. status - 获取状态
6. presets - 获取可用预设
7. help - 显示本帮助信息
"""
    await help_cmd.finish(help_message)


@help_pm.handle()
async def handle_help_pm():
    help_message = """
可用命令(私聊需加群号):
1. set_role <群号> <角色名> <角色设定>
2. role <群号>
3. calm <群号>
4. reset <群号>
5. status <群号>
6. presets <群号>
7. list_groups
8. help
"""
    await help_pm.finish(help_message)


@set_role.handle()
async def handle_set_role(event: GroupMessageEvent, args: Message = CommandArg()):
    role_args = args.extract_plain_text().strip().split(" ")
    if len(role_args) != 2:
        await set_role.finish("请提供<角色名> <角色设定>")
    await do_set_role(set_role, event.group_id, role_args[0], role_args[1])


@set_role_pm.handle()
async def handle_set_role_pm(args: Message = CommandArg()):
    role_args = args.extract_plain_text().strip().split(" ")
    if len(role_args) != 3:
        await set_role_pm.finish("请提供<群号> <角色名> <角色设定>")
    await do_set_role(set_role_pm, int(role_args[0]), role_args[1], role_args[2])


async def do_set_role(matcher: type[Matcher], group_id: int, name: str, role: str):
    state = ensure_group_state(group_id)
    if not state:
        return
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
    await do_get_role(get_role_pm, int(arg))


async def do_get_role(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        return
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
    await do_calm_down(calm_down_pm, int(arg))


async def do_calm_down(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        return
    async with state.session_lock:
        await state.session.load_session()
        await state.session.calm_down()
    await matcher.finish("已老实")


@reset.handle()
async def handle_reset(event: GroupMessageEvent):
    await do_reset(reset, event.group_id)


@reset_pm.handle()
async def handle_reset_pm(args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg == "":
        await reset_pm.finish("请提供<群号>")
    await do_reset(reset_pm, int(arg))


async def do_reset(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        return
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
    await do_status(get_status_pm, int(arg))


async def do_status(matcher: type[Matcher], group_id: int):
    state = ensure_group_state(group_id)
    if not state:
        return
    async with state.session_lock:
        await state.session.load_session()
        status_msg = state.session.status()
    await matcher.finish(status_msg)


@list_groups_pm.handle()
async def handle_list_groups_pm():
    allowed_groups = plugin_config.nyaturingtest_enabled_groups
    if not allowed_groups:
        await list_groups_pm.finish("没有启用的群组")
    msg = "启用的群组:\n"
    for group_id in allowed_groups:
        msg += f"- {group_id}\n"
    await list_groups_pm.finish(msg)


@auto_chat.handle()
async def handle_auto_chat(bot: Bot, event: GroupMessageEvent):
    group_id = event.group_id
    state = ensure_group_state(group_id)
    if not state:
        return

    user_id = event.get_user_id()

    # 获取 Bot 名字前，先确保加载
    async with state.session_lock:
        await state.session.load_session()
        bot_name = state.session.name()

    # 使用 logic 模块中的转换函数
    message_content = await message2BotMessage(
        bot_name=bot_name, group_id=group_id, message=event.original_message, bot=bot
    )
    if not message_content:
        return

    try:
        user_info = await bot.get_group_member_info(group_id=group_id, user_id=int(user_id))
        nickname = user_info.get("card") or user_info.get("nickname") or str(user_id)
    except Exception:
        nickname = str(user_id)

    async with state.data_lock:
        state.event = event
        state.bot = bot
        # 构建消息时，传入 user_id
        state.messages_chunk.append(
            MMessage(
                time=datetime.now(),
                user_name=nickname,
                content=message_content,
                id=str(event.message_id),
                user_id=str(user_id)  # 传递 QQ 号
            )
        )
