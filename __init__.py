import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
import random
import re
import ssl
import traceback

import anyio
import httpx
from nonebot import logger, on_command, on_message, require
from nonebot import get_driver
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    GroupMessageEvent,
    PrivateMessageEvent,
    MessageSegment,
    Message,
)
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from openai import AsyncOpenAI

require("nonebot_plugin_localstore")

from .client import LLMClient
from .config import Config, plugin_config
from .image_manager import IMAGE_CACHE_DIR, image_manager
from .mem import Message as MMessage
from .session import Session

__plugin_meta__ = PluginMetadata(
    name="NYATuringTest",
    description="群聊特化llm聊天机器人，具有长期记忆和情绪模拟能力",
    usage="群聊特化llm聊天机器人，具有长期记忆和情绪模拟能力",
    type="application",
    homepage="https://github.com/shadow3aaa/nonebot-plugin-nyaturingtest",
    config=Config,
    supported_adapters={"~onebot.v11"},
    extra={"author": "shadow3aaa <shadow3aaaa@gmail.com>"},
)


async def is_group_message(event: Event) -> bool:
    return isinstance(event, GroupMessageEvent)


async def is_private_message(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


# 提前定义全局客户端变量
_GLOBAL_HTTP_CLIENT: httpx.AsyncClient | None = None


# 优化连接池配置
def get_http_client() -> httpx.AsyncClient:
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT is None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.set_ciphers("ALL:@SECLEVEL=1")
        _GLOBAL_HTTP_CLIENT = httpx.AsyncClient(
            verify=ssl_context,
            timeout=30.0,
            # 增大连接池限制
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )
    return _GLOBAL_HTTP_CLIENT


@dataclass
class GroupState:
    event: Event | None = None
    bot: Bot | None = None
    # [关键] 确保 Session 初始化时传入了 ID 和 HTTP Client
    session: Session = field(
        default_factory=lambda: Session(siliconflow_api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                                        http_client=get_http_client())
    )
    messages_chunk: list[MMessage] = field(default_factory=list)

    # 复用全局 HTTP 客户端
    client: LLMClient = field(
        default_factory=lambda: LLMClient(
            client=AsyncOpenAI(
                api_key=plugin_config.nyaturingtest_chat_openai_api_key,
                base_url=plugin_config.nyaturingtest_chat_openai_base_url,
                http_client=get_http_client()
            )
        )
    )

    data_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_tasks: set[asyncio.Task] = set()


# 智能断句辅助函数
def _smart_split_text(text: str, max_chars: int = 40) -> list[str]:
    """
    将长文本切分为多条消息，模拟人类说话习惯
    :param text: 原始文本
    :param max_chars: 触发切分的阈值（小于此长度不切分）
    :return: 切分后的文本列表
    """
    text = text.strip()
    if not text:
        return []

    if len(text) < max_chars:
        return [text]

    # 使用正则按标点符号切分
    raw_parts = re.split(r'(?<=[。！？!?.~\n])\s*', text)

    final_parts = []
    current_buffer = ""

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if len(current_buffer) + len(part) < 15:
            current_buffer += part
        else:
            if current_buffer:
                final_parts.append(current_buffer)
            current_buffer = part

    if current_buffer:
        final_parts.append(current_buffer)

    return final_parts if final_parts else [text]


async def spawn_state(state: GroupState):
    """
    启动后台任务循环检查是否要回复
    """
    while True:
        await asyncio.sleep(random.uniform(5.0, 10.0))

        current_chunk = []
        async with state.data_lock:
            if state.bot is None or state.event is None:
                continue
            if len(state.messages_chunk) == 0:
                continue

            logger.debug(f"Snapshotting {len(state.messages_chunk)} messages for processing")
            current_chunk = state.messages_chunk.copy()
            state.messages_chunk.clear()

        async with state.session_lock:
            try:
                responses = await state.session.update(
                    messages_chunk=current_chunk, llm=lambda x: llm_response(state.client, x)
                )

                if responses:
                    total_responses = len(responses)

                    for r_idx, response in enumerate(responses):
                        raw_content = ""
                        reply_id = None

                        if isinstance(response, str):
                            raw_content = response
                        elif isinstance(response, dict):
                            raw_content = response.get("content", "")
                            reply_id = response.get("reply_to")

                        if not raw_content:
                            continue

                        # 智能断句
                        msg_parts = _smart_split_text(raw_content)

                        for i, part in enumerate(msg_parts):
                            # [修改] 去除句尾句号逻辑
                            part = part.strip()
                            # 如果是以 "。" 结尾，直接去掉
                            if part.endswith("。"):
                                part = part[:-1]
                            # 如果是以 "." 结尾，但不是 ".." 或 "..." (防止误伤英文省略号)
                            elif part.endswith(".") and not part.endswith(".."):
                                part = part[:-1]

                            msg_to_send = Message(part)

                            # 只在第一条消息的第一段挂载回复引用
                            if reply_id and r_idx == 0 and i == 0:
                                msg_to_send.insert(0, MessageSegment.reply(int(reply_id)))

                            await state.bot.send(message=msg_to_send, event=state.event)

                            # 统一延时逻辑
                            if i < len(msg_parts) - 1 or r_idx < total_responses - 1:
                                delay = random.uniform(1.5, 3.0)
                                logger.debug(f"模拟打字等待 {delay:.2f}s...")
                                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")
                traceback.print_exc()
                continue


group_states: dict[int, GroupState] = {}

# ... (Command 定义保持不变) ...
help = on_command(rule=is_group_message, permission=SUPERUSER, cmd="help", aliases={"帮助"}, priority=0, block=True)
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
reset = on_command(rule=is_group_message, permission=SUPERUSER, cmd="reset", aliases={"重置"}, priority=0, block=True)
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


def ensure_group_state(group_id: int):
    if group_id not in group_states:
        allowed_groups = plugin_config.nyaturingtest_enabled_groups
        if group_id not in allowed_groups:
            return None

        group_states[group_id] = GroupState(
            session=Session(
                id=f"{group_id}",  # 必须传，否则所有群混在一起
                siliconflow_api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                http_client=get_http_client()  # 必须传，否则无法复用连接池
            )
        )
        global _tasks
        task = asyncio.create_task(spawn_state(state=group_states[group_id]))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    return group_states[group_id]


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
        if await state.session.load_preset(filename=file):
            await matcher.finish(f"预设已加载: {file}")
        else:
            await matcher.finish(f"不存在的预设: {file}")


@help.handle()
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
    await help.finish(help_message)


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


async def llm_response(client: LLMClient, message: str) -> str:
    try:
        result = await client.generate_response(prompt=message, model=plugin_config.nyaturingtest_chat_openai_model)
        if result:
            return result
        else:
            return ""
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Error occurred while processing the message."


@auto_chat.handle()
async def handle_auto_chat(bot: Bot, event: GroupMessageEvent):
    group_id = event.group_id
    state = ensure_group_state(group_id)
    if not state:
        return

    user_id = event.get_user_id()
    bot_name = state.session.name()

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
        state.messages_chunk.append(
            MMessage(
                time=datetime.now(),
                user_name=nickname,
                content=message_content,
                id=str(event.message_id),
            )
        )


async def message2BotMessage(bot_name: str, group_id: int, message: Message, bot: Bot) -> str:
    """
    将消息转换为机器人可读的消息 (并发优化版)
    """

    async def process_segment(seg: MessageSegment) -> str:
        if seg.type == "text":
            return f"{seg.data.get('text', '')}"

        elif seg.type == "image" or seg.type == "emoji":
            try:
                url = seg.data.get("url", "")
                logger.debug(f"Image URL: {url}")

                cache_path = IMAGE_CACHE_DIR.joinpath("raw")
                cache_path.mkdir(parents=True, exist_ok=True)

                key = re.search(r"[?&]fileid=([a-zA-Z0-9_-]+)", url)
                key = key.group(1) if key else None

                image_bytes = None
                if key and cache_path.joinpath(key).exists():
                    async with await anyio.open_file(cache_path.joinpath(key), "rb") as f:
                        image_bytes = await f.read()
                else:
                    # 此处复用全局客户端，避免频繁创建连接
                    client = get_http_client()
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        image_bytes = resp.content
                        if key:
                            async with await anyio.open_file(cache_path.joinpath(key), "wb") as f:
                                await f.write(image_bytes)
                    except Exception as e:
                        logger.warning(f"下载图片失败: {e}")
                        return "\n[图片下载失败]\n"

                if not image_bytes:
                    return "\n[图片为空]\n"

                is_sticker = seg.data.get("sub_type") == 1
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                description = await image_manager.get_image_description(
                    image_base64=image_base64, is_sticker=is_sticker
                )
                if description:
                    if is_sticker:
                        return f"\n[表情包] [情感:{description.emotion}] [内容:{description.description}]\n"
                    else:
                        return f"\n[图片] {description.description}\n"
                return "\n[图片识别无结果]\n"

            except Exception as e:
                logger.error(f"Image process error: {e}")
                return "\n[图片处理出错]\n"

        elif seg.type == "at":
            id = seg.data.get("qq")
            if not id: return ""
            if id == str(bot.self_id):
                return f" @{bot_name} "
            else:
                try:
                    user_info = await bot.get_group_member_info(group_id=group_id, user_id=int(id))
                    nickname = user_info.get("card") or user_info.get("nickname") or str(id)
                    return f" @{nickname} "
                except Exception:
                    return f" @{id} "

        elif seg.type == "reply":
            reply_id = seg.data.get("id")
            if reply_id:
                try:
                    source_msg = await bot.get_msg(message_id=int(reply_id))
                    raw_chain = source_msg.get("message", [])
                    source_text = ""
                    sender_nickname = source_msg.get("sender", {}).get("nickname", "未知用户")

                    if isinstance(raw_chain, str):
                        source_text = raw_chain
                    else:
                        for chain_seg in raw_chain:
                            if chain_seg["type"] == "text":
                                source_text += chain_seg["data"].get("text", "")

                    return f" [回复 {sender_nickname}: {source_text[:20]}...] "
                except Exception:
                    return " [回复某条消息] "
            return ""

        return ""

    tasks = [process_segment(seg) for seg in message]
    results = await asyncio.gather(*tasks)

    return "".join(results).strip()


driver = get_driver()


@driver.on_shutdown
async def cleanup_tasks():
    """清理后台任务与资源"""
    logger.info("正在清理后台会话任务...")
    for task in _tasks:
        if not task.done():
            task.cancel()

    if _tasks:
        await asyncio.gather(*_tasks, return_exceptions=True)

    # 清理全局 HTTP 客户端
    global _GLOBAL_HTTP_CLIENT
    if _GLOBAL_HTTP_CLIENT:
        await _GLOBAL_HTTP_CLIENT.aclose()
        _GLOBAL_HTTP_CLIENT = None
        logger.info("全局 HTTP 客户端已关闭")

    logger.info("后台任务已清理")