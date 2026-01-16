# nyaturingtest/logic.py
import asyncio
import base64
import random
import re
import traceback
from datetime import datetime

import anyio
from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .client import LLMClient
from .config import plugin_config
from .image_manager import IMAGE_CACHE_DIR, image_manager
from .mem import Message as MMessage
from .state_manager import GroupState, SELF_SENT_MSG_IDS
from .utils import get_http_client, smart_split_text

_IMG_SEMAPHORE = asyncio.Semaphore(3)


async def llm_response(client: LLMClient, message: str, json_mode: bool = False) -> str:
    """
    封装 LLM 调用
    """
    try:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # kwargs["extra_body"] = {"include_reasoning": False}

        result = await client.generate_response(
            prompt=message,
            model=plugin_config.nyaturingtest_chat_openai_model,
            # Qwen 系列建议 Temperature: 0.7 ~ 0.8
            temperature=0.7,
            **kwargs
        )
        return result if result else ""
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Error occurred while processing the message."


async def message2BotMessage(bot_name: str, group_id: int, message: Message, bot: Bot) -> str:
    """
    将 OneBot 消息转换为 Bot 可读文本
    处理图片、表情包、@提及和回复引用
    """

    async def process_segment(seg: MessageSegment) -> str:
        if seg.type == "text":
            return f"{seg.data.get('text', '')}"

        elif seg.type == "image" or seg.type == "emoji":
            async with _IMG_SEMAPHORE:
                try:
                    url = seg.data.get("url", "")
                    file_unique = seg.data.get("file_unique", "")

                    if file_unique:
                        cached_desc = image_manager.get_from_cache(file_unique)
                        if cached_desc:
                            is_sticker = seg.data.get("sub_type") == 1
                            if is_sticker:
                                return f"\n[表情包] [情感:{cached_desc.emotion}] [内容:{cached_desc.description}]\n"
                            else:
                                return f"\n[图片] {cached_desc.description}\n"

                    cache_path = IMAGE_CACHE_DIR.joinpath("raw")
                    cache_path.mkdir(parents=True, exist_ok=True)

                    key = re.search(r"[?&]fileid=([a-zA-Z0-9_-]+)", url)
                    key = key.group(1) if key else (file_unique if file_unique else None)

                    image_bytes = None
                    if key and cache_path.joinpath(key).exists():
                        async with await anyio.open_file(cache_path.joinpath(key), "rb") as f:
                            image_bytes = await f.read()
                    else:
                        client = get_http_client()
                        try:
                            for _ in range(2):
                                try:
                                    resp = await client.get(url, timeout=5.0)
                                    resp.raise_for_status()
                                    image_bytes = resp.content
                                    break
                                except Exception:
                                    await asyncio.sleep(0.5)

                            if image_bytes and key:
                                async with await anyio.open_file(cache_path.joinpath(key), "wb") as f:
                                    await f.write(image_bytes)
                        except Exception as e:
                            logger.warning(f"下载图片失败: {e}")
                            return "\n[图片下载失败]\n"

                    if not image_bytes: return "\n[图片为空]\n"

                    is_sticker = seg.data.get("sub_type") == 1
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                    description = await image_manager.get_image_description(
                        image_base64=image_base64, is_sticker=is_sticker, cache_key=file_unique
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
                    sender = source_msg.get("sender", {}).get("nickname", "未知")
                    return f" [回复 {sender}] "
                except:
                    return " [回复] "
            return ""
        return ""

    tasks = [process_segment(seg) for seg in message]
    results = await asyncio.gather(*tasks)
    return "".join(results).strip()


async def spawn_state(state: GroupState):
    """
    后台思考循环 (Producer-Consumer 模式)
    负责从 Buffer 取消息 -> 调用 Session 处理 -> 发送回复
    """
    while True:
        try:
            # 1. 等待新消息信号 (debounce 2秒)
            try:
                await asyncio.wait_for(state.new_message_signal.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                continue

            state.new_message_signal.clear()
            await asyncio.sleep(2.0)

            # 2. 从 Buffer 取出消息
            current_chunk = []
            async with state.data_lock:
                if state.bot is None or state.event is None: continue
                if len(state.messages_chunk) == 0: continue
                current_chunk = state.messages_chunk.copy()
                state.messages_chunk.clear()

            bot_self_id = str(state.bot.self_id)
            is_echo_only = all(str(msg.user_id) == bot_self_id for msg in current_chunk)
            should_publish = not is_echo_only

            # 3. 加载 Session (加锁)
            # 只在加载数据时加锁，后续 LLM 生成时不加锁，允许查询命令插队
            async with state.session_lock:
                await state.session.load_session()

            # 4. 执行核心逻辑 (LLM 生成)
            try:
                responses = await state.session.update(
                    messages_chunk=current_chunk,
                    llm_func=lambda msg, json_mode=False: llm_response(state.client, msg, json_mode=json_mode),
                    publish=should_publish
                )

                # 5. 发送回复
                if responses:
                    total = len(responses)
                    for r_idx, response in enumerate(responses):
                        raw_content = ""
                        reply_id = None
                        if isinstance(response, str):
                            raw_content = response
                        elif isinstance(response, dict):
                            raw_content = response.get("content", "")
                            reply_id = response.get("reply_to")

                        if not raw_content: continue

                        # 智能分句发送
                        msg_parts = smart_split_text(raw_content)
                        for i, part in enumerate(msg_parts):
                            part = part.strip()
                            # 清理句末多余标点
                            if part and part[-1] in ["。", ".", "，"]:
                                part = part[:-1]

                            if not part: continue

                            msg_to_send = Message(part)
                            # 如果是回复某条特定消息，且是第一段，则带上 reply 引用
                            if reply_id and r_idx == 0 and i == 0:
                                msg_to_send.insert(0, MessageSegment.reply(int(reply_id)))

                            try:
                                result = await state.bot.send(message=msg_to_send, event=state.event)

                                if isinstance(result, dict) and "message_id" in result:
                                    msg_id = str(result["message_id"])
                                    SELF_SENT_MSG_IDS.append(msg_id)
                                    logger.debug(f"记录自身发送消息 ID: {msg_id}")

                            except ActionFailed as e:
                                if e.retcode == 1200 or "120" in str(e):
                                    logger.warning(f"风控拦截 (1200), 冷却中...")
                                    await asyncio.sleep(random.uniform(5.0, 10.0))
                                else:
                                    logger.error(f"发送失败: {e}")
                            except Exception as e:
                                logger.error(f"发送未知错误: {e}")

                            # 模拟人类打字延迟
                            if i < len(msg_parts) - 1 or r_idx < total - 1:
                                delay = 1.0 + len(part) * 0.1
                                delay = min(delay, 5.0)
                                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Processing cycle error: {e}")
                traceback.print_exc()
                continue

        except asyncio.CancelledError:
            logger.info("后台任务被取消")
            break
        except Exception as e:
            logger.error(f"Spawn loop fatal error: {e}")
            await asyncio.sleep(5.0)
