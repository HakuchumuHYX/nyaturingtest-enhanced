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


async def llm_response(client: LLMClient, message: str, model: str, temperature: float, json_mode: bool = False,
                       system_prompt: str = None, **kwargs) -> str:  # <--- 添加 **kwargs
    """
    封装 LLM 调用，支持高级参数透传
    """
    try:
        # 如果是 JSON 模式，合并到 kwargs
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        result = await client.generate_response(
            prompt=message,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs
        )
        return result if result else ""
    except Exception as e:
        logger.error(f"LLM Error [{model}]: {e}")
        return "Error occurred."


async def message2BotMessage(bot_name: str, group_id: int, message: Message, bot: Bot) -> str:
    """
    将 OneBot 消息转换为 Bot 可读文本
    支持解析引用消息(Reply)中的图片内容
    """

    # === 提取通用的图片解析逻辑 ===
    async def resolve_image(url: str, file_unique: str, is_sticker: bool) -> str:
        """内部函数：下载并分析图片，返回描述文本"""
        if not url:
            return "[无效图片]"

        async with _IMG_SEMAPHORE:
            try:
                # 1. 尝试从内存缓存获取
                if file_unique:
                    cached_desc = image_manager.get_from_cache(file_unique)
                    if cached_desc:
                        if is_sticker:
                            return f"\n[表情包] [情感:{cached_desc.emotion}] [内容:{cached_desc.description}]\n"
                        else:
                            return f"\n[图片] {cached_desc.description}\n"

                # 2. 准备文件缓存路径
                cache_path = IMAGE_CACHE_DIR.joinpath("raw")
                cache_path.mkdir(parents=True, exist_ok=True)

                # 尝试从 URL 或 file_unique 提取文件名
                key = None
                key_match = re.search(r"[?&]fileid=([a-zA-Z0-9_-]+)", url)
                if key_match:
                    key = key_match.group(1)
                elif file_unique:
                    key = file_unique

                image_bytes = None

                # 3. 尝试读取本地文件缓存
                if key and cache_path.joinpath(key).exists():
                    async with await anyio.open_file(cache_path.joinpath(key), "rb") as f:
                        image_bytes = await f.read()
                else:
                    # 4. 下载图片
                    client = get_http_client()
                    for _ in range(2):  # 重试2次
                        try:
                            resp = await client.get(url, timeout=10.0)  # 稍微增加超时
                            resp.raise_for_status()
                            image_bytes = resp.content
                            break
                        except Exception:
                            await asyncio.sleep(0.5)

                    # 下载成功后写入缓存
                    if image_bytes and key:
                        try:
                            async with await anyio.open_file(cache_path.joinpath(key), "wb") as f:
                                await f.write(image_bytes)
                        except Exception as e:
                            logger.warning(f"写入图片缓存失败: {e}")

                if not image_bytes:
                    return "\n[图片下载失败]\n"

                # 5. 调用 VLM 进行识别
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
                logger.error(f"Image resolve error: {e}")
                return "\n[图片处理出错]\n"

    # === 消息段处理逻辑 ===
    async def process_segment(seg: MessageSegment) -> str:
        if seg.type == "text":
            return f"{seg.data.get('text', '')}"

        elif seg.type == "image":
            url = seg.data.get("url", "")
            file_unique = seg.data.get("file_unique", "")
            is_sticker = seg.data.get("sub_type") == 1
            # 调用通用逻辑
            return await resolve_image(url, file_unique, is_sticker)

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

                    content_data = source_msg.get("message", [])
                    source_text = ""

                    # 统一转为列表处理
                    if isinstance(content_data, str):
                        # 如果是纯文本(这种情况较少，通常是列表)，直接当文本
                        source_text = content_data
                    elif isinstance(content_data, list):
                        for s in content_data:
                            msg_type = s.get("type")
                            data = s.get("data", {})

                            if msg_type == "text":
                                source_text += data.get("text", "")

                            elif msg_type == "image":
                                # 对引用消息里的图片也进行分析
                                img_url = data.get("url", "")
                                img_file_unique = data.get("file_unique", "")
                                # 引用里的图片通常不易判断是否为表情包，默认 False，或者尝试获取 sub_type
                                is_sticker_ref = str(data.get("sub_type", "")) == "1"

                                # Await 分析结果
                                img_desc = await resolve_image(img_url, img_file_unique, is_sticker_ref)
                                source_text += img_desc

                            elif msg_type == "face":
                                # 简单处理 QQ 表情
                                source_text += "[表情]"

                    # 截断过长文本 (图片描述通常比较长，这里稍微放宽一点限制，或者只截断纯文本部分)
                    # 简单策略：如果总长度超过 200 字符，截断
                    if len(source_text) > 200:
                        source_text = source_text[:200] + "..."

                    return f" [回复 {sender}: \"{source_text}\"] "
                except Exception as e:
                    logger.warning(f"获取回复内容失败: {e}")
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
            async with state.session_lock:
                await state.session.load_session()

            # 4. 组装 LLM 调用函数

            # 定义一个强制的 System Prompt 用于 Roleplay，使用中文指令
            rp_system_prompt = (
                "你是一个沉浸式的角色扮演者。你必须时刻保持角色设定。"
                "请使用中文进行思考和回答（除非人设要求使用其他语言）。"
                "最终输出必须是合法的 JSON 格式。"
            )

            gemini_extra_body = {
                "google": {
                    "model_safety_settings": {
                        # 必须全关，否则 RP 中稍微激动一点就会被 Google 掐断
                        "enabled": False
                    },
                    "thinking_config": {
                        # Gemini 3 Flash 特性：开启轻量思考，提升 RP 逻辑感
                        "include_thoughts": False,  # 只返回结果，不返回思考过程文本
                        "thinking_level": "low"  # 档位：minimal/low/medium/high
                    }
                }
            }

            # Chat 函数
            chat_func = lambda msg, json_mode=False: llm_response(
                state.client, msg,
                model=plugin_config.nyaturingtest_chat_openai_model,  # 确保 config 里填的是 "gemini-3-flash-preview"

                # Gemini 3 推荐保持 1.0，不要降温
                temperature=1.05,

                # OpenAI SDK 标准参数
                top_p=0.95,

                # 透传 Gemini 独有参数
                extra_body={
                    "top_k": 64,  # 锁住发散边界
                    **gemini_extra_body
                },

                json_mode=json_mode,
                system_prompt=rp_system_prompt
            )

            # Feedback 函数：温度极低，保证逻辑分析准确
            feedback_func = lambda msg, json_mode=False: llm_response(
                state.feedback_client,
                msg,
                model=plugin_config.nyaturingtest_feedback_openai_model,
                temperature=0.1,
                json_mode=json_mode
            )

            # 5. 执行核心逻辑 (LLM 生成)
            try:
                responses = await state.session.update(
                    messages_chunk=current_chunk,
                    chat_llm_func=chat_func,  # 传入 Chat 函数
                    feedback_llm_func=feedback_func,  # 传入 Feedback 函数
                    publish=should_publish
                )

                # 6. 发送回复 (保持不变)
                if responses:
                    total = len(responses)
                    for r_idx, response in enumerate(responses):
                        raw_content = ""
                        reply_id = None
                        if isinstance(response, str):
                            raw_content = response
                        elif isinstance(response, dict):
                            raw_content = response.get("content", "")
                            reply_id = response.get("target_id") or response.get("reply_to")

                        if not raw_content: continue

                        # 智能分句发送
                        msg_parts = smart_split_text(raw_content)
                        for i, part in enumerate(msg_parts):
                            part = part.strip()
                            if part and part[-1] in ["。", ".", "，"]:
                                part = part[:-1]

                            if not part: continue

                            msg_to_send = Message(part)
                            if reply_id and r_idx == 0 and i == 0:
                                try:
                                    msg_to_send.insert(0, MessageSegment.reply(int(reply_id)))
                                    logger.debug(f"添加引用回复: {reply_id}")
                                except ValueError:
                                    logger.warning(f"引用ID无效: {reply_id}")

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
