# nyaturingtest/logic.py
import asyncio
import random
import traceback

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .client import LLMClient
from .config import plugin_config, get_effective_chat_model, get_effective_feedback_model
from .image_manager import image_manager
from .mem import Message as MMessage
from .state_manager import GroupState, SELF_SENT_MSG_IDS, is_shutting_down
from .utils import smart_split_text
from .repository import SessionRepository


async def llm_response(client: LLMClient, message: str, model: str, temperature: float, json_mode: bool = False,
                       system_prompt: str | None = None, on_usage=None, **kwargs) -> str:  # <--- 添加 **kwargs
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
            on_usage=on_usage,
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

    # === 0. 预提取当前消息中的纯文本上下文 ===
    full_context_text = ""
    for seg in message:
        if seg.type == "text":
            full_context_text += seg.data.get("text", "")
    if len(full_context_text) > 200:
        full_context_text = full_context_text[:200]

    # === 消息段处理逻辑 ===

    # 定义 VLM 使用记录器
    def make_vlm_recorder(model_name: str):
        def _recorder(usage: dict):
            # 异步执行入库
            asyncio.create_task(
                SessionRepository.log_token_usage(
                    session_id=str(group_id),
                    model_name=model_name,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0)
                )
            )
        return _recorder

    async def process_segment(seg: MessageSegment) -> str:
        if seg.type == "text":
            return f"{seg.data.get('text', '')}"

        elif seg.type == "image":
            url = seg.data.get("url", "")
            file_unique = seg.data.get("file_unique", "")
            is_sticker = seg.data.get("sub_type") == 1

            # VLM disabled: skip recognition to avoid token cost / errors
            if not plugin_config.get("vlm", {}).get("enabled", True):
                return "\n[表情包]\n" if is_sticker else "\n[图片]\n"

            # Shutdown 检查：避免在关机时进入耗时的 VLM 请求
            if is_shutting_down():
                return "\n[表情包]\n" if is_sticker else "\n[图片]\n"
            
            # 使用逻辑层定义的 chat model 作为 VLM model 的近似记录（通常 VLM 和 Chat 用的是同一个 Key，或者 image_manager 内部用的就是 chat model）
            # image_manager 内部初始化时用的是 plugin_config.nyaturingtest_chat_openai_model
            # 所以这里记录为同一个模型名是准确的
            vlm_recorder = make_vlm_recorder(get_effective_chat_model())

            # 调用通用逻辑，传入提取到的上下文
            return await image_manager.resolve_image_from_url(
                url, file_unique, is_sticker, 
                context_text=full_context_text,
                on_usage=vlm_recorder
            )

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

                                # VLM disabled: skip recognition
                                if not getattr(plugin_config, "nyaturingtest_vlm_enabled", True):
                                    source_text += "\n[表情包]\n" if is_sticker_ref else "\n[图片]\n"
                                else:
                                    vlm_recorder = make_vlm_recorder(get_effective_chat_model())

                                    # Await 分析结果
                                    img_desc = await image_manager.resolve_image_from_url(
                                        img_url, img_file_unique, is_sticker_ref, 
                                        on_usage=vlm_recorder
                                    )
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
    logger.info(f"GroupState 后台任务启动: {id(state)}")
    while True:
        try:
            # 1. 等待新消息信号 (debounce 2秒)
            try:
                # 等待信号触发
                await asyncio.wait_for(state.new_message_signal.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                # 超时意味着长期无消息，检查任务是否被取消
                continue

            # 防抖逻辑：
            # 信号触发后，等待 2 秒让更多消息进入 buffer
            # 注意：在这 2 秒内如果有新消息，它们会被 append 到 chunk 中
            # 但不会再次触发 wait (因为我们还没回到 loop 顶部)
            await asyncio.sleep(2.0)
            
            # 清除信号，准备下一轮等待
            # 注意：要在取数据之前还是之后 clear？
            # 如果在 sleep 之后 clear，那么 sleep 期间进来的消息所触发的 set 会被 clear 掉
            # 但消息本身已经在 buffer 里了，会被接下来的代码取走
            # 所以这里 clear 是安全的，表示“直到此刻的消息我都处理了”
            state.new_message_signal.clear()

            # 2. 从 Buffer 取出消息
            current_chunk = []
            async with state.data_lock:
                if state.bot is None or state.event is None: 
                    # 只有当状态未完全初始化时才会发生
                    continue
                
                if len(state.messages_chunk) == 0: 
                    # 这是一个防御性检查，理论上信号触发了就该有消息
                    # 但可能被其他协程取走了（虽然目前只有一个消费者）
                    continue
                
                current_chunk = state.messages_chunk.copy()
                state.messages_chunk.clear()

            bot_self_id = str(state.bot.self_id)
            # 过滤掉只有 Bot 自己发的消息的 chunk (通常是回显)
            # 除非这些回显被某些逻辑标记为需要处理（目前没有）
            is_echo_only = all(str(msg.user_id) == bot_self_id for msg in current_chunk)
            
            # 如果全是回显，跳过生成回复，但需要更新记忆 (记录上下文)
            # 因为可能是其他进程发送的消息，或者是本进程的消息的回显(会被Session层去重)
            if is_echo_only:
                async with state.session_lock:
                    await state.session.load_session()
                    # 仅更新记忆，不触发 LLM
                    await state.session.update_without_trigger(current_chunk)
                continue
            
            # 既然已经过滤了回显，剩下的都是应该发布的消息
            should_publish = True

            # Shutdown 检查：避免在关机时进入耗时的 LLM 调用
            if is_shutting_down():
                logger.debug("Shutdown 检测，跳过 LLM 处理")
                continue

            # 3. 加载 Session (加锁)
            async with state.session_lock:
                await state.session.load_session()

            # 4. 组装 LLM 调用函数

            # --- 定义统计回调 ---
            # 使用闭包捕获 session.id
            current_session_id = str(state.session.id)
            
            def make_usage_recorder(model_name_record: str):
                def _recorder(usage: dict):
                    # 异步执行入库，不阻塞 LLM 返回
                    asyncio.create_task(
                        SessionRepository.log_token_usage(
                            session_id=current_session_id,
                            model_name=model_name_record,
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0)
                        )
                    )
                return _recorder

            # 读取当前人设角色，注入到 System Prompt 中
            current_role = state.session.role()

            # 定义一个强制的 System Prompt 用于 Roleplay，使用中文指令
            rp_system_prompt = (
                f"你是一个沉浸式的角色扮演者。你正在扮演的角色是：{current_role}。"
                "你必须时刻保持角色设定，严格遵循角色的性格描述，不得偏离。"
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
                model=get_effective_chat_model(),  # e.g. "gemini-3-flash-preview"

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
                system_prompt=rp_system_prompt,
                on_usage=make_usage_recorder(get_effective_chat_model())
            )

            # 定义 Feedback 专用的 System Prompt
            feedback_system_prompt = (
                "你是一个精确的对话情感分析器。"
                "你的任务是观察群聊消息，分析角色的情绪变化，并以 JSON 格式输出分析结果。"
                "你的输出必须包含 new_emotion 对象（含 valence、arousal、dominance 三个浮点数字段）。"
                "最终输出必须是合法的 JSON 格式，不要输出任何其他内容。"
            )

            # Feedback 函数：温度极低，保证逻辑分析准确
            feedback_func = lambda msg, json_mode=False: llm_response(
                state.feedback_client,
                msg,
                model=get_effective_feedback_model(),
                temperature=0.1,
                json_mode=json_mode,
                on_usage=make_usage_recorder(get_effective_feedback_model()),
                system_prompt=feedback_system_prompt
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
                            # 删除句末标点，保留逗号等句中标点，但保留问号以维持句意
                            if part and part[-1] in ["。", ".", "！", "!"]:
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
                                    
                                    # 主动写入记忆，确保"知道自己上一句说了什么"
                                    # 注意：这里需要 msg_to_send 的纯文本内容
                                    # 简单提取文本部分
                                    sent_content = msg_to_send.extract_plain_text()
                                    if not sent_content and len(msg_to_send) > 0:
                                        # 如果是图片或其他，尝试简单转义
                                        sent_content = str(msg_to_send)
                                    
                                    async with state.session_lock:
                                        await state.session.append_self_message(sent_content, msg_id)

                            except ActionFailed as e:
                                if getattr(e, "retcode", 0) == 1200 or "120" in str(e):
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
            logger.info(f"后台任务被取消: {id(state)}")
            break
        except Exception as e:
            logger.error(f"Spawn loop fatal error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5.0)
