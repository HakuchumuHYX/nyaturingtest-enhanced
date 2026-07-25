# nyaturingtest/logic.py
import asyncio
import hashlib
import time
import traceback
from collections.abc import Collection

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment

from ..llm.client import LLMClient
from ..config import (
    get_effective_chat_model,
    get_effective_vlm_model,
    get_runtime_settings,
)
from .. import config as config_module
try:
    from ..llm.vision import VisionInput
except ImportError:
    VisionInput = object
from ..memory.image import image_manager
from ..memory.image_schema import merge_segment_metas
from ..memory.short_term import Message as MMessage
from .metrics import metrics
from .debounced_inbox import DebouncedInbox
from .reply_dispatcher import ReplyDispatcher
from .structured_log import log_event
from .state_manager import GroupState, SELF_SENT_MSG_IDS, is_shutting_down
from .turn_call_factory import TurnCallFactory
from .usage import make_usage_recorder


get_vision_settings = getattr(
    config_module,
    "get_vision_settings",
    lambda endpoint_name: {
        "enabled": False,
        "detail": "low" if endpoint_name == "feedback" else "auto",
    },
)
native_vision_enabled = getattr(config_module, "native_vision_enabled", lambda: False)
should_use_standalone_vlm = getattr(
    config_module,
    "should_use_standalone_vlm",
    lambda: True,
)


def _is_sticker_segment_data(data: dict) -> bool:
    return str(data.get("sub_type", "")) == "1"


def _image_placeholder(is_sticker: bool) -> str:
    return "\n[表情包]\n" if is_sticker else "\n[图片]\n"


def _should_resolve_image(resolve_images: bool) -> bool:
    return (
        resolve_images
        and should_use_standalone_vlm()
        and not is_shutting_down()
    )


def _should_attach_native_image(attach_native_images: bool) -> bool:
    return attach_native_images and native_vision_enabled() and not is_shutting_down()


def _build_image_ref(
    message_scope: str,
    source: str,
    segment_index: int,
    identifier: str,
) -> str:
    digest = hashlib.sha1(str(identifier or "").encode("utf-8", "ignore")).hexdigest()[:12]
    scope_digest = hashlib.sha1(
        str(message_scope or "").encode("utf-8", "ignore")
    ).hexdigest()[:10]
    return f"{scope_digest}:{source}:{segment_index}:{digest}"


def _is_local_self_echo(msg: MMessage, bot_self_id: str, self_sent_ids: Collection[str]) -> bool:
    msg_id = str(getattr(msg, "id", "") or "")
    return bool(
        msg_id
        and str(getattr(msg, "user_id", "")) == str(bot_self_id)
        and msg_id in self_sent_ids
    )


def _filter_local_self_echoes(
        messages: list[MMessage],
        bot_self_id: str,
        self_sent_ids: Collection[str] | None = None,
) -> tuple[list[MMessage], list[MMessage]]:
    if self_sent_ids is None:
        self_sent_ids = set(SELF_SENT_MSG_IDS)
    filtered = []
    local_echoes = []
    for msg in messages:
        if _is_local_self_echo(msg, bot_self_id, self_sent_ids):
            local_echoes.append(msg)
        else:
            filtered.append(msg)
    return filtered, local_echoes


async def llm_response(
    client: LLMClient,
    message: str,
    model: str,
    temperature: float | None = None,
    json_mode: bool = False,
    system_prompt: str | None = None,
    on_usage=None,
    images: list[VisionInput] | None = None,
    **kwargs,
) -> str:
    """
    封装 LLM 调用，支持高级参数透传
    """
    started_at = time.perf_counter()
    try:
        # 如果是 JSON 模式，合并到 kwargs
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        result = await client.generate(
            prompt=message,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            on_usage=on_usage,
            images=images,
            **kwargs
        )
        if result.content:
            metrics.llm_success += 1
            log_event(
                "llm_success",
                provider=getattr(client, "provider", ""),
                model=model,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                tokens="recorded_by_usage_callback",
                decision="content",
            )
            return result.content
        metrics.llm_failure += 1
        log_event(
            "llm_failure",
            provider=getattr(client, "provider", ""),
            model=model,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            decision="empty",
        )
        return ""
    except Exception as e:
        metrics.llm_failure += 1
        log_event(
            "llm_error",
            provider=getattr(client, "provider", ""),
            model=model,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            decision="exception",
        )
        logger.error(f"LLM Error [{model}]: {e}")
        return "Error occurred."


def _vision_inputs_for_endpoint(
    messages: list[MMessage],
    endpoint_name: str,
) -> list[VisionInput]:
    settings = get_vision_settings(endpoint_name)
    if not settings["enabled"]:
        return []
    detail = settings["detail"]
    result = []
    seen_refs = set()
    for message in messages:
        for image_input in getattr(message, "image_inputs", []) or []:
            if not isinstance(image_input, VisionInput):
                continue
            if image_input.ref_id in seen_refs:
                continue
            seen_refs.add(image_input.ref_id)
            result.append(image_input.with_detail(detail))
    return result


async def message2BotMessage(
    bot_name: str,
    group_id: int,
    message: Message,
    bot: Bot,
    resolve_images: bool = True,
    *,
    attach_native_images: bool = False,
    image_inputs_out: list[VisionInput] | None = None,
    conversation_context: str = "",
    message_scope: str = "",
) -> tuple[str, dict | None]:
    """
    将 OneBot 消息转换为 Bot 可读文本，并附带图片的结构化元数据。
    返回 (text, image_meta)：
      - text: 拼接后的可读文本（与历史行为一致，含图片管道标签）
      - image_meta: 图片结构化观测，None 表示无图片/无结构信息。
        多图时结构为 {"primary": <首张非空 meta>, "referenced": [<引用消息图片 meta 列表>]}。
    支持解析引用消息(Reply)中的图片内容。
    """

    # === 0. 预提取当前消息中的纯文本上下文 ===
    full_context_text = str(conversation_context or "").strip()
    for seg in message:
        if seg.type == "text":
            text = seg.data.get("text", "")
            full_context_text = f"{full_context_text}\n当前消息：{text}".strip()
    if len(full_context_text) > 800:
        full_context_text = full_context_text[-800:]

    # === 消息段处理逻辑 ===
    # 每段返回 (text, meta)；meta 为该段产出的图片结构化观测（None 表示无）

    async def process_segment(
        seg: MessageSegment,
        segment_index: int,
    ) -> tuple[str, dict | None, list[VisionInput]]:
        if seg.type == "text":
            return (f"{seg.data.get('text', '')}", None, [])

        elif seg.type == "image":
            url = seg.data.get("url", "")
            file_unique = seg.data.get("file_unique", "")
            is_sticker = _is_sticker_segment_data(seg.data)
            should_describe = _should_resolve_image(resolve_images)
            should_attach = _should_attach_native_image(attach_native_images)

            if not should_describe and not should_attach:
                return (_image_placeholder(is_sticker), None, [])

            # 获取 VLM 的真实模型名称以准确记录 token 消耗
            # 兼容老配置，如果 vlm 未指定模型，则退而求其次使用 chat model
            vlm_model_name = get_effective_vlm_model() or get_effective_chat_model()
            vlm_recorder = make_usage_recorder(str(group_id), vlm_model_name)
            image_ref = _build_image_ref(
                message_scope,
                "primary",
                segment_index,
                file_unique or url,
            )

            # 调用通用逻辑，传入提取到的上下文
            text, meta, vision_input = await image_manager.resolve_image_from_url(
                url, file_unique, is_sticker,
                context_text=full_context_text,
                on_usage=vlm_recorder,
                describe=should_describe,
                include_native=should_attach,
                ref_id=image_ref,
                source="primary",
            )
            if meta:
                meta = dict(meta)
                meta["image_ref"] = image_ref
            return (text, meta, [vision_input] if vision_input else [])

        elif seg.type == "at":
            id = seg.data.get("qq")
            if not id: return ("", None, [])
            if id == str(bot.self_id):
                return (f" @{bot_name} ", None, [])
            else:
                try:
                    user_info = await bot.get_group_member_info(group_id=group_id, user_id=int(id))
                    nickname = user_info.get("card") or user_info.get("nickname") or str(id)
                    return (f" @{nickname} ", None, [])
                except Exception:
                    return (f" @{id} ", None, [])

        elif seg.type == "reply":
            reply_id = seg.data.get("id")
            if reply_id:
                try:
                    source_msg = await bot.get_msg(message_id=int(reply_id))
                    sender = source_msg.get("sender", {}).get("nickname", "未知")

                    content_data = source_msg.get("message", [])
                    source_text = ""
                    referenced_metas: list[dict] = []  # 引用消息里图片的结构化观测
                    referenced_inputs: list[VisionInput] = []

                    # 统一转为列表处理
                    if isinstance(content_data, str):
                        # 如果是纯文本(这种情况较少，通常是列表)，直接当文本
                        source_text = content_data
                    elif isinstance(content_data, list):
                        for reply_segment_index, s in enumerate(content_data):
                            msg_type = s.get("type")
                            data = s.get("data", {})

                            if msg_type == "text":
                                source_text += data.get("text", "")

                            elif msg_type == "image":
                                # 对引用消息里的图片也进行分析
                                img_url = data.get("url", "")
                                img_file_unique = data.get("file_unique", "")
                                # 引用里的图片通常不易判断是否为表情包，默认 False，或者尝试获取 sub_type
                                is_sticker_ref = _is_sticker_segment_data(data)

                                should_describe = _should_resolve_image(resolve_images)
                                should_attach = _should_attach_native_image(attach_native_images)
                                if not should_describe and not should_attach:
                                    source_text += _image_placeholder(is_sticker_ref)
                                    continue

                                vlm_model_name_ref = get_effective_vlm_model() or get_effective_chat_model()
                                vlm_recorder = make_usage_recorder(str(group_id), vlm_model_name_ref)

                                # Await 分析结果
                                image_ref = _build_image_ref(
                                    message_scope,
                                    "referenced",
                                    reply_segment_index,
                                    img_file_unique or img_url,
                                )
                                img_text, img_meta, vision_input = await image_manager.resolve_image_from_url(
                                    img_url, img_file_unique, is_sticker_ref,
                                    context_text=full_context_text,
                                    on_usage=vlm_recorder,
                                    describe=should_describe,
                                    include_native=should_attach,
                                    ref_id=image_ref,
                                    source="referenced",
                                )
                                source_text += img_text
                                if img_meta:
                                    img_meta = dict(img_meta)
                                    img_meta["image_ref"] = image_ref
                                    referenced_metas.append(img_meta)
                                if vision_input:
                                    referenced_inputs.append(vision_input)

                            elif msg_type == "face":
                                # 简单处理 QQ 表情
                                source_text += "[表情]"

                    # 截断过长文本 (图片描述通常比较长，这里稍微放宽一点限制，或者只截断纯文本部分)
                    # 简单策略：如果总长度超过 200 字符，截断
                    if len(source_text) > 800:
                        source_text = source_text[:800] + "..."

                    ref_meta: dict | None = {"referenced": referenced_metas} if referenced_metas else None
                    return (f" [回复 {sender}: \"{source_text}\"] ", ref_meta, referenced_inputs)
                except Exception as e:
                    logger.warning(f"获取回复内容失败: {e}")
                    return (" [回复] ", None, [])
            return ("", None, [])

        return ("", None, [])

    tasks = [process_segment(seg, index) for index, seg in enumerate(message)]
    results = await asyncio.gather(*tasks)

    # 聚合文本与元数据
    content = "".join(r[0] for r in results).strip()
    image_meta = merge_segment_metas([r[1] for r in results])
    if image_inputs_out is not None:
        for result in results:
            image_inputs_out.extend(result[2])

    return (content, image_meta)



async def _process_inbox_batch(
    state: GroupState,
    batch,
    *,
    call_factory: TurnCallFactory,
    dispatcher: ReplyDispatcher,
    runtime_settings: dict,
) -> None:
    bot_self_id = str(batch.bot.self_id)
    current_chunk, local_echoes = _filter_local_self_echoes(
        batch.messages,
        bot_self_id,
        set(SELF_SENT_MSG_IDS),
    )
    if local_echoes:
        logger.debug(f"过滤本机自身回显消息 {len(local_echoes)} 条")
    if not current_chunk:
        return

    if all(str(message.user_id) == bot_self_id for message in current_chunk):
        async with state.session_lock:
            await state.session.load_session()
            await state.session.update_without_trigger(current_chunk)
        return
    if is_shutting_down():
        return

    async with state.session_lock:
        await state.session.load_session()
        generation = getattr(state.session, "generation", 0)
        session_id = str(state.session.id)

    calls = call_factory.build(
        state=state,
        session_id=session_id,
        chat_images=_vision_inputs_for_endpoint(current_chunk, "chat"),
        feedback_images=_vision_inputs_for_endpoint(current_chunk, "feedback"),
    )
    try:
        responses = await state.session.update(
            messages_chunk=current_chunk,
            chat_llm_func=calls.chat,
            feedback_llm_func=calls.feedback,
            publish=True,
            expected_generation=generation,
        )
    finally:
        for message in current_chunk:
            image_inputs = getattr(message, "image_inputs", None)
            if isinstance(image_inputs, list):
                image_inputs.clear()

    await dispatcher.dispatch(
        state=state,
        responses=responses or [],
        bot=batch.bot,
        event=batch.event,
        generation=generation,
        runtime_settings=runtime_settings,
    )


async def spawn_state(state: GroupState):
    """Small worker boundary: debounce, invoke one turn, contain failures."""

    logger.info(f"GroupState 后台任务启动: {id(state)}")
    call_factory = TurnCallFactory(llm_response)
    dispatcher = ReplyDispatcher(SELF_SENT_MSG_IDS)

    while True:
        try:
            runtime_settings = get_runtime_settings()
            inbox = DebouncedInbox(
                state,
                debounce_seconds=runtime_settings["debounce_seconds"],
            )
            batch = await inbox.next_batch()
            if batch is None:
                continue
            await _process_inbox_batch(
                state,
                batch,
                call_factory=call_factory,
                dispatcher=dispatcher,
                runtime_settings=runtime_settings,
            )
        except asyncio.CancelledError:
            logger.info(f"后台任务被取消: {id(state)}")
            break
        except Exception as e:
            logger.error(f"Spawn loop error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5.0)
