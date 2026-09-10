import asyncio
import hashlib
import random
import re
import time
import traceback
from dataclasses import dataclass

from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from ..memory.image import fetch_image_input
from ..memory.short_term import Message as MMessage
from .llm import VisionInput, build_turn_calls
from .metrics import log_event, metrics
from .orchestrator import ConversationOrchestrator
from .state_manager import SELF_SENT_MSG_IDS, GroupState, is_shutting_down

DEBOUNCE_SECONDS = 2.0
QUEUE_MAX_SIZE = 200
MAX_REPLY_MESSAGES = 2

_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？!?~\n])\s*|(?<!\.)\.(?!\.)(?=\s|$|[\u4e00-\u9fff])\s*"
)
_SINGLE_TRAILING_PERIOD = re.compile(r"(?<!\.)\.$")


def _normalize_send_part(text: str) -> str:
    part = text.strip()
    if not part:
        return ""
    part = part.rstrip("。")
    part = _SINGLE_TRAILING_PERIOD.sub("", part)
    return part.strip()


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SPLIT_PATTERN.split(text) if part.strip()]
    return parts or [text]


def build_send_parts(text: str, max_messages: int = MAX_REPLY_MESSAGES) -> list[str]:
    parts = [_normalize_send_part(part) for part in _split_text(text)]
    parts = [part for part in parts if part]
    if len(parts) > max_messages:
        return [*parts[: max_messages - 1], " ".join(parts[max_messages - 1 :])]
    return parts


@dataclass(frozen=True)
class InboxBatch:
    messages: list
    bot: object
    event: object


async def next_inbox_batch(
    state: GroupState,
    *,
    debounce_seconds: float,
    idle_timeout: float = 20.0,
) -> InboxBatch | None:
    """等到静默窗口结束，再一次性取走积压的消息。"""

    try:
        await asyncio.wait_for(
            state.new_message_signal.wait(),
            timeout=idle_timeout,
        )
    except asyncio.TimeoutError:
        return None

    await asyncio.sleep(debounce_seconds)
    state.new_message_signal.clear()
    async with state.data_lock:
        if state.bot is None or state.event is None or not state.messages_chunk:
            return None
        batch = InboxBatch(
            messages=list(state.messages_chunk),
            bot=state.bot,
            event=state.event,
        )
        state.messages_chunk.clear()
        return batch


def _response_content(response) -> tuple[str, object | None]:
    if isinstance(response, str):
        return response, None
    if isinstance(response, dict):
        return (
            str(response.get("content") or ""),
            response.get("target_id") or response.get("reply_to"),
        )
    return "", None


def _delay_seconds(part: str) -> float:
    return min(1.0 + len(part) * 0.1, 5.0)


async def send_one(
    self_sent_ids,
    *,
    state,
    bot,
    event,
    message,
    generation: int,
) -> bool:
    try:
        result = await bot.send(message=message, event=event)
        sent_content = message.extract_plain_text()
        if not sent_content and len(message) > 0:
            sent_content = str(message)
        message_id = ""
        if isinstance(result, dict) and "message_id" in result:
            message_id = str(result["message_id"])
            self_sent_ids.append(message_id)

        if state.session.is_generation_stale(generation):
            state.session._log_stale_generation("append_self_message", generation)
            return True
        async with state.session_lock:
            if state.session.is_generation_stale(generation):
                state.session._log_stale_generation(
                    "append_self_message_locked",
                    generation,
                )
                return True
            await state.session.append_self_message(
                sent_content,
                message_id,
                str(bot.self_id),
            )
        return True
    except ActionFailed as e:
        if getattr(e, "retcode", 0) == 1200 or "120" in str(e):
            logger.warning("风控拦截 (1200), 冷却中...")
            await asyncio.sleep(random.uniform(5.0, 10.0))
        else:
            logger.error(f"发送失败: {e}")
    except Exception as e:
        logger.error(f"发送未知错误: {e}")
    return False


async def dispatch_replies(
    self_sent_ids,
    *,
    state,
    responses: list,
    bot,
    event,
    generation: int,
) -> int:
    if not responses:
        return 0
    if state.session.is_generation_stale(generation):
        state.session._log_stale_generation("pre_send", generation)
        return 0

    total = len(responses)
    sent_count = 0
    for response_index, response in enumerate(responses):
        if sent_count >= MAX_REPLY_MESSAGES:
            break
        raw_content, reply_id = _response_content(response)
        if not raw_content:
            continue
        parts = build_send_parts(raw_content, MAX_REPLY_MESSAGES - sent_count)
        for part_index, part in enumerate(parts):
            if sent_count >= MAX_REPLY_MESSAGES:
                break
            if state.session.is_generation_stale(generation):
                state.session._log_stale_generation("send_loop", generation)
                break
            part = part.strip()
            if not part:
                continue
            message = Message(part)
            if reply_id and response_index == 0 and part_index == 0:
                try:
                    message.insert(0, MessageSegment.reply(int(reply_id)))
                except ValueError:
                    logger.warning(f"引用ID无效: {reply_id}")

            sent = await send_one(
                self_sent_ids,
                state=state,
                bot=bot,
                event=event,
                message=message,
                generation=generation,
            )
            if sent:
                sent_count += 1

            has_more = part_index < len(parts) - 1 or response_index < total - 1
            if has_more:
                await asyncio.sleep(_delay_seconds(part))

    if sent_count:
        state.session._schedule_save_session()
    return sent_count


def _is_sticker_segment_data(data: dict) -> bool:
    return str(data.get("sub_type", "")) == "1"


def _build_image_ref(
    message_scope: str,
    source: str,
    segment_index: int,
    identifier: str,
) -> str:
    digest = hashlib.sha1(str(identifier or "").encode("utf-8", "ignore")).hexdigest()[
        :12
    ]
    scope_digest = hashlib.sha1(
        str(message_scope or "").encode("utf-8", "ignore")
    ).hexdigest()[:10]
    return f"{scope_digest}:{source}:{segment_index}:{digest}"


def _filter_local_self_echoes(
    messages: list[MMessage],
    bot_self_id: str,
    self_sent_ids,
) -> tuple[list[MMessage], list[MMessage]]:
    """按本地已发送消息 ID 分离自身回显。"""

    filtered, local_echoes = [], []
    for msg in messages:
        if (
            msg.id
            and str(msg.user_id) == str(bot_self_id)
            and str(msg.id) in self_sent_ids
        ):
            local_echoes.append(msg)
        else:
            filtered.append(msg)
    return filtered, local_echoes


async def llm_response(
    client,
    message: str,
    model: str,
    temperature: float | None = None,
    json_mode: bool = False,
    system_prompt: str | None = None,
    on_usage=None,
    images: list[VisionInput] | None = None,
    **kwargs,
) -> str:
    """封装 LLM 调用并记录指标；失败返回空串。"""

    started_at = time.perf_counter()
    try:
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        result = await client.generate(
            prompt=message,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            on_usage=on_usage,
            images=images,
            **kwargs,
        )
        if result:
            metrics.llm_success += 1
            log_event(
                "llm_success",
                provider=client.provider,
                model=model,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                tokens="recorded_by_usage_callback",
                decision="content",
            )
            return result
        metrics.llm_failure += 1
        log_event(
            "llm_failure",
            provider=client.provider,
            model=model,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            decision="empty",
        )
        return ""
    except Exception as e:
        metrics.llm_failure += 1
        log_event(
            "llm_error",
            provider=client.provider,
            model=model,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            decision="exception",
        )
        logger.error(f"LLM Error [{model}]: {e}")
        return "Error occurred."


async def message2BotMessage(
    bot_name: str,
    group_id: int,
    message: Message,
    bot: Bot,
    *,
    message_scope: str = "",
) -> tuple[str, list[VisionInput]]:
    """把 OneBot 消息转成可读文本，并收集原生图片输入。"""

    async def process_segment(
        seg: MessageSegment,
        segment_index: int,
    ) -> tuple[str, list[VisionInput]]:
        if seg.type == "text":
            return (f"{seg.data.get('text', '')}", [])

        if seg.type == "image":
            url = seg.data.get("url", "")
            file_unique = seg.data.get("file_unique", "")
            text, vision_input = await fetch_image_input(
                url,
                file_unique,
                is_sticker=_is_sticker_segment_data(seg.data),
                ref_id=_build_image_ref(
                    message_scope, "primary", segment_index, file_unique or url
                ),
                source="primary",
            )
            return (text, [vision_input] if vision_input else [])

        if seg.type == "at":
            target = seg.data.get("qq")
            if not target:
                return ("", [])
            if target == str(bot.self_id):
                return (f" @{bot_name} ", [])
            try:
                user_info = await bot.get_group_member_info(
                    group_id=group_id, user_id=int(target)
                )
                nickname = (
                    user_info.get("card") or user_info.get("nickname") or str(target)
                )
                return (f" @{nickname} ", [])
            except Exception:
                return (f" @{target} ", [])

        if seg.type == "reply":
            reply_id = seg.data.get("id")
            if not reply_id:
                return ("", [])
            try:
                source_msg = await bot.get_msg(message_id=int(reply_id))
                sender = source_msg.get("sender", {}).get("nickname", "未知")
                content_data = source_msg.get("message", [])
                source_text = ""
                image_inputs: list[VisionInput] = []

                if isinstance(content_data, str):
                    source_text = content_data
                elif isinstance(content_data, list):
                    for reply_index, segment in enumerate(content_data):
                        msg_type = segment.get("type")
                        data = segment.get("data", {})
                        if msg_type == "text":
                            source_text += data.get("text", "")
                        elif msg_type == "image":
                            img_url = data.get("url", "")
                            img_file_unique = data.get("file_unique", "")
                            img_text, vision_input = await fetch_image_input(
                                img_url,
                                img_file_unique,
                                is_sticker=_is_sticker_segment_data(data),
                                ref_id=_build_image_ref(
                                    message_scope,
                                    "referenced",
                                    reply_index,
                                    img_file_unique or img_url,
                                ),
                                source="referenced",
                            )
                            source_text += img_text
                            if vision_input:
                                image_inputs.append(vision_input)
                        elif msg_type == "face":
                            source_text += "[表情]"

                if len(source_text) > 800:
                    source_text = source_text[:800] + "..."
                return (f' [回复 {sender}: "{source_text}"] ', image_inputs)
            except Exception as e:
                logger.warning(f"获取回复内容失败: {e}")
                return (" [回复] ", [])

        return ("", [])

    tasks = [process_segment(seg, index) for index, seg in enumerate(message)]
    results = await asyncio.gather(*tasks)

    content = "".join(result[0] for result in results).strip()
    image_inputs = [item for result in results for item in result[1]]
    return (content, image_inputs)


async def _process_inbox_batch(state: GroupState, batch: InboxBatch) -> None:
    bot_self_id = str(batch.bot.self_id)
    current_chunk, local_echoes = _filter_local_self_echoes(
        batch.messages,
        bot_self_id,
        SELF_SENT_MSG_IDS,
    )
    if local_echoes:
        logger.debug(f"过滤本机自身回显消息 {len(local_echoes)} 条")
    if not current_chunk:
        return

    if all(str(message.user_id) == bot_self_id for message in current_chunk):
        # 自身账号发出的消息已由 append_self_message 写入记忆，这里直接丢弃
        return
    if is_shutting_down():
        return

    async with state.session_lock:
        await state.session.load_session()
        generation = state.session.state.generation
        session_id = str(state.session.id)

    images = [item for message in current_chunk for item in message.image_inputs]
    chat_call, feedback_call = build_turn_calls(
        llm_response,
        state=state,
        session_id=session_id,
        chat_images=images,
        feedback_images=images,
    )
    try:
        responses = await ConversationOrchestrator(state.session).process_chunk(
            messages_chunk=current_chunk,
            chat_llm_func=chat_call,
            feedback_llm_func=feedback_call,
            publish=True,
            expected_generation=generation,
        )
    finally:
        for message in current_chunk:
            message.image_inputs.clear()

    await dispatch_replies(
        SELF_SENT_MSG_IDS,
        state=state,
        responses=responses or [],
        bot=batch.bot,
        event=batch.event,
        generation=generation,
    )


async def spawn_state(state: GroupState):
    """Small worker boundary: debounce, invoke one turn, contain failures."""

    logger.info(f"GroupState 后台任务启动: {id(state)}")
    while True:
        try:
            batch = await next_inbox_batch(state, debounce_seconds=DEBOUNCE_SECONDS)
            if batch is None:
                continue
            await _process_inbox_batch(state, batch)
        except asyncio.CancelledError:
            logger.info(f"后台任务被取消: {id(state)}")
            break
        except Exception as e:
            logger.error(f"Spawn loop error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5.0)
