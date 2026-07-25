import asyncio
import hashlib
import random
import time

from nonebot import logger
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from .message_sender import build_send_parts


def build_self_message_id(content: str) -> str:
    digest = hashlib.sha1((content or "").encode("utf-8", "ignore")).hexdigest()[:12]
    return f"self:{time.time_ns()}:{digest}"


class ReplyDispatcher:
    def __init__(self, self_sent_message_ids):
        self._self_sent_message_ids = self_sent_message_ids

    async def dispatch(
        self,
        *,
        state,
        responses: list,
        bot,
        event,
        generation: int,
        runtime_settings: dict,
    ) -> int:
        if not responses:
            return 0
        if state.session.is_generation_stale(generation):
            state.session._log_stale_generation("pre_send", generation)
            return 0

        total = len(responses)
        sent_count = 0
        max_messages = max(0, int(runtime_settings["max_reply_messages"]))
        for response_index, response in enumerate(responses):
            if sent_count >= max_messages:
                break
            raw_content, reply_id = self._response_content(response)
            if not raw_content:
                continue
            parts = build_send_parts(
                raw_content,
                max_messages=max_messages - sent_count,
                strategy=runtime_settings["send_strategy"],
            )
            for part_index, part in enumerate(parts):
                if sent_count >= max_messages:
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

                sent = await self._send_one(
                    state=state,
                    bot=bot,
                    event=event,
                    message=message,
                    generation=generation,
                )
                if sent:
                    sent_count += 1

                has_more = (
                    part_index < len(parts) - 1
                    or response_index < total - 1
                )
                if has_more:
                    await asyncio.sleep(
                        self._delay_seconds(part, runtime_settings)
                    )

        if sent_count:
            schedule_save = getattr(state.session, "_schedule_save_session", None)
            if schedule_save is not None:
                schedule_save()
        return sent_count

    @staticmethod
    def _response_content(response) -> tuple[str, object | None]:
        if isinstance(response, str):
            return response, None
        if isinstance(response, dict):
            return (
                str(response.get("content") or ""),
                response.get("target_id") or response.get("reply_to"),
            )
        return "", None

    async def _send_one(
        self,
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
            if isinstance(result, dict) and "message_id" in result:
                message_id = str(result["message_id"])
                self._self_sent_message_ids.append(message_id)
            else:
                message_id = build_self_message_id(sent_content)

            if state.session.is_generation_stale(generation):
                state.session._log_stale_generation(
                    "append_self_message",
                    generation,
                )
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

    @staticmethod
    def _delay_seconds(part: str, runtime_settings: dict) -> float:
        if runtime_settings["send_strategy"] == "humanized_delay":
            delay = runtime_settings["humanized_delay_seconds"] + len(part) * 0.08
        else:
            delay = 1.0 + len(part) * 0.1
        return min(delay, 5.0)
