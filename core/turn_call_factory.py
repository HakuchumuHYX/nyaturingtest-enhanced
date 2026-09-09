from collections.abc import Callable
from dataclasses import dataclass

from ..config import (
    get_chat_max_tokens,
    get_reasoning_effort,
    get_chat_timeout,
    get_effective_chat_model,
    get_effective_feedback_model,
    get_feedback_max_tokens,
    get_feedback_timeout,
)
from .metrics import log_event
from .usage import make_usage_recorder


@dataclass(frozen=True)
class TurnCalls:
    chat: Callable
    feedback: Callable


CHAT_SYSTEM_PROMPT = (
    "你就是动态输入里的那个角色本人，正在群聊里用手机和人聊天。"
    "role 是你的性格与经历，examples_text 是你的说话习惯，search_result 是你的记忆，"
    "把它们当作自己的东西，不是别人给你的说明书。"
    "不要以 AI、助手、模型或角色扮演引擎的身份说话，不要解释设定。"
    "群聊回复要短、自然，像手机打字。"
    "请在内部完成分析，但最终输出只包含一个合法 JSON 对象，不要输出 Markdown、解释或思考过程。"
)

FEEDBACK_SYSTEM_PROMPT = (
    "你是一个对话分析引擎。你的输入是群聊消息日志和可选的群聊图片，"
    "输出是结构化的情感分析 JSON。"
    "这是一个纯数据处理任务：读取文本和图片 → 分析情感维度 → 输出 JSON。"
    "你不需要参与对话，不需要扮演任何角色，只需要做文本情感分析。"
    "你的输出必须包含 new_emotion 对象（含 valence、arousal、dominance 三个浮点数字段）。"
    "请在内部完成分析，但最终输出只包含一个合法 JSON 对象，不要输出 Markdown、解释或思考过程。"
)


class TurnCallFactory:
    def __init__(self, llm_response: Callable):
        self._llm_response = llm_response

    def build(
        self,
        *,
        state,
        session_id: str,
        chat_images: list,
        feedback_images: list,
    ) -> TurnCalls:
        def usage_recorder(model_name: str):
            def log_usage(usage: dict):
                log_event(
                    "token_usage",
                    session_id=session_id,
                    provider=usage.get("provider", ""),
                    model=model_name,
                    tokens=usage.get("total_tokens", 0),
                    decision=usage.get("finish_reason", ""),
                )

            return make_usage_recorder(
                session_id,
                model_name,
                event_logger=log_usage,
            )

        chat_model = get_effective_chat_model()
        feedback_model = get_effective_feedback_model()

        async def chat(message: str, json_mode: bool = False):
            return await self._llm_response(
                state.client,
                message,
                model=chat_model,
                temperature=0.7,
                reasoning_effort=get_reasoning_effort("chat"),
                json_mode=bool(json_mode),
                max_tokens=get_chat_max_tokens(),
                timeout=get_chat_timeout(),
                system_prompt=CHAT_SYSTEM_PROMPT,
                on_usage=usage_recorder(chat_model),
                images=chat_images,
            )

        async def feedback(message: str, json_mode: bool = False):
            return await self._llm_response(
                state.feedback_client,
                message,
                model=feedback_model,
                temperature=0.1,
                json_mode=bool(json_mode),
                reasoning_effort=get_reasoning_effort("feedback"),
                max_tokens=get_feedback_max_tokens(),
                timeout=get_feedback_timeout(),
                on_usage=usage_recorder(feedback_model),
                system_prompt=FEEDBACK_SYSTEM_PROMPT,
                images=feedback_images,
            )

        return TurnCalls(chat=chat, feedback=feedback)
