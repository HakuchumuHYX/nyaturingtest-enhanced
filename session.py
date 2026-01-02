# nyaturingtest/session.py
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import random
import traceback

import httpx
from nonebot import logger
import nonebot_plugin_localstore as store
from nonebot.utils import run_sync
from openai import AsyncOpenAI

from .client import LLMClient
from .config import plugin_config
from .emotion import EmotionState
from .vector_mem import VectorMemory
from .impression import Impression
from .mem import Memory, Message
from .presets import PRESETS
from .profile import PersonProfile
from .models import SessionModel, UserProfileModel, InteractionLogModel, GlobalMessageModel

# 导入拆分后的模块
from .utils import extract_and_parse_json, estimate_split_count, check_relevance
from .prompts import get_feedback_prompt, get_chat_prompt


@dataclass
class _SearchResult:
    """
    检索阶段的结果
    """
    mem_history: list[str]


class _ChattingState(Enum):
    ILDE = 0
    BUBBLE = 1
    ACTIVE = 2

    def __str__(self):
        match self:
            case _ChattingState.ILDE:
                return "潜水状态"
            case _ChattingState.BUBBLE:
                return "冒泡状态"
            case _ChattingState.ACTIVE:
                return "对话状态"


class Session:
    """
    群聊会话
    """

    def __init__(
            self,
            siliconflow_api_key: str,
            id: str = "global",
            name: str = "terminus",
            http_client: httpx.AsyncClient | None = None
    ):
        self.id = id
        if http_client:
            self._client_instance = http_client
        else:
            logger.debug(f"[Session {id}] 未传入全局 HTTP 客户端，创建局部客户端")
            self._client_instance = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                timeout=60.0
            )

        self.global_memory: Memory = Memory(
            llm_client=LLMClient(
                client=AsyncOpenAI(
                    api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                    base_url="https://api.siliconflow.cn/v1",
                    http_client=self._client_instance
                )
            )
        )

        self.long_term_memory: VectorMemory = VectorMemory(
            api_key=plugin_config.nyaturingtest_siliconflow_api_key,
            persist_directory=f"{store.get_plugin_data_dir()}/vector_index_{id}",
        )

        self.__name = name
        self.profiles: dict[str, PersonProfile] = {}
        self.global_emotion: EmotionState = EmotionState()
        self.last_response: list[Message] = []
        self.chat_summary = ""
        self.__role = "一个男性人类"
        self.__chatting_state = _ChattingState.ILDE
        self.__bubble_willing_sum = 0.0
        self.__update_hippo = False
        self.__search_result = None
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        self._active_count = 0
        self._loaded = False

    async def set_role(self, name: str, role: str):
        self.__role = role
        self.__name = name
        await self.save_session()

    def role(self) -> str:
        return f"{self.__name}（{self.__role}）"

    def name(self) -> str:
        return self.__name

    async def reset(self):
        self.__name = "terminus"
        self.__role = "一个男性人类"
        await self.global_memory.clear()
        self.long_term_memory.clear()
        self.profiles = {}
        self.global_emotion = EmotionState()
        self.last_response = []
        self.chat_summary = ""
        self.__chatting_state = _ChattingState.ILDE
        self.__bubble_willing_sum = 0.0
        self._active_count = 0
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        await self.save_session()

    async def calm_down(self):
        self.global_emotion = EmotionState()
        self.profiles = {}
        self.__chatting_state = _ChattingState.ILDE
        self._active_count = 0
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.now()
        await self.save_session()

    async def save_session(self, force_index: bool = False):
        try:
            session_db, created = await SessionModel.update_or_create(
                id=self.id,
                defaults={
                    "name": self.__name,
                    "role": self.__role,
                    "valence": self.global_emotion.valence,
                    "arousal": self.global_emotion.arousal,
                    "dominance": self.global_emotion.dominance,
                    "chat_summary": self.chat_summary,
                    "last_speak_time": self._last_speak_time,
                    "chatting_state": self.__chatting_state.value
                }
            )

            for user_id, profile in self.profiles.items():
                await UserProfileModel.update_or_create(
                    session=session_db,
                    user_id=user_id,
                    defaults={
                        "valence": profile.emotion.valence,
                        "arousal": profile.emotion.arousal,
                        "dominance": profile.emotion.dominance,
                    }
                )

            recent_msgs = self.global_memory.access().messages
            if recent_msgs:
                await GlobalMessageModel.filter(session=session_db).delete()
                bulk_msgs = []
                for msg in recent_msgs:
                    bulk_msgs.append(GlobalMessageModel(
                        session=session_db,
                        user_name=msg.user_name,
                        content=msg.content,
                        time=msg.time,
                        msg_id=msg.id
                    ))
                await GlobalMessageModel.bulk_create(bulk_msgs)

            logger.debug(f"[Session {self.id}] 数据库保存成功")
        except Exception as e:
            error_msg = str(e)
            if "no active connection" in error_msg or "closed database" in error_msg:
                logger.warning(f"[Session {self.id}] 放弃保存：数据库连接已关闭")
            else:
                logger.error(f"[Session {self.id}] 数据库保存失败: {e}")
                import traceback
                traceback.print_exc()

    async def load_session(self):
        if self._loaded:
            return

        session_db = await SessionModel.filter(id=self.id).first()
        if not session_db:
            logger.info(f"[Session {self.id}] 数据库中无记录，初始化新会话")
            self._loaded = True
            return

        self.__name = session_db.name
        self.__role = session_db.role
        self.chat_summary = session_db.chat_summary
        self.global_emotion.valence = session_db.valence
        self.global_emotion.arousal = session_db.arousal
        self.global_emotion.dominance = session_db.dominance
        if session_db.last_speak_time:
            self._last_speak_time = session_db.last_speak_time
        self.__chatting_state = _ChattingState(session_db.chatting_state)

        self.profiles = {}
        users_db = await UserProfileModel.filter(session=session_db).prefetch_related("interactions")

        for user_db in users_db:
            profile = PersonProfile(user_id=user_db.user_id)
            profile.emotion.valence = user_db.valence
            profile.emotion.arousal = user_db.arousal
            profile.emotion.dominance = user_db.dominance
            profile.last_update_time = user_db.last_update_time

            recent_logs = await user_db.interactions.all().order_by("-timestamp").limit(20)
            for log in reversed(recent_logs):
                imp = Impression(
                    timestamp=log.timestamp,
                    delta={
                        "valence": log.delta_valence,
                        "arousal": log.delta_arousal,
                        "dominance": log.delta_dominance
                    }
                )
                profile.interactions.append(imp)
            self.profiles[user_db.user_id] = profile

        msgs_db = await GlobalMessageModel.filter(session=session_db).order_by("-time").limit(50)
        history_msgs = []
        for msg_db in reversed(msgs_db):
            history_msgs.append(Message(
                time=msg_db.time,
                user_name=msg_db.user_name,
                content=msg_db.content,
                id=msg_db.msg_id
            ))

        self.global_memory = Memory(
            llm_client=self.global_memory._Memory__llm_client,
            compressed_message=self.global_memory.access().compressed_history,
            messages=history_msgs
        )

        self._loaded = True
        logger.info(f"[Session {self.id}] 数据库加载完成")

    def presets(self) -> list[str]:
        return [f"{filename}: {preset.name} {preset.role}" for filename, preset in PRESETS.items() if not preset.hidden]

    async def load_preset(self, filename: str) -> bool:
        if not filename.endswith(".json") and f"{filename}.json" in PRESETS.keys():
            filename = f"{filename}.json"

        if filename not in PRESETS.keys():
            logger.error(f"不存在的预设：{filename}")
            return False

        preset = PRESETS[filename]
        await self.set_role(preset.name, preset.role)

        to_add = (preset.knowledges + preset.relationships +
                  preset.events + preset.bot_self)
        await run_sync(self.long_term_memory.add_texts)(to_add)

        logger.info(f"加载预设：{filename} 成功")
        return True

    def status(self) -> str:
        recent_messages = self.global_memory.access().messages
        recent_messages_str = (
            "\n".join([f"{msg.user_name}: {msg.content}" for msg in recent_messages]) if recent_messages else "没有消息"
        )
        return f"""
名字：{self.__name}

设定：{self.__role}

情感状态：V:{self.global_emotion.valence:.2f} A:{self.global_emotion.arousal:.2f} D:{self.global_emotion.dominance:.2f}

最近消息：
{recent_messages_str}

过去总结：
{self.global_memory.access().compressed_history}

现状认识：{self.chat_summary}

状态: {self.__chatting_state}
疲劳度(气泡计数): {self._active_count}
"""

    async def __search_stage(self):
        logger.debug("检索阶段开始")
        recent_msgs = self.global_memory.access().messages
        retrieve_messages = (
                [f"'{msg.user_name}':'{msg.content}'" for msg in recent_msgs]
                + [self.global_memory.access().compressed_history]
                + [self.chat_summary]
        )

        should_retrieve = False
        if self.__chatting_state != _ChattingState.ILDE:
            should_retrieve = True
        else:
            if recent_msgs:
                for msg in list(recent_msgs)[-3:]:
                    has_at = f"@{self.__name}" in msg.content
                    has_reply = f"[回复 {self.__name}" in msg.content
                    if has_at or has_reply:
                        should_retrieve = True
                        break

        long_term_memory = []

        if should_retrieve:
            logger.debug(f"触发长期记忆检索 (状态: {self.__chatting_state})")
            long_term_memory = await run_sync(self.long_term_memory.retrieve)(retrieve_messages, k=2)
            if long_term_memory:
                logger.debug(f"搜索到的相关记忆：{long_term_memory}")
        else:
            logger.debug("潜水状态且无强关联，跳过长期记忆检索")

        self.__search_result = _SearchResult(mem_history=long_term_memory)
        logger.debug("检索阶段结束")

    async def __feedback_stage(self, messages_chunk: list[Message], llm: Callable[[str], Awaitable[str]], is_relevant: bool):
        logger.debug("反馈阶段开始")

        should_skip_llm = False
        if self.__chatting_state == _ChattingState.ILDE:
            if is_relevant:
                should_skip_llm = False
                logger.debug("检测到强关联（被@或回复），跳过潜水节流检查")
            else:
                curiosity_rate = 0.08
                if random.random() < curiosity_rate:
                    logger.debug("触发随机好奇心：虽然没叫我，但我决定通过 LLM 看看大家在聊什么")
                    should_skip_llm = False
                else:
                    should_skip_llm = True

        if should_skip_llm:
            logger.debug(f"触发节流：潜水状态且消息无强关联 ({len(messages_chunk)}条)，跳过 LLM 分析")
            for message in messages_chunk:
                if message.user_name not in self.profiles:
                    self.profiles[message.user_name] = PersonProfile(user_id=message.user_name)
                self.profiles[message.user_name].push_interaction(
                    Impression(timestamp=datetime.now().astimezone(), delta={})
                )
            self.__bubble_willing_sum += 0.03 * len(messages_chunk)
            random_value = random.uniform(0.8, 1.0)
            if self.__bubble_willing_sum > random_value:
                logger.debug(f"潜水观察积累意愿({self.__bubble_willing_sum:.2f}) > {random_value:.2f}，自动转入冒泡状态")
                self.__chatting_state = _ChattingState.BUBBLE
                self.__bubble_willing_sum = 0.0
            return

        reaction_users = self.global_memory.related_users()
        related_profiles = [profile for profile in self.profiles.values() if profile.user_id in reaction_users]
        related_profiles_json = json.dumps(
            [{"user_name": p.user_id, "emotion_tends_to_user": asdict(p.emotion)} for p in related_profiles],
            ensure_ascii=False, indent=2
        )
        search_stage_result = self.__search_result.mem_history if self.__search_result else []

        prompt = get_feedback_prompt(
            self.__name, self.__role, self.__chatting_state.value,
            self.global_memory.access().compressed_history,
            self.global_memory.access().messages,
            [f"{msg.user_name}: '{msg.content}'" for msg in messages_chunk],
            {"valence": self.global_emotion.valence, "arousal": self.global_emotion.arousal,
             "dominance": self.global_emotion.dominance},
            related_profiles_json, search_stage_result, self.chat_summary
        )

        MAX_RETRIES = 2
        response_dict = {}

        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    logger.warning(
                        f"反馈阶段 JSON 解析失败 (原因: 格式错误或幻觉)，正在进行第 {attempt}/{MAX_RETRIES} 次重试...")

                response = await llm(prompt)
                logger.debug(f"反馈阶段llm返回 (尝试 {attempt}): {response}")

                parsed = extract_and_parse_json(response)
                if parsed:
                    response_dict = parsed
                    if "emotion_tends" in response_dict or "willing" in response_dict:
                        break
            except Exception as e:
                logger.error(f"反馈阶段 LLM 请求或解析发生异常: {e}")

            if attempt == MAX_RETRIES:
                logger.error(f"反馈阶段重试 {MAX_RETRIES} 次后仍失败，执行降级处理。")
                response_dict = {}

        new_emotion = response_dict.get("new_emotion", {})
        self.global_emotion.valence = new_emotion.get("valence", self.global_emotion.valence)
        self.global_emotion.arousal = new_emotion.get("arousal", self.global_emotion.arousal)
        self.global_emotion.dominance = new_emotion.get("dominance", self.global_emotion.dominance)
        logger.debug(f"反馈阶段更新情感：{self.global_emotion}")

        emotion_tends = response_dict.get("emotion_tends", [])
        if not isinstance(emotion_tends, list):
            emotion_tends = []

        target_len = len(messages_chunk)
        current_len = len(emotion_tends)
        if current_len != target_len:
            if current_len < target_len:
                defaults = [{"valence": 0.0, "arousal": 0.0, "dominance": 0.0} for _ in
                            range(target_len - current_len)]
                emotion_tends.extend(defaults)
            else:
                emotion_tends = emotion_tends[:target_len]

        for index, message in enumerate(messages_chunk):
            if message.user_name not in self.profiles:
                self.profiles[message.user_name] = PersonProfile(user_id=message.user_name)
            delta = emotion_tends[index] if index < len(emotion_tends) else {}
            self.profiles[message.user_name].push_interaction(
                Impression(timestamp=datetime.now().astimezone(), delta=delta)
            )

        for profile in self.profiles.values():
            profile.update_emotion_tends()
            profile.merge_old_interactions()

        self.chat_summary = str(response_dict.get("summary", self.chat_summary))
        logger.debug(f"反馈阶段更新聊天总结：{self.chat_summary}")

        analyze_result = response_dict.get("analyze_result", [])
        if isinstance(analyze_result, list) and analyze_result:
            sanitized_result = []
            for item in analyze_result:
                if isinstance(item, str):
                    sanitized_result.append(item)
                elif isinstance(item, dict):
                    sanitized_result.append(json.dumps(item, ensure_ascii=False))
                else:
                    sanitized_result.append(str(item))
            if sanitized_result:
                await run_sync(self.long_term_memory.add_texts)(sanitized_result)
                logger.debug(f"反馈阶段更新长期记忆：{sanitized_result}")

        willing = response_dict.get("willing", {})
        if not isinstance(willing, dict):
            willing = {}

        idle_chance = float(willing.get("0", 0.0))
        if idle_chance > 1.0: idle_chance = 1.0
        logger.debug(f"nyabot潜水意愿(修正后)：{idle_chance}")

        bubble_chance = float(willing.get("1", 0.0))
        self.__bubble_willing_sum += bubble_chance
        logger.debug(f"nyabot本次冒泡意愿：{bubble_chance}")
        logger.debug(f"nyabot冒泡意愿累计(修正后)：{self.__bubble_willing_sum}")

        chat_chance = float(willing.get("2", 0.0))
        logger.debug(f"nyabot对话意愿(修正后)：{chat_chance}")

        # [修改] 降低门槛
        random_value = random.uniform(0.3, 0.7)
        logger.debug(f"意愿转变随机值：{random_value}")

        current_fatigue_factor = self._active_count * 0.15 if self.__chatting_state == _ChattingState.ACTIVE else 0.0

        match self.__chatting_state:
            case _ChattingState.ILDE:
                if self._last_speak_time.tzinfo is not None:
                    now_aware = datetime.now().astimezone()
                    seconds_since_speak = (now_aware - self._last_speak_time).total_seconds()
                else:
                    seconds_since_speak = (datetime.now() - self._last_speak_time).total_seconds()

                if seconds_since_speak < 90:
                    logger.debug(f"Bot 处于贤者时间 ({seconds_since_speak:.0f}s < 90s)，强制压制对话欲望")
                    chat_chance *= 0.5
                    self.__bubble_willing_sum = 0.0

                if chat_chance >= random_value:
                    self.__chatting_state = _ChattingState.ACTIVE
                    self.__bubble_willing_sum = 0.0
                elif self.__bubble_willing_sum >= random_value:
                    self.__chatting_state = _ChattingState.BUBBLE
                    self.__bubble_willing_sum = 0.0

            case _ChattingState.BUBBLE:
                if chat_chance >= random_value:
                    self.__chatting_state = _ChattingState.ACTIVE
                elif idle_chance >= random_value:
                    self.__chatting_state = _ChattingState.ILDE

            case _ChattingState.ACTIVE:
                fatigue_factor = self._active_count * 0.15
                final_idle_chance = (idle_chance * 1.2) + fatigue_factor
                logger.debug(
                    f"活跃退出判定: 基础意愿{idle_chance:.2f} + 疲劳({self._active_count}轮){fatigue_factor:.2f} = {final_idle_chance:.2f} (阈值: {random_value:.2f})")

                if final_idle_chance >= random_value:
                    logger.debug(f"Bot 聊累了(已聊{self._active_count}轮)，主动进入潜水状态")
                    self.__chatting_state = _ChattingState.ILDE
                    self._active_count = 0

        #  强制修正逻辑
        if is_relevant:
            # 如果被@了，但经过计算后依然是潜水状态，强制改为冒泡状态
            if self.__chatting_state == _ChattingState.ILDE:
                logger.info("检测到被@或回复，强制将状态从 [潜水] 修正为 [冒泡]")
                self.__chatting_state = _ChattingState.BUBBLE
                self.__bubble_willing_sum = 0.0

        logger.debug(
            f"[DECISION DEBUG] "
            f"状态: {self.__chatting_state.name} | "
            f"对话意愿(Chat): {chat_chance:.2f} | "
            f"潜水意愿(Idle): {idle_chance:.2f} | "
            f"疲劳值(Count): {self._active_count} (Factor: {current_fatigue_factor:.2f}) | "
            f"随机阈值: {random_value:.2f}"
        )

        logger.debug(f"反馈阶段更新对话状态：{self.__chatting_state!s}")
        logger.debug("反馈阶段结束")

    async def __chat_stage(self, messages_chunk: list[Message], llm: Callable[[str], Awaitable[str]]) -> list[dict]:
        logger.debug("对话阶段开始")
        reaction_users = self.global_memory.related_users()
        related_profiles = [profile for profile in self.profiles.values() if profile.user_id in reaction_users]
        related_profiles_json = json.dumps(
            [{"user_name": p.user_id, "emotion_tends_to_user": asdict(p.emotion)} for p in related_profiles],
            ensure_ascii=False, indent=2
        )
        search_stage_result = self.__search_result.mem_history if self.__search_result else []

        prompt = get_chat_prompt(
            self.__name, self.__role, self.__chatting_state.value,
            self.global_memory.access().compressed_history,
            self.global_memory.access().messages,
            [f"[ID:{getattr(msg, 'id', '')}] {msg.user_name}: '{msg.content}'" for msg in messages_chunk],
            {"valence": self.global_emotion.valence, "arousal": self.global_emotion.arousal,
             "dominance": self.global_emotion.dominance},
            related_profiles_json, search_stage_result, self.chat_summary
        )

        try:
            response = await llm(prompt)
            logger.debug(f"对话阶段llm返回：{response}")

            response_dict = extract_and_parse_json(response)

            if response_dict is None:
                logger.warning("对话阶段 JSON 解析失败，跳过本次回复")
                return []

            if isinstance(response_dict, list):
                logger.warning(f"对话阶段 LLM 返回了 list 而不是 dict，尝试自动修正。内容: {response_dict}")
                response_dict = {"reply": response_dict, "debug_reason": "自动修正:LLM返回了纯列表"}

            if not isinstance(response_dict, dict):
                logger.error(f"对话阶段 LLM 返回数据类型错误: {type(response_dict)}，无法处理。内容: {response_dict}")
                return []

            logger.debug(f"对话阶段回复内容：{response_dict.get('reply', [])}")
            logger.debug(f"对话阶段回复/不回复原因:{response_dict.get('debug_reason', '无原因')}")
            logger.debug("对话阶段结束")

            final_replies = []
            raw_replies = response_dict.get("reply", [])

            if isinstance(raw_replies, list):
                for item in raw_replies:
                    if isinstance(item, str):
                        final_replies.append({"content": item, "reply_to": None})
                    elif isinstance(item, dict):
                        content = item.get("content", "")
                        target_id = item.get("target_id")
                        if not target_id or str(target_id).lower() in ["null", "none", ""]:
                            target_id = None
                        final_replies.append({"content": content, "reply_to": target_id})

            return final_replies

        except Exception as e:
            logger.error(f"对话阶段发生未捕获异常: {e}")
            traceback.print_exc()
            return []

    async def update(self, messages_chunk: list[Message], llm: Callable[[str], Awaitable[str]]) -> list[dict] | None:
        now = datetime.now()
        time_since_last_active = (now - self._last_activity_time).total_seconds()

        if time_since_last_active > 300:
            if self.__chatting_state != _ChattingState.ILDE:
                logger.info(f"会话已冷却 ({time_since_last_active:.0f}s > 300s)，状态重置为 [潜水]")
                self.__chatting_state = _ChattingState.ILDE
                self.__bubble_willing_sum = 0.0
                self._active_count = 0

        self._last_activity_time = now

        #  计算是否强相关（被@或被回复）
        is_relevant = check_relevance(self.__name, messages_chunk)

        await self.__search_stage()
        #  传递 is_relevant
        await self.__feedback_stage(messages_chunk=messages_chunk, llm=llm, is_relevant=is_relevant)

        reply_messages = None
        match self.__chatting_state:
            case _ChattingState.ILDE:
                logger.debug("nyabot潜水中...")
                reply_messages = None
            case _ChattingState.BUBBLE:
                logger.debug("nyabot冒泡中...")
                reply_messages = await self.__chat_stage(messages_chunk=messages_chunk, llm=llm)
                if reply_messages:
                    self.__chatting_state = _ChattingState.ACTIVE
            case _ChattingState.ACTIVE:
                logger.debug("nyabot对话中...")
                chill_prob = 0.1 + (self._active_count * 0.05)

                # [修改] 如果强相关，强制跳过 Chill Mode
                if is_relevant:
                    logger.debug("检测到被@或回复，强制跳过 Chill Mode")
                    chill_prob = 0.0

                if random.random() < chill_prob:
                    logger.debug(f"Chill Mode触发 (概率{chill_prob:.2f}): 暂时不回消息")
                    reply_messages = None
                else:
                    reply_messages = await self.__chat_stage(messages_chunk=messages_chunk, llm=llm)

        def enable_update_hippo():
            self.__update_hippo = True

        text_to_add = [f"'{msg.user_name}':'{msg.content}'" for msg in messages_chunk]
        msgs_to_mem = messages_chunk.copy()

        if reply_messages:
            text_to_add.extend([f"'{self.__name}':'{msg['content']}'" for msg in reply_messages])
            msgs_to_mem.extend(
                [Message(user_name=self.__name, content=msg['content'], time=datetime.now()) for msg in reply_messages])

            actual_bubble_count = sum(estimate_split_count(msg.get('content', '')) for msg in reply_messages)
            self._active_count += actual_bubble_count
            logger.debug(
                f"Bot 发言 {len(reply_messages)} 条 (切分为 {actual_bubble_count} 个气泡)，疲劳值 +{actual_bubble_count}")
            self._last_speak_time = datetime.now()

        await run_sync(self.long_term_memory.add_texts)(text_to_add)

        await self.global_memory.update(msgs_to_mem, after_compress=enable_update_hippo)

        if self.__chatting_state == _ChattingState.ILDE:
            self._active_count = 0

        asyncio.create_task(self.save_session())
        return reply_messages