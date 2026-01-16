# nyaturingtest/session.py
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import random
import traceback
import uuid

import httpx
from nonebot import logger
import nonebot_plugin_localstore as store
from nonebot.utils import run_sync
from openai import AsyncOpenAI
from tortoise.transactions import in_transaction

from .client import LLMClient
from .config import plugin_config
from .emotion import EmotionState
from .vector_mem import VectorMemory
from .impression import Impression
from .mem import Memory, Message
from .presets import PRESETS
from .profile import PersonProfile
from .models import SessionModel, UserProfileModel, InteractionLogModel, GlobalMessageModel
from .utils import extract_and_parse_json, estimate_split_count, check_relevance
from .prompts import get_feedback_prompt, get_chat_prompt


@dataclass
class _SearchResult:
    mem_history: list[str]


class _ChattingState(Enum):
    ILDE = 0  # 潜水
    BUBBLE = 1  # 冒泡
    ACTIVE = 2  # 活跃

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
        self.__aliases: list[str] = []
        self.profiles: dict[str, PersonProfile] = {}
        self.global_emotion: EmotionState = EmotionState()
        self.chat_summary = ""
        self.__role = "一个男性人类"

        # 意愿值系统
        self.willingness: float = 0.0
        self.__chatting_state = _ChattingState.ILDE

        self.__search_result = None
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        self._active_count = 0
        self._loaded = False

        # 保持对后台任务的引用，防止被 GC
        self._background_tasks = set()

    def _sanitize(self, text: str) -> str:
        if not text: return ""
        try:
            return text.encode('utf-8', 'ignore').decode('utf-8')
        except:
            return ""

    def _escape_for_prompt(self, text: str) -> str:
        if not text: return ""
        return text.replace('"', '\\"').replace('\n', ' ')

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
        self.__aliases = []
        self.__role = "一个男性人类"
        await self.global_memory.clear()
        self.long_term_memory.clear()
        self.profiles = {}
        self.global_emotion = EmotionState()
        self.chat_summary = ""
        self.__chatting_state = _ChattingState.ILDE
        self.willingness = 0.0
        self._active_count = 0
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        await self.save_session()

    async def calm_down(self):
        self.global_emotion = EmotionState()
        self.profiles = {}
        self.__chatting_state = _ChattingState.ILDE
        self.willingness = 0.0
        self._active_count = 0
        self._last_activity_time = datetime.now()
        await self.save_session()

    async def save_session(self, force_index: bool = False):
        try:
            session_db, created = await SessionModel.update_or_create(
                id=self.id,
                defaults={
                    "name": self._sanitize(self.__name),
                    "role": self._sanitize(self.__role),
                    "aliases": self.__aliases,  # 保存别名
                    "valence": self.global_emotion.valence,
                    "arousal": self.global_emotion.arousal,
                    "dominance": self.global_emotion.dominance,
                    "chat_summary": self._sanitize(self.chat_summary),
                    "last_speak_time": self._last_speak_time,
                    "chatting_state": self.__chatting_state.value
                }
            )

            # 保存用户画像
            for user_id, profile in self.profiles.items():
                await UserProfileModel.update_or_create(
                    session=session_db,
                    user_id=str(user_id),
                    defaults={
                        "valence": profile.emotion.valence,
                        "arousal": profile.emotion.arousal,
                        "dominance": profile.emotion.dominance,
                    }
                )
            # 保存短时消息历史
            recent_msgs = self.global_memory.access().messages
            if recent_msgs:
                async with in_transaction():
                    await GlobalMessageModel.filter(session=session_db).delete()
                    bulk_msgs = []
                    for msg in recent_msgs:
                        bulk_msgs.append(GlobalMessageModel(
                            session=session_db,
                            user_name=self._sanitize(msg.user_name),
                            user_id=str(msg.user_id) if msg.user_id else "",
                            content=self._sanitize(msg.content),
                            time=msg.time,
                            msg_id=msg.id
                        ))
                    await GlobalMessageModel.bulk_create(bulk_msgs)

            # 记忆生命周期管理：1% 概率触发清理，或强制保存时触发
            if force_index or random.random() < 0.01:
                # 清理超过 90 天的旧记忆
                await run_sync(self.long_term_memory.cleanup)(days_retention=90)

            logger.debug(f"[Session {self.id}] 数据库保存成功")
        except Exception as e:
            logger.warning(f"[Session {self.id}] 数据库保存警告: {e}")

    async def load_session(self):
        if self._loaded: return

        session_db = await SessionModel.filter(id=self.id).first()
        if not session_db:
            logger.info(f"[Session {self.id}] 初始化新会话")
            self._loaded = True
            return

        self.__name = session_db.name
        self.__role = session_db.role
        self.__aliases = session_db.aliases if session_db.aliases else []  # 加载别名
        self.chat_summary = session_db.chat_summary
        self.global_emotion.valence = session_db.valence
        self.global_emotion.arousal = session_db.arousal
        self.global_emotion.dominance = session_db.dominance
        if session_db.last_speak_time:
            self._last_speak_time = session_db.last_speak_time
        self.__chatting_state = _ChattingState(session_db.chatting_state)

        # 重启后给一点点初始意愿，防止Bot彻底装死
        self.willingness = 0.1

        self.profiles = {}
        users_db = await UserProfileModel.filter(session=session_db).prefetch_related("interactions")
        for user_db in users_db:
            profile = PersonProfile(user_id=user_db.user_id)
            profile.emotion.valence = user_db.valence
            profile.emotion.arousal = user_db.arousal
            profile.emotion.dominance = user_db.dominance
            profile.last_update_time = user_db.last_update_time
            # 加载最近交互
            recent_logs = await user_db.interactions.all().order_by("-timestamp").limit(20)
            for log in reversed(recent_logs):
                imp = Impression(
                    timestamp=log.timestamp,
                    delta={"valence": log.delta_valence, "arousal": log.delta_arousal, "dominance": log.delta_dominance}
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
                id=msg_db.msg_id,
                user_id=msg_db.user_id if msg_db.user_id else ""
            ))

        self.global_memory = Memory(
            llm_client=self.global_memory._Memory__llm_client,
            compressed_message=self.global_memory.access().compressed_history,
            messages=history_msgs
        )

        self._loaded = True
        logger.info(f"[Session {self.id}] 加载完成 (别名: {self.__aliases})")

    def presets(self) -> list[str]:
        return [f"{filename}: {preset.name} {preset.role}" for filename, preset in PRESETS.items() if not preset.hidden]

    async def load_preset(self, filename: str) -> bool:
        if not filename.endswith(".json") and f"{filename}.json" in PRESETS.keys():
            filename = f"{filename}.json"
        if filename not in PRESETS.keys(): return False

        preset = PRESETS[filename]
        await self.set_role(preset.name, preset.role)
        self.__aliases = preset.aliases

        # 在添加新预设前，删除旧的预设记忆，防止重复
        await run_sync(self.long_term_memory.delete_by_metadata)({"source": "preset"})

        to_add = (preset.knowledges + preset.relationships + preset.events + preset.bot_self)
        if to_add:
            metadatas = [{"source": "preset", "type": "rule"} for _ in to_add]
            await run_sync(self.long_term_memory.add_texts)(to_add, metadatas=metadatas)

        await self.save_session()
        return True

    def status(self) -> str:
        recent_messages = self.global_memory.access().messages
        recent_str = "\n".join([f"{m.user_name}: {m.content}" for m in recent_messages]) if recent_messages else "无"
        return f"""
名字：{self.__name}
设定：{self.__role}
意愿值：{self.willingness:.2f}
状态: {self.__chatting_state}
情绪：V{self.global_emotion.valence:.2f} A{self.global_emotion.arousal:.2f} D{self.global_emotion.dominance:.2f}
摘要：{self.chat_summary}
最近消息：
{recent_str}
"""

    async def __search_stage(self, queries: list[str]):
        """
        优化检索阶段
        """
        logger.debug("检索阶段开始")

        # 增加当前话题摘要作为检索上下文
        if self.chat_summary:
            queries.append(self.chat_summary)

        # 去重并过滤
        queries = list(set([q for q in queries if q and q.strip()]))

        should_retrieve = self.willingness > 0.3  # 只有意愿尚可时才检索

        long_term_memory = []
        if should_retrieve and queries:
            logger.debug(f"触发长期记忆检索: {queries}")

            raw_results = await run_sync(self.long_term_memory.retrieve)(
                queries,
                k=20,
                where=None
            )

            if raw_results:
                formatted_results = []
                for item in raw_results:
                    content = item.get("content", "")
                    meta = item.get("metadata", {})
                    source = meta.get("source", "unknown")
                    # 日期格式化优化
                    date_str = str(meta.get("date", ""))
                    prefix = "【设定】" if source == "preset" else f"【记忆/d:{date_str}】"
                    formatted_results.append(f"{prefix} {content}")

                # 按时间倒序排列（虽然 RAG 是按相关性，但这里可以二次排序，或者直接交给 LLM）
                long_term_memory = formatted_results
                logger.debug(f"搜索结果：命中 {len(long_term_memory)} 条")

        self.__search_result = _SearchResult(mem_history=long_term_memory)

    async def __feedback_stage(self, messages_chunk: list[Message], llm_func: Callable):
        """
        反馈阶段：分析情绪、提取记忆、更新摘要
        """
        logger.debug(">> 反馈阶段 (Feedback) 开始")

        # 1. 准备画像数据
        reaction_users = list({msg.user_id if msg.user_id else msg.user_name for msg in messages_chunk})
        related_profiles = [self.profiles.get(uid, PersonProfile(user_id=uid)) for uid in reaction_users]
        for p in related_profiles:
            if p.user_id not in self.profiles: self.profiles[p.user_id] = p

        related_profiles_json = json.dumps(
            [{"user_name": p.user_id, "emotion_tends_to_user": asdict(p.emotion)} for p in related_profiles],
            ensure_ascii=False, indent=2
        )
        search_history = self.__search_result.mem_history if self.__search_result else []

        # 格式化消息时包含 UserID，供 LLM 识别
        formatted_msgs = [f"[ID:{msg.user_id}] {msg.user_name}: '{self._escape_for_prompt(msg.content)}'" for msg in
                          messages_chunk]

        # 2. 调用 LLM
        prompt = get_feedback_prompt(
            self.__name, self.__role, self.willingness,
            self.__chatting_state.value,
            self.global_memory.access().compressed_history,
            self.global_memory.access().messages,
            formatted_msgs,
            asdict(self.global_emotion),
            related_profiles_json, search_history, self.chat_summary
        )

        response_dict = {}
        try:
            # [Debug] 记录 LLM 原始输出
            response = await llm_func(prompt, json_mode=True)
            logger.debug(f"[Feedback LLM Output]:\n{response}")

            parsed = extract_and_parse_json(response)
            if parsed: response_dict = parsed
        except Exception as e:
            logger.error(f"反馈阶段 LLM 错误: {e}")

        # 3. 更新情绪
        new_emo = response_dict.get("new_emotion", {})
        self.global_emotion.valence = new_emo.get("valence", self.global_emotion.valence)
        self.global_emotion.arousal = new_emo.get("arousal", self.global_emotion.arousal)
        self.global_emotion.dominance = new_emo.get("dominance", self.global_emotion.dominance)

        # 4. 更新用户印象 (Emotion Tends)
        emo_tends = response_dict.get("emotion_tends", [])
        if isinstance(emo_tends, list):
            for i, msg in enumerate(messages_chunk):
                if i >= len(emo_tends): break
                uid = msg.user_id if msg.user_id else msg.user_name
                raw_delta = emo_tends[i]

                delta = {}
                if isinstance(raw_delta, (int, float)):
                    delta = {
                        "valence": float(raw_delta),
                        "arousal": abs(float(raw_delta)) * 0.5,
                        "dominance": 0.0
                    }
                elif isinstance(raw_delta, dict):
                    delta = raw_delta

                if uid in self.profiles and delta:
                    self.profiles[uid].push_interaction(
                        Impression(timestamp=datetime.now().astimezone(), delta=delta)
                    )

        for p in self.profiles.values():
            p.update_emotion_tends()
            p.merge_old_interactions()

        # 5. 更新摘要
        self.chat_summary = str(response_dict.get("summary", self.chat_summary))

        # 6. 记忆提取转为异步后台任务 (Fire-and-forget)
        analyze_result = response_dict.get("analyze_result", [])
        if isinstance(analyze_result, list) and analyze_result:
            unique_user_ids = {
                str(msg.user_id) for msg in messages_chunk
                if msg.user_id and str(msg.user_id).strip()
            }
            fallback_uid = list(unique_user_ids)[0] if len(unique_user_ids) == 1 else ""

            task = asyncio.create_task(
                self.__save_long_term_memory(analyze_result, default_user_id=fallback_uid)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # 7. 更新意愿值 (由 LLM 最终决定)
        try:
            new_willing = float(response_dict.get("willing", self.willingness))
            self.willingness = max(0.0, min(1.0, new_willing))
        except:
            pass

        # 8. 简单的状态流转逻辑
        random_threshold = random.uniform(0.4, 0.7)
        if self.willingness > random_threshold:
            if self.__chatting_state == _ChattingState.ILDE:
                self.__chatting_state = _ChattingState.BUBBLE
        elif self.willingness < 0.2:
            self.__chatting_state = _ChattingState.ILDE

        logger.debug(f"<< 反馈结束: 意愿 {self.willingness:.2f}, 状态 {self.__chatting_state}")

    async def __save_long_term_memory(self, analyze_result: list, default_user_id: str = ""):
        """
        后台任务：保存长期记忆到向量数据库
        """
        try:
            texts = []
            metadatas = []
            today = int(datetime.now().strftime("%Y%m%d"))

            for item in analyze_result:
                content = ""
                uid = ""

                # 情况A: LLM 返回了纯字符串 (偷懒格式)
                if isinstance(item, str) and item.strip():
                    content = item
                    uid = default_user_id
                # 情况B: LLM 返回了字典 (标准格式)
                elif isinstance(item, dict):
                    content = item.get("content", "")
                    uid = str(item.get("related_user_id", ""))
                    # 如果 LLM 漏填了 ID，尝试使用兜底 ID
                    if not uid and default_user_id:
                        uid = default_user_id

                if content:
                    texts.append(content)
                    metadatas.append({
                        "source": "memory",
                        "type": "event",
                        "date": today,
                        "user_id": uid
                    })

            if texts:
                # 这是一个阻塞IO操作，放入 run_sync
                await run_sync(self.long_term_memory.add_texts)(texts, metadatas=metadatas)
                logger.debug(f"[Async] 提取并保存长期记忆 ({len(texts)}条) [兜底ID: {default_user_id}]")
        except Exception as e:
            logger.error(f"[Async] 保存记忆失败: {e}")

    async def __chat_stage(self, messages_chunk: list[Message], llm_func: Callable) -> list[dict]:
        logger.debug(">> 对话阶段 (Chat) 开始")
        search_history = self.__search_result.mem_history if self.__search_result else []
        formatted_msgs = [f"[ID:{msg.id}] {msg.user_name}: '{self._escape_for_prompt(msg.content)}'" for msg in
                          messages_chunk]

        prompt = get_chat_prompt(
            self.__name, self.__role, self.__chatting_state.value,
            self.global_memory.access().compressed_history,
            self.global_memory.access().messages,
            formatted_msgs,
            asdict(self.global_emotion),
            "{}",
            search_history, self.chat_summary
        )

        try:
            # [Debug] 记录 LLM 原始输出
            response = await llm_func(prompt, json_mode=True)
            logger.debug(f"[Chat LLM Output]:\n{response}")

            response_data = extract_and_parse_json(response)

            # 兼容 List 返回类型，自动包装
            replies = []
            if isinstance(response_data, dict):
                replies = response_data.get("reply", [])
            elif isinstance(response_data, list):
                # 如果 LLM 直接返回了列表，我们假设这就是回复列表
                replies = response_data
                logger.warning("LLM 返回了 List 而非 Object，已自动兼容")

            if not isinstance(replies, list):
                return []

            if replies:
                self.willingness = max(0.0, self.willingness - 0.4)
                self.__chatting_state = _ChattingState.ACTIVE

            return replies
        except Exception as e:
            logger.error(f"对话阶段异常: {e}")
            return []

    async def update(self, messages_chunk: list[Message], llm_func: Callable[[str, bool], Awaitable[str]],
                     publish: bool = True) -> list[dict] | None:

        # 1. 更新短时记忆 (Buffer)
        def enable_hippo():
            self.__update_hippo = True

        await self.global_memory.update(messages_chunk, after_compress=enable_hippo)
        asyncio.create_task(self.save_session())

        if not publish: return None

        # 2. 意愿值计算 (确定性逻辑)
        # 时间衰减
        now = datetime.now()
        seconds_passed = (now - self._last_activity_time).total_seconds()
        decay = (seconds_passed / 60.0) * 0.05  # 每分钟衰减 0.05
        self.willingness = max(0.0, self.willingness - decay)
        self._last_activity_time = now

        # 触发增益
        is_relevant = check_relevance(self.__name, self.__aliases, messages_chunk)
        if is_relevant:
            self.willingness = max(self.willingness, 0.95)  # 被叫名字，意愿值设为较高
            logger.info("检测到强关联，意愿值提升")
        else:
            # 即使没叫我，如果有新消息，也稍微增加一点好奇心
            self.willingness = min(1.0, self.willingness + 0.05 * len(messages_chunk))

        # 3. 节流判断 (Gatekeeper)
        if self.willingness < 0.3 and not is_relevant:
            logger.debug(f"意愿值过低 ({self.willingness:.2f}) 且无强关联，跳过响应")
            return None

        # 4. 检索阶段 (提取关键词)
        # 混合“最近消息”和“当前话题摘要”来生成检索词，提高命中率
        queries = [msg.content for msg in messages_chunk[-2:]]
        if self.chat_summary and len(self.chat_summary) > 5:
            queries.append(self.chat_summary)
        await self.__search_stage(queries)

        # 5. 串行执行 (先思考/反馈，再决定是否说话)
        logger.debug("启用拟人化串行模式: Feedback -> Check -> Chat")

        # 5.1 反馈与思考 (LLM 更新情绪、提取记忆、最终决定意愿)
        # 这一步会更新 self.global_emotion 和 self.willingness
        await self.__feedback_stage(messages_chunk, llm_func)

        # 5.2 再次检查意愿 (Feedback 阶段可能会根据新消息调整意愿)
        # 如果 Feedback 后意愿降低(比如觉得无聊)，则停止回复
        if self.willingness < 0.4:
            return None

        # 5.3 对话执行
        # 此时 Chat 阶段使用的是 Feedback 更新后的最新情绪
        reply_messages = await self.__chat_stage(messages_chunk, llm_func)

        if reply_messages:
            self._last_speak_time = datetime.now()
            asyncio.create_task(self.save_session())

        return reply_messages
