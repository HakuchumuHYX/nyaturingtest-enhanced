# nyaturingtest/session.py
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
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
from .utils import extract_and_parse_json, check_relevance, sanitize_text, escape_for_prompt, get_time_description, should_store_memory
from .prompts import get_feedback_prompt, get_chat_prompt
from .repository import SessionRepository


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

        # 保存基础 LLM Client，供 Memory 和其他组件使用
        self._base_llm_client = LLMClient(
            client=AsyncOpenAI(
                api_key=plugin_config.nyaturingtest_siliconflow_api_key,
                base_url="https://api.siliconflow.cn/v1",
                http_client=self._client_instance
            )
        )

        self.global_memory: Memory = Memory(
            llm_client=self._base_llm_client
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
        self.__examples_str = ""

        # 意愿值系统
        self.willingness: float = 0.0
        self.__chatting_state = _ChattingState.ILDE

        self.__search_result = None
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        self._active_count = 0
        self._loaded = False
        self._background_tasks = set()

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
            # 1. 保存基础状态
            await SessionRepository.save_session_state(
                self.id,
                {
                    "name": self.__name,
                    "role": self.__role,
                    "aliases": self.__aliases,
                    "valence": self.global_emotion.valence,
                    "arousal": self.global_emotion.arousal,
                    "dominance": self.global_emotion.dominance,
                    "chat_summary": self.chat_summary,
                    "last_speak_time": self._last_speak_time,
                    "chatting_state": self.__chatting_state.value
                }
            )

            # 2. 更新画像
            await SessionRepository.update_user_profiles(self.id, self.profiles)

            # 3. 同步消息
            recent_msgs = self.global_memory.access().messages
            if recent_msgs:
                await SessionRepository.sync_messages(self.id, recent_msgs)

            if force_index or random.random() < 0.01:
                await run_sync(self.long_term_memory.cleanup)(days_retention=90)

            logger.debug(f"[Session {self.id}] 数据库保存成功")
        except Exception as e:
            logger.warning(f"[Session {self.id}] 数据库保存警告: {e}")

    async def load_session(self):
        if self._loaded: return

        # 使用 Repository 加载完整数据
        data = await SessionRepository.load_full_session_data(self.id)
        
        if not data:
            logger.info(f"[Session {self.id}] 初始化新会话")
            self._loaded = True
            return
            
        session_db = data["session"]
        
        self.__name = session_db.name
        self.__role = session_db.role
        self.__aliases = session_db.aliases if session_db.aliases else []
        self.chat_summary = session_db.chat_summary
        self.global_emotion.valence = session_db.valence
        self.global_emotion.arousal = session_db.arousal
        self.global_emotion.dominance = session_db.dominance
        
        if session_db.last_speak_time:
            t = session_db.last_speak_time
            if t.tzinfo is not None:
                t = t.astimezone(None).replace(tzinfo=None)
            self._last_speak_time = t
        self.__chatting_state = _ChattingState(session_db.chatting_state)

        if "[对话样本]" in self.__role:
            parts = self.__role.split("[对话样本]")
            if len(parts) > 1:
                self.__examples_str = parts[1].strip()

        self.willingness = 0.1
        self.profiles = {}
        
        # 恢复用户画像
        for user_data in data["users"]:
            user_id = user_data["user_id"]
            profile = PersonProfile(user_id=user_id)
            profile.emotion.valence = user_data["valence"]
            profile.emotion.arousal = user_data["arousal"]
            profile.emotion.dominance = user_data["dominance"]
            profile.last_update_time = user_data["last_update_time"]
            
            for log_data in user_data["recent_logs"]:
                imp = Impression(
                    timestamp=log_data["timestamp"],
                    delta=log_data["delta"]
                )
                profile.interactions.append(imp)
            self.profiles[user_id] = profile

        # 恢复短时记忆
        # 注意：这里将数据库中的 chat_summary 同步给 Memory，确保摘要不丢失
        self.global_memory = Memory(
            llm_client=self._base_llm_client,
            compressed_message=self.chat_summary,
            messages=data["messages"]
        )

        self._loaded = True
        logger.info(f"[Session {self.id}] 加载完成")

    def presets(self) -> list[str]:
        return [f"{filename}: {preset.name} {preset.role}" for filename, preset in PRESETS.items() if not preset.hidden]

    async def load_preset(self, filename: str) -> bool:
        if not filename.endswith(".json") and f"{filename}.json" in PRESETS.keys():
            filename = f"{filename}.json"
        if filename not in PRESETS.keys(): return False

        preset = PRESETS[filename]
        base_role = preset.role
        self.__name = preset.name
        self.__aliases = preset.aliases

        if preset.examples:
            ex_lines = []
            for ex in preset.examples:
                u = ex.get("user", "")
                b = ex.get("bot", "")
                if u and b:
                    ex_lines.append(f"User: {u}\n{preset.name}: {b}")
            self.__examples_str = "\n".join(ex_lines)
        else:
            self.__examples_str = ""

        if self.__examples_str:
            self.__role = f"{base_role}\n\n[对话样本]\n{self.__examples_str}"
        else:
            self.__role = base_role

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
后台任务数: {len(self._background_tasks)}
摘要：{self.chat_summary}
最近消息：
{recent_str}
"""

    async def __search_stage(self, queries: list[str], active_user_names: list[str], use_rerank: bool = True):
        """
        优化检索阶段
        """
        logger.debug(f"检索阶段开始 (Use Rerank: {use_rerank})")

        if self.chat_summary:
            queries.append(self.chat_summary)

        if active_user_names:
            queries.extend([f"关于{name}" for name in active_user_names])

        queries = list(set([q for q in queries if q and q.strip()]))

        should_retrieve = self.willingness > 0.3

        long_term_memory = []
        if should_retrieve and queries:
            logger.debug(f"触发长期记忆检索: {queries[:5]}...")

            where_filter = {
                "$or": [
                    {"source": {"$eq": "preset"}},
                    {"source": {"$eq": "memory"}}
                ]
            }

            raw_results = await run_sync(self.long_term_memory.retrieve)(
                queries,
                k=20,
                where=where_filter,
                use_rerank=use_rerank
            )

            if raw_results:
                formatted_results = []
                total_len = 0
                max_len = 1500

                for item in raw_results:
                    if total_len > max_len: break

                    content = item.get("content", "")
                    meta = item.get("metadata", {})
                    source = meta.get("source", "unknown")
                    date_str = str(meta.get("date", ""))

                    prefix = "【设定】" if source == "preset" else f"【记忆/d:{date_str}】"
                    line = f"{prefix} {content}"

                    formatted_results.append(line)
                    total_len += len(line)

                long_term_memory = formatted_results
                logger.debug(f"搜索结果：命中 {len(long_term_memory)} 条")

        self.__search_result = _SearchResult(mem_history=long_term_memory)

    async def __feedback_stage(self, messages_chunk: list[Message], llm_func: Callable,
                               is_relevant: bool = False) -> list[str]:
        """
        反馈阶段：分析情绪、提取记忆、更新摘要
        返回：recalled_history (溯源到的历史消息列表)
        """
        logger.debug(">> 反馈阶段 (Feedback) 开始")
        recalled_history = []

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

        formatted_msgs = [f"[ID:{msg.user_id}] {msg.user_name}: '{escape_for_prompt(msg.content)}'" for msg in
                          messages_chunk]

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        all_messages = self.global_memory.access().messages
        history_msgs = [m for m in all_messages if m not in messages_chunk]
        # 格式化一下 history_msgs，使其更易读 (不再直接 dump repr)
        # 格式化为：[ID:xxx] Name: Content
        # 但 get_feedback_prompt 原本接收 list，可能需要的是 raw object list 或者 dict list？
        # 原 Prompt 定义接收 list，然后直接放入 f-string。如果是 Object list，会显示 repr。
        # 为了 LLM 友好，我们这里转换成易读的文本列表
        history_msgs_formatted = [
            f"[{m.time.strftime('%H:%M')}] {m.user_name}: {escape_for_prompt(m.content)}" 
            for m in history_msgs
        ]

        # 2. 调用 LLM (使用传入的 feedback_llm_func)
        time_str = get_time_description(datetime.now())
        prompt = get_feedback_prompt(
            self.__name, self.__role, self.willingness,
            self.__chatting_state.value,
            self.global_memory.access().compressed_history,
            history_msgs_formatted, # 传入格式化后的历史
            formatted_msgs,
            asdict(self.global_emotion),
            related_profiles_json, search_history, self.chat_summary,
            is_relevant=is_relevant,
            time_info=time_str
        )

        response_dict = {}
        # 简单的重试逻辑
        for attempt in range(2):
            try:
                response = await llm_func(prompt, json_mode=True)
                parsed = extract_and_parse_json(response)
                if parsed and isinstance(parsed, dict):
                    response_dict = parsed
                    break
            except Exception as e:
                logger.warning(f"反馈阶段 LLM 错误 (尝试 {attempt + 1}/2): {e}")
                if attempt == 1:
                    logger.error("反馈阶段最终失败，跳过本次处理")
                    return []

        # 3. 更新情绪
        new_emo = response_dict.get("new_emotion", {})
        self.global_emotion.valence = new_emo.get("valence", self.global_emotion.valence)
        self.global_emotion.arousal = new_emo.get("arousal", self.global_emotion.arousal)
        self.global_emotion.dominance = new_emo.get("dominance", self.global_emotion.dominance)

        # 4. 更新用户印象
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
                    # 1. 更新内存
                    self.profiles[uid].push_interaction(
                        Impression(timestamp=datetime.now().astimezone(), delta=delta)
                    )

                    # 2. 异步写入数据库
                    # 启动一个后台任务去存库，不阻塞主流程
                    asyncio.create_task(self._save_interaction_log(uid, delta))

        for p in self.profiles.values():
            p.update_emotion_tends()
            p.merge_old_interactions()

        # 5. 更新摘要
        self.chat_summary = str(response_dict.get("summary", self.chat_summary))
        # 同步更新到 Memory，确保下一次 Prompt 使用最新摘要
        self.global_memory.update_summary(self.chat_summary)

        # 6. 记忆提取
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

        # 6.5 主动历史溯源 (Historical Recall)
        need_history = response_dict.get("need_history", False)
        if need_history:
            logger.info(f"[Session {self.id}] 观察者请求翻阅历史记录...")
            current_msgs = self.global_memory.access().messages
            if current_msgs:
                earliest_time = current_msgs[0].time
                # 使用 Repository 查库
                recalled_msgs = await SessionRepository.get_history_before(self.id, earliest_time, limit=20)

                if recalled_msgs:
                    formatted_history = []
                    for m in recalled_msgs:
                        time_str = m.time.strftime("%H:%M")
                        formatted_history.append(f"[{time_str}] {m.user_name}: {m.content}")

                    recalled_history = formatted_history
                    logger.info(f"[Session {self.id}] 成功回溯了 {len(formatted_history)} 条历史消息")

        # 7. 更新意愿值 (带强关联兜底)
        try:
            new_willing = float(response_dict.get("willing", self.willingness))
            self.willingness = max(0.0, min(1.0, new_willing))
            if is_relevant and self.willingness < 0.8:
                self.willingness = 0.85
                logger.debug(f"[Session {self.id}] 强关联强制提升意愿值至 0.85")
        except:
            pass

        # 8. 状态流转
        random_threshold = random.uniform(0.4, 0.7)
        if self.willingness > random_threshold:
            if self.__chatting_state == _ChattingState.ILDE:
                self.__chatting_state = _ChattingState.BUBBLE
        elif self.willingness < 0.2:
            self.__chatting_state = _ChattingState.ILDE

        logger.debug(f"<< 反馈结束: 意愿 {self.willingness:.2f}, 状态 {self.__chatting_state}")
        return recalled_history

    async def __save_long_term_memory(self, analyze_result: list, default_user_id: str = ""):
        """
        后台任务：保存长期记忆到向量数据库
        优化：增加质量过滤和去重
        """
        try:
            today = int(datetime.now().strftime("%Y%m%d"))
            saved_count = 0
            skipped_quality = 0
            skipped_dedup = 0

            for item in analyze_result:
                content = ""
                uid = ""

                # 情况 1: LLM 还是返回了字符串 (Prompt 没生效或模型太笨)
                if isinstance(item, str) and item.strip():
                    content = item.strip()
                    uid = default_user_id if default_user_id else ""

                # 情况 2: LLM 返回了我们要求的标准字典
                elif isinstance(item, dict):
                    content = item.get("content", "").strip()
                    uid = str(item.get("related_user_id", ""))
                    if not uid and default_user_id:
                        uid = default_user_id

                # 质量过滤：使用 should_store_memory 函数
                if not should_store_memory(content):
                    skipped_quality += 1
                    logger.debug(f"[Memory] 跳过低质量记忆: {content[:30]}...")
                    continue

                metadata = {
                    "source": "memory",
                    "type": "event",
                    "date": today,
                    "user_id": uid
                }

                # 去重添加
                added = await run_sync(self.long_term_memory.add_memory_with_dedup)(content, metadata)
                if added:
                    saved_count += 1
                else:
                    skipped_dedup += 1

            if saved_count > 0 or skipped_quality > 0 or skipped_dedup > 0:
                logger.info(
                    f"[Memory] 存储结果: 成功 {saved_count}, 质量过滤 {skipped_quality}, 去重跳过 {skipped_dedup}"
                )
        except Exception as e:
            logger.error(f"[Async] 保存记忆失败: {e}")

    async def __chat_stage(self, messages_chunk: list[Message], llm_func: Callable,
                           recalled_history: list[str]) -> list[dict]:
        logger.debug(">> 对话阶段 (Chat) 开始")
        search_history = self.__search_result.mem_history if self.__search_result else []
        formatted_msgs = [f"[ID:{msg.id}] {msg.user_name}: '{escape_for_prompt(msg.content)}'" for msg in
                          messages_chunk]

        # 格式化回溯的历史记录
        recalled_str = "\n".join(recalled_history) if recalled_history else "无"

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        all_messages = self.global_memory.access().messages
        history_msgs = [m for m in all_messages if m not in messages_chunk]
        history_msgs_formatted = [
            f"[{m.time.strftime('%H:%M')}] {m.user_name}: {escape_for_prompt(m.content)}" 
            for m in history_msgs
        ]

        time_str = get_time_description(datetime.now())
        prompt = get_chat_prompt(
            self.__name, self.__role, self.__chatting_state.value,
            self.global_memory.access().compressed_history,
            history_msgs_formatted, # 传入格式化后的历史
            formatted_msgs,
            asdict(self.global_emotion),
            "{}",
            search_history, self.chat_summary,
            examples_text="",  # examples 已经包含在 self.__role 里了
            recalled_history=recalled_str,
            time_info=time_str
        )

        last_error = None
        for attempt in range(2):
            try:
                # 使用传入的 chat_llm_func
                response = await llm_func(prompt, json_mode=True)
                response_data = extract_and_parse_json(response)

                replies = []
                if isinstance(response_data, dict):
                    replies = response_data.get("reply", [])
                elif isinstance(response_data, list):
                    replies = response_data
                    logger.warning("LLM 返回了 List 而非 Object，已自动兼容")

                if not isinstance(replies, list):
                    return []

                if replies:
                    # 发言后意愿值大幅扣除
                    self.willingness = max(0.0, self.willingness - 0.5)
                    self.__chatting_state = _ChattingState.ACTIVE

                return replies

            except Exception as e:
                last_error = e
                logger.warning(f"对话阶段异常 (尝试 {attempt + 1}/2): {e}")

        logger.error(f"对话阶段最终失败: {last_error}")
        return []

    # 提高插话阈值，防止连击
    async def append_self_message(self, content: str, msg_id: str):
        """
        主动记录 Bot 自己的发言 (防止等待回显导致记忆延迟)
        """
        logger.debug(f"[Session {self.id}] 主动写入自身记忆: {content[:20]}... (ID: {msg_id})")
        msg = Message(
            time=datetime.now(),
            user_name=self.__name,
            content=content,
            id=msg_id,
            user_id=self.id # Bot 自己的 ID 通常就是 group_id 或者 bot_id，这里用 self.id 只是标记
        )
        # 这里的 user_id 可能需要更准确，但这主要影响 Memory.related_users
        # 暂时用 self.id 或空字符串
        
        await self.global_memory.update([msg])
        asyncio.create_task(self.save_session())

    async def update_without_trigger(self, messages_chunk: list[Message]):
        """
        仅更新记忆，不触发 LLM 回复 (用于处理回显)
        """
        if not messages_chunk: return
        logger.debug(f"[Session {self.id}] 处理回显消息 (Count: {len(messages_chunk)})")
        await self.global_memory.update(messages_chunk)
        asyncio.create_task(self.save_session())

    async def update(self, messages_chunk: list[Message],
                     chat_llm_func: Callable[[str, bool], Awaitable[str]],
                     feedback_llm_func: Callable[[str, bool], Awaitable[str]],
                     publish: bool = True) -> list[dict] | None:
        
        # 1. 更新短时记忆 (Buffer)
        # 这里的 update 不再触发后台压缩，而是单纯的 Rolling Window 更新
        await self.global_memory.update(messages_chunk)
        asyncio.create_task(self.save_session())

        if not publish: return None

        # 2. 意愿值计算
        now = datetime.now()
        seconds_passed = (now - self._last_activity_time).total_seconds()

        # 加速非活跃期的意愿衰减
        decay_rate = 0.03 if seconds_passed < 300 else 0.06
        decay = (seconds_passed / 60.0) * decay_rate
        self.willingness = max(0.0, self.willingness - decay)
        self._last_activity_time = now

        is_relevant = check_relevance(self.__name, self.__aliases, messages_chunk)
        if is_relevant:
            self.willingness = max(self.willingness, 0.95)
            logger.info("检测到强关联，意愿值提升")
        else:
            # 降低自然增长幅度，避免看别人聊天自己越来越兴奋
            # 只有当目前意愿值较低时，才允许缓慢增长，防止无上限的自我激励
            if self.willingness < 0.7:
                self.willingness = min(1.0, self.willingness + 0.03 * len(messages_chunk))

        # 3. 节流判断 (Gatekeeper)
        if self.willingness < 0.35 and not is_relevant:
            logger.debug(f"意愿值过低 ({self.willingness:.2f}) 且无强关联，跳过响应")
            return None

        last_speak = self._last_speak_time
        if last_speak.tzinfo is not None:
            last_speak = last_speak.astimezone(None).replace(tzinfo=None)

        time_since_speak = (now - last_speak).total_seconds()

        if time_since_speak < 0:
            logger.warning(f"[Session {self.id}] 检测到最后发言时间在未来 ({time_since_speak:.1f}s)，忽略冷却限制")
            # 此时视为没有冷却，允许通行
        elif time_since_speak < 10.0 and not is_relevant:
            logger.debug(f"处于发言冷却期 ({time_since_speak:.1f}s)，跳过响应")
            return None

        # 4. 检索阶段
        queries = [msg.content for msg in messages_chunk[-2:]]
        active_users = [msg.user_name for msg in messages_chunk if msg.user_name]

        # 优化策略：仅当意愿值较高 (>0.6) 或强关联 (被点名) 时才启用重排序
        # 低意愿状态下使用纯向量检索，极其省钱且足够应付普通场景
        use_rerank_strategy = self.willingness > 0.6 or is_relevant

        await self.__search_stage(queries, active_user_names=active_users, use_rerank=use_rerank_strategy)

        logger.debug("启用拟人化串行模式: Feedback -> Check -> Chat")

        # 5. Feedback 阶段 (使用 feedback_llm_func)
        recalled_history = await self.__feedback_stage(messages_chunk, feedback_llm_func, is_relevant=is_relevant)

        # 再次检查意愿 (Feedback 可能会调整意愿)
        if self.willingness < 0.4 and not is_relevant:  # 这里同步提高阈值到 0.4
            return None

        # 6. Chat 阶段 (使用 chat_llm_func)
        reply_messages = await self.__chat_stage(messages_chunk, chat_llm_func, recalled_history=recalled_history)

        if reply_messages:
            self._last_speak_time = datetime.now()

        return reply_messages

    async def _save_interaction_log(self, user_id: str, delta: dict):
        await SessionRepository.log_interaction(self.id, user_id, delta)
