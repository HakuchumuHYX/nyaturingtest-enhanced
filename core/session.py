# nyaturingtest/session.py
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import random
import time
import traceback

import httpx
from nonebot import logger
import nonebot_plugin_localstore as store
from nonebot.utils import run_sync
from openai import AsyncOpenAI

from ..llm.client import LLMClient
from ..config import get_chat_thinking_settings, get_runtime_settings
from ..models.emotion import EmotionState, clamp_vad_value
from ..memory.vector import VectorMemory, where_any
from ..models.impression import Impression
from ..memory.short_term import Memory, Message
from ..prompts.presets import PRESETS
from ..models.profile import PersonProfile
from ..utils import extract_and_parse_json, check_relevance, sanitize_text, escape_for_prompt, get_time_description, should_store_memory
from ..prompts.templates import get_feedback_prompt, get_chat_prompt
from ..database.message_repository import MessageRepository
from ..database.profile_repository import ProfileRepository
from ..database.session_repository import SessionStateRepository
from .services import RagSearchService
from .orchestrator import ConversationOrchestrator
from .structured_log import log_event


def _limit_role_text(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[内容过长，已截断]"
    return text


def _history_without_current_chunk(all_messages: list[Message], messages_chunk: list[Message]) -> list[Message]:
    chunk_message_ids = {str(msg.id) for msg in messages_chunk if msg.id}
    return [
        m for m in all_messages
        if not (
            (m.id and str(m.id) in chunk_message_ids)
            or any(m is chunk_msg for chunk_msg in messages_chunk)
        )
    ]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _active_user_query_names(
    active_user_names: list[str] | None,
    active_users: list[dict] | None,
) -> list[str]:
    result = []
    seen = set()
    seen_names = set()

    for user in active_users or []:
        if not isinstance(user, dict):
            continue
        user_id = str(user.get("user_id") or "").strip()
        user_name = str(user.get("user_name") or "").strip()
        if not user_name:
            continue
        key = f"id:{user_id}" if user_id else f"name:{user_name}"
        if key in seen or user_name in seen_names:
            continue
        seen.add(key)
        seen_names.add(user_name)
        result.append(user_name)

    for user_name in active_user_names or []:
        user_name = str(user_name or "").strip()
        if not user_name:
            continue
        key = f"name:{user_name}"
        if key in seen or user_name in seen_names:
            continue
        seen.add(key)
        seen_names.add(user_name)
        result.append(user_name)

    return result


def _active_user_scope_ids(active_users: list[dict] | None) -> set[str]:
    result = set()
    for user in active_users or []:
        if not isinstance(user, dict):
            continue
        user_id = str(user.get("user_id") or "").strip()
        if user_id:
            result.add(user_id)
    return result


@dataclass
class _SearchResult:
    mem_history: list[str]
    raw_records: list[dict] | None = None
    stats: dict | None = None


def _score_stat_fields(stats: dict) -> dict:
    return {
        "adjusted_score_min": stats.get("adjusted_score_min"),
        "adjusted_score_p50": stats.get("adjusted_score_p50"),
        "adjusted_score_p90": stats.get("adjusted_score_p90"),
        "adjusted_score_max": stats.get("adjusted_score_max"),
    }


def _rag_debug_records(records: list[dict]) -> list[dict]:
    debug_items = []
    for item in records:
        content = item.get("content", "")
        meta = item.get("metadata", {}) or {}
        debug_items.append({
            "source": meta.get("source"),
            "type": meta.get("type"),
            "subtype": meta.get("subtype"),
            "retrieval_score": meta.get("retrieval_score"),
            "rerank_score": meta.get("rerank_score"),
            "adjusted_score": meta.get("adjusted_score"),
            "days_ago": meta.get("days_ago"),
            "content_preview": str(content)[:80],
        })
    return debug_items


def _existing_related_memories(
    raw_records: list[dict] | None,
    active_user_ids: set[str],
    *,
    ids_supported: bool,
    limit: int = 5,
) -> list[dict]:
    related = []
    for item in raw_records or []:
        content = str(item.get("content") or "")
        meta = item.get("metadata", {}) or {}
        if not content or str(meta.get("source") or "memory") != "memory":
            continue
        if str(meta.get("status") or "active") != "active":
            continue
        user_id = str(meta.get("user_id") or "").strip()
        if user_id and user_id not in active_user_ids:
            continue

        memory_ref = str(meta.get("memory_ref") or "").strip()
        if ids_supported and not memory_ref:
            continue

        entry = {
            "content_preview": content[:80],
            "source": str(meta.get("source") or "memory"),
            "type": str(meta.get("type") or "event"),
            "subtype": str(meta.get("subtype") or meta.get("type") or "event"),
            "category": str(meta.get("category") or meta.get("type") or "event"),
            "confidence": meta.get("confidence", 1.0),
        }
        if ids_supported:
            entry["memory_ref"] = memory_ref
        related.append(entry)
        if len(related) >= limit:
            break
    return related


class _ChattingState(Enum):
    IDLE = 0  # 潜水
    BUBBLE = 1  # 冒泡
    ACTIVE = 2  # 活跃

    def __str__(self):
        match self:
            case _ChattingState.IDLE:
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
        self._siliconflow_api_key = siliconflow_api_key
        if http_client:
            self._client_instance = http_client
            self._owns_http_client = False
        else:
            self._client_instance = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                timeout=60.0
            )
            self._owns_http_client = True

        # 保存基础 LLM Client，供 Memory 和其他组件使用
        # NOTE: this client is for internal memory/analysis components, keep SiliconFlow OpenAI-compatible here.
        self._base_llm_client = LLMClient(
            provider="openai_compatible",
            openai_client=AsyncOpenAI(
                api_key=self._siliconflow_api_key,
                base_url="https://api.siliconflow.cn/v1",
                http_client=self._client_instance,
            ),
        )

        self.global_memory: Memory = Memory()

        self.long_term_memory: VectorMemory = VectorMemory(
            api_key=self._siliconflow_api_key,
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
        self.__chatting_state = _ChattingState.IDLE

        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        self._active_count = 0
        self._passive_observe_skips = 0
        self._loaded = False
        self._background_tasks = set()

    async def set_role(self, name: str, role: str):
        self.__role = _limit_role_text(role, get_runtime_settings()["role_max_chars"])
        self.__name = name
        self.__aliases = []
        self.__examples_str = ""
        await self.save_session()

    def role(self) -> str:
        return f"{self.__name}（{self.__role}）"

    def name(self) -> str:
        return self.__name

    def aliases(self) -> list[str]:
        return list(self.__aliases)

    async def reset(self):
        self.__name = "terminus"
        self.__aliases = []
        self.__role = "一个男性人类"
        self.__examples_str = ""
        await self.global_memory.clear()
        self.long_term_memory.clear()
        self.profiles = {}
        self.global_emotion = EmotionState()
        self.chat_summary = ""
        self.__chatting_state = _ChattingState.IDLE
        self.willingness = 0.0
        self._active_count = 0
        self._last_activity_time = datetime.now()
        self._last_speak_time = datetime.min
        # 清理数据库中的所有关联数据
        await SessionStateRepository.delete_session_data(self.id)
        await self.save_session()
        logger.info(f"[Session {self.id}] 已完全重置（含数据库清理）")

    async def calm_down(self):
        self.global_emotion = EmotionState()
        self.profiles = {}
        self.__chatting_state = _ChattingState.IDLE
        self.willingness = 0.0
        self._active_count = 0
        self._last_activity_time = datetime.now()
        await self.save_session()

    async def reset_emotion(self):
        """仅重置情绪状态（VAD），不影响意愿值、聊天状态、记忆等"""
        self.global_emotion = EmotionState()
        # 同时重置所有用户画像的情绪
        for profile in self.profiles.values():
            profile.emotion = EmotionState()
            profile.mark_dirty()
        logger.info(f"[Session {self.id}] 情绪已初始化 (VAD -> 0, 0, 0)")
        await self.save_session()

    def _create_safe_task(self, coro):
        """创建带异常捕获的后台任务"""
        task = asyncio.create_task(coro)
        task.add_done_callback(self._on_task_done)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @staticmethod
    def _on_task_done(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[Session] 后台任务异常: {exc}")

    async def save_session(self, force_index: bool = False):
        try:
            # 1. 保存基础状态
            await SessionStateRepository.save_session_state(
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

            # 2. 更新变化过的画像，避免高频保存重复写全量 profiles。
            dirty_profiles = {
                user_id: profile
                for user_id, profile in self.profiles.items()
                if profile.is_dirty
            }
            if dirty_profiles:
                await ProfileRepository.update_user_profiles(self.id, dirty_profiles)
                for profile in dirty_profiles.values():
                    profile.mark_clean()

            # 3. 同步消息
            recent_msgs = self.global_memory.access().messages
            if recent_msgs:
                await MessageRepository.sync_messages(self.id, recent_msgs)

            if force_index or random.random() < 0.01:
                await run_sync(self.long_term_memory.cleanup)(days_retention=90)

            logger.debug(f"[Session {self.id}] 数据库保存成功")
        except Exception as e:
            logger.warning(f"[Session {self.id}] 数据库保存警告: {e}")

    async def load_session(self):
        if self._loaded: return

        # 使用 Repository 加载完整数据
        data = await SessionStateRepository.load_full_session_data(self.id)
        
        if not data:
            logger.info(f"[Session {self.id}] 初始化新会话")
            self._loaded = True
            return
            
        session_db = data["session"]
        
        self.__name = session_db.name
        self.__role = _limit_role_text(session_db.role, get_runtime_settings()["role_max_chars"])
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
            profile.mark_clean()
            self.profiles[user_id] = profile

        # 恢复短时记忆
        # 注意：这里将数据库中的 chat_summary 同步给 Memory，确保摘要不丢失
        self.global_memory = Memory(
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
                    ex_lines.append(f"用户: {u}\n{preset.name}: {b}")
            self.__examples_str = _limit_role_text("\n".join(ex_lines), get_runtime_settings()["examples_max_chars"])
        else:
            self.__examples_str = ""

        if self.__examples_str:
            self.__role = _limit_role_text(
                f"{base_role}\n\n[对话样本]\n{self.__examples_str}",
                get_runtime_settings()["role_max_chars"],
            )
        else:
            self.__role = _limit_role_text(base_role, get_runtime_settings()["role_max_chars"])

        await run_sync(self.long_term_memory.delete_by_metadata)({"source": "preset"})

        preset_items: list[tuple[str, str]] = []
        preset_items.extend((item, "knowledge") for item in preset.knowledges)
        preset_items.extend((item, "relationship") for item in preset.relationships)
        preset_items.extend((item, "event") for item in preset.events)
        preset_items.extend((item, "bot_self") for item in preset.bot_self)
        to_add = [item for item, _ in preset_items]
        if to_add:
            metadatas = [
                {"source": "preset", "type": "rule", "subtype": subtype}
                for _, subtype in preset_items
            ]
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

    async def search_stage(
        self,
        queries: list[str],
        active_user_names: list[str] | None = None,
        *,
        active_users: list[dict] | None = None,
        use_rerank: bool = True,
    ):
        """
        优化检索阶段
        """
        started_at = time.perf_counter()
        logger.debug(f"检索阶段开始 (Use Rerank: {use_rerank})")
        runtime_settings = get_runtime_settings()
        rag_stats = {
            "session_id": self.id,
            "query_count": 0,
            "queries_preview": [],
            "use_rerank": bool(use_rerank),
            "skip_reason": "none",
            "fallback_reason": "none",
            "candidate_count": 0,
            "returned_count": 0,
            "injected_count": 0,
            "injected_chars": 0,
            "elapsed_ms": 0,
            "adjusted_score_min": None,
            "adjusted_score_p50": None,
            "adjusted_score_p90": None,
            "adjusted_score_max": None,
        }

        if self.chat_summary:
            queries.append(self.chat_summary)

        active_query_names = _active_user_query_names(active_user_names, active_users)
        active_scope_user_ids = _active_user_scope_ids(active_users)
        if active_query_names:
            queries.extend([f"关于{name}" for name in active_query_names])

        queries = _dedupe_preserve_order([q for q in queries if q and q.strip()])
        rag_stats["query_count"] = len(queries)
        rag_stats["queries_preview"] = [q[:40] for q in queries[:3]]

        should_retrieve = self.willingness > 0.3

        long_term_memory = []
        raw_results = []
        search_result = _SearchResult(mem_history=[], raw_records=[], stats=rag_stats)
        try:
            if not queries:
                rag_stats["skip_reason"] = "no_queries"
            elif not should_retrieve:
                rag_stats["skip_reason"] = "low_willingness"
            else:
                logger.debug(f"触发长期记忆检索: {queries[:5]}...")

                where_filter = where_any("source", ["preset", "memory"])

                raw_results = await RagSearchService(self.long_term_memory).search_for_chat(
                    queries,
                    k=runtime_settings["rag_final_k"],
                    where=where_filter,
                    use_rerank=use_rerank,
                    candidate_k=runtime_settings["rag_candidate_k"],
                    active_user_ids=active_scope_user_ids,
                )
                retrieval_stats = getattr(self.long_term_memory, "last_retrieval_stats", {}) or {}
                rag_stats.update({
                    "candidate_count": int(retrieval_stats.get("candidate_count") or 0),
                    "returned_count": int(retrieval_stats.get("returned_count") or len(raw_results)),
                    "fallback_reason": str(retrieval_stats.get("fallback_reason") or "none"),
                })
                rag_stats.update(_score_stat_fields(retrieval_stats))

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

                        if source == "preset":
                            subtype = str(meta.get("subtype") or "legacy_rule")
                            prefix = f"【设定/{subtype}】"
                        else:
                            prefix = f"【记忆/d:{date_str}】"
                        line = f"{prefix} {content}"

                        formatted_results.append(line)
                        total_len += len(line)

                    long_term_memory = formatted_results
                    rag_stats["injected_count"] = len(long_term_memory)
                    rag_stats["injected_chars"] = sum(len(item) for item in long_term_memory)
                    logger.debug(f"搜索结果：命中 {len(long_term_memory)} 条")
        finally:
            rag_stats["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
            if runtime_settings["rag_debug_log"]:
                rag_stats["result_debug"] = _rag_debug_records(raw_results)
            log_event("rag_search", **rag_stats)
            search_result = _SearchResult(
                mem_history=long_term_memory,
                raw_records=raw_results,
                stats=rag_stats,
            )
        return search_result

    async def feedback_stage(self, messages_chunk: list[Message], llm_func: Callable,
                               is_relevant: bool = False,
                               search_result: _SearchResult | None = None) -> list[str]:
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
            [{"user_id": p.user_id, "emotion_tends_to_user": asdict(p.emotion)} for p in related_profiles],
            ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        search_history = search_result.mem_history if search_result else []
        active_user_ids = {
            str(msg.user_id)
            for msg in messages_chunk
            if msg.user_id and str(msg.user_id).strip()
        }
        ids_supported = bool(getattr(self.long_term_memory, "ids_supported", False))
        existing_related_memories = _existing_related_memories(
            search_result.raw_records if search_result else [],
            active_user_ids,
            ids_supported=ids_supported,
        )
        allow_memory_supersede = ids_supported and any(
            item.get("memory_ref") for item in existing_related_memories
        )

        formatted_msgs = [f"[ID:{msg.user_id}] {msg.user_name}: '{escape_for_prompt(msg.content)}'" for msg in
                          messages_chunk]
        new_msg_speakers = [
            {
                "index": index,
                "user_id": str(msg.user_id or ""),
                "user_name": msg.user_name,
            }
            for index, msg in enumerate(messages_chunk)
        ]

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        context_record = self.global_memory.access_context(limit=get_runtime_settings()["short_context_limit"])
        all_messages = context_record.messages
        history_msgs = _history_without_current_chunk(all_messages, messages_chunk)
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
            context_record.compressed_history,
            history_msgs_formatted, # 传入格式化后的历史
            formatted_msgs,
            asdict(self.global_emotion),
            related_profiles_json, search_history, self.chat_summary,
            is_relevant=is_relevant,
            time_info=time_str,
            existing_related_memories=existing_related_memories,
            allow_memory_supersede=allow_memory_supersede,
            new_msg_speakers=new_msg_speakers,
        )

        response_dict = {}
        # 简单的重试逻辑
        for attempt in range(2):
            try:
                response = await llm_func(prompt, json_mode=True)
                # logger.debug(f"[Session {self.id}] Feedback 原始返回 (attempt {attempt+1}): {response[:500] if response else '<empty>'}")
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
        if not new_emo:
            logger.warning(f"[Session {self.id}] Feedback 未返回 new_emotion，跳过情绪更新。response_dict keys: {list(response_dict.keys())}")
        else:
            logger.debug(f"[Session {self.id}] Feedback 返回 emotion: V={new_emo.get('valence')}, A={new_emo.get('arousal')}, D={new_emo.get('dominance')}")
            self.global_emotion.valence = clamp_vad_value(new_emo.get("valence"), -1.0, 1.0, self.global_emotion.valence)
            self.global_emotion.arousal = clamp_vad_value(new_emo.get("arousal"), 0.0, 1.0, self.global_emotion.arousal)
            self.global_emotion.dominance = clamp_vad_value(new_emo.get("dominance"), -1.0, 1.0, self.global_emotion.dominance)

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
                    self._create_safe_task(self._save_interaction_log(uid, delta))

        for p in self.profiles.values():
            p.update_emotion_tends()
            p.merge_old_interactions()

        # 5. 更新摘要
        summary = response_dict.get("summary")
        if summary is not None:
            self.chat_summary = str(summary)
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

            self._create_safe_task(
                self.save_long_term_memory(
                    analyze_result,
                    default_user_id=fallback_uid,
                    supersede_candidates=existing_related_memories,
                )
            )

        # 6.5 主动历史溯源 (Historical Recall)
        need_history = response_dict.get("need_history", False)
        if need_history:
            logger.info(f"[Session {self.id}] 观察者请求翻阅历史记录...")
            current_msgs = self.global_memory.access().messages
            if current_msgs:
                earliest_time = current_msgs[0].time
                # 使用 Repository 查库
                recalled_msgs = await MessageRepository.get_history_before(
                    self.id,
                    earliest_time,
                    limit=get_runtime_settings()["history_recall_limit"],
                )

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
            relevance_floor = get_runtime_settings()["relevance_willingness_floor"]
            if is_relevant and self.willingness < relevance_floor:
                self.willingness = relevance_floor
                logger.debug(f"[Session {self.id}] 强关联强制提升意愿值至 {relevance_floor:.2f}")
        except:
            pass

        # 8. 状态流转
        runtime_settings = get_runtime_settings()
        random_threshold = random.uniform(0.4, 0.7)
        if self.willingness < 0.2:
            self.__chatting_state = _ChattingState.IDLE
        elif (
            self.__chatting_state == _ChattingState.ACTIVE
            and self.willingness < runtime_settings["active_to_bubble_threshold"]
        ):
            self.__chatting_state = _ChattingState.BUBBLE
        elif self.willingness > random_threshold:
            if self.__chatting_state == _ChattingState.IDLE:
                self.__chatting_state = _ChattingState.BUBBLE

        logger.debug(f"<< 反馈结束: 意愿 {self.willingness:.2f}, 状态 {self.__chatting_state}")
        return recalled_history

    async def save_long_term_memory(
            self,
            analyze_result: list,
            default_user_id: str = "",
            supersede_candidates: list[dict] | None = None,
    ):
        """
        后台任务：保存长期记忆到向量数据库
        优化：增加质量过滤和去重
        """
        try:
            today = int(datetime.now().strftime("%Y%m%d"))
            runtime_settings = get_runtime_settings()
            skipped_quality = 0
            superseded_count = 0
            pending_memories: list[tuple[str, dict]] = []
            allowed_supersede_refs = {
                str(item.get("memory_ref")): item
                for item in supersede_candidates or []
                if isinstance(item, dict) and item.get("memory_ref")
            }

            for item in analyze_result:
                content = ""
                subject_user_id = ""
                subject_user_name = ""
                speaker_user_id = ""
                speaker_user_name = ""
                action = "add"
                category = "event"
                confidence = 0.7
                importance = 0.5

                # 情况 1: LLM 还是返回了字符串 (Prompt 没生效或模型太笨)
                if isinstance(item, str) and item.strip():
                    content = item.strip()
                    subject_user_id = default_user_id if default_user_id else ""

                # 情况 2: LLM 返回了我们要求的标准字典
                elif isinstance(item, dict):
                    action = str(item.get("action") or "add").strip().lower()
                    if action == "ignore":
                        continue
                    if action not in {"add", "supersede"}:
                        logger.debug(f"[Memory] 暂不处理的记忆 action: {action}")
                        continue
                    content = item.get("content", "").strip()
                    subject_user_id = str(item.get("subject_user_id") or item.get("related_user_id") or "").strip()
                    subject_user_name = str(item.get("subject_user_name") or "").strip()
                    speaker_user_id = str(item.get("speaker_user_id") or "").strip()
                    speaker_user_name = str(item.get("speaker_user_name") or "").strip()
                    if not subject_user_id and default_user_id:
                        subject_user_id = default_user_id
                    category = str(item.get("category") or "event").strip() or "event"
                    try:
                        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.7))))
                    except (TypeError, ValueError):
                        confidence = 0.7
                    try:
                        importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
                    except (TypeError, ValueError):
                        importance = 0.5

                if action == "supersede":
                    target_ref = str(item.get("target_ref") or "").strip() if isinstance(item, dict) else ""
                    candidate = allowed_supersede_refs.get(target_ref)
                    if not candidate:
                        log_event(
                            "rag_action_hallucination",
                            session_id=self.id,
                            action=action,
                            target_ref=target_ref,
                            reason="target_ref_not_in_current_candidates",
                        )
                        continue

                    target_metadata = await run_sync(self.long_term_memory.get_metadata_by_id)(target_ref)
                    if not target_metadata:
                        log_event(
                            "rag_action_hallucination",
                            session_id=self.id,
                            action=action,
                            target_ref=target_ref,
                            reason="target_ref_missing_in_vector_store",
                        )
                        continue

                    target_source = str(target_metadata.get("source") or candidate.get("source") or "memory")
                    target_type = str(target_metadata.get("type") or candidate.get("type") or "event")
                    target_subtype = str(target_metadata.get("subtype") or candidate.get("subtype") or target_type)
                    target_category = str(target_metadata.get("category") or candidate.get("category") or target_type)
                    allowed_target_types = {"event", "preference", "profile", "relationship"}
                    if (
                        target_source != "memory"
                        or target_subtype == "bot_self"
                        or (target_type not in allowed_target_types and target_category not in allowed_target_types)
                    ):
                        log_event(
                            "rag_action_rejected",
                            session_id=self.id,
                            action=action,
                            target_ref=target_ref,
                            source=target_source,
                            type=target_type,
                            subtype=target_subtype,
                            category=target_category,
                            reason="target_not_supersedable",
                        )
                        continue

                # 质量过滤：使用 should_store_memory 函数
                if not should_store_memory(content):
                    skipped_quality += 1
                    logger.debug(f"[Memory] 跳过低质量记忆: {content[:30]}...")
                    continue

                metadata = {
                    "schema_version": 2,
                    "source": "memory",
                    "type": category,
                    "date": today,
                    "user_id": subject_user_id,
                    "subject_user_id": subject_user_id,
                    "subject_user_name": subject_user_name,
                    "speaker_user_id": speaker_user_id,
                    "speaker_user_name": speaker_user_name,
                    "status": "active",
                    "category": category,
                    "confidence": confidence,
                    "importance": importance,
                    "ttl_days": runtime_settings["rag_default_event_ttl_days"],
                }

                if action == "supersede":
                    target_ref = str(item.get("target_ref") or "").strip()
                    metadata["supersedes"] = target_ref
                    await run_sync(self.long_term_memory.add_texts)([content], metadatas=[metadata])
                    updated_target_metadata = dict(target_metadata)
                    updated_target_metadata["status"] = "superseded"
                    updated_target_metadata["superseded_at"] = datetime.now().astimezone().isoformat()
                    updated_target_metadata["superseded_reason"] = str(item.get("reason") or "")[:200]
                    await run_sync(self.long_term_memory.update_metadata_by_id)(target_ref, updated_target_metadata)
                    superseded_count += 1
                else:
                    pending_memories.append((content, metadata))

            store_result = {"added": 0, "skipped_dedup": 0}
            if pending_memories:
                store_result = await run_sync(self.long_term_memory.add_memories_with_dedup)(pending_memories)

            saved_count = store_result.get("added", 0)
            skipped_dedup = store_result.get("skipped_dedup", 0)
            if saved_count > 0 or skipped_quality > 0 or skipped_dedup > 0 or superseded_count > 0:
                logger.info(
                    f"[Memory] 存储结果: 成功 {saved_count}, 替换 {superseded_count}, 质量过滤 {skipped_quality}, 去重跳过 {skipped_dedup}"
                )
        except Exception as e:
            logger.error(f"[Async] 保存记忆失败: {e}")

    async def chat_stage(self, messages_chunk: list[Message], llm_func: Callable,
                           recalled_history: list[str],
                           search_result: _SearchResult | None = None) -> list[dict]:
        logger.debug(">> 对话阶段 (Chat) 开始")
        search_history = search_result.mem_history if search_result else []
        formatted_msgs = [f"[ID:{msg.id}] {msg.user_name}: '{escape_for_prompt(msg.content)}'" for msg in
                          messages_chunk]

        # 格式化回溯的历史记录
        recalled_str = "\n".join(recalled_history) if recalled_history else "无"

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        context_record = self.global_memory.access_context(limit=get_runtime_settings()["short_context_limit"])
        all_messages = context_record.messages
        history_msgs = _history_without_current_chunk(all_messages, messages_chunk)
        history_msgs_formatted = [
            f"[{m.time.strftime('%H:%M')}] {m.user_name}: {escape_for_prompt(m.content)}" 
            for m in history_msgs
        ]

        time_str = get_time_description(datetime.now())
        # 分离 role 和 examples：role 中可能包含 [对话样本] 后缀，需要去除避免重复
        chat_role = self.__role.split("[对话样本]")[0].strip() if "[对话样本]" in self.__role else self.__role
        reaction_users = list({msg.user_id if msg.user_id else msg.user_name for msg in messages_chunk})
        related_profiles = [self.profiles.get(uid, PersonProfile(user_id=uid)) for uid in reaction_users]
        related_profiles_json = json.dumps(
            [{"user_id": p.user_id, "emotion_tends_to_user": asdict(p.emotion)} for p in related_profiles],
            ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        prompt = get_chat_prompt(
            self.__name, chat_role, self.__chatting_state.value,
            context_record.compressed_history,
            history_msgs_formatted, # 传入格式化后的历史
            formatted_msgs,
            asdict(self.global_emotion),
            related_profiles_json,
            search_history, self.chat_summary,
            examples_text=self.__examples_str,
            recalled_history=recalled_str,
            time_info=time_str,
            rp_style=get_chat_thinking_settings().get("rp_style", "off"),
        )
        log_event("rag_prompt_budget",
            session_id=self.id,
            chat_prompt_total_chars=len(prompt),
            rag_injected_count=len(search_history),
            rag_injected_chars=sum(len(item) for item in search_history),
            history_chars=len(context_record.compressed_history or "") + sum(len(item) for item in history_msgs_formatted),
            recent_chars=sum(len(item) for item in formatted_msgs),
            recalled_history_chars=len(recalled_str),
            examples_chars=len(self.__examples_str or ""),
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
    async def append_self_message(self, content: str, msg_id: str, bot_user_id: str):
        """
        主动记录 Bot 自己的发言 (防止等待回显导致记忆延迟)
        """
        logger.debug(f"[Session {self.id}] 主动写入自身记忆: {content[:20]}... (ID: {msg_id})")
        msg = Message(
            time=datetime.now(),
            user_name=self.__name,
            content=content,
            id=msg_id,
            user_id=bot_user_id
        )
        
        await self.global_memory.update([msg])
        self._create_safe_task(self.save_session())

    async def update_without_trigger(self, messages_chunk: list[Message]):
        """
        仅更新记忆，不触发 LLM 回复 (用于处理回显)
        """
        if not messages_chunk: return
        logger.debug(f"[Session {self.id}] 处理回显消息 (Count: {len(messages_chunk)})")
        await self.global_memory.update(messages_chunk)
        self._create_safe_task(self.save_session())

    async def drain_background_tasks(self, timeout: float = 10.0):
        pending = [task for task in self._background_tasks if not task.done()]
        if not pending:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[Session {self.id}] 等待后台任务超时，取消 {len(pending)} 个任务")
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self):
        try:
            self.long_term_memory.close()
        except Exception as e:
            logger.warning(f"[Session {self.id}] 关闭向量记忆失败: {e}")
        if getattr(self, "_client_instance", None) and getattr(self, "_owns_http_client", False):
            try:
                await self._client_instance.aclose()
            except Exception as e:
                logger.warning(f"[Session {self.id}] 关闭 HTTP 客户端失败: {e}")

    async def update(self, messages_chunk: list[Message],
                     chat_llm_func: Callable[[str, bool], Awaitable[str]],
                     feedback_llm_func: Callable[[str, bool], Awaitable[str]],
                     publish: bool = True) -> list[dict] | None:
        return await ConversationOrchestrator(self).process_chunk(
            messages_chunk,
            chat_llm_func,
            feedback_llm_func,
            publish=publish,
        )

    async def _save_interaction_log(self, user_id: str, delta: dict):
        await ProfileRepository.log_interaction(self.id, user_id, delta)
