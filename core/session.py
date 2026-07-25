# nyaturingtest/session.py
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import random
import time
import traceback

import httpx
from nonebot import logger
import nonebot_plugin_localstore as store
from nonebot.utils import run_sync
from ..config import get_chat_thinking_settings, get_runtime_settings
from .. import config as config_module
from ..models.emotion import EmotionState, clamp_vad_value
from ..memory.vector import VectorMemory, where_any
from ..memory.validation import validate_memory_candidate
from ..models.impression import Impression
from ..memory.short_term import Memory, Message
from ..prompts.presets import PRESETS, reload_presets
from ..models.profile import PersonProfile
from .text_utils import (
    check_relevance,
    extract_and_parse_json,
    sanitize_text,
    should_store_memory,
)
from .time_context import get_time_description
from ..prompts.templates import PromptBudget, get_feedback_prompt, get_chat_prompt
from ..database.message_repository import MessageRepository
from ..database.profile_repository import ProfileRepository
from ..database.session_repository import SessionStateRepository
from ..database.backup_lock import BACKUP_IO_LOCK
from .services import RagSearchService
from .orchestrator import ConversationOrchestrator
from .structured_log import log_event
from .rag_query import build_chat_rag_queries
from .turn_models import FeedbackOutcome
from .persistence import PersistenceCoordinator
from .session_state import ChattingState as _ChattingState, SessionState
from .session_runtime import SessionRuntime
from .feedback_processor import FeedbackProcessor
from .chat_planner import ChatPlanner
from .memory_writer import MemoryWriteService


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


def _message_image_refs(message: Message) -> list[str]:
    getter = getattr(message, "image_refs", None)
    if callable(getter):
        return getter()
    return [
        str(getattr(item, "ref_id", "") or "")
        for item in (getattr(message, "image_inputs", []) or [])
        if getattr(item, "ref_id", "")
    ]


def _endpoint_uses_native_vision(endpoint_name: str) -> bool:
    getter = getattr(config_module, "get_vision_settings", None)
    if not callable(getter):
        return False
    return bool(getter(endpoint_name).get("enabled", False))


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


@dataclass
class _FeedbackContext:
    response_dict: dict
    existing_related_memories: list[dict]
    allow_memory_supersede: bool
    active_user_ids: set[str]


_STALE_GENERATION_WRITE = object()


def _normalize_feedback_response(
    parsed: object,
    current_emotion: EmotionState,
) -> tuple[dict | None, str]:
    """Compatibility facade for callers that already parsed Feedback JSON."""

    result = FeedbackProcessor.parse(
        json.dumps(parsed, ensure_ascii=False),
        current_emotion,
    )
    return result.payload, result.failure_reason


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
        subject_user_id = str(meta.get("subject_user_id") or meta.get("user_id") or "").strip()
        if subject_user_id and subject_user_id not in active_user_ids:
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
            "subject_user_id": subject_user_id,
            "subject_user_name": str(meta.get("subject_user_name") or ""),
            "speaker_user_id": str(meta.get("speaker_user_id") or ""),
            "speaker_user_name": str(meta.get("speaker_user_name") or ""),
        }
        if ids_supported:
            entry["memory_ref"] = memory_ref
        related.append(entry)
        if len(related) >= limit:
            break
    return related


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

        runtime_settings = get_runtime_settings()
        short_term_memory = Memory(
            context_limit=runtime_settings["short_context_limit"],
            buffer_size=runtime_settings["short_term_buffer_size"],
        )

        vector_memory = VectorMemory(
            api_key=self._siliconflow_api_key,
            persist_directory=f"{store.get_plugin_data_dir()}/vector_index_{id}",
        )
        self._state = SessionState(name=name)
        self._runtime = SessionRuntime(
            short_term_memory=short_term_memory,
            vector_memory=vector_memory,
            http_client=self._client_instance,
            owns_http_client=self._owns_http_client,
            memory_writer=MemoryWriteService(vector_memory),
        )
        self._runtime.persistence = PersistenceCoordinator(
            self._save_coordinated,
            task_factory=self._create_safe_task,
        )

    def _domain_state(self) -> SessionState:
        state = self.__dict__.get("_state")
        if state is None:
            state = SessionState()
            self.__dict__["_state"] = state
        return state

    def _runtime_state(self) -> SessionRuntime:
        runtime = self.__dict__.get("_runtime")
        if runtime is None:
            runtime = SessionRuntime()
            self.__dict__["_runtime"] = runtime
        return runtime

    _Session__name = property(
        lambda self: self._domain_state().name,
        lambda self, value: setattr(self._domain_state(), "name", value),
    )
    _Session__role = property(
        lambda self: self._domain_state().role,
        lambda self, value: setattr(self._domain_state(), "role", value),
    )
    _Session__aliases = property(
        lambda self: self._domain_state().aliases,
        lambda self, value: setattr(self._domain_state(), "aliases", value),
    )
    _Session__examples_str = property(
        lambda self: self._domain_state().examples,
        lambda self, value: setattr(self._domain_state(), "examples", value),
    )
    _Session__chatting_state = property(
        lambda self: self._domain_state().chatting_state,
        lambda self, value: setattr(self._domain_state(), "chatting_state", value),
    )
    profiles = property(
        lambda self: self._domain_state().profiles,
        lambda self, value: setattr(self._domain_state(), "profiles", value),
    )
    global_emotion = property(
        lambda self: self._domain_state().global_emotion,
        lambda self, value: setattr(self._domain_state(), "global_emotion", value),
    )
    chat_summary = property(
        lambda self: self._domain_state().chat_summary,
        lambda self, value: setattr(self._domain_state(), "chat_summary", value),
    )
    willingness = property(
        lambda self: self._domain_state().willingness,
        lambda self, value: setattr(self._domain_state(), "willingness", value),
    )
    generation = property(
        lambda self: self._domain_state().generation,
        lambda self, value: setattr(self._domain_state(), "generation", value),
    )
    _last_activity_time = property(
        lambda self: self._domain_state().last_activity_time,
        lambda self, value: setattr(self._domain_state(), "last_activity_time", value),
    )
    _last_decay_time = property(
        lambda self: self._domain_state().last_decay_time,
        lambda self, value: setattr(self._domain_state(), "last_decay_time", value),
    )
    _last_speak_time = property(
        lambda self: self._domain_state().last_speak_time,
        lambda self, value: setattr(self._domain_state(), "last_speak_time", value),
    )
    _active_count = property(
        lambda self: self._domain_state().active_count,
        lambda self, value: setattr(self._domain_state(), "active_count", value),
    )
    _engaged = property(
        lambda self: self._domain_state().engaged,
        lambda self, value: setattr(self._domain_state(), "engaged", value),
    )
    last_consolidated_time = property(
        lambda self: self._domain_state().last_consolidated_time,
        lambda self, value: setattr(
            self._domain_state(),
            "last_consolidated_time",
            value,
        ),
    )
    _messages_since_consolidation = property(
        lambda self: self._domain_state().messages_since_consolidation,
        lambda self, value: setattr(
            self._domain_state(),
            "messages_since_consolidation",
            value,
        ),
    )
    _last_consolidation_attempt = property(
        lambda self: self._domain_state().last_consolidation_attempt,
        lambda self, value: setattr(
            self._domain_state(),
            "last_consolidation_attempt",
            value,
        ),
    )
    _loaded = property(
        lambda self: self._domain_state().loaded,
        lambda self, value: setattr(self._domain_state(), "loaded", value),
    )
    global_memory = property(
        lambda self: self._runtime_state().short_term_memory,
        lambda self, value: setattr(self._runtime_state(), "short_term_memory", value),
    )
    long_term_memory = property(
        lambda self: self._runtime_state().vector_memory,
        lambda self, value: setattr(self._runtime_state(), "vector_memory", value),
    )
    _background_tasks = property(
        lambda self: self._runtime_state().background_tasks,
        lambda self, value: setattr(self._runtime_state(), "background_tasks", value),
    )
    _save_lock = property(
        lambda self: self._runtime_state().save_lock,
        lambda self, value: setattr(self._runtime_state(), "save_lock", value),
    )
    _persistence = property(
        lambda self: self._runtime_state().persistence,
        lambda self, value: setattr(self._runtime_state(), "persistence", value),
    )

    @property
    def memory_writer(self) -> MemoryWriteService:
        writer = self._runtime_state().memory_writer
        if writer is None or writer.vector_memory is not self.long_term_memory:
            writer = MemoryWriteService(self.long_term_memory)
            self._runtime_state().memory_writer = writer
        return writer

    def bump_generation(self, reason: str = "") -> int:
        self.generation += 1
        log_event("session_generation_bumped",
            session_id=self.id,
            generation=self.generation,
            reason=reason,
        )
        return self.generation

    def is_generation_stale(self, expected_generation: int | None) -> bool:
        return expected_generation is not None and self.generation != expected_generation

    def _log_stale_generation(self, stage: str, expected_generation: int | None) -> None:
        log_event("stale_turn_discarded",
            session_id=self.id,
            stage=stage,
            expected_generation=expected_generation,
            current_generation=self.generation,
        )

    async def _run_sync_if_generation_current(
        self,
        func,
        *args,
        expected_generation: int | None = None,
        stage: str,
        **kwargs,
    ):
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation(stage, expected_generation)
            return _STALE_GENERATION_WRITE

        def guarded():
            with BACKUP_IO_LOCK:
                if self.is_generation_stale(expected_generation):
                    return _STALE_GENERATION_WRITE
                return func(*args, **kwargs)

        result = await run_sync(guarded)()
        if result is _STALE_GENERATION_WRITE:
            self._log_stale_generation(stage, expected_generation)
        return result

    async def set_role(self, name: str, role: str):
        self.bump_generation("set_role")
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
        self.bump_generation("reset")
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
        self._last_decay_time = datetime.now()
        self._last_speak_time = datetime.min
        self._engaged = False
        self.last_consolidated_time = None
        self._messages_since_consolidation = 0
        self._last_consolidation_attempt = datetime.min
        # 清理数据库中的所有关联数据，并与后台持久化共用同一把锁：
        # 旧 generation 的后台写入要么已在删除前完成，要么拿锁后被跳过。
        async with self._save_lock:
            await SessionStateRepository.delete_session_data(self.id)
            await self._save_session_locked()
        logger.info(f"[Session {self.id}] 已完全重置（含数据库清理）")

    async def calm_down(self):
        self.bump_generation("calm_down")
        self.global_emotion = EmotionState()
        self.profiles = {}
        self.__chatting_state = _ChattingState.IDLE
        self.willingness = 0.0
        self._active_count = 0
        self._last_activity_time = datetime.now()
        self._engaged = False
        await self.save_session()

    async def reset_emotion(self):
        """仅重置情绪状态（VAD），不影响意愿值、聊天状态、记忆等"""
        self.bump_generation("reset_emotion")
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

    def _schedule_save_session(self, force_index: bool = False):
        coordinator = self._get_persistence_coordinator()
        coordinator.request(force_index=force_index)
        return coordinator

    def _get_persistence_coordinator(self) -> PersistenceCoordinator:
        coordinator = getattr(self, "_persistence", None)
        if coordinator is None:
            coordinator = PersistenceCoordinator(
                self._save_coordinated,
                task_factory=self._create_safe_task,
            )
            self._persistence = coordinator
        return coordinator

    def begin_persistence_batch(self) -> None:
        self._get_persistence_coordinator().begin_batch()

    async def end_persistence_batch(self, *, flush: bool = False) -> bool:
        return await self._get_persistence_coordinator().end_batch(flush=flush)

    async def flush_persistence(self) -> bool:
        return await self._get_persistence_coordinator().flush()

    async def _save_coordinated(self, force_index: bool = False) -> bool:
        async with self._save_lock:
            return await self._save_session_locked(force_index=force_index)

    @staticmethod
    def _on_task_done(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[Session] 后台任务异常: {exc}")

    async def save_session(
        self,
        force_index: bool = False,
        expected_generation: int | None = None,
    ) -> bool:
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("save_session", expected_generation)
            return False
        async with self._save_lock:
            if self.is_generation_stale(expected_generation):
                self._log_stale_generation("save_session_locked", expected_generation)
                return False
            result = await self._save_session_locked(force_index=force_index)
            if result and getattr(self, "_persistence", None) is not None:
                self._persistence.mark_current_persisted()
            return result

    async def _save_session_locked(self, force_index: bool = False) -> bool:
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
                    "last_consolidated_time": self.last_consolidated_time,
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

            # 3. 只同步新增或内容被图片观察丰富过的消息。
            pending_messages = self.global_memory.pending_messages()
            if pending_messages:
                await MessageRepository.sync_messages(
                    self.id,
                    [message for message, _ in pending_messages],
                )
                self.global_memory.mark_persisted(pending_messages)

            logger.debug(f"[Session {self.id}] 数据库保存成功")
            return True
        except Exception as e:
            logger.warning(f"[Session {self.id}] 数据库保存警告: {e}")
            return False

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
        self.last_consolidated_time = session_db.last_consolidated_time
        self.__chatting_state = _ChattingState(session_db.chatting_state)

        if "[对话样本]" in self.__role:
            parts = self.__role.split("[对话样本]")
            if len(parts) > 1:
                self.__examples_str = parts[1].strip()

        self.willingness = get_runtime_settings()["willingness_load_value"]
        # 重启一致性：低意愿时强制回到潜水态，避免「状态=对话中但意愿=静音」的矛盾
        if self.willingness < get_runtime_settings()["low_willingness_skip_threshold"]:
            self.__chatting_state = _ChattingState.IDLE
            self._engaged = False
        self.profiles = {}
        
        # 恢复用户画像
        for user_data in data["users"]:
            user_id = user_data["user_id"]
            profile = PersonProfile(user_id=user_id)
            profile.emotion.valence = user_data["valence"]
            profile.emotion.arousal = user_data["arousal"]
            profile.emotion.dominance = user_data["dominance"]
            profile.last_update_time = user_data["last_update_time"]
            profile.interaction_count = int(user_data.get("interaction_count") or 0)
            profile.first_interaction_at = user_data.get("first_interaction_at")
            profile.last_interaction_at = user_data.get("last_interaction_at")

            profile.mark_clean()
            self.profiles[user_id] = profile

        # 恢复短时记忆
        # 注意：这里将数据库中的 chat_summary 同步给 Memory，确保摘要不丢失
        rt = get_runtime_settings()
        self.global_memory = Memory(
            compressed_message=self.chat_summary,
            messages=data["messages"],
            context_limit=rt["short_context_limit"],
            buffer_size=rt["short_term_buffer_size"],
        )

        self._loaded = True
        logger.info(f"[Session {self.id}] 加载完成")

    def presets(self) -> list[str]:
        reload_presets()
        return [
            f"{filename}: {preset.name} {preset.role}"
            for filename, preset in PRESETS.items()
            if not preset.hidden
        ]

    async def load_preset(self, filename: str) -> bool:
        reload_presets()
        if not filename.endswith(".json") and f"{filename}.json" in PRESETS.keys():
            filename = f"{filename}.json"
        if filename not in PRESETS:
            return False

        self.bump_generation("load_preset")
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
        force_retrieve: bool = False,
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
            "other_subject_downweighted_count": 0,
            "legacy_subject_count": 0,
            "scope_counts": {},
        }

        active_scope_user_ids = _active_user_scope_ids(active_users)
        queries = build_chat_rag_queries(
            queries,
            chat_summary=self.chat_summary,
            active_user_names=active_user_names,
            active_users=active_users,
        )
        rag_stats["query_count"] = len(queries)
        rag_stats["queries_preview"] = [q[:40] for q in queries[:3]]

        should_retrieve = force_retrieve or self.willingness > runtime_settings["low_willingness_skip_threshold"]

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

                retrieval_result = await RagSearchService(self.long_term_memory).search_for_chat(
                    queries,
                    k=runtime_settings["rag_final_k"],
                    where=where_filter,
                    use_rerank=use_rerank,
                    candidate_k=runtime_settings["rag_per_query_recall_k"],
                    merged_candidate_cap=runtime_settings["rag_merged_candidate_cap"],
                    active_user_ids=active_scope_user_ids,
                )
                raw_results = retrieval_result.records
                retrieval_stats = retrieval_result.stats
                rag_stats.update({
                    "candidate_count": int(retrieval_stats.get("candidate_count") or 0),
                    "returned_count": int(retrieval_stats.get("returned_count") or len(raw_results)),
                    "fallback_reason": str(retrieval_stats.get("fallback_reason") or "none"),
                    "other_subject_downweighted_count": int(retrieval_stats.get("other_subject_downweighted_count") or 0),
                    "legacy_subject_count": int(retrieval_stats.get("legacy_subject_count") or 0),
                    "scope_counts": dict(retrieval_stats.get("scope_counts") or {}),
                })
                rag_stats.update(_score_stat_fields(retrieval_stats))

                if raw_results:
                    formatted_results = []
                    total_len = 0
                    max_len = runtime_settings["rag_memory_char_budget"]

                    for item in raw_results:
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

                        remaining = max_len - total_len
                        if remaining <= 0:
                            break
                        if len(line) > remaining:
                            line = line[:remaining].rstrip()
                        if not line:
                            break
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

    async def _run_feedback_llm(
            self,
            messages_chunk: list[Message],
            llm_func: Callable,
            is_relevant: bool = False,
            search_result: _SearchResult | None = None,
    ) -> tuple[_FeedbackContext | None, str]:
        """运行 Feedback LLM 并返回可复用的分析上下文。"""
        reaction_users = list({msg.user_id if msg.user_id else msg.user_name for msg in messages_chunk})
        related_profiles = [self.profiles.get(uid, PersonProfile(user_id=uid)) for uid in reaction_users]
        for p in related_profiles:
            if p.user_id not in self.profiles:
                self.profiles[p.user_id] = p

        related_profiles_data = [
            {"user_id": p.user_id, "emotion_tends_to_user": asdict(p.emotion)}
            for p in related_profiles
        ]
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

        formatted_msgs = [
            {
                "id": str(msg.user_id or ""),
                "name": msg.user_name,
                "content": msg.content,
                "image_meta": msg.image_meta,
                "image_refs": (
                    _message_image_refs(msg)
                    if _endpoint_uses_native_vision("feedback")
                    else []
                ),
            }
            for msg in messages_chunk
        ]
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
        # 历史消息格式化为结构化 dict（含 image_meta，重启后/过窗口的图多为 None）
        history_msgs_formatted = [
            {
                "time": m.time.strftime('%H:%M'),
                "name": m.user_name,
                "content": m.content,
                "image_meta": m.image_meta,
                "image_refs": [],
            }
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
            related_profiles_data, search_history, self.chat_summary,
            is_relevant=is_relevant,
            time_info=time_str,
            existing_related_memories=existing_related_memories,
            allow_memory_supersede=allow_memory_supersede,
            new_msg_speakers=new_msg_speakers,
            budget=PromptBudget.from_runtime(get_runtime_settings()),
        )

        try:
            response = await llm_func(prompt, json_mode=True)
        except Exception as e:
            logger.error(f"反馈阶段 LLM 错误，跳过本次处理: {e}")
            return None, "llm_error"

        parsed_feedback = FeedbackProcessor.parse(response, self.global_emotion)
        response_dict = parsed_feedback.payload
        if response_dict is None:
            log_event(
                "feedback_rejected",
                session_id=self.id,
                failure_reason=parsed_feedback.failure_reason,
            )
            return None, parsed_feedback.failure_reason

        expected_feedback_fields = [
            "analyze_result",
            "willing",
            "new_emotion",
            "emotion_tends",
            "summary",
            "need_history",
        ]
        missing_feedback_fields = [
            field for field in expected_feedback_fields
            if field not in response_dict
        ]
        if missing_feedback_fields:
            log_event("feedback_fields_missing",
                session_id=self.id,
                missing_feedback_fields=missing_feedback_fields,
                response_keys=sorted(str(key) for key in response_dict.keys()),
            )

        return (
            _FeedbackContext(
                response_dict=response_dict,
                existing_related_memories=existing_related_memories,
                allow_memory_supersede=allow_memory_supersede,
                active_user_ids=active_user_ids,
            ),
            "",
        )

    def _apply_native_image_observations(
        self,
        response_dict: dict,
        messages_chunk: list[Message],
    ) -> None:
        from ..memory.image_schema import (
            merge_segment_metas,
            parse_vlm_response,
            render_image_text,
        )

        raw_observations = response_dict.get("image_observations", [])
        if not isinstance(raw_observations, list):
            return
        observations_by_ref = {
            str(item.get("image_ref") or ""): item
            for item in raw_observations
            if isinstance(item, dict) and str(item.get("image_ref") or "")
        }
        if not observations_by_ref:
            return

        for msg in messages_chunk:
            if msg.image_meta or not msg.image_inputs:
                continue
            segment_metas = []
            rendered_labels = []
            observed_inputs = []
            for image_input in msg.image_inputs:
                image_ref = str(getattr(image_input, "ref_id", "") or "")
                observation = observations_by_ref.get(image_ref)
                if not observation:
                    continue
                description = parse_vlm_response(
                    json.dumps(observation, ensure_ascii=False),
                    is_sticker=bool(getattr(image_input, "is_sticker", False)),
                )
                meta = description.to_meta()
                meta["image_ref"] = image_ref
                if getattr(image_input, "source", "primary") == "referenced":
                    segment_metas.append({"referenced": [meta]})
                else:
                    segment_metas.append(meta)
                rendered_labels.append(
                    render_image_text(description, description.is_sticker)
                )
                observed_inputs.append(image_input)

            merged_meta = merge_segment_metas(segment_metas)
            if not merged_meta:
                continue
            msg.image_meta = merged_meta
            enriched_content = msg.content
            for image_input in observed_inputs:
                placeholder = "[表情包]" if getattr(image_input, "is_sticker", False) else "[图片]"
                enriched_content = enriched_content.replace(placeholder, "", 1)
            msg.content = f"{enriched_content.strip()}{''.join(rendered_labels)}".strip()
            mark_dirty = getattr(self.global_memory, "mark_dirty", None)
            if mark_dirty is not None:
                mark_dirty(msg)

    def _apply_sediment(
        self,
        ctx: _FeedbackContext,
        messages_chunk: list[Message],
        expected_generation: int | None = None,
    ) -> None:
        """应用 Feedback 的沉淀结果：情绪、画像、摘要、长期记忆。"""
        response_dict = ctx.response_dict

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
        interaction_updates: list[tuple[str, dict]] = []
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
                    interaction_updates.append((uid, delta))

        if interaction_updates:
            self._create_safe_task(
                self._save_interaction_logs(
                    interaction_updates,
                    expected_generation=expected_generation,
                )
            )

        for p in self.profiles.values():
            p.update_emotion_tends()
            p.merge_old_interactions()

        # 5. 更新摘要
        summary = response_dict.get("summary")
        if summary is not None:
            prompt_budget = PromptBudget.from_runtime(get_runtime_settings())
            self.chat_summary = str(summary)[:prompt_budget.summary_chars]
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
                    supersede_candidates=ctx.existing_related_memories,
                    expected_generation=expected_generation,
                )
            )

    async def _apply_decision(
            self,
            ctx: _FeedbackContext,
            messages_chunk: list[Message],
            is_relevant: bool,
            expected_generation: int | None = None,
    ) -> list[str]:
        """应用 Feedback 的发言决策结果：历史溯源、意愿、状态。"""
        response_dict = ctx.response_dict
        recalled_history = []

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

        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("feedback_decision", expected_generation)
            return []

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

        return recalled_history

    async def feedback_stage(self, messages_chunk: list[Message], llm_func: Callable,
                               is_relevant: bool = False,
                               search_result: _SearchResult | None = None,
                               expected_generation: int | None = None) -> FeedbackOutcome:
        """
        反馈阶段：分析情绪、提取记忆、更新摘要
        返回：recalled_history (溯源到的历史消息列表)
        """
        logger.debug(">> 反馈阶段 (Feedback) 开始")
        ctx, failure_reason = await self._run_feedback_llm(
            messages_chunk,
            llm_func,
            is_relevant,
            search_result,
        )
        if ctx is None:
            return FeedbackOutcome.rejected(failure_reason)
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("feedback_sediment", expected_generation)
            return FeedbackOutcome.rejected("stale_generation")
        self._apply_native_image_observations(ctx.response_dict, messages_chunk)
        self._apply_sediment(ctx, messages_chunk, expected_generation=expected_generation)
        recalled_history = await self._apply_decision(ctx, messages_chunk, is_relevant, expected_generation)
        if self.is_generation_stale(expected_generation):
            return FeedbackOutcome.rejected("stale_generation")
        logger.debug(f"<< 反馈结束: 意愿 {self.willingness:.2f}, 状态 {self.__chatting_state}")
        return FeedbackOutcome(
            accepted=True,
            recalled_history=recalled_history,
            state_changed=True,
        )

    async def consolidate_stage(
        self,
        messages_chunk: list[Message],
        feedback_llm_func: Callable,
        expected_generation: int | None = None,
    ) -> FeedbackOutcome:
        """常驻记忆固化：分析+沉淀，但不改回复意愿、不做发言决策。"""
        if not messages_chunk:
            return FeedbackOutcome.rejected("no_messages")
        self._last_consolidation_attempt = datetime.now()
        logger.debug(f"[Session {self.id}] >> 记忆固化 (Consolidate) {len(messages_chunk)} 条")
        queries = [m.content for m in reversed(messages_chunk[-3:])]
        active_user_names = [m.user_name for m in messages_chunk if m.user_name]
        active_users = [
            {"user_id": str(m.user_id or ""), "user_name": m.user_name}
            for m in messages_chunk if m.user_name
        ]
        search_result = await self.search_stage(
            queries,
            active_user_names=active_user_names,
            active_users=active_users,
            use_rerank=False,
            force_retrieve=True,
        )
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("consolidation_search", expected_generation)
            return FeedbackOutcome.rejected("stale_generation")
        ctx, failure_reason = await self._run_feedback_llm(
            messages_chunk,
            feedback_llm_func,
            is_relevant=False,
            search_result=search_result,
        )
        if ctx is None:
            self._schedule_save_session()
            return FeedbackOutcome.rejected(failure_reason)
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("consolidation_sediment", expected_generation)
            return FeedbackOutcome.rejected("stale_generation")
        self._apply_native_image_observations(ctx.response_dict, messages_chunk)
        self._apply_sediment(ctx, messages_chunk, expected_generation=expected_generation)
        latest = max((m.time for m in messages_chunk), default=None)
        if latest is not None:
            if self.last_consolidated_time is None or latest > self.last_consolidated_time:
                self.last_consolidated_time = latest
        self._messages_since_consolidation = 0
        self._schedule_save_session()
        return FeedbackOutcome(accepted=True, state_changed=True)

    async def save_long_term_memory(
            self,
            analyze_result: list,
            default_user_id: str = "",
            supersede_candidates: list[dict] | None = None,
            expected_generation: int | None = None,
    ):
        """
        后台任务：保存长期记忆到向量数据库
        优化：增加质量过滤和去重
        """
        try:
            if self.is_generation_stale(expected_generation):
                self._log_stale_generation("long_term_memory", expected_generation)
                return

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
                    category = str(item.get("category") or "event").strip().lower() or "event"
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

                # 质量过滤：基础长度/噪声过滤 + 服务端事实边界验证。
                if not should_store_memory(content):
                    skipped_quality += 1
                    logger.debug(f"[Memory] 跳过低质量记忆: {content[:30]}...")
                    continue
                validation_result = validate_memory_candidate(
                    content=content,
                    category=category,
                    confidence=confidence,
                    subject_user_id=subject_user_id,
                    subject_user_name=subject_user_name,
                    reason=str(item.get("reason") or "") if isinstance(item, dict) else "",
                )
                if not validation_result.valid:
                    skipped_quality += 1
                    log_event(
                        "memory_candidate_rejected",
                        session_id=self.id,
                        action=action,
                        category=category,
                        reason=validation_result.reason,
                    )
                    logger.debug(
                        f"[Memory] 跳过不可靠记忆({validation_result.reason}): {content[:30]}..."
                    )
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
                    operation_result = await self._run_sync_if_generation_current(
                        self.memory_writer.supersede,
                        content,
                        metadata,
                        target_ref,
                        reason=str(item.get("reason") or ""),
                        expected_generation=expected_generation,
                        stage="long_term_memory_supersede",
                    )
                    if operation_result is _STALE_GENERATION_WRITE:
                        return
                    if not (
                        isinstance(operation_result, dict)
                        and operation_result.get("completed")
                    ):
                        log_event(
                            "rag_action_rejected",
                            session_id=self.id,
                            action=action,
                            target_ref=target_ref,
                            reason="supersede_queued_for_repair",
                            queued_repair=(
                                operation_result.get("queued_repair")
                                if isinstance(operation_result, dict)
                                else 0
                            ),
                        )
                        continue
                    superseded_count += 1
                else:
                    pending_memories.append((content, metadata))

            store_result = {"added": 0, "skipped_dedup": 0}
            if pending_memories:
                store_result = await self._run_sync_if_generation_current(
                    self.memory_writer.add_candidates,
                    pending_memories,
                    expected_generation=expected_generation,
                    stage="long_term_memory_bulk",
                )
                if store_result is _STALE_GENERATION_WRITE:
                    return

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
                           search_result: _SearchResult | None = None,
                           expected_generation: int | None = None) -> list[dict]:
        logger.debug(">> 对话阶段 (Chat) 开始")
        search_history = search_result.mem_history if search_result else []
        formatted_msgs = [
            {
                "id": str(msg.id or ""),
                "name": msg.user_name,
                "content": msg.content,
                "image_meta": msg.image_meta,
                "image_refs": (
                    _message_image_refs(msg)
                    if _endpoint_uses_native_vision("chat")
                    else []
                ),
            }
            for msg in messages_chunk
        ]

        # 格式化回溯的历史记录
        recalled_str = "\n".join(recalled_history) if recalled_history else "无"

        # 过滤掉本次的新消息，避免 Prompt 上下文重复
        context_record = self.global_memory.access_context(limit=get_runtime_settings()["short_context_limit"])
        all_messages = context_record.messages
        history_msgs = _history_without_current_chunk(all_messages, messages_chunk)
        history_msgs_formatted = [
            {
                "time": m.time.strftime('%H:%M'),
                "name": m.user_name,
                "content": m.content,
                "image_meta": m.image_meta,
                "image_refs": [],
            }
            for m in history_msgs
        ]

        time_str = get_time_description(datetime.now())
        # 分离 role 和 examples：role 中可能包含 [对话样本] 后缀，需要去除避免重复
        chat_role = self.__role.split("[对话样本]")[0].strip() if "[对话样本]" in self.__role else self.__role
        reaction_users = list({msg.user_id if msg.user_id else msg.user_name for msg in messages_chunk})
        related_profiles = [self.profiles.get(uid, PersonProfile(user_id=uid)) for uid in reaction_users]
        related_profiles_data = [
            {"user_id": p.user_id, "emotion_tends_to_user": asdict(p.emotion)}
            for p in related_profiles
        ]
        prompt = get_chat_prompt(
            self.__name, chat_role, self.__chatting_state.value,
            context_record.compressed_history,
            history_msgs_formatted, # 传入格式化后的历史
            formatted_msgs,
            asdict(self.global_emotion),
            related_profiles_data,
            search_history, self.chat_summary,
            examples_text=self.__examples_str,
            recalled_history=recalled_str,
            time_info=time_str,
            rp_style=get_chat_thinking_settings().get("rp_style", "off"),
            budget=PromptBudget.from_runtime(get_runtime_settings()),
        )
        log_event("rag_prompt_budget",
            session_id=self.id,
            chat_prompt_total_chars=len(prompt),
            rag_injected_count=len(search_history),
            rag_injected_chars=sum(len(item) for item in search_history),
            history_chars=len(context_record.compressed_history or "") + sum(len(item.get("content", "")) for item in history_msgs_formatted),
            recent_chars=sum(len(item.get("content", "")) for item in formatted_msgs),
            recalled_history_chars=len(recalled_str),
            examples_chars=len(self.__examples_str or ""),
        )

        try:
            # 使用传入的 chat_llm_func
            response = await llm_func(prompt, json_mode=True)
            replies = ChatPlanner.parse(response).replies

            if self.is_generation_stale(expected_generation):
                self._log_stale_generation("chat_reply", expected_generation)
                return []

            if replies:
                retain = get_runtime_settings()["speak_willingness_retain_factor"]
                self.willingness = max(0.0, self.willingness * retain)
                self.__chatting_state = _ChattingState.ACTIVE

            return replies

        except Exception as e:
            logger.error(f"对话阶段异常: {e}")
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

    async def update_without_trigger(self, messages_chunk: list[Message]):
        """
        仅更新记忆，不触发 LLM 回复 (用于处理回显)
        """
        if not messages_chunk: return
        logger.debug(f"[Session {self.id}] 处理回显消息 (Count: {len(messages_chunk)})")
        await self.global_memory.update(messages_chunk)
        self._schedule_save_session()

    async def drain_background_tasks(self, timeout: float | None = None):
        if timeout is None:
            timeout = get_runtime_settings()["memory_drain_timeout_seconds"]
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
                     publish: bool = True,
                     expected_generation: int | None = None) -> list[dict] | None:
        return await ConversationOrchestrator(self).process_chunk(
            messages_chunk,
            chat_llm_func,
            feedback_llm_func,
            publish=publish,
            expected_generation=expected_generation,
        )

    async def _save_interaction_log(
        self,
        user_id: str,
        delta: dict,
        expected_generation: int | None = None,
    ):
        await self._save_interaction_logs(
            [(user_id, delta)],
            expected_generation=expected_generation,
        )

    async def _save_interaction_logs(
        self,
        interactions: list[tuple[str, dict]],
        expected_generation: int | None = None,
    ):
        if self.is_generation_stale(expected_generation):
            self._log_stale_generation("interaction_log", expected_generation)
            return
        async with self._save_lock:
            if self.is_generation_stale(expected_generation):
                self._log_stale_generation("interaction_log_locked", expected_generation)
                return
            await ProfileRepository.log_interactions(self.id, interactions)
