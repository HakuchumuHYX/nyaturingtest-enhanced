import asyncio
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
import numpy as np
from nonebot import logger
from openai import AsyncOpenAI
from tortoise.transactions import in_transaction

from ..config import get_app_settings
from ..storage.models import MemoryModel
from ..webconsole_store import record_llm_call

RAG_FINAL_K = 20
RAG_PER_QUERY_RECALL_K = 40
RAG_MERGED_CANDIDATE_CAP = 64
RAG_MEMORY_CHAR_BUDGET = 1500
RAG_ITEM_CHARS = 500

# 碎片只是原料：长期价值由用户档案/群志承载，所以所有类别都会过期；episode 按 event 算
EVENT_TTL_DAYS = 90
FACT_TTL_DAYS = 180

EMBEDDING_BATCH_SIZE = 32
# Qwen3-Embedding 只给查询加任务指令、记忆正文不加
QUERY_INSTRUCTION = (
    "Instruct: Given a group chat message, retrieve memories about the chat members "
    "that are relevant to it\nQuery: "
)

# 印象（玩笑、调侃）不放进每轮都给的档案，走检索：话题碰到才想起，不用另外判断该不该提
IMPRESSION_CATEGORY = "impression"
IMPRESSION_CONFIDENCE = 0.8
IMPRESSION_IMPORTANCE = 0.5

MEMORY_TYPE_WEIGHT = {
    "episode": 1.0,
    "event": 1.0,
    "preference": 1.05,
    "profile": 1.05,
    "relationship": 1.0,
    IMPRESSION_CATEGORY: 1.05,
}
MEMORY_TYPE_DECAY_RATE = {
    "episode": 0.02,
    "event": 0.02,
    "preference": 0.003,
    "profile": 0.003,
    "relationship": 0.003,
    IMPRESSION_CATEGORY: 0.003,
}
SCOPE_WEIGHT = {
    "active_subject": 1.10,
    "mentioned_subject": 1.08,
    "active_speaker": 1.04,
    "global": 1.0,
    "other_subject": 0.5,
}


@dataclass
class RetrievedMemory:
    """检索到的一条碎片和它的分数。"""

    row: MemoryModel
    retrieval_score: float
    rerank_score: float | None = None
    # 以下由 retrieve 加权排序时填
    adjusted_score: float = 0.0
    scope: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    records: list[RetrievedMemory]
    stats: dict[str, Any]


@dataclass(frozen=True)
class NewMemory:
    """待写入的一条碎片；由 candidates.build_memory 构造，source_msg_ids / participant_ids 已去重拼好。"""

    content: str
    category: str
    date: int
    confidence: float
    importance: float
    source_msg_ids: str
    participant_ids: str
    subject_user_id: str = ""
    subject_user_name: str = ""
    speaker_user_id: str = ""
    speaker_user_name: str = ""
    is_correction: bool = False
    replaces: str = ""


def _retrieval_stats(
    candidates: list, results: list, fallback_reason: str = "none"
) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "returned_count": len(results),
        "fallback_reason": fallback_reason,
    }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _query_mentions_name(queries: list[str], name: str) -> bool:
    if len(name) < 2:
        return False
    return any(name in query for query in queries)


def _memory_scope(
    row: MemoryModel, active_scope_ids: set[str], queries: list[str]
) -> tuple[str, float]:
    subject_user_id = row.subject_user_id
    if active_scope_ids and subject_user_id and subject_user_id in active_scope_ids:
        return "active_subject", SCOPE_WEIGHT["active_subject"]
    if _query_mentions_name(queries, row.subject_user_name):
        return "mentioned_subject", SCOPE_WEIGHT["mentioned_subject"]
    speaker_user_id = row.speaker_user_id
    if active_scope_ids and speaker_user_id and speaker_user_id in active_scope_ids:
        return "active_speaker", SCOPE_WEIGHT["active_speaker"]
    if subject_user_id:
        return "other_subject", SCOPE_WEIGHT["other_subject"]
    return "global", SCOPE_WEIGHT["global"]


def _parse_date(date: int) -> datetime:
    return datetime.strptime(str(date), "%Y%m%d")


def memory_expires_at(category: str, date: int, importance: float) -> datetime:
    """date 之后满 基础TTL×(1+importance) 天的次日删除；event / episode 90 天，其余 180 天。"""

    base_days = EVENT_TTL_DAYS if category in ("event", "episode") else FACT_TTL_DAYS
    ttl_days = int(base_days * (1.0 + importance))
    return _parse_date(date) + timedelta(days=ttl_days + 1)


# Embedding / Rerank 客户端全进程共享，首次使用时按配置创建
_embedding_client: AsyncOpenAI | None = None
_rerank_client: httpx.AsyncClient | None = None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def embed_texts(texts: list[str], *, instruction: str = "") -> np.ndarray:
    """返回逐行 L2 归一化的 float32 矩阵，余弦相似度直接用点积。

    instruction 只给查询加（各检索场景自己写一句），被检索的正文不加。
    """

    global _embedding_client
    settings = get_app_settings()
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(
            api_key=settings.siliconflow_api_key,
            base_url=settings.memory.base_url,
            timeout=settings.memory.timeout,
            max_retries=0,
        )
    started = time.perf_counter()
    try:
        response = await _embedding_client.embeddings.create(
            model=settings.memory.model,
            input=[instruction + text.replace("\n", " ") for text in texts],
            encoding_format="float",
        )
    except Exception as e:
        record_llm_call(
            "embedding", settings.memory.model, None, False, _elapsed_ms(started), type(e).__name__
        )
        raise
    record_llm_call("embedding", settings.memory.model, None, True, _elapsed_ms(started))
    vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


async def rerank(query: str, documents: list[str]) -> list[dict[str, Any]]:
    global _rerank_client
    settings = get_app_settings()
    if _rerank_client is None:
        _rerank_client = httpx.AsyncClient(
            timeout=settings.memory.rerank_timeout, trust_env=False
        )
    started = time.perf_counter()
    try:
        response = await _rerank_client.post(
            settings.memory.rerank_base_url,
            headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
            json={
                "model": settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),  # 全排，然后本地过滤
                "return_documents": False,
            },
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as e:
        record_llm_call(
            "rerank", settings.rerank_model, None, False, _elapsed_ms(started), type(e).__name__
        )
        logger.error(f"Rerank API Error: {e}")
        return []
    record_llm_call("rerank", settings.rerank_model, None, True, _elapsed_ms(started))
    return results


async def close_clients() -> None:
    global _embedding_client, _rerank_client
    if _embedding_client is not None:
        await _embedding_client.close()
        _embedding_client = None
    if _rerank_client is not None:
        await _rerank_client.aclose()
        _rerank_client = None


class VectorMemory:
    """一个群的长期记忆。

    向量存在 nyabot_memories 表；首次使用时把本群有效向量整体读成矩阵，检索是一次矩阵乘法。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        # 加载与写入共用一把锁：否则加载期间提交的新行可能既不在快照里、也追加不进缓存
        self._lock = asyncio.Lock()
        self._ids: list[str] | None = None
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._participants: list[frozenset[str]] = []

    async def _load_locked(self) -> None:
        rows = await MemoryModel.filter(
            session_id=self.session_id, embedding__not_isnull=True
        ).values_list(
            "id", "embedding", "embedding_model", "participant_ids"
        )
        model = get_app_settings().memory.model
        mismatched = {row[2] for row in rows if row[2] != model}
        if mismatched:
            # 不同模型的向量混在一起检索会静默变成噪声，必须先重嵌入
            raise RuntimeError(
                f"群 {self.session_id} 的记忆向量来自 {sorted(mismatched)}，"
                f"与当前 embedding 模型 {model} 不一致，需要重嵌入"
            )
        self._ids = [row[0] for row in rows]
        if rows:
            self._matrix = np.frombuffer(
                b"".join(row[1] for row in rows), dtype=np.float32
            ).reshape(len(rows), -1)
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)
        self._participants = [frozenset(row[3].split()) for row in rows]

    async def _ensure_loaded(self) -> None:
        if self._ids is not None:
            return
        async with self._lock:
            if self._ids is None:
                await self._load_locked()

    def _append_to_cache(self, rows: list[MemoryModel]) -> None:
        rows = [row for row in rows if row.embedding is not None]
        if self._ids is None or not rows:
            return
        vectors = np.frombuffer(
            b"".join(row.embedding for row in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        self._matrix = vectors if not self._ids else np.vstack([self._matrix, vectors])
        self._ids.extend(row.id for row in rows)
        self._participants.extend(frozenset(row.participant_ids.split()) for row in rows)

    def drop_cache(self) -> None:
        """表被外部改动（每日维护）后调用，下次使用时重新加载。"""

        self._ids = None

    async def clear(self) -> None:
        await MemoryModel.filter(session_id=self.session_id).delete()
        self._ids = None

    async def recent(self, limit: int) -> list[MemoryModel]:
        """最近写入的记忆，新的在前；主动发起话题时当素材，不需要 query。
        印象每次整理都重写，算不上最近发生的事。"""

        return (
            await MemoryModel.filter(session_id=self.session_id)
            .exclude(category=IMPRESSION_CATEGORY)
            .order_by("-created_at")
            .limit(limit)
        )

    async def count_by_user(self, user_id: str) -> int:
        return await MemoryModel.filter(
            session_id=self.session_id, subject_user_id=user_id
        ).count()

    async def retrieve(
        self,
        queries: list[str],
        *,
        k: int,
        user_ids: set[str] | None = None,
        active_user_ids: set[str] = frozenset(),
    ) -> RetrievalResult:
        """检索后按时间衰减/类型/置信度/作用域加权排序，取前 k 条。

        user_ids 不为 None 时只在参与者含这些人的记忆里找；带 "" 时参与者为空的记忆也算。
        rerank 接口失败时回退到向量分；rerank 把候选全筛掉不回退：向量分在闲聊上挤在 0.5 上下分不出好坏，回退只会塞进无关记忆。
        """

        unique_queries = _dedupe_preserve_order([q for q in queries if q.strip()])
        if not unique_queries:
            return RetrievalResult([], _retrieval_stats([], []))

        try:
            await self._ensure_loaded()
            if not self._ids:
                return RetrievalResult([], _retrieval_stats([], []))

            scores = (
                await embed_texts(unique_queries, instruction=QUERY_INSTRUCTION)
                @ self._matrix.T
            )
            if user_ids is not None:
                wanted = frozenset(user_ids)
                mask = np.fromiter(
                    (
                        not wanted.isdisjoint(participants)
                        or ("" in wanted and not participants)
                        for participants in self._participants
                    ),
                    dtype=bool,
                    count=len(self._participants),
                )
                scores[:, ~mask] = -np.inf

            # 每条 query 召回 top 条，多 query 命中同一条取最高分
            top_n = min(RAG_PER_QUERY_RECALL_K, len(self._ids))
            best_by_index: dict[int, float] = {}
            for row_scores in scores:
                for index in np.argpartition(-row_scores, top_n - 1)[:top_n]:
                    score = float(row_scores[index])
                    if score == -np.inf:
                        continue
                    if score > best_by_index.get(index, -np.inf):
                        best_by_index[index] = score
            ref_scores = {self._ids[index]: score for index, score in best_by_index.items()}
            if not ref_scores:
                return RetrievalResult([], _retrieval_stats([], []))
            rows = await MemoryModel.filter(id__in=list(ref_scores))

            # 同一正文只留最高分（不同主体可能被抽取出完全相同的句子）
            candidate_by_content: dict[str, RetrievedMemory] = {}
            for row in rows:
                score = max(0.0, min(1.0, ref_scores[row.id]))
                existing = candidate_by_content.get(row.content)
                if existing is None or score > existing.retrieval_score:
                    candidate_by_content[row.content] = RetrievedMemory(row, score)
            candidates = sorted(
                candidate_by_content.values(),
                key=lambda item: item.retrieval_score,
                reverse=True,
            )[:RAG_MERGED_CANDIDATE_CAP]
            if not candidates:
                return RetrievalResult([], _retrieval_stats([], []))

            # Rerank 加阈值；接口失败或没配 rerank_model 时按向量分取前若干条。
            # search_stage 已把最新有效消息排在第一位；summary query 只做补充召回。
            settings = get_app_settings()
            if not settings.rerank_model:
                picked = candidates[:RAG_PER_QUERY_RECALL_K]
                fallback_reason = "rerank_disabled"
            else:
                rerank_results = await rerank(
                    unique_queries[0], [item.row.content for item in candidates]
                )
                if not rerank_results:
                    picked = candidates[:RAG_PER_QUERY_RECALL_K]
                    fallback_reason = "rerank_api_empty"
                else:
                    picked = []
                    for res in rerank_results:
                        score = res.get("relevance_score", 0.0)
                        if score < settings.rerank_threshold:
                            continue
                        item = candidates[res["index"]]
                        item.rerank_score = score
                        picked.append(item)
                        if len(picked) >= RAG_PER_QUERY_RECALL_K:
                            break
                    if not picked:
                        return RetrievalResult(
                            [], _retrieval_stats(candidates, [], "rerank_all_filtered")
                        )
                    fallback_reason = "none"

            # 时间衰减与加权，取前 k
            active_scope_ids = set(active_user_ids)
            today_dt = datetime.now()
            scope_counts: dict[str, int] = {}
            for item in picked:
                row = item.row
                scope, scope_weight = _memory_scope(row, active_scope_ids, queries)
                scope_counts[scope] = scope_counts.get(scope, 0) + 1

                days_ago = max(0, (today_dt - _parse_date(row.date)).days)
                decay_rate = MEMORY_TYPE_DECAY_RATE[row.category]
                original_score = (
                    item.retrieval_score if item.rerank_score is None else item.rerank_score
                )
                item.adjusted_score = (
                    original_score
                    * math.exp(-decay_rate * days_ago)
                    * MEMORY_TYPE_WEIGHT[row.category]
                    * (0.7 + row.confidence * 0.3)
                    * (1.0 + row.importance * 0.15)
                    * scope_weight
                )
                item.scope = scope

            final_results = sorted(picked, key=lambda x: x.adjusted_score, reverse=True)[:k]
            stats = _retrieval_stats(candidates, final_results, fallback_reason)
            stats["scope_counts"] = scope_counts
            return RetrievalResult(final_results, stats)

        except Exception as e:
            logger.error(f"Vector retrieve failed: {e}")
            return RetrievalResult([], _retrieval_stats([], [], "retrieve_error"))

    async def add_memories(
        self,
        memories: list[NewMemory],
        *,
        still_current: Callable[[], bool],
    ) -> dict[str, int] | None:
        """批量写入长期记忆；会话在写入前被 reset/load_preset 作废则返回 None。

        只去掉同一批里正文完全相同的条目，不和已有记忆比向量：Qwen3-Embedding 比两条记忆时
        主要看开头人名，同一人不相干甚至相反的两件事余弦也有 0.97，找不到能分开真重复的阈值。
        更正条的 replaces 指向的旧行在同一事务里删除。
        embedding 调用失败时照样落库（embedding 为 NULL），由每日维护补算。
        """

        result = {"added": 0, "skipped_dedup": 0, "corrected": 0}
        valid: list[NewMemory] = []
        seen_batch = set()
        for memory in memories:
            batch_key = (memory.content, memory.subject_user_id, memory.category)
            if not memory.content or batch_key in seen_batch:
                result["skipped_dedup"] += bool(memory.content)
                continue
            seen_batch.add(batch_key)
            valid.append(memory)
        if not valid:
            return result

        try:
            vectors = await embed_texts([memory.content for memory in valid])
        except Exception as e:
            logger.error(f"[Memory] Embedding 失败，记忆先不带向量落库: {e}")
            vectors = None

        embedding_model = get_app_settings().memory.model
        async with self._lock:
            new_rows: list[MemoryModel] = []
            for index, memory in enumerate(valid):
                vector = None if vectors is None else vectors[index]
                new_rows.append(
                    MemoryModel(
                        id=str(uuid.uuid4()),
                        session_id=self.session_id,
                        content=memory.content,
                        embedding=None if vector is None else vector.tobytes(),
                        embedding_model="" if vector is None else embedding_model,
                        category=memory.category,
                        subject_user_id=memory.subject_user_id,
                        subject_user_name=memory.subject_user_name,
                        speaker_user_id=memory.speaker_user_id,
                        speaker_user_name=memory.speaker_user_name,
                        confidence=memory.confidence,
                        importance=memory.importance,
                        date=memory.date,
                        expires_at=memory_expires_at(
                            memory.category, memory.date, memory.importance
                        ),
                        source_msg_ids=memory.source_msg_ids,
                        participant_ids=memory.participant_ids,
                        is_correction=memory.is_correction,
                    )
                )
            replaces = {memory.replaces: memory.content for memory in valid if memory.replaces}

            async with in_transaction():
                # 在事务内确认代际：reset 先递增 generation 再删库，所以这里放行的写入一定早于删除
                if not still_current():
                    return None
                replaced_rows = (
                    await MemoryModel.filter(session_id=self.session_id, id__in=list(replaces))
                    if replaces
                    else []
                )
                for row in replaced_rows:
                    logger.info(f"[Memory] 更正替换: 旧「{row.content}」→ 新「{replaces[row.id]}」")
                    await row.delete()
                result["corrected"] = len(replaced_rows)
                if new_rows:
                    await MemoryModel.bulk_create(new_rows)
            if replaced_rows:
                # 更正很少发生：整体丢掉矩阵缓存，下次检索重新加载，不维护增量删除
                self.drop_cache()
            else:
                self._append_to_cache(new_rows)

        result["added"] = len(new_rows)
        return result

    async def impressions_of(self, user_id: str) -> list[str]:
        return await (
            MemoryModel.filter(
                session_id=self.session_id,
                subject_user_id=user_id,
                category=IMPRESSION_CATEGORY,
            )
            .order_by("created_at")
            .values_list("content", flat=True)
        )

    async def replace_impressions(
        self,
        user_id: str,
        user_name: str,
        contents: list[str],
        *,
        still_current: Callable[[], bool],
    ) -> None:
        """整组换掉此人的印象：每次整理都是在旧印象上重写的完整一份。"""

        await MemoryModel.filter(
            session_id=self.session_id,
            subject_user_id=user_id,
            category=IMPRESSION_CATEGORY,
        ).delete()
        self.drop_cache()
        date = int(f"{datetime.now():%Y%m%d}")
        await self.add_memories(
            [
                NewMemory(
                    content=content,
                    category=IMPRESSION_CATEGORY,
                    date=date,
                    confidence=IMPRESSION_CONFIDENCE,
                    importance=IMPRESSION_IMPORTANCE,
                    source_msg_ids="",
                    participant_ids=user_id,
                    subject_user_id=user_id,
                    subject_user_name=user_name,
                )
                for content in contents
            ],
            still_current=still_current,
        )


async def maintain_memories() -> dict:
    """每日维护：删除过期记忆，补算写入时 embedding 失败的行。返回 {deleted, filled, pending}，pending 是还没补上的。"""

    deleted = await MemoryModel.filter(expires_at__lt=datetime.now()).delete()
    pending = await MemoryModel.filter(embedding__isnull=True)
    embedding_model = get_app_settings().memory.model
    filled = 0
    for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
        batch = pending[start : start + EMBEDDING_BATCH_SIZE]
        try:
            vectors = await embed_texts([row.content for row in batch])
        except Exception as e:
            logger.error(f"[Memory] 补算 embedding 失败，明天再试: {e}")
            break
        for row, vector in zip(batch, vectors):
            row.embedding = vector.tobytes()
            row.embedding_model = embedding_model
            await row.save(update_fields=["embedding", "embedding_model", "updated_at"])
        filled += len(batch)
    logger.info(f"[Memory] 维护完成：删除过期 {deleted} 条，补算向量 {filled}/{len(pending)} 条")
    return {"deleted": deleted, "filled": filled, "pending": len(pending) - filled}

