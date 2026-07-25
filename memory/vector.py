# nyaturingtest/vector_mem.py
import os
import uuid
import math
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List
import chromadb
from nonebot import logger
from ..config import (
    get_app_settings,
    get_memory_endpoint_settings,
    get_runtime_settings,
)
from ..database.backup_lock import BACKUP_IO_LOCK
from .vector_clients import (
    SiliconFlowEmbeddingFunction,
    SiliconFlowReranker,
)


MEMORY_COLLECTION_NAME = "nyabot_memory"
MEMORY_COLLECTION_METADATA = {"hnsw:space": "cosine"}
PRESET_TYPE_WEIGHT = {
    "bot_self": 0.95,
    "relationship": 0.90,
    "legacy_rule": 0.85,
    "knowledge": 0.82,
    "event": 0.75,
}
MEMORY_TYPE_WEIGHT = {
    "event": 1.0,
    "preference": 1.05,
    "profile": 1.05,
    "relationship": 1.0,
}
MEMORY_TYPE_DECAY_RATE = {
    "event": 0.02,
    "preference": 0.003,
    "profile": 0.003,
    "relationship": 0.003,
}
SCOPE_WEIGHT = {
    "active_subject": 1.10,
    "mentioned_subject": 1.08,
    "active_speaker": 1.04,
    "global": 1.0,
    "legacy_subject": 0.75,
    "other_subject": 0.5,
}
_metric_check_done: set[str] = set()


@dataclass(frozen=True)
class RetrievalResult(Sequence[dict[str, Any]]):
    records: list[dict[str, Any]]
    stats: dict[str, Any]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def _score_from_distance(distance: float | int | None) -> float:
    if distance is None:
        return 0.5
    try:
        score = 1.0 - float(distance)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


def _empty_retrieval_stats(*, use_rerank: bool = False, fallback_reason: str = "none") -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "returned_count": 0,
        "use_rerank": bool(use_rerank),
        "fallback_reason": fallback_reason,
        "adjusted_score_min": None,
        "adjusted_score_p50": None,
        "adjusted_score_p90": None,
        "adjusted_score_max": None,
    }


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * ratio + 0.5)
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def _score_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "adjusted_score_min": None,
            "adjusted_score_p50": None,
            "adjusted_score_p90": None,
            "adjusted_score_max": None,
        }
    return {
        "adjusted_score_min": min(values),
        "adjusted_score_p50": _percentile(values, 0.50),
        "adjusted_score_p90": _percentile(values, 0.90),
        "adjusted_score_max": max(values),
    }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _memory_operation_id(
    operation: str,
    content: str,
    metadata: dict,
    target_ref: str = "",
) -> str:
    payload = json.dumps(
        {
            "operation": operation,
            "content": content,
            "metadata": metadata,
            "target_ref": target_ref,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def where_any(field: str, values: list[Any]) -> dict:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return {}
    if len(cleaned) == 1:
        return {field: {"$eq": cleaned[0]}}
    return {"$or": [{field: {"$eq": value}} for value in cleaned]}


def where_all(*conditions: dict) -> dict:
    cleaned = [condition for condition in conditions if condition]
    if not cleaned:
        return {}
    if len(cleaned) == 1:
        return cleaned[0]
    return {"$and": cleaned}


def _subject_user_where(user_ids: set[str]) -> dict:
    subject_conditions = []
    for user_id in sorted(user_ids):
        subject_conditions.append({"subject_user_id": {"$eq": user_id}})
        subject_conditions.append({"user_id": {"$eq": user_id}})
    return where_all(
        {"source": {"$eq": "memory"}},
        {"$or": subject_conditions},
    )


def _clamp_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return max(lower, min(upper, number))


def _clean_metadata_string(value: Any) -> str:
    return str(value or "").strip()


def _metadata_schema_version(value: Any) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError):
        return 1
    return version if version > 0 else 1


def _normalized_metadata(meta: dict | None) -> dict[str, Any]:
    data = dict(meta or {})
    source = str(data.get("source") or "memory")
    memory_type = str(data.get("type") or "event")
    subtype = str(data.get("subtype") or ("legacy_rule" if source == "preset" else memory_type))
    subject_user_id = _clean_metadata_string(data.get("subject_user_id") or data.get("user_id"))
    data["source"] = source
    data["type"] = memory_type
    data["subtype"] = subtype
    data["status"] = str(data.get("status") or "active")
    data["category"] = str(data.get("category") or memory_type)
    data["confidence"] = _clamp_float(data.get("confidence"), 1.0, 0.0, 1.0)
    data["importance"] = _clamp_float(data.get("importance"), 0.0, 0.0, 1.0)
    data["schema_version"] = _metadata_schema_version(data.get("schema_version"))
    data["subject_user_id"] = subject_user_id
    data["subject_user_name"] = _clean_metadata_string(data.get("subject_user_name"))
    data["speaker_user_id"] = _clean_metadata_string(data.get("speaker_user_id"))
    data["speaker_user_name"] = _clean_metadata_string(data.get("speaker_user_name"))
    data["user_id"] = subject_user_id
    return data


def _dedup_scope_key(meta: dict | None) -> tuple[str, str, str]:
    metadata = _normalized_metadata(meta)
    source_class = "preset" if metadata["source"] == "preset" else "memory"
    subject_user_id = "" if source_class == "preset" else metadata["subject_user_id"]
    category = _clean_metadata_string(metadata.get("category") or metadata.get("type"))
    return source_class, subject_user_id, category


def _same_dedup_scope(candidate: dict | None, existing: dict | None) -> bool:
    existing_metadata = _normalized_metadata(existing)
    if existing_metadata["source"] == "memory" and _metadata_status(existing_metadata) != "active":
        return False
    return _dedup_scope_key(candidate) == _dedup_scope_key(existing_metadata)


def _dedup_where(metadata: dict) -> dict:
    source_class, subject_user_id, category = _dedup_scope_key(metadata)
    conditions = [
        {"source": {"$eq": source_class}},
        {
            "$or": [
                {"category": {"$eq": category}},
                {"type": {"$eq": category}},
            ]
        },
    ]
    if source_class == "memory" and subject_user_id:
        conditions.append({
            "$or": [
                {"subject_user_id": {"$eq": subject_user_id}},
                {"user_id": {"$eq": subject_user_id}},
            ]
        })
    return where_all(*conditions)


def _source_type_weight(meta: dict) -> float:
    source = str(meta.get("source") or "memory")
    memory_type = str(meta.get("type") or "event")
    subtype = str(meta.get("subtype") or ("legacy_rule" if source == "preset" else memory_type))
    if source == "preset":
        return PRESET_TYPE_WEIGHT.get(subtype, PRESET_TYPE_WEIGHT["legacy_rule"])
    return MEMORY_TYPE_WEIGHT.get(memory_type, 1.0)


def _memory_decay_rate(meta: dict, default_decay_rate: float) -> float:
    if str(meta.get("source") or "memory") == "preset":
        return 0.0
    if default_decay_rate <= 0:
        return 0.0
    memory_type = str(meta.get("type") or "event")
    return MEMORY_TYPE_DECAY_RATE.get(memory_type, default_decay_rate)


def _confidence_weight(meta: dict) -> float:
    confidence = _clamp_float(meta.get("confidence"), 1.0, 0.0, 1.0)
    return 0.7 + confidence * 0.3


def _query_mentions_name(queries: list[str], name: str) -> bool:
    clean_name = _clean_metadata_string(name)
    if len(clean_name) < 2:
        return False
    return any(clean_name in str(query or "") for query in queries or [])


def _memory_scope(meta: dict, active_scope_ids: set[str], queries: list[str]) -> tuple[str, float]:
    if meta.get("source") == "preset":
        return "global", SCOPE_WEIGHT["global"]

    subject_user_id = _clean_metadata_string(meta.get("subject_user_id") or meta.get("user_id"))
    subject_user_name = _clean_metadata_string(meta.get("subject_user_name"))
    speaker_user_id = _clean_metadata_string(meta.get("speaker_user_id"))

    if active_scope_ids and subject_user_id and subject_user_id in active_scope_ids:
        return "active_subject", SCOPE_WEIGHT["active_subject"]
    if _query_mentions_name(queries, subject_user_name):
        return "mentioned_subject", SCOPE_WEIGHT["mentioned_subject"]
    if active_scope_ids and speaker_user_id and speaker_user_id in active_scope_ids:
        return "active_speaker", SCOPE_WEIGHT["active_speaker"]
    if subject_user_id:
        if int(meta.get("schema_version") or 1) < 2 and not subject_user_name:
            return "legacy_subject", SCOPE_WEIGHT["legacy_subject"]
        return "other_subject", SCOPE_WEIGHT["other_subject"]
    return "global", SCOPE_WEIGHT["global"]


def _collection_metric_state(collection) -> str:
    metadata = getattr(collection, "metadata", None)
    if not isinstance(metadata, dict):
        return "unknown"
    space = metadata.get("hnsw:space")
    if not space:
        return "unknown"
    return "cosine" if str(space).lower() == "cosine" else "mismatch"


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    safe_size = max(1, int(batch_size or 1))
    return [items[index:index + safe_size] for index in range(0, len(items), safe_size)]


def _metadata_status(meta: dict | None) -> str:
    return str((meta or {}).get("status") or "active")


def _date_days_ago(meta: dict, *, now: datetime) -> int | None:
    date = meta.get("date")
    if not date or not isinstance(date, int) or date <= 0:
        return None
    try:
        memory_dt = datetime.strptime(str(date), "%Y%m%d")
    except ValueError:
        return None
    return max(0, (now - memory_dt).days)


class VectorMemory:
    """
    Synchronous vector store wrapper.

    Call ChromaDB, embedding, and rerank operations from async code through
    nonebot.utils.run_sync or another thread-pool adapter.
    """

    def __init__(self, api_key: str, persist_directory: str, session_id: str = "global"):
        self.persist_directory = persist_directory
        self._version = 0
        os.makedirs(self.persist_directory, exist_ok=True)
        memory_settings = get_memory_endpoint_settings()
        self.emb_fn = SiliconFlowEmbeddingFunction(
            api_key=api_key,
            session_id=session_id,
            model=str(memory_settings["model"]),
            base_url=str(memory_settings["base_url"]),
            timeout=float(memory_settings["timeout"]),
        )
        
        # 初始化 Reranker
        self.reranker = None
        app_settings = get_app_settings()
        if app_settings.rerank_model:
            self.reranker = SiliconFlowReranker(
                api_key=api_key, 
                model=app_settings.rerank_model,
                api_url=str(memory_settings["rerank_base_url"]),
                timeout=float(memory_settings["rerank_timeout"]),
            )

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=self.emb_fn,
            metadata=MEMORY_COLLECTION_METADATA
        )
        self._check_collection_metric_once()
        self._ids_supported = self._probe_ids_support()
        self.replay_pending()

    @property
    def version(self) -> int:
        return int(getattr(self, "_version", 0) or 0)

    def _bump_version(self) -> None:
        self._version = self.version + 1

    def _check_collection_metric_once(self) -> str:
        state = _collection_metric_state(self.collection)
        if self.persist_directory in _metric_check_done:
            return state
        _metric_check_done.add(self.persist_directory)
        if state == "unknown":
            logger.warning(f"Vector collection metric metadata unknown: {self.persist_directory}")
        elif state == "mismatch":
            metadata = getattr(self.collection, "metadata", None)
            logger.error(f"Vector collection metric mismatch: {self.persist_directory} metadata={metadata}")
        return state

    @property
    def ids_supported(self) -> bool:
        return bool(getattr(self, "_ids_supported", False))

    def _probe_ids_support(self) -> bool:
        try:
            with BACKUP_IO_LOCK:
                probe = self.collection.get(limit=1, include=[])
            supported = isinstance(probe, dict) and "ids" in probe
            logger.info(f"Vector collection capability: ids_supported={supported} path={self.persist_directory}")
            return supported
        except Exception as e:
            logger.warning(f"Vector collection ids probe failed: {e}")
            return False

    def _wal_path(self) -> str:
        return os.path.join(self.persist_directory, "pending_memories.jsonl")

    def _append_wal(self, items: list[tuple[str, dict]]) -> bool:
        operations = []
        for content, metadata in items:
            operation_id = str(metadata.get("operation_id") or "") or _memory_operation_id(
                "add",
                content,
                metadata,
            )
            normalized_metadata = dict(metadata)
            normalized_metadata["operation_id"] = operation_id
            operations.append({
                "operation_id": operation_id,
                "operation": "add",
                "content": content,
                "metadata": normalized_metadata,
                "target_ref": "",
            })
        return self._append_wal_operations(operations)

    def _append_wal_operations(self, operations: list[dict]) -> bool:
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            with open(self._wal_path(), "a", encoding="utf-8") as handle:
                for operation in operations:
                    handle.write(json.dumps(operation, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            logger.error(f"WAL append failed: {e}")
            return False

    def replay_pending(self) -> int:
        path = self._wal_path()
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"WAL read failed: {e}")
            return 0

        operations: list[dict] = []
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = str(obj.get("content") or "").strip()
            if not content:
                continue
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            operation = str(obj.get("operation") or "add")
            operation_id = str(obj.get("operation_id") or metadata.get("operation_id") or "")
            if not operation_id:
                operation_id = _memory_operation_id(
                    operation,
                    content,
                    metadata,
                    str(obj.get("target_ref") or ""),
                )
            operations.append({
                **obj,
                "operation": operation,
                "operation_id": operation_id,
                "content": content,
                "metadata": metadata,
            })

        if not operations:
            try:
                os.remove(path)
            except OSError:
                pass
            return 0

        completed = 0
        remaining = []
        for operation in operations:
            try:
                if operation["operation"] == "supersede":
                    result = self.supersede_memory(
                        operation["content"],
                        operation["metadata"],
                        str(operation.get("target_ref") or ""),
                        reason=str(operation.get("reason") or ""),
                        operation_id=operation["operation_id"],
                        queue_on_failure=False,
                    )
                    success = bool(result.get("completed"))
                else:
                    result = self.add_texts(
                        [operation["content"]],
                        metadatas=[{
                            **operation["metadata"],
                            "operation_id": operation["operation_id"],
                        }],
                        queue_on_failure=False,
                    )
                    success = int(result.get("confirmed") or result.get("added") or 0) >= 1
                if success:
                    completed += 1
                else:
                    remaining.append(operation)
            except Exception:
                remaining.append(operation)

        try:
            if remaining:
                with open(path, "w", encoding="utf-8") as handle:
                    for operation in remaining:
                        handle.write(json.dumps(operation, ensure_ascii=False) + "\n")
            else:
                os.remove(path)
        except OSError as e:
            logger.error(f"WAL rewrite failed: {e}")
            return 0
        logger.info(f"Replayed {completed} pending memory operations from WAL")
        return completed

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[dict] | None = None,
        *,
        queue_on_failure: bool = True,
    ) -> dict[str, Any]:
        empty_result = {
            "added": 0,
            "confirmed": 0,
            "queued_wal": 0,
            "failed": 0,
            "memory_refs": [],
        }
        if not texts:
            return empty_result
        valid_data = [(t, metadatas[i] if metadatas and i < len(metadatas) else {})
                      for i, t in enumerate(texts) if t and t.strip()]
        if not valid_data:
            return empty_result

        runtime = get_runtime_settings()
        max_retries = int(runtime.get("memory_write_max_retries", 0) or 0)
        base_delay = float(runtime.get("memory_write_retry_base_delay", 0.5) or 0.0)
        prepared_data = []
        ids = []
        for content, metadata in valid_data:
            prepared_metadata = dict(metadata or {})
            operation_id = str(prepared_metadata.get("operation_id") or "")
            if not operation_id:
                operation_id = _memory_operation_id("add", content, prepared_metadata)
            prepared_metadata["operation_id"] = operation_id
            prepared_data.append((content, prepared_metadata))
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, operation_id)))
        for attempt in range(max_retries + 1):
            try:
                existing_ids = set()
                try:
                    with BACKUP_IO_LOCK:
                        existing = self.collection.get(ids=ids, include=[])
                    existing_ids = set(existing.get("ids") or []) if isinstance(existing, dict) else set()
                except Exception:
                    existing_ids = set()
                missing = [
                    (item_id, data)
                    for item_id, data in zip(ids, prepared_data)
                    if item_id not in existing_ids
                ]
                if not missing:
                    return {
                        "added": 0,
                        "confirmed": len(prepared_data),
                        "queued_wal": 0,
                        "failed": 0,
                        "memory_refs": ids,
                    }
                with BACKUP_IO_LOCK:
                    self.collection.add(
                        documents=[data[0] for _, data in missing],
                        metadatas=[data[1] for _, data in missing],
                        ids=[item_id for item_id, _ in missing],
                    )
                self._bump_version()
                return {
                    "added": len(missing),
                    "confirmed": len(prepared_data),
                    "queued_wal": 0,
                    "failed": 0,
                    "memory_refs": ids,
                }
            except Exception as e:
                logger.error(f"Vector add failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries and base_delay > 0:
                    time.sleep(base_delay * (2 ** attempt))

        if queue_on_failure and self._append_wal(prepared_data):
            logger.warning(f"Vector add exhausted retries, wrote {len(prepared_data)} memories to WAL")
            return {
                **empty_result,
                "queued_wal": len(prepared_data),
                "memory_refs": ids,
            }
        logger.error(f"Vector add exhausted retries and WAL append failed for {len(prepared_data)} memories")
        return {
            **empty_result,
            "failed": len(prepared_data),
            "memory_refs": ids,
        }

    def retrieve(
        self,
        queries: List[str],
        k: int = 5,
        where: dict | None = None,
        use_rerank: bool = True,
        merged_candidate_cap: int | None = None,
    ) -> RetrievalResult:
        """
        检索逻辑：
        1. k 表示每条 query 的召回数量
        2. 如果未启用，直接召回 Top K
        """
        if not queries:
            return RetrievalResult([], _empty_retrieval_stats(use_rerank=use_rerank))
        unique_queries = _dedupe_preserve_order([q for q in queries if q.strip()])
        if not unique_queries:
            return RetrievalResult([], _empty_retrieval_stats(use_rerank=use_rerank))
        
        initial_k = max(1, int(k or 1))
        
        try:
            results = self.collection.query(query_texts=unique_queries, n_results=initial_k, where=where)
            
            # 第一步：合并去重初筛结果
            candidate_by_content: dict[str, dict[str, Any]] = {}
            
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            distances = results.get("distances") or []
            ids = results.get("ids") or []

            if documents:
                for i, docs in enumerate(documents):
                    metas = metadatas[i] if i < len(metadatas) else []
                    row_distances = distances[i] if i < len(distances) else []
                    row_ids = ids[i] if i < len(ids) else []
                    
                    for j, doc in enumerate(docs):
                        if not doc:
                            continue
                        metadata = _normalized_metadata(metas[j] if j < len(metas) else {})
                        distance = row_distances[j] if j < len(row_distances) else None
                        metadata["retrieval_score"] = _score_from_distance(distance)
                        if j < len(row_ids):
                            metadata["memory_ref"] = row_ids[j]
                        existing = candidate_by_content.get(doc)
                        if (
                            existing is None
                            or metadata["retrieval_score"] > existing["metadata"].get("retrieval_score", 0.0)
                        ):
                            candidate_by_content[doc] = {
                                "content": doc,
                                "metadata": metadata
                            }
            flattened_candidates = sorted(
                candidate_by_content.values(),
                key=lambda item: item.get("metadata", {}).get("retrieval_score", 0.0),
                reverse=True,
            )
            if merged_candidate_cap is not None:
                cap = max(1, int(merged_candidate_cap or 1))
                flattened_candidates = flattened_candidates[:cap]
            
            # 如果没有结果，直接返回
            if not flattened_candidates:
                return RetrievalResult([], _empty_retrieval_stats(use_rerank=use_rerank))

            # 如果不使用 Rerank 或 Reranker 未初始化，直接截断返回
            if not use_rerank or not self.reranker:
                results = flattened_candidates[:k]
                stats = {
                    **_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="rerank_disabled"),
                    "candidate_count": len(flattened_candidates),
                    "returned_count": len(results),
                }
                return RetrievalResult(results, stats)

            # 第二步：Rerank
            # search_stage 已把最新有效消息排在第一位；summary/name query 只做补充召回。
            main_query = unique_queries[0]
            
            candidate_docs = [item["content"] for item in flattened_candidates]
            
            rerank_results = self.reranker.rerank(
                query=main_query,
                documents=candidate_docs,
                top_n=len(candidate_docs), # 全排，然后本地过滤
            )
            if not rerank_results:
                logger.debug("Rerank无结果，回退到初筛候选")
                results = flattened_candidates[:k]
                stats = {
                    **_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="rerank_api_empty"),
                    "candidate_count": len(flattened_candidates),
                    "returned_count": len(results),
                }
                return RetrievalResult(results, stats)
            
            final_results = []
            threshold = get_app_settings().rerank_threshold
            
            for res in rerank_results:
                idx = res.get("index")
                score = res.get("relevance_score", 0.0)
                
                if score < threshold:
                    continue
                    
                if isinstance(idx, int) and 0 <= idx < len(flattened_candidates):
                    item = flattened_candidates[idx]
                    # 可以把分数附加上去，方便调试
                    item["metadata"]["rerank_score"] = score
                    final_results.append(item)
                    
                if len(final_results) >= k:
                    break
            
            logger.debug(f"Rerank完成: 初筛{len(candidate_docs)} -> 终选{len(final_results)} (阈值{threshold})")
            if not final_results:
                logger.debug("Rerank结果全部被过滤，回退到初筛候选")
                results = flattened_candidates[:k]
                stats = {
                    **_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="rerank_all_filtered"),
                    "candidate_count": len(flattened_candidates),
                    "returned_count": len(results),
                }
                return RetrievalResult(results, stats)
            stats = {
                **_empty_retrieval_stats(use_rerank=use_rerank),
                "candidate_count": len(flattened_candidates),
                "returned_count": len(final_results),
            }
            return RetrievalResult(final_results, stats)

        except Exception as e:
            logger.error(f"Vector retrieve failed: {e}")
            return RetrievalResult(
                [],
                _empty_retrieval_stats(
                    use_rerank=use_rerank,
                    fallback_reason="retrieve_error",
                ),
            )

    def _retrieve_active_subject_records(
        self,
        active_user_ids: set[str],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not active_user_ids:
            return []
        where = _subject_user_where(active_user_ids)
        if not where:
            return []
        try:
            with BACKUP_IO_LOCK:
                result = self.collection.get(where=where, include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Active subject memory recall failed: {e}")
            return []

        ids = result.get("ids") or [] if isinstance(result, dict) else []
        documents = result.get("documents") or [] if isinstance(result, dict) else []
        metadatas = result.get("metadatas") or [] if isinstance(result, dict) else []
        records = []
        for index, document in enumerate(documents):
            if not document:
                continue
            metadata = _normalized_metadata(metadatas[index] if index < len(metadatas) else {})
            if metadata.get("source") != "memory" or _metadata_status(metadata) != "active":
                continue
            subject_user_id = _clean_metadata_string(metadata.get("subject_user_id") or metadata.get("user_id"))
            if subject_user_id not in active_user_ids:
                continue
            if index < len(ids):
                metadata["memory_ref"] = ids[index]
            metadata["retrieval_score"] = _clamp_float(metadata.get("retrieval_score"), 0.5, 0.0, 1.0)
            records.append({
                "content": document,
                "metadata": metadata,
            })

        records.sort(
            key=lambda item: (
                _clamp_float(item.get("metadata", {}).get("importance"), 0.0, 0.0, 1.0),
                int(item.get("metadata", {}).get("date") or 0),
            ),
            reverse=True,
        )
        return records[:max(1, int(limit or 1))]

    def delete_by_metadata(self, where: dict):
        """删除指定条件的记忆"""
        try:
            with BACKUP_IO_LOCK:
                self.collection.delete(where=where)
            self._bump_version()
            logger.info(f"Deleted vectors where {where}")
        except Exception as e:
            logger.error(f"Vector delete failed: {e}")

    def cleanup(self, days_retention: int = 90):
        """生命周期管理：清理过期事件"""
        try:
            ids, metadatas = self._get_all_ids_metadatas()
            now = datetime.now()
            delete_ids = []
            for item_id, meta in zip(ids, metadatas):
                metadata = _normalized_metadata(meta)
                if metadata.get("source") == "preset":
                    continue
                days_ago = _date_days_ago(metadata, now=now)
                if days_ago is None:
                    continue
                status = _metadata_status(metadata)
                ttl_days = int(metadata.get("ttl_days") or days_retention)
                importance = _clamp_float(metadata.get("importance"), 0.0, 0.0, 1.0)
                effective_ttl_days = int(ttl_days * (1.0 + importance))
                should_delete = (
                    (status in {"archived", "superseded"} and days_ago > days_retention)
                    or (metadata.get("type") == "event" and days_ago > effective_ttl_days)
                )
                if should_delete:
                    delete_ids.append(item_id)

            for batch in _batched(delete_ids, 200):
                with BACKUP_IO_LOCK:
                    self.collection.delete(ids=batch)
            if delete_ids:
                self._bump_version()
            logger.info(f"Cleaned up {len(delete_ids)} expired vector memories")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _get_all_ids_metadatas(self) -> tuple[list[str], list[dict]]:
        with BACKUP_IO_LOCK:
            result = self.collection.get(include=["metadatas"])
        ids = result.get("ids", []) if isinstance(result, dict) else []
        metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
        return list(ids or []), [dict(meta or {}) for meta in (metadatas or [])]

    def get_metadata_by_id(self, memory_ref: str) -> dict | None:
        if not memory_ref:
            return None
        with BACKUP_IO_LOCK:
            result = self.collection.get(ids=[memory_ref], include=["metadatas"])
        if not isinstance(result, dict):
            return None
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return dict(metadatas[0] or {})

    def update_metadata_by_id(self, memory_ref: str, metadata: dict) -> None:
        if not memory_ref:
            return
        with BACKUP_IO_LOCK:
            self.collection.update(ids=[memory_ref], metadatas=[dict(metadata or {})])
        self._bump_version()

    def supersede_memory(
        self,
        content: str,
        metadata: dict,
        target_ref: str,
        *,
        reason: str = "",
        operation_id: str = "",
        queue_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Apply an idempotent, repairable supersede operation."""

        target_metadata = self.get_metadata_by_id(target_ref)
        if not target_metadata:
            return {"completed": False, "queued_repair": 0, "reason": "target_missing"}
        normalized_target = _normalized_metadata(target_metadata)
        if (
            normalized_target.get("source") != "memory"
            or normalized_target.get("subtype") == "bot_self"
        ):
            return {"completed": False, "queued_repair": 0, "reason": "target_not_supersedable"}

        operation_id = operation_id or _memory_operation_id(
            "supersede",
            content,
            metadata,
            target_ref,
        )
        replacement_metadata = _normalized_metadata(metadata)
        replacement_metadata.update({
            "operation_id": operation_id,
            "supersede_operation_id": operation_id,
            "supersedes": target_ref,
            "status": "pending_supersede",
        })
        replacement_ref = str(uuid.uuid5(uuid.NAMESPACE_URL, operation_id))
        operation = {
            "operation_id": operation_id,
            "operation": "supersede",
            "content": content,
            "metadata": replacement_metadata,
            "target_ref": target_ref,
            "reason": str(reason or "")[:200],
        }

        try:
            add_result = self.add_texts(
                [content],
                metadatas=[replacement_metadata],
                queue_on_failure=False,
            )
            if int(add_result.get("confirmed") or 0) < 1:
                raise RuntimeError("replacement_not_confirmed")

            updated_target = dict(normalized_target)
            updated_target["status"] = "superseded"
            updated_target["superseded_at"] = datetime.now().astimezone().isoformat()
            updated_target["superseded_reason"] = operation["reason"]
            updated_target["supersede_operation_id"] = operation_id
            self.update_metadata_by_id(target_ref, updated_target)

            active_replacement = dict(replacement_metadata)
            active_replacement["status"] = "active"
            self.update_metadata_by_id(replacement_ref, active_replacement)
            return {
                "completed": True,
                "queued_repair": 0,
                "operation_id": operation_id,
                "memory_ref": replacement_ref,
            }
        except Exception as e:
            queued = int(queue_on_failure and self._append_wal_operations([operation]))
            logger.error(f"Supersede operation failed ({operation_id}): {e}")
            return {
                "completed": False,
                "queued_repair": queued,
                "operation_id": operation_id,
                "memory_ref": replacement_ref,
                "reason": type(e).__name__,
            }

    def backfill_active_status(self, *, dry_run: bool = True, batch_size: int = 200, max_rounds: int = 5) -> dict[str, Any]:
        """Backfill missing status metadata without re-embedding records."""
        marker_path = os.path.join(
            self.persist_directory,
            ".rag_status_backfill_complete.json",
        )
        if not dry_run and os.path.exists(marker_path):
            return {
                "dry_run": False,
                "total_count": 0,
                "missing_status_count": 0,
                "backfilled_count": 0,
                "verify_rounds": 0,
                "complete": True,
                "already_complete": True,
            }
        report = {
            "dry_run": dry_run,
            "total_count": 0,
            "missing_status_count": 0,
            "backfilled_count": 0,
            "verify_rounds": 0,
            "complete": False,
        }
        zero_rounds = 0
        for _ in range(max(1, max_rounds)):
            ids, metadatas = self._get_all_ids_metadatas()
            missing = [
                (item_id, metadata)
                for item_id, metadata in zip(ids, metadatas)
                if not metadata.get("status")
            ]
            report["total_count"] = len(ids)
            report["missing_status_count"] = len(missing)
            if dry_run:
                report["complete"] = not missing
                return report
            if not missing:
                zero_rounds += 1
                report["verify_rounds"] = zero_rounds
                if zero_rounds >= 2:
                    report["complete"] = True
                    self._write_status_backfill_marker(report)
                    return report
                continue
            zero_rounds = 0
            for batch in _batched(missing, batch_size):
                batch_ids = [item_id for item_id, _ in batch]
                batch_metadatas = []
                for _, metadata in batch:
                    updated = dict(metadata or {})
                    updated["status"] = "active"
                    batch_metadatas.append(updated)
                with BACKUP_IO_LOCK:
                    self.collection.update(ids=batch_ids, metadatas=batch_metadatas)
                report["backfilled_count"] += len(batch)
                self._bump_version()
        return report

    def _write_status_backfill_marker(self, report: dict[str, Any]) -> None:
        marker_path = os.path.join(self.persist_directory, ".rag_status_backfill_complete.json")
        payload = {
            "collection": MEMORY_COLLECTION_NAME,
            "completed_at": datetime.now().astimezone().isoformat(),
            "total_count": report.get("total_count", 0),
            "backfilled_count": report.get("backfilled_count", 0),
            "verify_rounds": report.get("verify_rounds", 0),
        }
        with open(marker_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)

    def clear(self):
        try:
            with BACKUP_IO_LOCK:
                self.client.delete_collection(MEMORY_COLLECTION_NAME)
                self.collection = self.client.get_or_create_collection(
                    name=MEMORY_COLLECTION_NAME,
                    embedding_function=self.emb_fn,
                    metadata=MEMORY_COLLECTION_METADATA,
                )
            self._bump_version()
        except Exception as e:
            logger.error(f"Clear failed: {e}")

    def close(self):
        try:
            self.emb_fn.close()
        except Exception as e:
            logger.warning(f"Close embedding client failed: {e}")
        if self.reranker:
            try:
                self.reranker.close()
            except Exception as e:
                logger.warning(f"Close reranker client failed: {e}")

    def count_by_user(self, user_id: str) -> int:
        """统计某用户的记忆数量"""
        try:
            if not user_id or not user_id.strip():
                return 0
            
            results = self.collection.get(
                where={"user_id": {"$eq": user_id}},
                include=[]  # 不需要实际内容，只需要 ID
            )
            return len(results.get("ids", []))
        except Exception as e:
            logger.warning(f"统计记忆数量失败: {e}")
            return 0

    def add_memory_with_dedup(self, content: str, metadata: dict, threshold: float = 0.9) -> bool:
        """
        带去重的记忆添加
        
        Args:
            content: 记忆内容
            metadata: 元数据
            threshold: 相似度阈值，超过此值视为重复
            
        Returns:
            是否成功添加（False 表示重复跳过）
        """
        result = self.add_memories_with_dedup([(content, metadata)], threshold=threshold)
        return result["added"] > 0

    def _reinforce_duplicate_memory(self, memory_ref: str, existing_metadata: dict, new_metadata: dict) -> bool:
        if not memory_ref:
            return False
        metadata = _normalized_metadata(existing_metadata)
        if metadata.get("source") != "memory" or _metadata_status(metadata) != "active":
            return False
        if metadata.get("subtype") == "bot_self":
            return False
        if not _same_dedup_scope(new_metadata, metadata):
            return False

        old_confidence = _clamp_float(metadata.get("confidence"), 1.0, 0.0, 1.0)
        metadata["confidence"] = min(1.0, old_confidence + (1.0 - old_confidence) * 0.2)
        new_date = new_metadata.get("date")
        if isinstance(new_date, int) and new_date > 0:
            metadata["date"] = new_date
        try:
            reaffirm_count = int(metadata.get("reaffirm_count") or 0)
        except (TypeError, ValueError):
            reaffirm_count = 0
        metadata["reaffirm_count"] = reaffirm_count + 1
        metadata["last_reaffirmed_at"] = datetime.now().astimezone().isoformat()
        metadata["user_id"] = metadata.get("subject_user_id") or metadata.get("user_id") or ""
        self.update_metadata_by_id(memory_ref, metadata)
        return True

    def add_memories_with_dedup(self, memories: list[tuple[str, dict]], threshold: float = 0.9) -> dict[str, int]:
        """
        批量去重并添加长期记忆。

        对同一批候选记忆只做一次 Chroma query 和一次 add，避免逐条 embedding/query/add。
        """
        result = {"added": 0, "skipped_empty": 0, "skipped_dedup": 0, "reinforced": 0, "dedup_errors": 0}
        valid: list[tuple[str, dict]] = []
        seen_batch = set()
        for content, metadata in memories:
            normalized = (content or "").strip()
            if not normalized:
                result["skipped_empty"] += 1
                continue
            normalized_metadata = _normalized_metadata(metadata)
            batch_key = (normalized, _dedup_scope_key(normalized_metadata))
            if batch_key in seen_batch:
                result["skipped_dedup"] += 1
                continue
            seen_batch.add(batch_key)
            valid.append((normalized, normalized_metadata))

        if not valid:
            return result

        try:
            to_add: list[tuple[str, dict]] = []
            grouped: dict[tuple[str, str, str], list[tuple[str, dict]]] = {}
            for item in valid:
                grouped.setdefault(_dedup_scope_key(item[1]), []).append(item)

            for group in grouped.values():
                existing = self.collection.query(
                    query_texts=[content for content, _ in group],
                    n_results=5,
                    where=_dedup_where(group[0][1]),
                )
                distances = existing.get("distances") or []
                ids = existing.get("ids") or []
                metadatas = existing.get("metadatas") or []
                for idx, (content, metadata) in enumerate(group):
                    row_distances = distances[idx] if idx < len(distances) else []
                    row_ids = ids[idx] if idx < len(ids) else []
                    row_metadatas = metadatas[idx] if idx < len(metadatas) else []
                    duplicate = None
                    for candidate_index, distance in enumerate(row_distances):
                        if distance is None:
                            continue
                        existing_metadata = (
                            row_metadatas[candidate_index]
                            if candidate_index < len(row_metadatas)
                            else {}
                        )
                        if not _same_dedup_scope(metadata, existing_metadata):
                            continue
                        similarity = 1 - distance
                        if similarity > threshold:
                            memory_ref = (
                                str(row_ids[candidate_index] or "").strip()
                                if candidate_index < len(row_ids)
                                else ""
                            )
                            duplicate = (similarity, memory_ref, existing_metadata)
                            break

                    if duplicate is None:
                        to_add.append((content, metadata))
                        continue

                    similarity, memory_ref, existing_metadata = duplicate
                    logger.debug(
                        f"[Memory] 跳过同 scope 重复记忆 "
                        f"(相似度 {similarity:.2f}): {content[:30]}..."
                    )
                    result["skipped_dedup"] += 1
                    if self._reinforce_duplicate_memory(memory_ref, existing_metadata, metadata):
                        result["reinforced"] += 1

            if to_add:
                write_result = self.add_texts(
                    [content for content, _ in to_add],
                    metadatas=[metadata for _, metadata in to_add],
                )
                if isinstance(write_result, dict):
                    result["added"] = int(write_result.get("added") or 0)
                else:
                    result["added"] = len(to_add)
            return result

        except Exception as e:
            logger.error(f"批量去重添加记忆失败: {e}")
            result["dedup_errors"] += 1
            result["added"] = 0
            result["skipped_dedup"] = 0
            self._append_wal(valid)
            return result

    def retrieve_with_decay(
        self, 
        queries: List[str], 
        k: int = 5, 
        where: dict | None = None, 
        use_rerank: bool = True,
        decay_rate: float = 0.02,
        candidate_k: int | None = None,
        merged_candidate_cap: int | None = None,
        active_user_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> RetrievalResult:
        """
        带时间衰减的检索
        
        Args:
            queries: 查询语句列表
            k: 返回结果数量
            where: 过滤条件
            use_rerank: 是否使用 Rerank
            decay_rate: 时间衰减率（默认 0.02，约 35 天半衰期）
            candidate_k: 每条 query 的召回数量，None 时使用 k
            merged_candidate_cap: 合并去重后送入 rerank 的候选上限
            active_user_ids: 当前活跃用户 ID；不传时保持旧 caller 行为
            
        Returns:
            检索结果列表，按综合分数排序
        """
        active_scope_ids = {
            str(user_id).strip()
            for user_id in active_user_ids or []
            if str(user_id).strip()
        }
        # 1. 调用原有语义检索方法，再补充当前主体的结构化 metadata 召回。
        effective_candidate_k = candidate_k if candidate_k is not None else k
        retrieval_result = self.retrieve(
            queries,
            k=effective_candidate_k,
            where=where,
            use_rerank=use_rerank,
            merged_candidate_cap=merged_candidate_cap,
        )
        if isinstance(retrieval_result, RetrievalResult):
            raw_results = list(retrieval_result.records)
            stats = dict(retrieval_result.stats)
        else:
            # Compatibility for injected/fake stores during gradual migration.
            raw_results = list(retrieval_result or [])
            stats = {
                **_empty_retrieval_stats(
                    use_rerank=use_rerank,
                    fallback_reason="legacy_result",
                ),
                "candidate_count": len(raw_results),
                "returned_count": len(raw_results),
            }
        subject_results = self._retrieve_active_subject_records(active_scope_ids, limit=min(5, max(1, k)))
        if subject_results:
            merged_results = []
            seen = set()
            for item in list(raw_results or []) + subject_results:
                content = str(item.get("content") or "")
                metadata = item.get("metadata", {}) or {}
                memory_ref = str(metadata.get("memory_ref") or "").strip()
                key = f"ref:{memory_ref}" if memory_ref else f"content:{content}"
                if not content or key in seen:
                    continue
                seen.add(key)
                merged_results.append(item)
            subject_added_count = max(0, len(merged_results) - len(raw_results or []))
            raw_results = merged_results
            stats["subject_recall_count"] = len(subject_results)
            stats["candidate_count"] = int(stats.get("candidate_count") or len(raw_results)) + subject_added_count

        if not raw_results:
            return RetrievalResult([], stats)

        # 2. 应用生命周期过滤和时间衰减
        today_dt = datetime.now()
        active_results = []
        other_subject_downweighted_count = 0
        legacy_subject_count = 0
        scope_counts: dict[str, int] = {}
        
        for item in raw_results:
            meta = item.get("metadata", {})
            meta.update(_normalized_metadata(meta))
            if _metadata_status(meta) != "active":
                continue
            scope, scope_weight = _memory_scope(meta, active_scope_ids, queries)
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
            if scope == "other_subject":
                other_subject_downweighted_count += 1
            elif scope == "legacy_subject":
                legacy_subject_count += 1
            active_results.append(item)
            date = meta.get("date", 0)

            if meta.get("source") == "preset":
                days_ago = 0
                decay_factor = 1.0
                effective_decay_rate = 0.0
            elif date and isinstance(date, int) and date > 0:
                # 计算天数差
                try:
                    memory_dt = datetime.strptime(str(date), "%Y%m%d")
                    days_ago = max(0, (today_dt - memory_dt).days)
                except ValueError:
                    days_ago = 60
                effective_decay_rate = _memory_decay_rate(meta, decay_rate)
                decay_factor = math.exp(-effective_decay_rate * days_ago)
            else:
                # 没有日期的记忆，视为较久以前
                days_ago = 60
                effective_decay_rate = _memory_decay_rate(meta, decay_rate)
                decay_factor = math.exp(-effective_decay_rate * days_ago)
            
            # 获取原始分数
            original_score = meta.get("rerank_score")
            if original_score is None:
                original_score = meta.get("retrieval_score", 0.5)
            
            # 计算调整后的分数
            source_type_weight = _source_type_weight(meta)
            confidence_weight = _confidence_weight(meta)
            importance_weight = 1.0 + _clamp_float(meta.get("importance"), 0.0, 0.0, 1.0) * 0.15
            adjusted_score = original_score * decay_factor * source_type_weight * confidence_weight * importance_weight * scope_weight
            meta["adjusted_score"] = adjusted_score
            meta["days_ago"] = days_ago
            meta["decay_rate"] = effective_decay_rate
            meta["decay_factor"] = decay_factor
            meta["source_type_weight"] = source_type_weight
            meta["confidence_weight"] = confidence_weight
            meta["importance_weight"] = importance_weight
            meta["scope"] = scope
            meta["scope_weight"] = scope_weight
        
        # 3. 重新排序
        sorted_results = sorted(
            active_results,
            key=lambda x: x.get("metadata", {}).get("adjusted_score", 0), 
            reverse=True
        )
        
        # 4. 截取前 k 个
        final_results = sorted_results[:k]
        adjusted_scores = [
            float(item.get("metadata", {}).get("adjusted_score", 0.0))
            for item in active_results
        ]
        stats.update(_score_distribution(adjusted_scores))
        stats["returned_count"] = len(final_results)
        stats["other_subject_downweighted_count"] = other_subject_downweighted_count
        stats["legacy_subject_count"] = legacy_subject_count
        stats["scope_counts"] = dict(scope_counts)
        return RetrievalResult(final_results, stats)
