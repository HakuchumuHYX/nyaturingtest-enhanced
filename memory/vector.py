# nyaturingtest/vector_mem.py
import os
import uuid
import math
import json
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from nonebot import logger
from openai import OpenAI
from ..config import plugin_config, get_memory_endpoint_settings
from ..database.backup_lock import BACKUP_IO_LOCK


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
SCOPE_WEIGHT = {
    "active_user": 1.10,
    "active_subject": 1.10,
    "mentioned_subject": 1.08,
    "active_speaker": 1.04,
    "global": 1.0,
    "legacy_subject": 0.75,
    "other_user": 0.5,
    "other_subject": 0.5,
}
_metric_check_done: set[str] = set()


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


def _source_type_weight(meta: dict) -> float:
    source = str(meta.get("source") or "memory")
    memory_type = str(meta.get("type") or "event")
    subtype = str(meta.get("subtype") or ("legacy_rule" if source == "preset" else memory_type))
    if source == "preset":
        return PRESET_TYPE_WEIGHT.get(subtype, PRESET_TYPE_WEIGHT["legacy_rule"])
    return MEMORY_TYPE_WEIGHT.get(memory_type, 1.0)


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


class SiliconFlowReranker:
    def __init__(self, api_key: str, model: str, api_url: str | None = None, timeout: float | None = None):
        settings = get_memory_endpoint_settings()
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or str(settings["rerank_base_url"])
        self._client = httpx.Client(timeout=timeout or float(settings["rerank_timeout"]), trust_env=False)

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        返回格式: [{"index": int, "relevance_score": float}, ...] 
        注意: SiliconFlow API 返回的结果中 document 索引对应传入 documents 的顺序
        """
        if not documents:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False  # 不需要返回文档内容，只要索引和分数，省流
        }
        
        try:
            response = self._client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # 兼容不同厂商的返回格式，通常 SiliconFlow (BGE) 返回 results 列表
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Rerank API Error: {e}")
            return []

    def close(self):
        self._client.close()


class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        api_key: str,
        session_id: str,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        settings = get_memory_endpoint_settings()
        self.api_key = api_key
        self.session_id = session_id
        self.model = model or str(settings["model"])
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or str(settings["base_url"]),
            timeout=timeout or float(settings["timeout"]),
            max_retries=1,
        )

    def __call__(self, input: Documents) -> Embeddings:
        if not input: return []
        cleaned_input = [text.replace("\n", " ") for text in input]
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=cleaned_input,
                encoding_format="float",
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding API Error: {e}")
            raise e

    def close(self):
        self._client.close()


class VectorMemory:
    """
    Synchronous vector store wrapper.

    Call ChromaDB, embedding, and rerank operations from async code through
    nonebot.utils.run_sync or another thread-pool adapter.
    """

    def __init__(self, api_key: str, persist_directory: str, session_id: str = "global"):
        self.persist_directory = persist_directory
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
        if plugin_config.get("rerank", {}).get("model"):
            self.reranker = SiliconFlowReranker(
                api_key=api_key, 
                model=plugin_config.get("rerank", {}).get("model", ""),
                api_url=str(memory_settings["rerank_base_url"]),
                timeout=float(memory_settings["rerank_timeout"]),
            )

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=self.emb_fn,
            metadata=MEMORY_COLLECTION_METADATA
        )
        self._last_retrieval_stats = _empty_retrieval_stats()
        self._check_collection_metric_once()
        self._ids_supported = self._probe_ids_support()

    @property
    def last_retrieval_stats(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_retrieval_stats", _empty_retrieval_stats()))

    def _set_retrieval_stats(self, stats: dict[str, Any]) -> None:
        self._last_retrieval_stats = dict(stats)

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

    def add_texts(self, texts: List[str], metadatas: List[dict] | None = None):
        if not texts: return
        valid_data = [(t, metadatas[i] if metadatas and i < len(metadatas) else {})
                      for i, t in enumerate(texts) if t and t.strip()]
        if not valid_data: return

        # 使用 UUID 防止重复覆盖
        ids = [str(uuid.uuid4()) for _ in valid_data]
        try:
            with BACKUP_IO_LOCK:
                self.collection.add(
                    documents=[d[0] for d in valid_data],
                    metadatas=[d[1] for d in valid_data],
                    ids=ids
                )
        except Exception as e:
            logger.error(f"Vector add failed: {e}")

    def retrieve(self, queries: List[str], k: int = 5, where: dict | None = None, use_rerank: bool = True) -> List[Dict[str, Any]]:
        """
        检索逻辑：
        1. 如果启用 Rerank，先扩大召回 (k * 2, max 50)，然后 Rerank 取 Top K
        2. 如果未启用，直接召回 Top K
        """
        if not queries:
            self._set_retrieval_stats(_empty_retrieval_stats(use_rerank=use_rerank))
            return []
        unique_queries = _dedupe_preserve_order([q for q in queries if q.strip()])
        if not unique_queries:
            self._set_retrieval_stats(_empty_retrieval_stats(use_rerank=use_rerank))
            return []
        
        # 决定初筛数量 (优化：降低倍率至2倍，且设置硬性上限50，防止Token爆炸)
        if use_rerank and self.reranker:
            initial_k = min(k * 2, 50) 
        else:
            initial_k = k
        
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
            
            # 如果没有结果，直接返回
            if not flattened_candidates:
                self._set_retrieval_stats(_empty_retrieval_stats(use_rerank=use_rerank))
                return []

            # 如果不使用 Rerank 或 Reranker 未初始化，直接截断返回
            if not use_rerank or not self.reranker:
                results = flattened_candidates[:k]
                self._set_retrieval_stats({
                    **_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="rerank_disabled"),
                    "candidate_count": len(flattened_candidates),
                    "returned_count": len(results),
                })
                return results

            # 第二步：Rerank
            # 由于 Rerank 通常是一对多（一个 Query 对多个 Doc），这里简化处理：
            # 将所有 Query 拼接（或者只取第一个 Query）作为 Rerank 的基准 Query
            # 这里的业务场景通常是 "关于XXX的记忆"，语义比较接近，取第一个 Query 往往足够
            # 或者，更严谨的做法是对每个 Query 分别 Rerank 再融合，但耗时。
            # 这里采用：拼接最长的两个 Query 作为基准语义
            sorted_queries = sorted(unique_queries, key=lambda x: len(x), reverse=True)
            main_query = " ".join(sorted_queries[:2]) 
            
            candidate_docs = [item["content"] for item in flattened_candidates]
            
            rerank_results = self.reranker.rerank(
                query=main_query,
                documents=candidate_docs,
                top_n=len(candidate_docs), # 全排，然后本地过滤
            )
            if not rerank_results:
                logger.debug("Rerank无结果，回退到初筛候选")
                results = flattened_candidates[:k]
                self._set_retrieval_stats({
                    **_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="rerank_api_empty"),
                    "candidate_count": len(flattened_candidates),
                    "returned_count": len(results),
                })
                return results
            
            final_results = []
            threshold = plugin_config.get("rerank", {}).get("threshold", 0.05)
            
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
                self._set_retrieval_stats({
                    **_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="rerank_all_filtered"),
                    "candidate_count": len(flattened_candidates),
                    "returned_count": len(results),
                })
                return results
            self._set_retrieval_stats({
                **_empty_retrieval_stats(use_rerank=use_rerank),
                "candidate_count": len(flattened_candidates),
                "returned_count": len(final_results),
            })
            return final_results

        except Exception as e:
            logger.error(f"Vector retrieve failed: {e}")
            self._set_retrieval_stats(_empty_retrieval_stats(use_rerank=use_rerank, fallback_reason="retrieve_error"))
            return []

    def delete_by_metadata(self, where: dict):
        """删除指定条件的记忆"""
        try:
            with BACKUP_IO_LOCK:
                self.collection.delete(where=where)
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
                should_delete = (
                    (status in {"archived", "superseded"} and days_ago > days_retention)
                    or (metadata.get("type") == "event" and days_ago > ttl_days)
                )
                if should_delete:
                    delete_ids.append(item_id)

            for batch in _batched(delete_ids, 200):
                with BACKUP_IO_LOCK:
                    self.collection.delete(ids=batch)
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

    def backfill_active_status(self, *, dry_run: bool = True, batch_size: int = 200, max_rounds: int = 5) -> dict[str, Any]:
        """Backfill missing status metadata without re-embedding records."""
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

    def add_memories_with_dedup(self, memories: list[tuple[str, dict]], threshold: float = 0.9) -> dict[str, int]:
        """
        批量去重并添加长期记忆。

        对同一批候选记忆只做一次 Chroma query 和一次 add，避免逐条 embedding/query/add。
        """
        result = {"added": 0, "skipped_empty": 0, "skipped_dedup": 0}
        valid: list[tuple[str, dict]] = []
        seen_batch = set()
        for content, metadata in memories:
            normalized = (content or "").strip()
            if not normalized:
                result["skipped_empty"] += 1
                continue
            if normalized in seen_batch:
                result["skipped_dedup"] += 1
                continue
            seen_batch.add(normalized)
            valid.append((normalized, metadata or {}))

        if not valid:
            return result

        try:
            existing = self.collection.query(
                query_texts=[content for content, _ in valid],
                n_results=1,
                where={
                    "$or": [
                        {"source": {"$eq": "memory"}},
                        {"source": {"$eq": "preset"}},
                    ]
                },
            )

            to_add: list[tuple[str, dict]] = []
            distances = existing.get("distances") or []
            for idx, (content, metadata) in enumerate(valid):
                row = distances[idx] if idx < len(distances) else []
                distance = row[0] if row else None
                if distance is None:
                    to_add.append((content, metadata))
                    continue

                similarity = 1 - distance
                if similarity > threshold:
                    logger.debug(f"[Memory] 跳过重复记忆 (相似度 {similarity:.2f}): {content[:30]}...")
                    result["skipped_dedup"] += 1
                else:
                    to_add.append((content, metadata))

            if to_add:
                self.add_texts(
                    [content for content, _ in to_add],
                    metadatas=[metadata for _, metadata in to_add],
                )
                result["added"] = len(to_add)
            return result

        except Exception as e:
            logger.error(f"批量去重添加记忆失败: {e}")
            self.add_texts(
                [content for content, _ in valid],
                metadatas=[metadata for _, metadata in valid],
            )
            result["added"] = len(valid)
            result["skipped_dedup"] = 0
            return result

    def retrieve_with_decay(
        self, 
        queries: List[str], 
        k: int = 5, 
        where: dict | None = None, 
        use_rerank: bool = True,
        decay_rate: float = 0.02,
        candidate_k: int | None = None,
        active_user_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        带时间衰减的检索
        
        Args:
            queries: 查询语句列表
            k: 返回结果数量
            where: 过滤条件
            use_rerank: 是否使用 Rerank
            decay_rate: 时间衰减率（默认 0.02，约 35 天半衰期）
            candidate_k: 召回候选数量，None 时保持旧 k*2 行为
            active_user_ids: 当前活跃用户 ID；不传时保持旧 caller 行为
            
        Returns:
            检索结果列表，按综合分数排序
        """
        # 1. 调用原有检索方法
        effective_candidate_k = candidate_k if candidate_k is not None else k * 2
        raw_results = self.retrieve(queries, k=effective_candidate_k, where=where, use_rerank=use_rerank)
        stats = self.last_retrieval_stats
        
        if not raw_results:
            self._set_retrieval_stats(stats)
            return []
        
        # 2. 应用生命周期过滤和时间衰减
        today_dt = datetime.now()
        active_results = []
        active_scope_ids = {
            str(user_id).strip()
            for user_id in active_user_ids or []
            if str(user_id).strip()
        }
        other_user_filtered_count = 0
        other_subject_downweighted_count = 0
        legacy_subject_count = 0
        
        for item in raw_results:
            meta = item.get("metadata", {})
            meta.update(_normalized_metadata(meta))
            if _metadata_status(meta) != "active":
                continue
            scope, scope_weight = _memory_scope(meta, active_scope_ids, queries)
            if scope == "other_subject":
                other_subject_downweighted_count += 1
            elif scope == "legacy_subject":
                legacy_subject_count += 1
            active_results.append(item)
            date = meta.get("date", 0)

            if meta.get("source") == "preset":
                days_ago = 0
                decay_factor = 1.0
            elif date and isinstance(date, int) and date > 0:
                # 计算天数差
                try:
                    memory_dt = datetime.strptime(str(date), "%Y%m%d")
                    days_ago = max(0, (today_dt - memory_dt).days)
                except ValueError:
                    days_ago = 60
                decay_factor = math.exp(-decay_rate * days_ago)
            else:
                # 没有日期的记忆，视为较久以前
                days_ago = 60
                decay_factor = math.exp(-decay_rate * days_ago)
            
            # 获取原始分数
            original_score = meta.get("rerank_score")
            if original_score is None:
                original_score = meta.get("retrieval_score", 0.5)
            
            # 计算调整后的分数
            source_type_weight = _source_type_weight(meta)
            confidence_weight = _clamp_float(meta.get("confidence"), 1.0, 0.0, 1.0)
            importance_weight = 1.0 + _clamp_float(meta.get("importance"), 0.0, 0.0, 1.0) * 0.15
            adjusted_score = original_score * decay_factor * source_type_weight * confidence_weight * importance_weight * scope_weight
            meta["adjusted_score"] = adjusted_score
            meta["days_ago"] = days_ago
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
        stats["other_user_filtered_count"] = other_user_filtered_count
        stats["other_subject_downweighted_count"] = other_subject_downweighted_count
        stats["legacy_subject_count"] = legacy_subject_count
        self._set_retrieval_stats(stats)
        return final_results
