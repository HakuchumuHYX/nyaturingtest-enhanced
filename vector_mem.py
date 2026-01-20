# nyaturingtest/vector_mem.py
import os
import asyncio
import uuid
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from nonebot import logger
from .config import plugin_config
from .repository import SessionRepository


class SiliconFlowReranker:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.siliconflow.cn/v1/rerank"

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
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                # 兼容不同厂商的返回格式，通常 SiliconFlow (BGE) 返回 results 列表
                return data.get("results", [])
        except Exception as e:
            logger.error(f"Rerank API Error: {e}")
            return []


class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, session_id: str, model: str = "BAAI/bge-m3"):
        self.api_key = api_key
        self.session_id = session_id
        self.model = model
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        if not input: return []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        cleaned_input = [text.replace("\n", " ") for text in input]
        payload = {"model": self.model, "input": cleaned_input, "encoding_format": "float"}
        try:
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Embedding API Error: {e}")
            raise e


class VectorMemory:
    def __init__(self, api_key: str, persist_directory: str, session_id: str = "global"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.emb_fn = SiliconFlowEmbeddingFunction(api_key=api_key, session_id=session_id)
        
        # 初始化 Reranker
        self.reranker = None
        if plugin_config.nyaturingtest_rerank_model:
            self.reranker = SiliconFlowReranker(
                api_key=api_key, 
                model=plugin_config.nyaturingtest_rerank_model
            )

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="nyabot_memory",
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_texts(self, texts: List[str], metadatas: List[dict] | None = None):
        if not texts: return
        valid_data = [(t, metadatas[i] if metadatas and i < len(metadatas) else {})
                      for i, t in enumerate(texts) if t and t.strip()]
        if not valid_data: return

        # 使用 UUID 防止重复覆盖
        ids = [str(uuid.uuid4()) for _ in valid_data]
        try:
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
        1. 如果启用 Rerank，先扩大召回 (k * 4)，然后 Rerank 取 Top K
        2. 如果未启用，直接召回 Top K
        """
        if not queries: return []
        unique_queries = list(set([q for q in queries if q.strip()]))
        
        # 决定初筛数量
        initial_k = k * 4 if (use_rerank and self.reranker) else k
        
        try:
            results = self.collection.query(query_texts=unique_queries, n_results=initial_k, where=where)
            
            # 第一步：合并去重初筛结果
            flattened_candidates = []
            seen = set()
            
            if results['documents']:
                for i, docs in enumerate(results['documents']):
                    metas = results['metadatas'][i]
                    # distances = results['distances'][i] # 如果需要
                    
                    for j, doc in enumerate(docs):
                        if doc and doc not in seen:
                            flattened_candidates.append({
                                "content": doc, 
                                "metadata": metas[j]
                            })
                            seen.add(doc)
            
            # 如果没有结果，直接返回
            if not flattened_candidates:
                return []

            # 如果不使用 Rerank 或 Reranker 未初始化，直接截断返回
            if not use_rerank or not self.reranker:
                return flattened_candidates[:k]

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
            
            final_results = []
            threshold = plugin_config.nyaturingtest_rerank_threshold
            
            for res in rerank_results:
                idx = res.get("index")
                score = res.get("relevance_score", 0.0)
                
                if score < threshold:
                    continue
                    
                if idx < len(flattened_candidates):
                    item = flattened_candidates[idx]
                    # 可以把分数附加上去，方便调试
                    item["metadata"]["rerank_score"] = score
                    final_results.append(item)
                    
                if len(final_results) >= k:
                    break
            
            logger.debug(f"Rerank完成: 初筛{len(candidate_docs)} -> 终选{len(final_results)} (阈值{threshold})")
            return final_results

        except Exception as e:
            logger.error(f"Vector retrieve failed: {e}")
            return []

    def delete_by_metadata(self, where: dict):
        """删除指定条件的记忆"""
        try:
            self.collection.delete(where=where)
            logger.info(f"Deleted vectors where {where}")
        except Exception as e:
            logger.error(f"Vector delete failed: {e}")

    def cleanup(self, days_retention: int = 90):
        """生命周期管理：清理过期事件"""
        try:
            # 将 cutoff_date 转为整数 (例如 20251018)
            cutoff_date = int((datetime.now() - timedelta(days=days_retention)).strftime("%Y%m%d"))

            # 仅清理 type=event 且日期早于 cutoff 的记录
            where_filter = {
                "$and": [
                    {"type": {"$eq": "event"}},
                    {"date": {"$lt": cutoff_date}}  # 现在是整数比较整数，ChromaDB 支持
                ]
            }
            self.collection.delete(where=where_filter)
            logger.info(f"Cleaned up memories before {cutoff_date}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def clear(self):
        try:
            self.client.delete_collection("nyabot_memory")
            self.collection = self.client.get_or_create_collection(name="nyabot_memory", embedding_function=self.emb_fn)
        except Exception as e:
            logger.error(f"Clear failed: {e}")
