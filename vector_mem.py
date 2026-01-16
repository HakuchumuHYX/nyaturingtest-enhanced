# nyaturingtest/vector_mem.py
import os
import uuid
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from nonebot import logger


class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model: str = "BAAI/bge-m3"):
        self.api_key = api_key
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
                return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            logger.error(f"Embedding API Error: {e}")
            raise e


class VectorMemory:
    def __init__(self, api_key: str, persist_directory: str):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.emb_fn = SiliconFlowEmbeddingFunction(api_key=api_key)
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

    def retrieve(self, queries: List[str], k: int = 5, where: dict | None = None) -> List[Dict[str, Any]]:
        if not queries: return []
        unique_queries = list(set([q for q in queries if q.strip()]))
        try:
            results = self.collection.query(query_texts=unique_queries, n_results=k, where=where)
            flattened = []
            seen = set()
            if results['documents']:
                for i, docs in enumerate(results['documents']):
                    metas = results['metadatas'][i]
                    for j, doc in enumerate(docs):
                        if doc not in seen:
                            flattened.append({"content": doc, "metadata": metas[j]})
                            seen.add(doc)
            return flattened
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
