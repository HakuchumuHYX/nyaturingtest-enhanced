# nyaturingtest/vector_mem.py
import os
import hashlib
import logging
import uuid  # [新增]
from typing import List
import httpx
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from nonebot import logger


class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    """
    使用 SiliconFlow API 计算向量
    """

    def __init__(self, api_key: str, model: str = "BAAI/bge-m3"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        cleaned_input = [text.replace("\n", " ") for text in input]

        payload = {
            "model": self.model,
            "input": cleaned_input,
            "encoding_format": "float"
        }

        try:
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Embedding API 请求失败: {e}")
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
        logger.info(f"VectorMemory 已加载，路径: {self.persist_directory}")

    def add_texts(self, texts: List[str]):
        """
        添加记忆文本
        """
        if not texts:
            return

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return

        # [修改] 使用 UUID 生成随机 ID，允许重复内容入库（因为它们代表不同时刻的记忆）
        ids = [str(uuid.uuid4()) for _ in valid_texts]

        try:
            # [修改] 使用 add 而不是 upsert，因为 ID 是唯一的
            self.collection.add(
                documents=valid_texts,
                ids=ids
            )
            logger.debug(f"已处理 {len(valid_texts)} 条长期记忆 (Add)")
        except Exception as e:
            logger.error(f"VectorMemory add_texts 失败: {e}")

    def retrieve(self, queries: List[str], k: int = 5) -> List[str]:
        if not queries:
            return []

        unique_queries = list(set([q for q in queries if q and q.strip()]))
        if not unique_queries:
            return []

        results_set = set()
        try:
            query_results = self.collection.query(
                query_texts=unique_queries,
                n_results=k
            )
            if query_results['documents']:
                for doc_list in query_results['documents']:
                    for doc in doc_list:
                        if doc:
                            results_set.add(doc)
            return list(results_set)
        except Exception as e:
            logger.error(f"VectorMemory retrieve 失败: {e}")
            return []

    def clear(self):
        try:
            self.client.delete_collection("nyabot_memory")
            self.collection = self.client.get_or_create_collection(
                name="nyabot_memory",
                embedding_function=self.emb_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("长期记忆已清空")
        except Exception as e:
            logger.error(f"VectorMemory clear 失败: {e}")
