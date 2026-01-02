import os
import hashlib
import logging
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
        # 移除换行符
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
            raise e  # 抛出异常以便在上层看到具体的错误堆栈

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
        添加记忆文本，自动去重
        """
        if not texts:
            return

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return

        # [关键优化] 使用内容哈希作为ID，实现幂等性（重复添加不会重复存储）
        ids = [hashlib.md5(t.encode('utf-8')).hexdigest() for t in valid_texts]

        try:
            # 使用 upsert：存在则更新，不存在则插入
            self.collection.upsert(
                documents=valid_texts,
                ids=ids
            )
            logger.debug(f"已处理 {len(valid_texts)} 条长期记忆 (Upsert)")
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
