# nyaturingtest/vector_mem.py
import os
import hashlib
import logging
import uuid
from typing import List, Dict, Any
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

    # 增加 metadatas 参数
    def add_texts(self, texts: List[str], metadatas: List[dict] | None = None):
        """
        添加记忆文本，支持元数据
        """
        if not texts:
            return

        valid_data = []
        for i, t in enumerate(texts):
            if t and t.strip():
                # 如果有元数据就带上，没有就给空字典
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                valid_data.append((t, meta))

        if not valid_data:
            return

        valid_texts = [d[0] for d in valid_data]
        valid_metadatas = [d[1] for d in valid_data]

        # 使用 MD5 哈希作为 ID，防止重复存储相同的记忆/预设
        ids = [hashlib.md5(t.encode('utf-8')).hexdigest() for t in valid_texts]

        try:
            self.collection.add(
                documents=valid_texts,
                metadatas=valid_metadatas,  # 存入元数据
                ids=ids
            )
            logger.debug(f"已存入 {len(valid_texts)} 条记忆 (含元数据)")
        except Exception as e:
            logger.error(f"VectorMemory add_texts 失败: {e}")

    # 返回类型改为 List[dict] 以便携带元数据
    def retrieve(self, queries: List[str], k: int = 5, where: dict | None = None) -> List[Dict[str, Any]]:
        if not queries:
            return []

        unique_queries = list(set([q for q in queries if q and q.strip()]))
        if not unique_queries:
            return []

        results_list = []
        seen_contents = set()

        try:
            query_results = self.collection.query(
                query_texts=unique_queries,
                n_results=k,
                where=where  # 传递过滤条件，例如 {"user_id": "123456"}
            )

            # 解析 ChromaDB 的返回格式 (documents 和 metadatas 都是二维列表)
            if query_results['documents']:
                for i, doc_list in enumerate(query_results['documents']):
                    meta_list = query_results['metadatas'][i] if query_results['metadatas'] else []

                    for j, doc in enumerate(doc_list):
                        if doc and doc not in seen_contents:
                            meta = meta_list[j] if meta_list and j < len(meta_list) else {}
                            results_list.append({
                                "content": doc,
                                "metadata": meta
                            })
                            seen_contents.add(doc)

            return results_list
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
