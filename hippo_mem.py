from datetime import datetime
import os
import shutil
import numpy as np  # [新增] 引入 numpy 以便做更严谨的判断(如果需要)

from hipporag import HippoRAG
from nonebot import logger
from transformers.models.auto.tokenization_auto import AutoTokenizer

# [新增] 全局分词器变量，防止多实例重复加载占用内存
_GLOBAL_TOKENIZER = None


def _get_tokenizer():
    """
    获取全局唯一的 Tokenizer 实例
    """
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        logger.info("正在加载全局分词器 (BAAI/bge-m3)...")
        _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
    return _GLOBAL_TOKENIZER


class HippoMemory:
    def __init__(
            self,
            llm_model: str,
            llm_base_url: str,
            llm_api_key: str,
            embedding_api_key: str,
            persist_directory: str = "./hippo_index",
            collection_name: str = "hippo_collection",
    ):
        # 确保存储目录存在
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化HippoRAG
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 使用HippoRAG初始化记忆库
        try:
            self.hippo = HippoRAG(
                llm_model_name=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_model_name="BAAI/bge-m3",
                embedding_api_key=embedding_api_key,
                embedding_base_url="https://api.siliconflow.cn/v1",
                save_dir=persist_directory,
            )
            logger.info(f"已创建/加载新的HippoRAG集合: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create HippoRAG collection: {e}")

        # 用于跟踪上次清理的时间
        self._last_forget = datetime.now()
        # 缓存要索引的文本
        self._cache = ""
        # [修改] 使用全局单例初始化分词器
        self.tokenizer = _get_tokenizer()

    def _now_str(self) -> str:
        """返回当前时间的 ISO 格式字符串"""
        return datetime.now().isoformat()

    def clear(self) -> None:
        """
        清除所有记忆
        """
        # 删除索引文件
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except Exception as e:
                logger.error(f"Failed to delete persist directory: {e}")
        else:
            logger.warning(f"Persist directory {self.persist_directory} does not exist.")
        # 重新创建索引
        try:
            self.hippo = HippoRAG(
                llm_model_name=self.hippo.global_config.llm_name,
                llm_base_url=self.hippo.global_config.llm_base_url,
                llm_api_key=self.hippo.global_config.llm_api_key,
                embedding_model_name=self.hippo.global_config.embedding_model_name,
                embedding_api_key=self.hippo.global_config.embedding_api_key,
                embedding_base_url=self.hippo.global_config.embedding_base_url,
                save_dir=self.persist_directory,
            )
        except Exception as e:
            logger.error(f"Failed to recreate HippoRAG collection: {e}")
            return
        logger.info("已清除所有记忆")

    def add_texts(self, texts: list[str]) -> None:
        """
        添加文本到缓存

        Args:
            texts: 要添加的文本列表
        """
        for text in texts:
            self._cache += text + "\n"

    def index(self):
        """
        对缓存的文本进行索引，整理到长期记忆
        """
        # 1. 检查缓存是否为空，为空直接返回，不要输出日志（防止刷屏）
        if not self._cache or not self._cache.strip():
            return

        try:
            # 2. 切割文本 (BAAI/bge-m3上限为8192tokens)
            texts = _split_text_by_tokens(self._cache, self.tokenizer, max_tokens=2048, overlap=200)

            if not texts:
                return

            logger.info(f"开始构建索引，共 {len(texts)} 条文本段...")

            # 3. 执行索引
            self.hippo.index(texts)

            logger.info(f"已成功索引 {len(texts)} 条缓存文本")

            # 4. 只有索引成功后，才清空缓存
            self._cache = ""

        except Exception as e:
            # 5. 捕获异常，保留 self._cache 不清空，以便下次重试
            logger.error(f"HippoRAG 索引构建失败，缓存已保留: {e}")
            # 抛出异常供 session.py 捕获感知
            raise e

    def retrieve(self, queries: list[str], k: int = 5) -> list[str]:
        """
        检索与查询相关的文本

        Args:
            queries: 查询文本列表
            k: 返回的最大结果数

        Returns:
            包含检索结果的Document列表
        """
        # [关键修复] 检查索引是否为空
        # HippoRAG 的 passage_embeddings 为空时会导致 numpy shape mismatch (0,) vs (1024,)
        try:
            if hasattr(self.hippo, "passage_embeddings"):
                # 如果 embedding 是 None 或者长度为 0
                if self.hippo.passage_embeddings is None or len(self.hippo.passage_embeddings) == 0:
                    # 静默返回，因为这在刚启动时很正常
                    return []
        except Exception:
            # 如果访问属性出错，也直接返回空以保平安
            return []

        # 切割(BAAI/bge-m3上限为8192tokens)
        logger.debug(f"查询文本: {queries}")
        splited_queries = []
        for query in queries:
            splited_queries += _split_text_by_tokens(query, self.tokenizer, max_tokens=2048, overlap=100)

        # logger.debug(f"分割后的查询: {splited_queries}")

        try:
            results = self.hippo.retrieve(queries=splited_queries, num_to_retrieve=k)
            # make ruff happy
            assert isinstance(results, list)
            docs = [doc for result in results for doc in result.docs]
            # 去重
            return list(set(docs))
        except ValueError as e:
            # 双重保险：捕获 numpy 的 shapes mismatch 错误
            if "shapes" in str(e) and "not aligned" in str(e):
                logger.warning("长期记忆索引为空，跳过检索")
                return []
            raise e
        except Exception as e:
            logger.error(f"检索过程发生未知错误: {e}")
            return []

    @property
    def pending_count(self) -> int:
        """
        返回缓存中等待索引的文本行数/条数
        """
        if not self._cache:
            return 0
        # 简单通过换行符计算积累了多少条记忆
        return len(self._cache.strip().split('\n'))


def _split_text_by_tokens(
        text: str, tokenizer, max_tokens=2048, overlap=100
) -> list[str]:
    """
    按照指定的最大 token 数量和重叠数量将文本分割成多个块
    Args:
        text: 要分割的文本
        tokenizer: 用于分割文本的分词器
        max_tokens: 每个块的最大 token 数量
        overlap: 重叠的 token 数量
    Returns:
        分割后的文本块列表
    """
    if not text:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks