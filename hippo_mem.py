import gc
import time
from datetime import datetime
import os
import shutil
# import numpy as np

from hipporag import HippoRAG
from nonebot import logger
from transformers.models.auto.tokenization_auto import AutoTokenizer

# 全局分词器变量
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

        # 初始化参数保存
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = "https://api.siliconflow.cn/v1"

        # 尝试初始化
        self._init_hipporag()

        # 用于跟踪上次清理的时间
        self._last_forget = datetime.now()
        # 缓存要索引的文本
        self._cache = ""
        # 使用全局单例初始化分词器
        self.tokenizer = _get_tokenizer()

    def _init_hipporag(self, retry: bool = True) -> bool:
        """
        尝试初始化 HippoRAG 实例 (带自动修复功能)
        """
        try:
            self.hippo = HippoRAG(
                llm_model_name=self.llm_model,
                llm_base_url=self.llm_base_url,
                llm_api_key=self.llm_api_key,
                embedding_model_name="BAAI/bge-m3",
                embedding_api_key=self.embedding_api_key,
                embedding_base_url=self.embedding_base_url,
                save_dir=self.persist_directory,
            )
            logger.info(f"已创建/加载 HippoRAG 集合: {self.collection_name}")
            return True
        except Exception as e:
            error_msg = str(e)

            # 检测文件损坏错误
            if retry and ("Ran out of input" in error_msg or "EOFError" in error_msg or "unpickling" in error_msg):
                logger.warning(f"检测到 HippoRAG 索引文件损坏 ({error_msg})，正在准备自动修复...")

                # 1. 强制断开引用 & GC
                if hasattr(self, 'hippo'):
                    del self.hippo
                import gc
                import time
                gc.collect()
                time.sleep(1.0)  # 给 Windows 一点时间释放锁

                if os.path.exists(self.persist_directory):
                    try:
                        shutil.rmtree(self.persist_directory, ignore_errors=True)
                        logger.info("已清理损坏的索引目录")

                        if os.path.exists(self.persist_directory):
                            logger.warning("目录删除似乎未完全生效，再次尝试...")
                            time.sleep(1.0)
                            shutil.rmtree(self.persist_directory, ignore_errors=True)

                    except Exception as rm_e:
                        logger.error(f"删除损坏目录失败: {rm_e}")
                        return False

                logger.info("正在尝试重新构建索引...")
                return self._init_hipporag(retry=False)

            logger.error(f"HippoRAG 初始化失败: {e}")
            if hasattr(self, 'hippo'):
                delattr(self, 'hippo')
            return False

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

        # 重新初始化
        self._init_hipporag()
        logger.info("已清除所有记忆")

    def add_texts(self, texts: list[str]) -> None:
        """
        添加文本到缓存
        """
        for text in texts:
            self._cache += text + "\n"

    def index(self):
        """
        对缓存的文本进行索引，整理到长期记忆
        """
        # 1. 检查 HippoRAG 是否初始化成功
        if not hasattr(self, 'hippo'):
            logger.warning("HippoRAG 未初始化，尝试重新初始化...")
            if not self._init_hipporag():
                logger.error("HippoRAG 重新初始化失败，跳过本次索引构建 (数据保留在缓存中)")
                return

        # 2. 先检查并“快照”取走当前缓存
        content_to_index = self._cache
        if not content_to_index or not content_to_index.strip():
            return

        # 3. 立即清空缓存
        self._cache = ""

        try:
            # 4. 使用快照数据进行切分
            texts = _split_text_by_tokens(content_to_index, self.tokenizer, max_tokens=2048, overlap=200)

            if not texts:
                return

            logger.info(f"开始构建索引，共 {len(texts)} 条文本段...")

            # 5. 执行索引 (耗时操作)
            self.hippo.index(texts)

            logger.info(f"已成功索引 {len(texts)} 条缓存文本")

        except Exception as e:
            # 6. 发生错误时进行“回滚”
            logger.error(f"HippoRAG 索引构建失败，正在回滚缓存: {e}")
            self._cache = content_to_index + self._cache
            # 不抛出异常

    def retrieve(self, queries: list[str], k: int = 5) -> list[str]:
        """
        检索与查询相关的文本
        """
        # 检查实例是否存在
        if not hasattr(self, 'hippo'):
            if not self._init_hipporag():
                return []

        try:
            # 检查索引是否为空
            if hasattr(self.hippo, "passage_embeddings"):
                if self.hippo.passage_embeddings is None or len(self.hippo.passage_embeddings) == 0:
                    return []
        except Exception:
            return []

        splited_queries = []
        for query in queries:
            splited_queries += _split_text_by_tokens(query, self.tokenizer, max_tokens=2048, overlap=100)

        try:
            results = self.hippo.retrieve(queries=splited_queries, num_to_retrieve=k)
            # make ruff happy
            assert isinstance(results, list)
            docs = [doc for result in results for doc in result.docs]
            # 去重
            return list(set(docs))
        except ValueError as e:
            if "shapes" in str(e) and "not aligned" in str(e):
                logger.warning("长期记忆索引为空或数据对齐错误，跳过检索")
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
        return len(self._cache.strip().split('\n'))


def _split_text_by_tokens(
        text: str, tokenizer, max_tokens=2048, overlap=100
) -> list[str]:
    """
    按照指定的最大 token 数量和重叠数量将文本分割成多个块
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
