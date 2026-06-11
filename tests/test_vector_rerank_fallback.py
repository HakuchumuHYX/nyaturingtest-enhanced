import unittest

from test_vector_batch import _load_vector_module


class FakeCollection:
    def query(self, *, query_texts, n_results, where=None):
        return {
            "documents": [["memory alpha", "memory beta"]],
            "metadatas": [[{"source": "memory"}, {"source": "memory"}]],
        }


class EmptyReranker:
    def rerank(self, query, documents, top_n):
        return []


class InvalidIndexReranker:
    def rerank(self, query, documents, top_n):
        return [
            {"index": None, "relevance_score": 0.99},
            {"index": "1", "relevance_score": 0.95},
            {"index": 99, "relevance_score": 0.90},
        ]


class RecordingReranker:
    def __init__(self):
        self.queries = []

    def rerank(self, query, documents, top_n):
        self.queries.append(query)
        return [{"index": 0, "relevance_score": 0.99}]


class VectorRerankFallbackTests(unittest.TestCase):
    def _memory_with_reranker(self, reranker):
        module = _load_vector_module()
        memory = object.__new__(module.VectorMemory)
        memory.collection = FakeCollection()
        memory.reranker = reranker
        return memory

    def test_retrieve_falls_back_to_initial_candidates_when_rerank_returns_empty(self):
        memory = self._memory_with_reranker(EmptyReranker())

        result = memory.retrieve(["alpha"], k=1, use_rerank=True)

        self.assertEqual([item["content"] for item in result], ["memory alpha"])

    def test_retrieve_ignores_invalid_rerank_indexes_and_falls_back_when_none_are_usable(self):
        memory = self._memory_with_reranker(InvalidIndexReranker())

        result = memory.retrieve(["alpha"], k=2, use_rerank=True)

        self.assertEqual([item["content"] for item in result], ["memory alpha", "memory beta"])

    def test_retrieve_uses_first_effective_query_as_rerank_anchor(self):
        reranker = RecordingReranker()
        memory = self._memory_with_reranker(reranker)

        memory.retrieve(["最新有效消息", "这是一段很长很长很长的上一轮摘要"], k=1, use_rerank=True)

        self.assertEqual(["最新有效消息"], reranker.queries)


if __name__ == "__main__":
    unittest.main()
