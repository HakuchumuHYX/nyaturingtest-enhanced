import unittest

from tests.test_vector_batch import _load_vector_module


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


if __name__ == "__main__":
    unittest.main()
