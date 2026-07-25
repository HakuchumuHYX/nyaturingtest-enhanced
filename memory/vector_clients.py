from typing import Any

import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from nonebot import logger
from openai import OpenAI

from ..config import get_memory_endpoint_settings


class SiliconFlowReranker:
    """Small synchronous adapter for the configured rerank endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str | None = None,
        timeout: float | None = None,
    ):
        settings = get_memory_endpoint_settings()
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or str(settings["rerank_base_url"])
        self._client = httpx.Client(
            timeout=timeout or float(settings["rerank_timeout"]),
            trust_env=False,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        try:
            response = self._client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": False,
                },
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"Rerank API Error: {e}")
            return []

    def close(self) -> None:
        self._client.close()


class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding adapter with an owned OpenAI-compatible client."""

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
            max_retries=0,
        )

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=[text.replace("\n", " ") for text in input],
                encoding_format="float",
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding API Error: {e}")
            raise

    def close(self) -> None:
        self._client.close()
