from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool
from qdrant_client import QdrantClient

from config_utils import load_config


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        if "embedding" in data:
            return data["embedding"]
        if "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]
        raise ValueError("Unexpected Ollama embed response format")


class RagRetriever:
    def __init__(self, config_path) -> None:
        config = load_config(config_path)
        qdrant_cfg = config["qdrant"]
        ollama_cfg = config["ollama"]

        self.collection_name = qdrant_cfg["collection_name"]
        self.client = QdrantClient(
            url=qdrant_cfg["url"],
            api_key=qdrant_cfg.get("api_key") or None,
        )
        self.embedder = OllamaEmbedder(
            base_url=ollama_cfg["base_url"],
            model=ollama_cfg["embedding_model"],
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        vec = self.embedder.embed_query(query)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=top_k,
        )

        docs: List[Dict[str, Any]] = []
        for point in hits.points:
            payload = point.payload or {}
            docs.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "text": payload.get("text_ru") or "",
                    **payload,
                }
            )
        return docs


def make_rag_tool(retriever: RagRetriever):
    @tool("rag_search")
    def rag_search(query: str) -> Dict[str, Any]:
        """
        Поиск по локальной базе фильмов (RAG).
        Обязательно указывать:
        - query: запрос с примерным описанием об искомом фильме
        Возвращает:
        - "results": список документов
        """
        return {"results": retriever.retrieve(query, top_k=10)}

    return rag_search
