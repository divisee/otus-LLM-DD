"""
Простое RAG-приложение для поиска фильмов.

Использует OpenAI embeddings (text-embedding-3-small) для векторного поиска
и gpt-4o-mini для генерации ответов на основе найденного контекста.
"""

import os
import csv
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "Ты — помощник по поиску фильмов. Тебе даны описания фильмов из базы данных "
    "и вопрос пользователя.\n"
    "Определи, о каком фильме идёт речь, и дай краткий ответ.\n"
    "Отвечай на основе предоставленного контекста. "
    "Выбери наиболее подходящий фильм из контекста, даже если совпадение частичное.\n"
    "Обязательно укажи название фильма в ответе."
)


def load_movies(csv_path: str) -> List[Dict[str, str]]:
    """Загрузка фильмов из CSV (title, description)."""
    movies = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            description = row.get("description", "").strip()
            if title and description:
                movies.append({"title": title, "description": description})
    return movies


def get_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Получение эмбеддингов через OpenAI API (батчами до 2048)."""
    all_embeddings = []
    batch_size = 2048
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
    return np.array(all_embeddings)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Косинусное сходство между запросом и набором документов."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
    doc_norms = doc_vecs / norms
    return doc_norms @ query_norm


def retrieve(
    query: str,
    movies: List[Dict[str, str]],
    movie_embeddings: np.ndarray,
    client: OpenAI,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Поиск top-k наиболее релевантных фильмов по запросу."""
    query_vec = get_embeddings([query], client)[0]
    scores = cosine_similarity(query_vec, movie_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**movies[i], "score": float(scores[i])} for i in top_indices]


def generate_answer(question: str, contexts: List[str], client: OpenAI) -> str:
    """Генерация ответа на основе контекста через OpenAI."""
    context_text = "\n---\n".join(contexts)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Контекст:\n{context_text}\n\nВопрос: {question}",
            },
        ],
        temperature=0,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


class MovieRAG:
    """RAG-система для поиска фильмов: эмбеддинги + генерация ответа."""

    def __init__(self, csv_path: str, client: Optional[OpenAI] = None):
        self.client = client or OpenAI()
        self.movies = load_movies(csv_path)
        if not self.movies:
            raise ValueError(f"Не удалось загрузить фильмы из {csv_path}")
        texts = [f"{m['title']}. {m['description']}" for m in self.movies]
        self.embeddings = get_embeddings(texts, self.client)

    def query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """Полный RAG-пайплайн: retrieve → generate."""
        retrieved = retrieve(
            question, self.movies, self.embeddings, self.client, top_k
        )
        contexts = [
            f"Фильм: {m['title']}\n{m['description']}" for m in retrieved
        ]
        answer = generate_answer(question, contexts, self.client)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "retrieved_titles": [m["title"] for m in retrieved],
        }


if __name__ == "__main__":
    data_path = Path(__file__).parent / "data" / "movies.csv"
    rag = MovieRAG(str(data_path))
    result = rag.query("Помоги вспомнить фильм, где главный герой застрял в одном и том же дне.")
    print(f"Ответ: {result['answer']}")
    print(f"Найдены: {result['retrieved_titles']}")
