"""Общие фикстуры для pytest: загрузка RAG, goldens, прогон пайплайна."""

import json
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Загружаем переменные окружения из .env (OPENAI_API_KEY и др.)
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_app import MovieRAG  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
GOLDENS_PATH = Path(__file__).parent / "goldens.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


@pytest.fixture(scope="session")
def rag():
    """Инициализация RAG-системы (один раз за сессию)."""
    return MovieRAG(str(DATA_DIR / "movies.csv"))


@pytest.fixture(scope="session")
def goldens():
    """Загрузка золотых примеров."""
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def rag_results(rag, goldens):
    """Прогон RAG-пайплайна по всем goldens (один раз за сессию).

    Результат каждого примера содержит: question, answer, contexts,
    retrieved_titles, expected_title, ground_truth.
    """
    results = []
    for g in goldens:
        result = rag.query(g["question"])
        result["expected_title"] = g["expected_title"]
        result["ground_truth"] = g["ground_truth"]
        results.append(result)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "rag_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
