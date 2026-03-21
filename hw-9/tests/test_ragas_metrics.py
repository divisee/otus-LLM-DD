"""
Тестирование RAG-приложения с помощью RAGAS (collections API).

Метрики (основные — с жёсткими порогами):
  - Faithfulness (проверка на галлюцинации)
  - Context Recall (полнота извлечённого контекста)

Метрики (дополнительные — с мягкими порогами):
  - Context Precision (релевантность извлечённых документов)
  - Answer Relevancy (релевантность ответа вопросу)
  - Factual Correctness (фактическая правильность ответа vs ground truth)
  - Semantic Similarity (семантическое сходство ответа и ground truth)

Пороги задаются в THRESHOLDS. Результаты сохраняются в results/.
Подробные логи — в logs/.
"""

import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import pytest
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    AnswerRelevancy,
    FactualCorrectness,
    SemanticSimilarity,
)

THRESHOLDS = {
    "faithfulness": 0.7,
    "context_recall": 0.5,
}

SOFT_THRESHOLDS = {
    "context_precision": 0.4,
    "answer_relevancy": 0.4,
    "factual_correctness": 0.3,
    "semantic_similarity": 0.5,
}

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"


class _Logger:
    """Пишет одновременно в stdout и в лог-файл."""

    def __init__(self, log_path: Path):
        LOGS_DIR.mkdir(exist_ok=True)
        self._f = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str, end: str = "\n", flush: bool = True):
        print(msg, end=end, flush=flush)
        self._f.write(msg + end)
        if flush:
            self._f.flush()

    def close(self):
        self._f.close()


def _build_components():
    """Создание LLM и embeddings через новый RAGAS API."""
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory(provider="openai", model="text-embedding-3-small")
    return llm, embeddings


def _build_metrics(llm, embeddings):
    """Создание всех метрик с переданными LLM/embeddings."""
    metrics = {}
    metrics["faithfulness"] = Faithfulness(llm=llm)
    metrics["context_recall"] = ContextRecall(llm=llm)

    try:
        metrics["context_precision"] = ContextPrecision(llm=llm)
    except Exception:
        pass

    try:
        metrics["answer_relevancy"] = AnswerRelevancy(llm=llm, embeddings=embeddings)
    except Exception:
        pass

    try:
        metrics["factual_correctness"] = FactualCorrectness(llm=llm)
    except Exception:
        pass

    try:
        metrics["semantic_similarity"] = SemanticSimilarity(embeddings=embeddings)
    except Exception:
        pass

    return metrics


METRIC_FIELDS = {
    "faithfulness": lambda s: dict(
        user_input=s["question"], response=s["answer"], retrieved_contexts=s["contexts"]
    ),
    "context_recall": lambda s: dict(
        user_input=s["question"], retrieved_contexts=s["contexts"], reference=s["ground_truth"]
    ),
    "context_precision": lambda s: dict(
        user_input=s["question"], reference=s["ground_truth"], retrieved_contexts=s["contexts"]
    ),
    "answer_relevancy": lambda s: dict(
        user_input=s["question"], response=s["answer"]
    ),
    "factual_correctness": lambda s: dict(
        response=s["answer"], reference=s["ground_truth"]
    ),
    "semantic_similarity": lambda s: dict(
        reference=s["ground_truth"], response=s["answer"]
    ),
}


def _truncate(text: str, max_len: int = 120) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


async def _score_all(metrics, samples, log: _Logger):
    """Оценить все примеры по всем метрикам, вернуть per-sample и средние."""
    total_steps = len(samples) * len(metrics)
    done_steps = 0
    t_start = time.time()

    per_sample = []
    for i, sample in enumerate(samples):
        log.log(f"\n{'='*80}")
        log.log(f"  SAMPLE {i+1}/{len(samples)}")
        log.log(f"  Question:     {sample['question']}")
        log.log(f"  Answer:       {_truncate(sample['answer'])}")
        log.log(f"  Ground Truth: {_truncate(sample['ground_truth'])}")
        log.log(f"  Expected:     {sample.get('expected_title', '—')}")
        log.log(f"{'='*80}")

        row = {"idx": i, "question": sample["question"]}
        for name, metric in metrics.items():
            step_t0 = time.time()
            done_steps += 1
            log.log(
                f"  [{done_steps}/{total_steps}] metric={name}  ...",
                end="",
            )
            try:
                kwargs = METRIC_FIELDS[name](sample)
                result = await metric.ascore(**kwargs)
                val = float(result.value) if result.value is not None else None
                row[name] = val
                elapsed = time.time() - step_t0
                log.log(f"  score={val}  ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.time() - step_t0
                log.log(f"  ERROR ({elapsed:.1f}s): {exc}")
                row[name] = None

        per_sample.append(row)
        elapsed_total = time.time() - t_start
        remaining = (elapsed_total / (i + 1)) * (len(samples) - i - 1)
        log.log(
            f"  >>> sample {i+1}/{len(samples)} done  "
            f"elapsed={elapsed_total:.0f}s  est. remaining={remaining:.0f}s"
        )

    averages = {}
    for name in metrics:
        vals = [r[name] for r in per_sample if r.get(name) is not None]
        if vals:
            averages[name] = statistics.mean(vals)

    log.log(f"\n  Total scoring time: {time.time() - t_start:.0f}s")
    return averages, per_sample


@pytest.fixture(scope="module")
def ragas_scores(rag_results):
    """Запуск RAGAS метрик и возврат средних значений."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = _Logger(LOGS_DIR / f"ragas_{ts}.log")

    log.log(f"RAGAS evaluation started at {ts}")
    log.log(f"Samples: {len(rag_results)}")

    llm, embeddings = _build_components()
    metrics = _build_metrics(llm, embeddings)

    loaded_names = list(metrics.keys())
    log.log(f"RAGAS metrics loaded: {loaded_names}")

    averages, per_sample = asyncio.run(_score_all(metrics, rag_results, log))

    RESULTS_DIR.mkdir(exist_ok=True)
    report = {
        "thresholds_hard": THRESHOLDS,
        "thresholds_soft": SOFT_THRESHOLDS,
        "average_scores": averages,
        "per_sample": per_sample,
    }
    with open(RESULTS_DIR / "ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    log.log("\n=== RAGAS Results (averages) ===")
    for name, val in averages.items():
        kind = "HARD" if name in THRESHOLDS else "soft"
        thresh = THRESHOLDS.get(name, SOFT_THRESHOLDS.get(name, "—"))
        status = "PASS" if val >= (THRESHOLDS.get(name) or SOFT_THRESHOLDS.get(name, 0)) else "FAIL"
        log.log(f"  {name:25s}: {val:.4f}  (порог {kind}: {thresh})  [{status}]")

    log.log(f"\nLog saved to {log._f.name}")
    log.close()
    return averages


# ── Основные тесты (жёсткие пороги) ──────────────────────────────


def test_faithfulness(ragas_scores):
    """Faithfulness >= 0.7 — ответы должны быть основаны на контексте."""
    score = ragas_scores.get("faithfulness", 0)
    threshold = THRESHOLDS["faithfulness"]
    print(f"\nFaithfulness: {score:.4f} (порог: {threshold})")
    assert score >= threshold, (
        f"Faithfulness {score:.4f} ниже порога {threshold}"
    )


def test_context_recall(ragas_scores):
    """Context Recall >= 0.5 — нужный контекст должен быть извлечён."""
    score = ragas_scores.get("context_recall", 0)
    threshold = THRESHOLDS["context_recall"]
    print(f"\nContext Recall: {score:.4f} (порог: {threshold})")
    assert score >= threshold, (
        f"Context Recall {score:.4f} ниже порога {threshold}"
    )


# ── Дополнительные тесты (мягкие пороги) ─────────────────────────


def test_context_precision(ragas_scores):
    """Context Precision — доля релевантных документов среди извлечённых."""
    score = ragas_scores.get("context_precision")
    if score is None:
        pytest.skip("context_precision не вычислена (метрика недоступна)")
    threshold = SOFT_THRESHOLDS["context_precision"]
    print(f"\nContext Precision: {score:.4f} (мягкий порог: {threshold})")
    assert score >= threshold, (
        f"Context Precision {score:.4f} ниже мягкого порога {threshold}"
    )


def test_answer_relevancy(ragas_scores):
    """Answer Relevancy — насколько ответ релевантен вопросу."""
    score = ragas_scores.get("answer_relevancy")
    if score is None:
        pytest.skip("answer_relevancy не вычислена (метрика недоступна)")
    threshold = SOFT_THRESHOLDS["answer_relevancy"]
    print(f"\nAnswer Relevancy: {score:.4f} (мягкий порог: {threshold})")
    assert score >= threshold, (
        f"Answer Relevancy {score:.4f} ниже мягкого порога {threshold}"
    )


def test_factual_correctness(ragas_scores):
    """Factual Correctness — фактическая правильность ответа vs ground truth."""
    score = ragas_scores.get("factual_correctness")
    if score is None:
        pytest.skip("factual_correctness не вычислена (метрика недоступна)")
    threshold = SOFT_THRESHOLDS["factual_correctness"]
    print(f"\nFactual Correctness: {score:.4f} (мягкий порог: {threshold})")
    assert score >= threshold, (
        f"Factual Correctness {score:.4f} ниже мягкого порога {threshold}"
    )


def test_semantic_similarity(ragas_scores):
    """Semantic Similarity — семантическое сходство ответа и ground truth."""
    score = ragas_scores.get("semantic_similarity")
    if score is None:
        pytest.skip("semantic_similarity не вычислена (метрика недоступна)")
    threshold = SOFT_THRESHOLDS["semantic_similarity"]
    print(f"\nSemantic Similarity: {score:.4f} (мягкий порог: {threshold})")
    assert score >= threshold, (
        f"Semantic Similarity {score:.4f} ниже мягкого порога {threshold}"
    )
