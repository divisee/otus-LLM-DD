#!/usr/bin/env python3
"""Retrieval evaluation for chunked collection.

Compares: movies (no chunking) vs movies_chunked (with chunking).
For chunked collection, deduplicates by movie title (returns best chunk per movie).
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from qdrant_client import QdrantClient

HW6_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW6_ROOT))

from config_utils import load_config

logger = logging.getLogger(__name__)
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def norm(s: str) -> str:
    return " ".join(WORD_RE.findall((s or "").lower().replace("ё", "е")))


def find_rank(expected: str, docs: list[dict]) -> int | None:
    exp = norm(expected)
    for i, d in enumerate(docs, 1):
        if norm(d.get("title", "")) == exp:
            return i
    return None


def ndcg(expected: str, docs: list[dict], k: int) -> float:
    rels = [1 if norm(expected) == norm(d.get("title", "")) else 0 for d in docs[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(rels), k)))
    return dcg / ideal if ideal else 0.0


@dataclass
class Metrics:
    hit: dict = field(default_factory=dict)
    mrr: dict = field(default_factory=dict)
    ndcg: dict = field(default_factory=dict)
    mean_rank: float = 0.0
    not_found: float = 0.0
    coverage: float = 0.0
    latency_ms: float = 0.0


class VectorRetriever:
    """Vector retriever with configurable collection."""

    def __init__(self, config_path: Path, collection_name: str):
        config = load_config(config_path)
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=config["qdrant"]["url"],
            api_key=config["qdrant"].get("api_key"),
        )
        ollama_cfg = config["ollama"]
        self.embed_url = f"{ollama_cfg['base_url'].rstrip('/')}/api/embed"
        self.embed_model = ollama_cfg["embedding_model"]
        self.vector_name = "dense"

    def embed(self, text: str) -> list[float]:
        resp = requests.post(self.embed_url, json={"model": self.embed_model, "input": text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [data.get("embedding")])[0]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        vec = self.embed(query)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using="dense",
            limit=top_k * 3,  # Get more for deduplication
            score_threshold=0.3,
        )
        docs = []
        for p in hits.points:
            payload = p.payload or {}
            docs.append({
                "id": p.id,
                "score": p.score,
                "title": payload.get("title", ""),
                "text": payload.get("chunk_text") or payload.get("text_ru", ""),
                **payload,
            })
        return docs

    def retrieve_dedupe(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve and deduplicate by title (keep best score per movie)."""
        all_docs = self.retrieve(query, top_k * 3)
        seen_titles = set()
        deduped = []
        for d in all_docs:
            title_norm = norm(d.get("title", ""))
            if title_norm and title_norm not in seen_titles:
                seen_titles.add(title_norm)
                deduped.append(d)
            if len(deduped) >= top_k:
                break
        return deduped


def evaluate(questions: list[dict], results: list[list[dict]], ks: list[int], corpus_size: int) -> Metrics:
    hit = {k: 0 for k in ks}
    mrr = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    ranks, not_found = [], 0
    unique_ids = set()

    for q, docs in zip(questions, results):
        r = find_rank(q["title"], docs)
        if r:
            ranks.append(r)
        else:
            not_found += 1
        for k in ks:
            if r and r <= k:
                hit[k] += 1
                mrr[k] += 1 / r
            ndcg_sum[k] += ndcg(q["title"], docs, k)
        for d in docs:
            unique_ids.add(d.get("id"))

    n = len(questions)
    return Metrics(
        hit={k: hit[k] / n for k in ks},
        mrr={k: mrr[k] / n for k in ks},
        ndcg={k: ndcg_sum[k] / n for k in ks},
        mean_rank=sum(ranks) / len(ranks) if ranks else 0,
        not_found=not_found / n,
        coverage=len(unique_ids) / corpus_size if corpus_size else 0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HW6_ROOT / "config.yaml")
    p.add_argument("--questions", type=Path, default=HW6_ROOT / "rag_analysis/data/rag_questions.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=HW6_ROOT / "rag_analysis/out")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ks = [1, 3, 5, 10]

    # Load questions
    df = pd.read_csv(args.questions)
    if args.limit:
        df = df.head(args.limit)
    questions = df.to_dict("records")
    logger.info(f"Loaded {len(questions)} questions")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 1: movies (no chunking) - vector only
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Running vector_no_chunk (collection: movies)...")
    ret1 = VectorRetriever(args.config, "movies")
    corpus1 = ret1.client.get_collection("movies").points_count

    t0 = time.time()
    docs1 = [ret1.retrieve(q["question"], args.top_k) for q in questions]
    lat1 = (time.time() - t0) * 1000 / len(questions)

    m1 = evaluate(questions, docs1, ks, corpus1)
    m1.latency_ms = lat1
    results["vector_no_chunk"] = m1

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 2: movies_chunked (with chunking) - deduplicated
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Running vector_chunked (collection: movies_chunked)...")
    ret2 = VectorRetriever(args.config, "movies_chunked")
    corpus2 = ret2.client.get_collection("movies_chunked").points_count

    t0 = time.time()
    docs2 = [ret2.retrieve_dedupe(q["question"], args.top_k) for q in questions]
    lat2 = (time.time() - t0) * 1000 / len(questions)

    m2 = evaluate(questions, docs2, ks, corpus2)
    m2.latency_ms = lat2
    results["vector_chunked"] = m2

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────
    for name, m in results.items():
        out_dir = args.out / name / f"{name}-{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "experiment": name,
            "run": f"{name}-{ts}",
            "metrics": {
                "hit": m.hit, "mrr": m.mrr, "ndcg": m.ndcg,
                "mean_rank": m.mean_rank, "not_found": m.not_found,
                "coverage": m.coverage, "latency_ms": m.latency_ms,
            }
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        logger.info(f"{name}: Hit@10={m.hit[10]:.3f} MRR@10={m.mrr[10]:.3f} latency={m.latency_ms:.1f}ms")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON: No Chunking vs With Chunking (chunk=500, overlap=100)")
    print("=" * 70)
    print(f"{'Metric':<20} {'No Chunk':<15} {'Chunked':<15} {'Diff':<15}")
    print("-" * 70)

    m1, m2 = results["vector_no_chunk"], results["vector_chunked"]
    for metric, name in [
        (10, "Hit@10"),
        (10, "MRR@10"),
        (10, "nDCG@10"),
    ]:
        if name == "Hit@10":
            v1, v2 = m1.hit[metric], m2.hit[metric]
        elif name == "MRR@10":
            v1, v2 = m1.mrr[metric], m2.mrr[metric]
        else:
            v1, v2 = m1.ndcg[metric], m2.ndcg[metric]
        diff = v2 - v1
        sign = "+" if diff >= 0 else ""
        print(f"{name:<20} {v1:<15.3f} {v2:<15.3f} {sign}{diff:<14.3f}")

    print(f"{'Not Found':<20} {m1.not_found:<15.1%} {m2.not_found:<15.1%} {m2.not_found - m1.not_found:+.1%}")
    print(f"{'Latency (ms)':<20} {m1.latency_ms:<15.1f} {m2.latency_ms:<15.1f} {m2.latency_ms - m1.latency_ms:+.1f}")
    print("=" * 70)

    logger.info(f"Done! Results in {args.out}")


if __name__ == "__main__":
    main()

