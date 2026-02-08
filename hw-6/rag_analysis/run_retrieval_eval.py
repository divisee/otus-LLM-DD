#!/usr/bin/env python3
"""Retrieval evaluation: vector, BM25, RRF, DBSF."""

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
from typing import Any

import pandas as pd

HW6_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW6_ROOT))

from config_utils import load_config
from tools_rag import RagRetriever

logger = logging.getLogger(__name__)
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────────────────────────────────────

def rrf(a: list[dict], b: list[dict], top_k: int, c=60) -> list[dict]:
    """Reciprocal Rank Fusion."""
    ra = {str(d["id"]): i + 1 for i, d in enumerate(a)}
    rb = {str(d["id"]): i + 1 for i, d in enumerate(b)}
    by_id = {str(d["id"]): d for d in a + b}
    scores = {}
    for pid in set(ra) | set(rb):
        scores[pid] = 1 / (c + ra.get(pid, 1e9)) + 1 / (c + rb.get(pid, 1e9))
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [{**by_id[pid], "rrf_score": s} for pid, s in ranked[:top_k]]


def dbsf(a: list[dict], b: list[dict], top_k: int) -> list[dict]:
    """Distribution-Based Score Fusion (min-max norm + weighted sum)."""
    def get_scores(docs, field):
        return {str(d["id"]): d.get(field, 0) for d in docs if field in d}

    def minmax(scores):
        if not scores:
            return {}
        mn, mx = min(scores.values()), max(scores.values())
        rng = mx - mn if mx > mn else 1
        return {k: (v - mn) / rng for k, v in scores.items()}

    sa = minmax(get_scores(a, "score"))
    sb = minmax(get_scores(b, "bm25_score"))
    by_id = {str(d["id"]): d for d in a + b}
    scores = {pid: 0.5 * sa.get(pid, 0) + 0.5 * sb.get(pid, 0) for pid in set(sa) | set(sb)}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [{**by_id[pid], "dbsf_score": s} for pid, s in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    hit: dict = field(default_factory=dict)
    mrr: dict = field(default_factory=dict)
    ndcg: dict = field(default_factory=dict)
    mean_rank: float = 0.0
    not_found: float = 0.0
    coverage: float = 0.0
    latency_ms: float = 0.0


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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HW6_ROOT / "config.yaml")
    p.add_argument("--questions", type=Path, default=HW6_ROOT / "rag_analysis/data/rag_questions.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=HW6_ROOT / "rag_analysis/out")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Load questions
    df = pd.read_csv(args.questions)
    if args.limit:
        df = df.head(args.limit)
    questions = df.to_dict("records")
    logger.info(f"Loaded {len(questions)} questions")

    # Load corpus from Qdrant
    cfg = load_config(args.config)
    from qdrant_client import QdrantClient, models
    qdr = QdrantClient(url=cfg["qdrant"]["url"])
    corpus = []
    offset = None
    while True:
        pts, offset = qdr.scroll(cfg["qdrant"]["collection_name"], limit=500, offset=offset, with_payload=True)
        corpus.extend({"id": p.id, **p.payload} for p in pts)
        if offset is None:
            break
    logger.info(f"Corpus: {len(corpus)} docs")

    # Retrievers
    vec = RagRetriever(args.config)
    ks = [1, 3, 5, 10]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    experiments = {}

    # Vector
    logger.info("Running vector_rag...")
    t0 = time.time()
    vec_res = [vec.retrieve(q["question"], args.top_k) for q in questions]
    experiments["vector_rag"] = (vec_res, (time.time() - t0) * 1000 / len(questions))

    # BM25 (server-side)
    logger.info("Running bm25...")
    t0 = time.time()
    bm25_res = []
    for q in questions:
        hits = qdr.query_points(
            collection_name=cfg["qdrant"]["collection_name"],
            query=models.Document(text=q["question"], model="Qdrant/bm25"),
            using="bm25",
            limit=args.top_k
        )
        docs = []
        for point in hits.points:
            payload = point.payload or {}
            docs.append({
                "id": point.id,
                "bm25_score": point.score,
                **payload
            })
        bm25_res.append(docs)
    experiments["bm25"] = (bm25_res, (time.time() - t0) * 1000 / len(questions))

    # RRF
    logger.info("Running hybrid_rrf...")
    rrf_res = [rrf(v, b, args.top_k) for v, b in zip(vec_res, bm25_res)]
    experiments["hybrid_rrf"] = (rrf_res, experiments["vector_rag"][1] + experiments["bm25"][1])

    # DBSF
    logger.info("Running hybrid_dbsf...")
    dbsf_res = [dbsf(v, b, args.top_k) for v, b in zip(vec_res, bm25_res)]
    experiments["hybrid_dbsf"] = (dbsf_res, experiments["vector_rag"][1] + experiments["bm25"][1])

    # Evaluate & save
    for name, (results, latency) in experiments.items():
        m = evaluate(questions, results, ks, len(corpus))
        m.latency_ms = latency

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

        per_query = []
        for q, docs in zip(questions, results):
            per_query.append({
                "title": q["title"], "question": q["question"],
                "rank": find_rank(q["title"], docs),
                "top_titles": [d.get("title") for d in docs[:5]],
            })
        (out_dir / "per_query.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_query))

        logger.info(f"{name}: Hit@10={m.hit[10]:.3f} MRR@10={m.mrr[10]:.3f} latency={m.latency_ms:.1f}ms")

    logger.info(f"Done! Results in {args.out}")


if __name__ == "__main__":
    main()

