#!/usr/bin/env python3
"""Offline retrieval evaluation for movie search.

Reads `rag_analysis/data/rag_questions.csv` with columns:
- title: expected movie title
- question: query text

Runs 3 retrieval variants:
1) vector_rag: Qdrant vector search (existing RagRetriever)
2) bm25_text_ru: BM25 over `text_ru` payload (downloaded from Qdrant via scroll)
3) hybrid_rrf: Reciprocal Rank Fusion of (1) and (2)

Saves per-query results + aggregated metrics to `rag_analysis/out/<experiment>/`.
Also logs runs to Langfuse dataset for later comparisons.

NOTE: This script evaluates retrieval only (ranking quality), not the final LLM answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Allow importing from hw-6 root.
# This file lives in hw-6/rag_analysis/, so parents[1] is hw-6.
HW6_ROOT = Path(__file__).resolve().parents[1]
if str(HW6_ROOT) not in sys.path:
    sys.path.insert(0, str(HW6_ROOT))

# PyCharm/IDE may not resolve these imports due to dynamic sys.path.
# At runtime they exist in hw-6/.
try:
    from config_utils import load_config  # type: ignore
    from tools_rag import RagRetriever  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(f"Failed to import hw-6 modules via sys.path={HW6_ROOT}: {e}")


logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def normalize_title(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    # keep only word chars
    tokens = _WORD_RE.findall(s)
    return " ".join(tokens)


def title_soft_match(expected: str, got: str) -> bool:
    e = normalize_title(expected)
    g = normalize_title(got)
    if not e or not g:
        return False
    if e == g:
        return True
    e_set = set(e.split())
    g_set = set(g.split())
    jacc = len(e_set & g_set) / max(len(e_set | g_set), 1)
    if jacc >= 0.8:
        return True
    return e in g or g in e


def rank_of_expected(expected_title: str, retrieved: Sequence[Dict[str, Any]]) -> Optional[int]:
    exp_norm = normalize_title(expected_title)
    if not exp_norm:
        return None
    for i, doc in enumerate(retrieved, start=1):
        got = str(doc.get("title") or "")
        if normalize_title(got) == exp_norm:
            return i
    return None


def dcg_at_k(rels: Sequence[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel:
            s += 1.0 / math.log2(i + 1)
    return s


def ndcg_at_k(expected_title: str, retrieved: Sequence[Dict[str, Any]], k: int) -> float:
    rels = [1 if title_soft_match(expected_title, str(d.get("title") or "")) else 0 for d in retrieved]
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return (dcg / ideal) if ideal > 0 else 0.0


@dataclass
class Metrics:
    hit: Dict[int, float]
    mrr: Dict[int, float]
    ndcg: Dict[int, float]
    median_rank: Optional[float]
    mean_rank: Optional[float]
    not_found_share: float
    title_exact_at_1: float
    title_soft_at_1: float


class VectorRetriever:
    def __init__(self, config_path: Path, *, top_k: int, langfuse=None) -> None:
        self.top_k = top_k
        self.rag = RagRetriever(config_path, langfuse=langfuse)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        return self.rag.retrieve(query, top_k=self.top_k)


class BM25Retriever:
    """Simple BM25 over full corpus of Qdrant payload `text_ru`.

    We download documents via Qdrant scroll once per run (fast for ~477 docs).
    """

    def __init__(self, config_path: Path, *, top_k: int, k1: float = 1.5, b: float = 0.75) -> None:
        self.top_k = top_k
        self.k1 = k1
        self.b = b

        cfg = load_config(config_path)
        qdr = cfg["qdrant"]
        from qdrant_client import QdrantClient  # local import to keep deps explicit

        self.collection_name = qdr["collection_name"]
        self.client = QdrantClient(url=qdr["url"], api_key=qdr.get("api_key") or None)

        self._docs: List[Dict[str, Any]] = []
        self._doc_tokens: List[List[str]] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0

        self._build_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _WORD_RE.findall((text or "").lower().replace("ё", "е"))

    def _scroll_all_payloads(self) -> List[Dict[str, Any]]:
        # qdrant-client scroll API returns (points, next_page_offset)
        points, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        all_points = list(points)
        while next_offset is not None and len(all_points) < 10000:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
        docs: List[Dict[str, Any]] = []
        for p in all_points:
            payload = p.payload or {}
            docs.append({"id": p.id, "title": payload.get("title"), "text_ru": payload.get("text_ru") or "", **payload})
        return docs

    def _build_index(self) -> None:
        t0 = time.time()
        self._docs = self._scroll_all_payloads()
        self._doc_tokens = []
        self._df = {}

        lengths = []
        for d in self._docs:
            tokens = self._tokenize("%s %s" % (d.get("title") or "", d.get("text_ru") or ""))
            self._doc_tokens.append(tokens)
            lengths.append(len(tokens))
            for tok in set(tokens):
                self._df[tok] = self._df.get(tok, 0) + 1

        n = len(self._docs)
        self._avgdl = sum(lengths) / max(n, 1)
        self._idf = {}
        for tok, df in self._df.items():
            # BM25+ style idf
            self._idf[tok] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        logger.info("BM25 index built: n_docs=%s avgdl=%.2f vocab=%s (%.2fs)", n, self._avgdl, len(self._idf), time.time() - t0)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores: List[Tuple[float, int]] = []
        for idx, tokens in enumerate(self._doc_tokens):
            if not tokens:
                continue
            dl = len(tokens)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            score = 0.0
            for qt in q_tokens:
                idf = self._idf.get(qt)
                if idf is None:
                    continue
                f = tf.get(qt, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (dl / max(self._avgdl, 1e-9)))
                score += idf * (f * (self.k1 + 1) / denom)
            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, idx in scores[: self.top_k]:
            d = dict(self._docs[idx])
            d["bm25_score"] = score
            out.append(d)
        return out


def rrf_fuse(
    a: Sequence[Dict[str, Any]],
    b: Sequence[Dict[str, Any]],
    *,
    key: str = "id",
    c: int = 60,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    ranks: Dict[str, Dict[str, int]] = {"a": {}, "b": {}}
    for i, d in enumerate(a, start=1):
        ranks["a"][str(d.get(key))] = i
    for i, d in enumerate(b, start=1):
        ranks["b"][str(d.get(key))] = i

    all_ids = set(ranks["a"]) | set(ranks["b"])
    scored: List[Tuple[float, str]] = []
    for pid in all_ids:
        s = 0.0
        if pid in ranks["a"]:
            s += 1.0 / (c + ranks["a"][pid])
        if pid in ranks["b"]:
            s += 1.0 / (c + ranks["b"][pid])
        scored.append((s, pid))
    scored.sort(key=lambda x: x[0], reverse=True)

    by_id: Dict[str, Dict[str, Any]] = {}
    for d in list(a) + list(b):
        pid = str(d.get(key))
        if pid not in by_id:
            by_id[pid] = dict(d)

    out: List[Dict[str, Any]] = []
    for s, pid in scored[:top_k]:
        d = dict(by_id.get(pid, {"id": pid}))
        d["rrf_score"] = s
        d["rrf_rank_a"] = ranks["a"].get(pid)
        d["rrf_rank_b"] = ranks["b"].get(pid)
        out.append(d)
    return out


def _extract_scores(docs: Sequence[Dict[str, Any]]) -> Tuple[List[Optional[float]], List[str]]:
    """Extract common score fields from docs.

    Returns:
        (scores, fields_used)

    Notes:
    - vector retrieval from Qdrant typically uses `score`
    - BM25 retriever uses `bm25_score`
    - RRF fusion uses `rrf_score`
    """

    candidate_fields = ["rrf_score", "bm25_score", "score"]
    scores: List[Optional[float]] = []
    used_fields: List[str] = []

    # determine preferred field (first present in any doc)
    preferred: Optional[str] = None
    for f in candidate_fields:
        if any((isinstance(d.get(f), (int, float)) for d in docs)):
            preferred = f
            break

    if preferred:
        used_fields.append(preferred)
        for d in docs:
            v = d.get(preferred)
            scores.append(float(v) if isinstance(v, (int, float)) else None)
        return scores, used_fields

    # fallback: no known field
    return [None for _ in docs], used_fields


def evaluate_rows(
    rows: List[Dict[str, Any]],
    retrieved: List[List[Dict[str, Any]]],
    ks: Sequence[int],
) -> Tuple[Metrics, List[Dict[str, Any]]]:
    assert len(rows) == len(retrieved)

    hit_counts = {k: 0 for k in ks}
    mrr_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}

    ranks: List[int] = []
    n_not_found = 0

    exact1 = 0
    soft1 = 0

    per_query: List[Dict[str, Any]] = []

    for row, docs in zip(rows, retrieved):
        expected_title = str(row["title"])
        q = str(row["question"])

        r = rank_of_expected(expected_title, docs)
        if r is None:
            n_not_found += 1
        else:
            ranks.append(r)

        top1_title = str(docs[0].get("title") or "") if docs else ""
        exact_match = 1 if normalize_title(expected_title) and normalize_title(expected_title) == normalize_title(top1_title) else 0
        soft_match = 1 if title_soft_match(expected_title, top1_title) else 0
        exact1 += exact_match
        soft1 += soft_match

        for k in ks:
            sub = docs[:k]
            hit = 1 if rank_of_expected(expected_title, sub) is not None else 0
            hit_counts[k] += hit

            rr = 0.0
            pos = rank_of_expected(expected_title, sub)
            if pos is not None:
                rr = 1.0 / pos
            mrr_sum[k] += rr

            ndcg_sum[k] += ndcg_at_k(expected_title, sub, k)

        score_list, score_fields = _extract_scores(docs)

        per_query.append(
            {
                "title": expected_title,
                "question": q,
                "rank": r if r is not None else None,
                "top1_title": top1_title,
                "title_exact@1": exact_match,
                "title_soft@1": soft_match,
                "top_titles": [str(d.get("title") or "") for d in docs],
                "top_ids": [str(d.get("id")) for d in docs],
                "top_scores": score_list,
                "top_score_fields": score_fields,
                "top_vector_scores": [float(d.get("score")) if isinstance(d.get("score"), (int, float)) else None for d in docs],
                "top_bm25_scores": [float(d.get("bm25_score")) if isinstance(d.get("bm25_score"), (int, float)) else None for d in docs],
                "top_rrf_scores": [float(d.get("rrf_score")) if isinstance(d.get("rrf_score"), (int, float)) else None for d in docs],
                "top_rrf_rank_a": [d.get("rrf_rank_a") for d in docs],
                "top_rrf_rank_b": [d.get("rrf_rank_b") for d in docs],
            }
        )

    n = len(rows)
    metrics = Metrics(
        hit={k: hit_counts[k] / max(n, 1) for k in ks},
        mrr={k: mrr_sum[k] / max(n, 1) for k in ks},
        ndcg={k: ndcg_sum[k] / max(n, 1) for k in ks},
        median_rank=(float(pd.Series(ranks).median()) if ranks else None),
        mean_rank=(float(pd.Series(ranks).mean()) if ranks else None),
        not_found_share=n_not_found / max(n, 1),
        title_exact_at_1=exact1 / max(n, 1),
        title_soft_at_1=soft1 / max(n, 1),
    )
    return metrics, per_query


def ensure_langfuse_env_from_config(config_path: Path) -> None:
    """If LANGFUSE_* env vars are absent, fill them from hw-6 config.yaml."""

    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    if all(os.getenv(k) for k in required):
        return

    cfg = load_config(config_path)
    lf = cfg.get("langfuse", {}) if isinstance(cfg, dict) else {}
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", str(lf.get("public_key") or ""))
    os.environ.setdefault("LANGFUSE_SECRET_KEY", str(lf.get("secret_key") or ""))
    os.environ.setdefault("LANGFUSE_BASE_URL", str(lf.get("base_url") or ""))


def _call_with_retries(
    fn,
    *,
    what: str,
    max_retries: int = 8,
    base_sleep_s: float = 0.5,
    max_sleep_s: float = 20.0,
):
    """Call Langfuse API with retries on rate limiting / transient errors.

    We can't rely on a specific exception class across langfuse SDK versions,
    so we match by HTTP status code in the error message.
    """

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:  # pragma: no cover
            msg = str(e)
            is_retryable = (
                " 429" in msg
                or "status_code: 429" in msg
                or "rate limit" in msg.lower()
                or "status_code: 5" in msg
                or " 5" in msg and "status_code" in msg
            )
            if not is_retryable or attempt >= max_retries:
                raise

            sleep_s = min(max_sleep_s, base_sleep_s * (2**attempt))
            # small deterministic jitter to avoid thundering herd
            sleep_s = sleep_s + (0.05 * (attempt + 1))
            logger.warning(
                "Langfuse transient error (%s). Retry %s/%s in %.2fs: %s",
                what,
                attempt + 1,
                max_retries,
                sleep_s,
                msg,
            )
            time.sleep(sleep_s)


def get_or_create_dataset(langfuse, *, name: str, items: List[Dict[str, Any]]):
    try:
        ds = langfuse.get_dataset(name=name)
        return ds
    except Exception:
        pass

    ds = _call_with_retries(lambda: langfuse.create_dataset(name=name), what=f"create_dataset:{name}")

    # Creating items is the noisiest part for rate limits; throttle a bit.
    throttle_s = float(os.getenv("LANGFUSE_THROTTLE_S", "0.05") or "0.05")

    for idx, it in enumerate(items, start=1):
        _call_with_retries(
            lambda it=it: langfuse.create_dataset_item(
                dataset_name=name,
                input={"query": it["question"]},
                expected_output={"title": it["title"], "difficulty": it.get("difficulty")},
            ),
            what=f"create_dataset_item:{name}:{idx}",
        )
        if throttle_s > 0:
            time.sleep(throttle_s)

    return _call_with_retries(lambda: langfuse.get_dataset(name=name), what=f"get_dataset:{name}")


def log_langfuse_run(
    *,
    config_path: Path,
    dataset_name: str,
    run_name: str,
    run_description: str,
    run_metadata: Dict[str, Any],
    rows: List[Dict[str, Any]],
    per_query: List[Dict[str, Any]],
    metrics: Metrics,
) -> List[str]:
    ensure_langfuse_env_from_config(config_path)

    from langfuse import get_client  # noqa

    langfuse = get_client()
    dataset = get_or_create_dataset(langfuse, name=dataset_name, items=rows)

    # Map by query text for quick join (queries are unique in this dataset)
    pq_by_query = {str(r["question"]): r for r in per_query}

    trace_ids: List[str] = []

    per_item_throttle_s = float(os.getenv("LANGFUSE_THROTTLE_S", "0.05") or "0.05")

    for idx, item in enumerate(dataset.items, start=1):
        query = (item.input or {}).get("query", "")
        pq = pq_by_query.get(str(query))
        if pq is None:
            continue

        def _log_one():
            with item.run(
                run_name=run_name,
                run_description=run_description,
                run_metadata=run_metadata,
            ) as root:
                root.update_trace(
                    input={"query": query},
                    output={
                        "expected_title": (item.expected_output or {}).get("title"),
                        "rank": pq.get("rank"),
                        "top1_title": pq.get("top1_title"),
                    },
                )

                # Per-item scores
                root.score(name="rank", value=float(pq["rank"]) if pq.get("rank") else 0.0, data_type="NUMERIC")
                root.score(name="found@10", value=float(1.0 if pq.get("rank") and pq["rank"] <= 10 else 0.0))
                root.score(name="title_exact@1", value=float(pq.get("title_exact@1") or 0.0))
                root.score(name="title_soft@1", value=float(pq.get("title_soft@1") or 0.0))

                trace_ids.append(root.trace_id)

        _call_with_retries(_log_one, what=f"item.run:{run_name}:{idx}")
        if per_item_throttle_s > 0:
            time.sleep(per_item_throttle_s)

    # Also attach aggregated metrics to a standalone trace (dataset run summary)
    def _log_summary():
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"retrieval_eval_summary::{run_name}",
            input={"dataset": dataset_name, "run": run_name},
        ) as summary:
            summary.update(
                output={
                    "hit": metrics.hit,
                    "mrr": metrics.mrr,
                    "ndcg": metrics.ndcg,
                    "median_rank": metrics.median_rank,
                    "mean_rank": metrics.mean_rank,
                    "not_found_share": metrics.not_found_share,
                    "title_exact@1": metrics.title_exact_at_1,
                    "title_soft@1": metrics.title_soft_at_1,
                },
                metadata=run_metadata,
            )

    _call_with_retries(_log_summary, what=f"summary:{run_name}")

    _call_with_retries(langfuse.flush, what=f"flush:{run_name}")
    _call_with_retries(langfuse.shutdown, what=f"shutdown:{run_name}")
    return trace_ids


def save_outputs(
    out_dir: Path,
    *,
    exp_name: str,
    run_name: str,
    rows: List[Dict[str, Any]],
    per_query: List[Dict[str, Any]],
    metrics: Metrics,
    trace_ids: Optional[List[str]],
    metadata: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "per_query.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_query) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(per_query).to_csv(out_dir / "per_query.csv", index=False)

    summary = {
        "experiment": exp_name,
        "run_name": run_name,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "hit": metrics.hit,
            "mrr": metrics.mrr,
            "ndcg": metrics.ndcg,
            "median_rank": metrics.median_rank,
            "mean_rank": metrics.mean_rank,
            "not_found_share": metrics.not_found_share,
            "title_exact@1": metrics.title_exact_at_1,
            "title_soft@1": metrics.title_soft_at_1,
        },
        "metadata": metadata,
        "trace_ids": trace_ids or [],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def read_questions(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    need = {"title", "question"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {sorted(need)} in {path}, got {list(df.columns)}")

    if limit is not None:
        df = df.head(limit)

    raw_rows = df.to_dict(orient="records")
    out: List[Dict[str, Any]] = []
    for r in raw_rows:
        out.append(
            {
                "title": str(r.get("title") or ""),
                "question": str(r.get("question") or ""),
                "difficulty": (
                    int(r.get("difficulty") or 0)
                    if str(r.get("difficulty") or "").isdigit()
                    else r.get("difficulty")
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for movie RAG")
    parser.add_argument(
        "--config",
        type=Path,
        default=HW6_ROOT / "config.yaml",
        help="Path to hw-6 config.yaml",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=HW6_ROOT / "rag_analysis" / "data" / "rag_questions.csv",
        help="Path to rag_questions.csv",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--top-k", type=int, default=10, help="Retriever top_k")
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated ks to compute metrics for",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HW6_ROOT / "rag_analysis" / "out",
        help="Output base dir",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Disable Langfuse logging",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="hw-6/rag-questions-movies",
        help="Langfuse dataset name",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    rows = read_questions(args.questions, limit=args.limit)

    run_name_base = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # 1) vector
    logger.info("Running vector_rag...")
    vec_ret = VectorRetriever(args.config, top_k=args.top_k, langfuse=None)
    vec_docs = [vec_ret.retrieve(r["question"]) for r in rows]
    vec_metrics, vec_per_query = evaluate_rows(rows, vec_docs, ks)

    # 2) bm25
    logger.info("Running bm25_text_ru...")
    bm25_ret = BM25Retriever(args.config, top_k=args.top_k)
    bm25_docs = [bm25_ret.retrieve(r["question"]) for r in rows]
    bm25_metrics, bm25_per_query = evaluate_rows(rows, bm25_docs, ks)

    # 3) hybrid rrf
    logger.info("Running hybrid_rrf...")
    hybrid_docs: List[List[Dict[str, Any]]] = []
    for a, b in zip(vec_docs, bm25_docs):
        hybrid_docs.append(rrf_fuse(a, b, top_k=args.top_k))
    hybrid_metrics, hybrid_per_query = evaluate_rows(rows, hybrid_docs, ks)

    # Save outputs + (optional) Langfuse
    experiments = [
        (
            "vector_rag",
            vec_metrics,
            vec_per_query,
            {"top_k": args.top_k, "variant": "vector", "qdrant_collection": load_config(args.config)["qdrant"]["collection_name"]},
        ),
        (
            "bm25_text_ru",
            bm25_metrics,
            bm25_per_query,
            {"top_k": args.top_k, "variant": "bm25", "bm25_k1": bm25_ret.k1, "bm25_b": bm25_ret.b},
        ),
        (
            "hybrid_rrf",
            hybrid_metrics,
            hybrid_per_query,
            {"top_k": args.top_k, "variant": "rrf", "rrf_c": 60},
        ),
    ]

    for exp_name, metrics, per_query, metadata in experiments:
        run_name = f"{exp_name}-{run_name_base}"
        out_dir = args.out / exp_name / run_name

        trace_ids = None
        if not args.no_langfuse:
            try:
                logger.info("Logging to Langfuse: exp=%s run=%s", exp_name, run_name)
                trace_ids = log_langfuse_run(
                    config_path=args.config,
                    dataset_name=args.dataset_name,
                    run_name=run_name,
                    run_description=f"Retrieval eval: {exp_name}",
                    run_metadata=metadata,
                    rows=rows,
                    per_query=per_query,
                    metrics=metrics,
                )
            except Exception as e:
                logger.warning("Langfuse logging failed for %s: %s", exp_name, e)
                trace_ids = None

        save_outputs(
            out_dir,
            exp_name=exp_name,
            run_name=run_name,
            rows=rows,
            per_query=per_query,
            metrics=metrics,
            trace_ids=trace_ids,
            metadata=metadata,
        )

        logger.info(
            "Done %s | hit@%s=%s | mrr@%s=%s | not_found=%.2f",
            exp_name,
            max(ks),
            round(metrics.hit[max(ks)], 3) if max(ks) in metrics.hit else None,
            max(ks),
            round(metrics.mrr[max(ks)], 3) if max(ks) in metrics.mrr else None,
            metrics.not_found_share,
        )


if __name__ == "__main__":
    main()
