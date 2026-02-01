import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

DATASET_NAME = "hw-6/films-retrieval"
OUT_DIR = Path("hw-5/out")
TOP_K = int(os.getenv("TOP_K", "5"))


def require_env() -> None:
    load_dotenv()
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit("Missing env vars: " + ", ".join(missing))


def has_qdrant() -> bool:
    return bool(os.getenv("QDRANT_URL") and os.getenv("QDRANT_COLLECTION"))


def has_embeddings() -> bool:
    return bool(os.getenv("EMBEDDING_BASE_URL") and os.getenv("EMBEDDING_MODEL"))


_word_re = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def norm_tokens(s: str) -> set:
    return set(_word_re.findall((s or "").lower()))


def embed_openai_compatible(text: str) -> List[float]:
    base = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
    url = f"{base}/embeddings"
    model = os.getenv("EMBEDDING_MODEL")
    key = os.getenv("EMBEDDING_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    r = requests.post(
        url,
        headers=headers,
        json={"model": model, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data["data"][0]["embedding"]


def qdrant_search(vector: List[float], limit: int) -> List[Dict[str, Any]]:
    base = os.getenv("QDRANT_URL", "").rstrip("/")
    collection = os.getenv("QDRANT_COLLECTION")
    key = os.getenv("QDRANT_API_KEY", "")

    url = f"{base}/collections/{collection}/points/search"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["api-key"] = key

    payload = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    res = r.json().get("result", [])
    return res


def local_fallback_search(query: str, dataset_items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    q = norm_tokens(query)
    scored = []
    for it in dataset_items:
        pid = it.get("expected_point_id")
        title = it.get("title") or ""
        year = it.get("year") or ""
        doc = f"{title} {year}"
        d = norm_tokens(doc)
        score = len(q & d) / max(len(q | d), 1)
        scored.append({"id": pid, "score": score, "payload": {"title": title, "year": year}})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def hit_at_k(expected_id: str, retrieved: List[Dict[str, Any]]) -> int:
    ids = [str(x.get("id")) for x in retrieved]
    return 1 if str(expected_id) in ids else 0


def mrr(expected_id: str, retrieved: List[Dict[str, Any]]) -> float:
    for i, x in enumerate(retrieved, start=1):
        if str(x.get("id")) == str(expected_id):
            return 1.0 / i
    return 0.0


def main() -> None:
    require_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    langfuse = get_client()
    run_name = f"hw6-films-retrieval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    trace_ids: List[str] = []

    try:
        dataset = langfuse.get_dataset(name=DATASET_NAME)

        fallback_docs = []
        for item in dataset.items:
            exp = item.expected_output or {}
            fallback_docs.append(
                {
                    "expected_point_id": exp.get("expected_point_id"),
                    "title": exp.get("title"),
                    "year": exp.get("year"),
                }
            )

        with propagate_attributes(
            session_id=run_name,
            trace_name="hw6-films-retrieval",
            metadata={"run": run_name, "top_k": TOP_K},
        ):
            for item in dataset.items:
                query = (item.input or {}).get("query", "")
                exp = item.expected_output or {}
                expected_id = exp.get("expected_point_id")

                with item.run(
                    run_name=run_name,
                    run_description="Films retrieval experiment (Qdrant or fallback)",
                    run_metadata={"variant": "qdrant+embeddings" if has_qdrant() else "local-fallback"},
                ) as root:
                    root.update_trace(
                        input={"query": query},
                        output={"expected_point_id": expected_id},
                    )

                    with root.start_as_current_observation(
                        as_type="retriever",
                        name="films_retriever",
                        input={"query": query, "top_k": TOP_K},
                    ) as retr_span:
                        if has_qdrant() and has_embeddings():
                            with retr_span.start_as_current_observation(
                                as_type="embedding",
                                name="embed_query",
                                model=os.getenv("EMBEDDING_MODEL"),
                                input={"text": query},
                            ) as emb_span:
                                vec = embed_openai_compatible(query)
                                emb_span.update(output={"dim": len(vec)})

                            hits = qdrant_search(vec, TOP_K)
                        else:
                            hits = local_fallback_search(query, fallback_docs, TOP_K)

                        retrieved_ids = [str(h.get("id")) for h in hits]
                        retr_span.update(output={"retrieved_ids": retrieved_ids, "top1": hits[0] if hits else None})

                    h = hit_at_k(expected_id, hits)
                    r = mrr(expected_id, hits)

                    top1_title = ""
                    if hits:
                        payload = hits[0].get("payload") or {}
                        top1_title = str(payload.get("title") or payload.get("name") or "")

                    expected_title = str(exp.get("title") or "")
                    title_match = 1 if expected_title and top1_title and expected_title.strip().lower() == top1_title.strip().lower() else 0

                    with root.start_as_current_observation(
                        as_type="evaluator",
                        name="retrieval_metrics",
                        input={"expected_id": expected_id},
                    ) as ev:
                        ev.update(output={"hit_at_k": h, "mrr": r, "title_match": title_match})

                    root.score(name=f"hit@{TOP_K}", value=float(h))
                    root.score(name="mrr", value=float(r))
                    root.score(name="title_match", value=float(title_match))

                    root.update_trace(
                        output={
                            "expected_point_id": expected_id,
                            "retrieved_top1_id": (str(hits[0].get("id")) if hits else None),
                            "retrieved_top1_title": top1_title,
                        }
                    )

                    trace_ids.append(root.trace_id)

        out_path = OUT_DIR / f"trace_ids_{run_name}.json"
        out_path.write_text(json.dumps(trace_ids, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved trace ids:", out_path)

    finally:
        langfuse.flush()
        langfuse.shutdown()


if __name__ == "__main__":
    main()
