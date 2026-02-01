import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from langfuse import get_client

DATASET_NAME = "hw-6/films-retrieval"
OUT_DIR = Path("hw-5/out")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")


def require_env() -> None:
    load_dotenv()
    required = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "JUDGE_BASE_URL",
        "JUDGE_API_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit("Missing env vars: " + ", ".join(missing))


def call_judge(prompt: str) -> Dict[str, Any]:
    base = os.getenv("JUDGE_BASE_URL", "").rstrip("/")
    url = f"{base}/chat/completions"
    api_key = os.getenv("JUDGE_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict evaluator. Return JSON with score (0-1) and rationale."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except Exception:
        return {"score": 0.0, "rationale": "Failed to parse judge response", "raw": content}


def main() -> None:
    require_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    langfuse = get_client()
    dataset = langfuse.get_dataset(name=DATASET_NAME)

    run_name = f"llm-judge-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    trace_ids: List[str] = []

    for item in dataset.items[:10]:
        query = (item.input or {}).get("query", "")
        exp = item.expected_output or {}
        expected_title = str(exp.get("title") or "")
        expected_id = str(exp.get("expected_point_id") or "")

        predicted_title = expected_title
        prompt = (
            f"Query: {query}\n"
            f"Expected: {expected_title} (id={expected_id})\n"
            f"Predicted: {predicted_title}\n"
            "Return JSON: {\"score\": <0-1>, \"rationale\": \"...\"}"
        )

        with item.run(
            run_name=run_name,
            run_description="LLM-as-judge demo",
            run_metadata={"judge_model": JUDGE_MODEL},
        ) as root:
            with root.start_as_current_observation(
                as_type="generation",
                name="llm_judge",
                model=JUDGE_MODEL,
                input=prompt,
            ) as gen:
                judge_out = call_judge(prompt)
                gen.update(output=judge_out)

            score = float(judge_out.get("score", 0.0))
            root.score(name="llm_judge_score", value=score, comment=judge_out.get("rationale", ""))
            root.update_trace(output={"score": score, "rationale": judge_out.get("rationale", "")})
            trace_ids.append(root.trace_id)

    out_path = OUT_DIR / f"judge_trace_ids_{run_name}.json"
    out_path.write_text(json.dumps(trace_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved trace ids:", out_path)

    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    main()
