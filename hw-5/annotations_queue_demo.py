import os
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from langfuse import get_client

OUT_DIR = Path("hw-5/out")
ANNOTATION_LIMIT = int(os.getenv("ANNOTATION_LIMIT", "3"))


def require_env() -> None:
    load_dotenv()
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit("Missing env vars: " + ", ".join(missing))


def main() -> None:
    require_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    langfuse = get_client()
    trace_ids = []

    queries = [
        "Найди фильм про мальчика с волшебными силами",
        "Посоветуй топ 5 комедий 2025 года",
        "Подбери фильмы для семейного просмотра",
    ]

    for q in queries[:ANNOTATION_LIMIT]:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="annotation_queue_item",
            input={"query": q},
        ) as root:
            root.update(output={"status": "ready_for_annotation"})
            trace_ids.append(root.trace_id)

    out_path = OUT_DIR / f"annotation_queue_trace_ids_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(trace_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved trace ids:", out_path)

    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    main()
