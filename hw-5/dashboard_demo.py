import os
from dotenv import load_dotenv
from langfuse import get_client


def require_env() -> None:
    load_dotenv()
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit("Missing env vars: " + ", ".join(missing))


def main() -> None:
    require_env()
    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="dashboard_demo",
        input={"query": "Найди фильм про ограбление казино"},
    ) as root:
        root.score(name="latency_s", value=1.23)
        root.score(name="cost_usd", value=0.0009)
        root.score(name="quality", value=0.92)
        root.update(output={"status": "ok"})

    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    main()
