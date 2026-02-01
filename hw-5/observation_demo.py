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
        name="observations_demo",
        input={"query": "Фильм про путешествие во времени"},
    ) as root:
        with root.start_as_current_observation(
            as_type="agent",
            name="demo_agent",
            input={"step": "plan"},
        ) as agent_span:
            agent_span.update(output={"status": "ok"})

        with root.start_as_current_observation(
            as_type="generation",
            name="demo_generation",
            model="gpt-4o-mini",
            input="Сгенерируй короткий ответ",
        ) as gen:
            gen.update(output="Краткий ответ о фильме про путешествие во времени.")

        with root.start_as_current_observation(
            as_type="embedding",
            name="demo_embedding",
            model="demo-embedding",
            input={"text": "time travel"},
        ) as emb:
            emb.update(output={"vector_len": 768})

        with root.start_as_current_observation(
            as_type="retriever",
            name="demo_retriever",
            input={"query": "time travel", "top_k": 3},
        ) as retr:
            retr.update(output={"hits": 3})

        with root.start_as_current_observation(
            as_type="tool",
            name="demo_tool",
            input={"query": "time travel movies"},
        ) as tool:
            tool.update(output={"hits": 2})

        with root.start_as_current_observation(
            as_type="evaluator",
            name="demo_evaluator",
            input={"expected": "Terminator"},
        ) as ev:
            ev.update(output={"score": 0.8})

        root.score(name="demo_score", value=0.8)

    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    main()
