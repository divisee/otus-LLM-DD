import os
from dotenv import load_dotenv
import yaml
from langfuse import get_client, propagate_attributes

# Загрузка конфигурации из config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Инициализация Langfuse клиента через переменные окружения
load_dotenv()

required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
missing = [key for key in required_vars if not os.getenv(key)]
if missing:
    print(
        "Ошибка: не найдены переменные окружения: "
        + ", ".join(missing)
        + ". Заполните langfuse-demo/.env или экспортируйте их в окружение."
    )
    raise SystemExit(1)

langfuse = get_client()

try:
    # ROOT observation (span) = старт новой trace (теперь trace создаётся автоматически)
    with langfuse.start_as_current_observation(
        as_type="span",
        name="example_trace",
        input={"started": True},
    ) as root:

        # Атрибуты трассы/наблюдений (user_id, metadata, version, tags, trace_name и т.д.)
        with propagate_attributes(
            user_id="user_123",
            metadata={"version": "1.0.0"},
            trace_name="example_trace",
        ):
            # Пример 2: generation (LLM)
            with root.start_as_current_observation(
                as_type="generation",
                name="llm_call",
                model="gpt-4o-mini",
                model_parameters={"temperature": "0.7"},
                input={"prompt": "What is the capital of France?"},
            ) as gen:
                gen.update(
                    output="The capital of France is Paris.",
                    usage_details={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )

            # Пример 2b: embedding
            with root.start_as_current_observation(
                as_type="embedding",
                name="embed_query",
                model="text-embedding-3-small",
                input={"text": "Paris is the capital of France."},
            ) as emb:
                emb.update(
                    output={"vector_preview": [0.12, 0.34, 0.56]},
                    usage_details={"prompt_tokens": 7, "total_tokens": 7},
                )

            # Пример 2c: agent
            with root.start_as_current_observation(
                as_type="agent",
                name="planner_agent",
                input={"goal": "Find capital of France"},
            ) as agent_span:
                agent_span.update(output={"status": "completed"})

            # Пример 2d: tool
            with root.start_as_current_observation(
                as_type="tool",
                name="wiki_search",
                input={"query": "capital of France"},
            ) as tool_span:
                tool_span.update(output={"hits": 3})

            # Пример 2e: chain
            with root.start_as_current_observation(
                as_type="chain",
                name="qa_chain",
                input={"question": "What is the capital of France?"},
            ) as chain_span:
                chain_span.update(output={"answer": "Paris"})

            # Пример 2f: retriever
            with root.start_as_current_observation(
                as_type="retriever",
                name="vector_retriever",
                input={"query": "capital of France"},
            ) as retriever_span:
                retriever_span.update(output={"documents": ["doc1", "doc2"]})

            # Пример 2g: evaluator
            with root.start_as_current_observation(
                as_type="evaluator",
                name="answer_evaluator",
                input={"answer": "Paris"},
            ) as evaluator_span:
                evaluator_span.update(output={"score": 0.95})

            # Пример 2h: guardrail
            with root.start_as_current_observation(
                as_type="guardrail",
                name="policy_check",
                input={"content": "Paris is the capital of France."},
            ) as guardrail_span:
                guardrail_span.update(output={"allowed": True})

            # Пример 3: span (операция)
            with root.start_as_current_observation(
                as_type="span",
                name="database_query",
                input={"query": "SELECT * FROM users"},
            ) as db_span:
                db_span.update(output={"results": ["user1", "user2"]})

            # Пример 4: score (оценка)
            root.score(
                name="accuracy",
                value=0.95,
                comment="High accuracy on this query",
            )

            # Пример 5: event (замена на span)
            with root.start_as_current_observation(
                as_type="span",
                name="file_upload_event",
                metadata={"file": "data.csv", "size": 1024},
            ) as file_upload_span:
                file_upload_span.update(output={"status": "ok"})

            # Пример 6: event с ошибкой (замена на span)
            with root.start_as_current_observation(
                as_type="span",
                name="error_occurred",
                level="ERROR",
                status_message="Connection timeout",
                metadata={"error": "Connection timeout", "code": 500},
            ) as error_span:
                error_span.update(output={"retry": False})

        # Завершение “трейса” = обновляем output у root span (по умолчанию станет trace output)
        root.update(output="Completed successfully")

finally:
    # В короткоживущих скриптах обязательно, чтобы не потерять данные
    langfuse.flush()
    langfuse.shutdown()

print("Примеры записей в Langfuse созданы. Проверьте дашборд Langfuse.")
