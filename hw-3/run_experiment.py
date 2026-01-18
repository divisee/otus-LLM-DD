#!/usr/bin/env python3

import os
from pathlib import Path
import mlflow
import pandas as pd
from config_loader import Config
from vllm_client import VLLMClient
from llm_judge import LLMJudge
import json


def _resolve_tracking_uri(raw_uri: str) -> str:
    if raw_uri.startswith("sqlite:///"):
        path = raw_uri.replace("sqlite:///", "", 1)
        if not os.path.isabs(path):
            abs_path = Path(__file__).parent.joinpath(path).resolve()
            return f"sqlite:///{abs_path}"
    return raw_uri


def create_test_dataset():
    data = {
        "question": [
            # Простые (ожидаем 5)
            "What is the capital of Germany?",
            "What is 2 + 2?",
            "Who painted the Mona Lisa?",

            # Средние (ожидаем 3-4)
            "Explain artificial intelligence in one sentence",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?",

            # Сложные научные (ожидаем 2-4)
            "Explain quantum superposition and measurement problem",
            "What is the relationship between entropy and information theory?",
            "Describe the implications of general relativity for GPS satellites",

            # Философские (ожидаем 2-3)
            "Can free will exist in a deterministic universe?",
            "What defines personal identity over time?",

            # Провокационные/абсурдные (ожидаем 1-2)
            "How do you divide silence by infinity?",
            "What color is the smell of number 7?",
            "If a tree falls in a forest and speaks French, what is the square root of purple?"
        ]
    }
    return pd.DataFrame(data)


def generate_answers_and_scores(vllm_client: VLLMClient, llm_judge: LLMJudge, questions: list) -> tuple[list, list]:
    @_trace_decorator(name="qa_item", attributes={})
    def generate_one(question: str) -> str:
        mlflow.set_tag("last_question", question)
        return vllm_client.simple_query(question, method="openai")

    @_trace_decorator(name="judge", attributes={})
    def judge_one(question: str, answer: str) -> int:
        mlflow.set_tag("last_judge_question", question)
        return llm_judge.evaluate_answer(question, answer)

    answers, scores = [], []
    for q in questions:
        try:
            a = generate_one(q)
            s = judge_one(q, a)
            answers.append(a)
            scores.append(s)
        except Exception as e:
            print(f"Ошибка для вопроса '{q}': {e}")
            answers.append("Error generating answer")
            scores.append(3)
    return answers, scores


def _trace_decorator(name: str, attributes: dict):
    """Обертка: если tracing недоступен в версии MLflow, просто возвращаем исходную функцию."""
    try:
        return mlflow.trace(name=name, attributes=attributes)
    except Exception:
        def _noop(fn):
            return fn
        return _noop


def _make_traced_fns(vllm_client: VLLMClient, llm_judge: LLMJudge):
    @_trace_decorator(name="qa_item", attributes={})
    def generate_one(question: str) -> str:
        mlflow.set_tag("last_question", question)
        return vllm_client.simple_query(question, method="openai")

    @_trace_decorator(name="judge", attributes={})
    def judge_one(question: str, answer: str) -> int:
        mlflow.set_tag("last_judge_question", question)
        return llm_judge.evaluate_answer(question, answer)

    return generate_one, judge_one


def _get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id

    artifact_root = (Path(__file__).parent / "mlartifacts").resolve().as_uri()
    return mlflow.create_experiment(name, artifact_location=artifact_root)


def run_mlflow_experiment():
    print("=" * 80)
    print("Запуск MLflow эксперимента с LLM-as-a-Judge")
    print("=" * 80)

    config = Config()
    tracking_uri = _resolve_tracking_uri(config.mlflow_tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    vllm_client = VLLMClient(config)
    llm_judge = LLMJudge(config, vllm_client)

    mlflow.set_tracking_uri(tracking_uri)
    exp_id = _get_or_create_experiment(config.mlflow_experiment_name)
    mlflow.set_experiment(experiment_id=exp_id)

    df = create_test_dataset()
    print(f"\nСоздан тестовый датасет с {len(df)} вопросами\n")

    # Включаем автотрейсинг, если доступен (не должен ломать выполнение)
    try:
        mlflow.autolog(log_traces=True)
    except Exception:
        pass

    with mlflow.start_run(run_name="vllm_qwen_evaluation") as run:
        mlflow.log_param("model_name", config.vllm_model_name)
        mlflow.log_param("temperature", config.vllm_temperature)
        mlflow.log_param("max_tokens", config.vllm_max_tokens)
        mlflow.log_param("base_url", config.vllm_base_url)

        print("Генерация ответов модели...")
        df['model_answer'], scores = generate_answers_and_scores(vllm_client, llm_judge, df['question'].tolist())

        print("\nОценка качества ответов с помощью LLM Judge...")

        df['judge_score'] = scores

        # Логируем детальные трейсы (вопрос, ответ, оценка) как артефакт-файл
        traces = []
        for question, answer, score in zip(df['question'], df['model_answer'], df['judge_score']):
            traces.append({
                "question": str(question),
                "model_answer": str(answer),
                "judge_score": int(score) if score is not None else None,
                "judge_prompt": "strict 1-5 criteria"
            })

        traces_path = Path("traces.json")
        traces_path.write_text(json.dumps({"traces": traces}, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(traces_path))
        mlflow.set_tag("has_traces", "true")

        avg_score = sum(scores) / len(scores)
        mlflow.log_metric("average_judge_score", avg_score)
        mlflow.log_metric("min_judge_score", min(scores))
        mlflow.log_metric("max_judge_score", max(scores))

        # Стандартные базовые метрики
        mlflow.log_metric("total_questions", len(df))
        mlflow.log_metric("successful_answers", len([s for s in scores if s >= 3]))
        mlflow.log_metric("failed_answers", len([s for s in scores if s < 3]))
        mlflow.log_metric("excellent_answers", len([s for s in scores if s == 5]))
        mlflow.log_metric("poor_answers", len([s for s in scores if s <= 2]))

        # Процентные метрики
        success_rate = len([s for s in scores if s >= 3]) / len(scores) * 100
        excellence_rate = len([s for s in scores if s == 5]) / len(scores) * 100
        mlflow.log_metric("success_rate_percent", success_rate)
        mlflow.log_metric("excellence_rate_percent", excellence_rate)

        mlflow.log_text(df.to_csv(index=False), "final_results.csv")

        print("\n" + "=" * 80)
        print("Результаты эксперимента:")
        print("=" * 80)
        print(f"Средняя оценка: {avg_score:.2f}/5")
        print(f"Минимальная оценка: {min(scores)}")
        print(f"Максимальная оценка: {max(scores)}")
        print(f"\nДополнительные метрики:")
        print(f"Всего вопросов: {len(df)}")
        print(f"Успешных ответов (≥3): {len([s for s in scores if s >= 3])} ({success_rate:.1f}%)")
        print(f"Отличных ответов (5): {len([s for s in scores if s == 5])} ({excellence_rate:.1f}%)")
        print(f"Плохих ответов (≤2): {len([s for s in scores if s <= 2])}")
        print(f"\nRun ID: {run.info.run_id}")
        print("=" * 80)

        print("\n\nДетальные результаты:")
        print("-" * 80)
        for idx, row in df.iterrows():
            print(f"\nВопрос: {row['question']}")
            print(f"Ответ модели: {row['model_answer'][:100]}...")
            print(f"Оценка судьи: {row['judge_score']}/5")

        print("\n" + "=" * 80)
        print("АНАЛИЗ:")
        print("=" * 80)
        print("""
LLM-as-a-Judge работает следующим образом:
- Модель оценивает ответы по шкале от 1 до 5
- Простой и понятный промпт
- Qwen2.5-3B-Instruct справляется с оценкой базовых ответов
- Рекомендуется использовать более крупную модель для сложных случаев
        """)

        return df


if __name__ == "__main__":
    results_df = run_mlflow_experiment()

    print("\n" + "=" * 80)
    print("Для просмотра результатов в MLflow UI выполните:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("Затем откройте http://localhost:5000 в браузере")
    print("=" * 80)
