#!/usr/bin/env python3

import os
from pathlib import Path
import mlflow
import pandas as pd
from config_loader import Config
from vllm_client import VLLMClient
from llm_judge import LLMJudge


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


def generate_answers(client: VLLMClient, questions: list) -> list:
    answers = []
    for question in questions:
        try:
            answer = client.simple_query(question, method="openai")
            answers.append(answer)
        except Exception as e:
            print(f"Ошибка при генерации ответа для '{question}': {e}")
            answers.append("Error generating answer")
    return answers


def run_mlflow_experiment():
    print("=" * 80)
    print("Запуск MLflow эксперимента с LLM-as-a-Judge")
    print("=" * 80)

    config = Config()
    tracking_uri = _resolve_tracking_uri(config.mlflow_tracking_uri)
    vllm_client = VLLMClient(config)
    llm_judge = LLMJudge(config, vllm_client)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    df = create_test_dataset()
    print(f"\nСоздан тестовый датасет с {len(df)} вопросами\n")

    with mlflow.start_run(run_name="vllm_qwen_evaluation") as run:
        mlflow.log_param("model_name", config.vllm_model_name)
        mlflow.log_param("temperature", config.vllm_temperature)
        mlflow.log_param("max_tokens", config.vllm_max_tokens)
        mlflow.log_param("base_url", config.vllm_base_url)

        print("Генерация ответов модели...")
        df['model_answer'] = generate_answers(vllm_client, df['question'].tolist())

        mlflow.log_text(df.to_csv(index=False), "test_dataset_with_answers.csv")

        print("\nОценка качества ответов с помощью LLM Judge...")
        scores = []
        for idx, row in df.iterrows():
            try:
                score = llm_judge.evaluate_answer(row['question'], row['model_answer'])
                scores.append(score)
                print(f"  Вопрос {int(idx)+1}/{len(df)}: оценка = {score}")
            except Exception as e:
                print(f"  Вопрос {int(idx)+1}/{len(df)}: ошибка - {e}")
                scores.append(3)

        df['judge_score'] = scores

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
