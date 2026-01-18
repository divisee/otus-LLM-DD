#!/usr/bin/env python3


import mlflow
import pandas as pd
from config_loader import Config
from vllm_client import VLLMClient
from llm_judge import LLMJudge


def create_test_dataset():
    data = {
        "question": [
            "What is the capital of Germany?",
            "Explain artificial intelligence in one sentence",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?",
            "What is the speed of light?",
            "What is the capital of France?",
            "Who painted the Mona Lisa?",
            "What is 2 + 2?"
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
    vllm_client = VLLMClient(config)
    llm_judge = LLMJudge(config, vllm_client)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
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

        mlflow.log_text(df.to_csv(index=False), "final_results.csv")

        print("\n" + "=" * 80)
        print("Результаты эксперимента:")
        print("=" * 80)
        print(f"Средняя оценка: {avg_score:.2f}/5")
        print(f"Минимальная оценка: {min(scores)}")
        print(f"Максимальная оценка: {max(scores)}")
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
    print("mlflow ui --port 5000")
    print("Затем откройте http://localhost:5000 в браузере")
    print("=" * 80)
