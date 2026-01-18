#!/usr/bin/env python3


import mlflow
import pandas as pd
from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from config_loader import Config
from vllm_client import VLLMClient


def create_custom_metric_with_local_model(config: Config):
    grading_prompt = """Оцените качество ответа на вопрос по шкале от 1 до 5.

Вопрос: {inputs}
Ответ: {output}

Верните только число от 1 до 5."""

    examples = [
        EvaluationExample(
            input="What is the capital of France?",
            output="The capital of France is Paris.",
            score=5,
            justification="Идеальный ответ."
        ),
        EvaluationExample(
            input="What is the capital of France?",
            output="I don't know.",
            score=1,
            justification="Нет ответа."
        ),
        EvaluationExample(
            input="What is 2+2?",
            output="2+2 equals 4.",
            score=5,
            justification="Правильный ответ."
        )
    ]

    metric = make_genai_metric(
        name="local_llm_judge",
        definition="Оценка качества ответа от 1 до 5 с использованием локальной LLM.",
        grading_prompt=grading_prompt,
        examples=examples,
        model=f"openai:/{config.vllm_model_name}",
        grading_context_columns=["inputs"],
        parameters={"temperature": 0.1, "max_tokens": 10},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True
    )

    return metric


def create_evaluation_dataset():
    data = {
        "input": [
            "What is the capital of Germany?",
            "Explain artificial intelligence",
            "What is the largest planet?",
            "Who wrote Romeo and Juliet?",
            "What is the speed of light?",
            "What is quantum computing?",
            "Name the continents",
            "What is the chemical symbol for gold?",
            "Explain photosynthesis",
            "What is the tallest mountain?"
        ]
    }
    return pd.DataFrame(data)


def model_function(inputs_df, client: VLLMClient):
    outputs = []

    # MLflow может передать данные по-разному, проверяем все варианты
    if isinstance(inputs_df, pd.DataFrame):
        # Пробуем разные названия колонок
        if 'input' in inputs_df.columns:
            questions = inputs_df['input'].tolist()
        elif 'question' in inputs_df.columns:
            questions = inputs_df['question'].tolist()
        else:
            # Берем первую колонку
            questions = inputs_df.iloc[:, 0].tolist()
    else:
        # Если не DataFrame, возможно это Series или список
        questions = [inputs_df] if isinstance(inputs_df, str) else list(inputs_df)

    for question in questions:
        try:
            answer = client.simple_query(question, method="openai")
            outputs.append(answer)
        except Exception as e:
            print(f"Ошибка для вопроса '{question}': {e}")
            outputs.append(f"Error: {str(e)}")

    return outputs


def run_advanced_experiment():
    print("=" * 80)
    print("Расширенный эксперимент MLflow с make_genai_metric")
    print("=" * 80)

    config = Config()
    vllm_client = VLLMClient(config)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    eval_df = create_evaluation_dataset()
    print(f"\nСоздан датасет для оценки с {len(eval_df)} вопросами\n")

    custom_metric = create_custom_metric_with_local_model(config)
    print("Создана кастомная метрика с использованием локальной модели\n")

    with mlflow.start_run(run_name="advanced_vllm_evaluation") as run:
        mlflow.log_param("model_name", config.vllm_model_name)
        mlflow.log_param("temperature", config.vllm_temperature)
        mlflow.log_param("max_tokens", config.vllm_max_tokens)
        mlflow.log_param("judge_model", config.vllm_model_name)
        mlflow.log_param("evaluation_method", "make_genai_metric")

        print("Генерация ответов и оценка с помощью MLflow evaluate...")

        def wrapped_model(inputs_df):
            return model_function(inputs_df, vllm_client)

        results = mlflow.evaluate(
            model=wrapped_model,
            data=eval_df,
            model_type="text",
            extra_metrics=[custom_metric],
            evaluator_config={
                "col_mapping": {
                    "inputs": "input",
                    "predictions": "predictions"
                }
            }
        )

        # Дополнительные стандартные метрики
        mlflow.log_metric("dataset_size", len(eval_df))
        mlflow.log_metric("total_questions_evaluated", len(eval_df))

        # Если в results.metrics есть наши метрики судьи, логируем дополнительную статистику
        if 'local_llm_judge/mean' in results.metrics:
            mean_score = results.metrics['local_llm_judge/mean']
            mlflow.log_metric("judge_mean_normalized", mean_score / 5.0)  # Нормализованная оценка 0-1
            mlflow.log_metric("judge_quality_percent", (mean_score / 5.0) * 100)  # В процентах

        print("\n" + "=" * 80)
        print("Результаты оценки:")
        print("=" * 80)
        print(results.metrics)

        print(f"\nДополнительные метрики:")
        print(f"Размер датасета: {len(eval_df)} вопросов")
        if 'local_llm_judge/mean' in results.metrics:
            mean_score = results.metrics['local_llm_judge/mean']
            print(f"Средняя оценка судьи: {mean_score:.2f}/5 ({(mean_score/5.0)*100:.1f}%)")
        if 'local_llm_judge/variance' in results.metrics:
            print(f"Разброс оценок: {results.metrics['local_llm_judge/variance']:.3f}")
        if 'local_llm_judge/p90' in results.metrics:
            print(f"90-й перцентиль: {results.metrics['local_llm_judge/p90']:.2f}")

        print(f"\nRun ID: {run.info.run_id}")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
        print("=" * 80)
        print("""
Этот эксперимент демонстрирует:

1. Интеграцию локальной vLLM модели с MLflow
2. Использование make_genai_metric для создания кастомной метрики
3. Автоматическую оценку качества ответов моделью-судьей
4. Агрегацию метрик (mean, variance, p90)

Особенности модели Qwen2.5-3B-Instruct:
- Поддерживает chat-шаблоны с system/user ролями
- OpenAI-совместимый API через vLLM
- Хорошо справляется с фактическими вопросами
- Адекватна для роли судьи в простых случаях

Рекомендации:
- Использовать более крупную модель в роли судьи для сложных случаев
- Расширить набор примеров (examples)
- Комбинировать с традиционными метриками
        """)

        return results


if __name__ == "__main__":
    try:
        results = run_advanced_experiment()

    except Exception as e:
        print(f"\nОшибка при выполнении эксперимента: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Для просмотра результатов запустите MLflow UI:")
    print("mlflow ui --port 5000")
    print("=" * 80)
