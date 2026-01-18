#!/usr/bin/env python3

import os
from pathlib import Path
import mlflow
import pandas as pd
from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from config_loader import Config
from vllm_client import VLLMClient


def _resolve_tracking_uri(raw_uri: str) -> str:
    if raw_uri.startswith("sqlite:///"):
        path = raw_uri.replace("sqlite:///", "", 1)
        if not os.path.isabs(path):
            abs_path = Path(__file__).parent.joinpath(path).resolve()
            return f"sqlite:///{abs_path}"
    return raw_uri


def create_custom_metric_with_local_model(config: Config):
    grading_prompt = """Вы СТРОГИЙ эксперт-оценщик. Оцените ответ по 5-балльной шкале:

СТРОГИЕ КРИТЕРИИ:
5 - ОТЛИЧНО: Полный, точный, развернутый ответ с деталями и примерами. Без ошибок.
4 - ХОРОШО: Правильный ответ, но недостаточно подробный или есть мелкие неточности.
3 - УДОВЛЕТВОРИТЕЛЬНО: Частично правильный, есть существенные пробелы или упрощения.
2 - ПЛОХО: Ответ содержит ошибки, слишком поверхностный или уклончивый.
1 - ОЧЕНЬ ПЛОХО: Неправильный, бессмысленный или отказ от ответа.

Вопрос: {input}
Ответ: {output}

ВАЖНО: Будьте СТРОГИ! Краткие ответы = максимум 4 балла. Поверхностные = 2-3 балла.
Верните ТОЛЬКО число от 1 до 5."""

    examples = [
        EvaluationExample(
            input="Explain quantum entanglement",
            output="Quantum entanglement is when particles are connected and measuring one affects the other instantly.",
            score=3,
            justification="Поверхностно, нет объяснения механизма"
        ),
        EvaluationExample(
            input="What is the capital of France?",
            output="Paris",
            score=4,
            justification="Правильно, но слишком кратко"
        ),
        EvaluationExample(
            input="Explain photosynthesis",
            output="Plants make food from sunlight",
            score=2,
            justification="Слишком упрощенно, нет деталей"
        ),
        EvaluationExample(
            input="What is 2+2?",
            output="The answer is 4 (two plus two equals four)",
            score=5,
            justification="Полный точный ответ"
        ),
        EvaluationExample(
            input="Explain consciousness",
            output="I'm not sure about that",
            score=1,
            justification="Отказ от ответа"
        )
    ]

    metric = make_genai_metric(
        name="local_llm_judge",
        definition="СТРОГАЯ оценка качества ответа от 1 до 5 с четкими критериями глубины и полноты.",
        grading_prompt=grading_prompt,
        examples=examples,
        model=f"openai:/{config.vllm_model_name}",
        parameters={"temperature": 0.3, "max_tokens": 20},
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True
    )

    return metric


def create_evaluation_dataset():
    data = {
        "input": [
            # Простые вопросы (ожидаем 4-5, если развернуто)
            "What is the capital of Germany?",
            "What is 2 + 2?",

            # Средней сложности (ожидаем 2-4)
            "Explain artificial intelligence in one sentence",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?",

            # Научные - требуют глубины (ожидаем 1-3)
            "Explain quantum entanglement and discuss whether it allows faster-than-light communication",
            "What is the relationship between Gödel's incompleteness theorems and the halting problem in computer science?",
            "Describe how general relativity explains gravity through spacetime curvature",
            "What are the leading theories of consciousness and their main criticisms?",

            # Философские/неоднозначные (ожидаем 1-3)
            "Can free will exist in a deterministic universe? Provide arguments for both sides",
            "Is mathematics invented or discovered? Defend your position",
            "What is the nature of time? Does it flow or is that an illusion of consciousness?",

            # Междисциплинарные сложные (ожидаем 1-3)
            "How would you design an experiment to test if an AI has achieved consciousness?",
            "Explain the P vs NP problem and why it matters for cryptography",
            "What connects entropy in thermodynamics, information theory, and black holes?",

            # Провокационные/абсурдные (ожидаем 1)
            "How many fingers does a blue elephant have in a parallel universe where logic is reversed?",
            "Calculate the emotional temperature of Tuesday in degrees of happiness",
            "What color is the smell of the number 7 when mixed with jazz music?",
            "If silence had mass, how much would a cubic meter of Tuesday weigh?",

            # Трюковые логические (ожидаем 1-3)
            "If you have 3 apples and eat 2 oranges, how many purple fruits remain?",
            "What sound does the color blue make in a vacuum?",
            "How many corners does a perfect circle have? Explain mathematically",

            # Экстремально специфичные научные (ожидаем 1-2)
            "Explain the Yang-Mills mass gap problem and why it's a Millennium Prize Problem",
            "What is the cosmological constant problem in quantum field theory?",
            "Describe quorum sensing in bacterial biofilms and its role in antibiotic resistance",
            "Explain the AdS/CFT correspondence and its implications for quantum gravity",

            # Нерешенные проблемы (ожидаем 1-2)
            "Solve the Riemann hypothesis and explain your proof",
            "What is dark matter made of? Provide experimental evidence",
            "How do you reconcile quantum mechanics with general relativity?"
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
    tracking_uri = _resolve_tracking_uri(config.mlflow_tracking_uri)
    vllm_client = VLLMClient(config)

    mlflow.set_tracking_uri(tracking_uri)
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
                    "inputs": "input"
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
