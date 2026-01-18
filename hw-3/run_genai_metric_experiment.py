#!/usr/bin/env python3

import os
from pathlib import Path

import mlflow
import pandas as pd

from config_loader import Config
from vllm_client import VLLMClient
from llm_judge import LLMJudge
from datasets import create_test_dataset


def _resolve_tracking_uri(raw_uri: str) -> str:
    if raw_uri.startswith("sqlite:///"):
        path = raw_uri.replace("sqlite:///", "", 1)
        if not os.path.isabs(path):
            abs_path = Path(__file__).parent.joinpath(path).resolve()
            return f"sqlite:///{abs_path}"
    return raw_uri


def run_experiment() -> None:
    config = Config()
    tracking_uri = _resolve_tracking_uri(config.mlflow_tracking_uri)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    client = VLLMClient(config)
    judge = LLMJudge(config, client)

    # Важно: используем тот же набор вопросов, что и в обычном эксперименте.
    df = create_test_dataset()

    def predict_fn(inputs_df: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for q in inputs_df["question"].tolist():
            outputs.append(client.simple_query(q, method="openai"))

        # Для model_type="text" mlflow.evaluate ожидает табличный вывод
        # c колонкой `predictions`.
        return pd.DataFrame({"predictions": outputs})

    metric = judge.create_mlflow_metric(metric_name="local_genai_judge")

    with mlflow.start_run(run_name="genai_metric_evaluation"):
        mlflow.log_param("model_name", config.vllm_model_name)
        mlflow.log_param("temperature", config.vllm_temperature)
        mlflow.log_param("max_tokens", config.vllm_max_tokens)
        mlflow.log_param("base_url", config.vllm_base_url)

        results = mlflow.evaluate(
            model=predict_fn,
            data=df,
            model_type="text",
            extra_metrics=[metric],
            evaluator_config={
                "col_mapping": {
                    "question": "question",
                    # В таблицах evaluate предсказания лежат в колонке `predictions`,
                    # а genai-метрики ожидают стандартное имя `output`.
                    "output": "predictions",
                }
            },
        )

        # Явно логируем метрики evaluate в текущий run
        for k, v in (results.metrics or {}).items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # Сохраняем результаты
        try:
            if hasattr(results, "tables") and results.tables:
                for name, table in results.tables.items():
                    try:
                        mlflow.log_text(table.to_csv(index=False), f"eval_table_{name}.csv")
                    except Exception:
                        pass
        except Exception:
            pass


if __name__ == "__main__":
    run_experiment()
