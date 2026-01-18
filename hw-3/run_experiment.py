#!/usr/bin/env python3

import os
from pathlib import Path
import json

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


def _trace_decorator(name: str):
    try:
        return mlflow.trace(name=name)
    except Exception:
        def _noop(fn):
            return fn
        return _noop


def run_mlflow_experiment() -> pd.DataFrame:
    config = Config()
    tracking_uri = _resolve_tracking_uri(config.mlflow_tracking_uri)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    client = VLLMClient(config)
    judge = LLMJudge(config, client)

    df = create_test_dataset()

    @_trace_decorator("qa_item")
    def ask(question: str) -> str:
        return client.simple_query(question, method="openai")

    @_trace_decorator("judge")
    def grade(question: str, answer: str) -> int:
        return judge.evaluate_answer(question, answer)

    with mlflow.start_run(run_name="vllm_qwen_evaluation") as run:
        mlflow.log_param("model_name", config.vllm_model_name)
        mlflow.log_param("temperature", config.vllm_temperature)
        mlflow.log_param("max_tokens", config.vllm_max_tokens)
        mlflow.log_param("base_url", config.vllm_base_url)

        answers, scores = [], []
        for q in df["question"].tolist():
            a = ask(q)
            s = grade(q, a)
            answers.append(a)
            scores.append(s)

        df["model_answer"] = answers
        df["judge_score"] = scores

        avg_score = float(sum(scores) / len(scores))
        mlflow.log_metric("average_judge_score", avg_score)
        mlflow.log_metric("min_judge_score", float(min(scores)))
        mlflow.log_metric("max_judge_score", float(max(scores)))
        mlflow.log_metric("total_questions", float(len(df)))

        # Q/A как артефакт
        mlflow.log_text(df.to_csv(index=False), "qa_results.csv")

        # Читабельный json на всякий случай
        mlflow.log_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), "qa_results.json")

        print(f"Run ID: {run.info.run_id}")

    return df


if __name__ == "__main__":
    run_mlflow_experiment()

