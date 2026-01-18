import re
from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from vllm_client import VLLMClient
from config_loader import Config


class LLMJudge:
    def __init__(self, config: Config, vllm_client: VLLMClient):
        self.config = config
        self.vllm_client = vllm_client

    def evaluate_answer(self, question: str, answer: str) -> int:
        prompt = f"""Оцените качество ответа на вопрос по шкале от 1 до 5.

Вопрос: {question}
Ответ: {answer}

Верните только число от 1 до 5."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.vllm_client.chat_with_openai(
            messages=messages,
            temperature=0.1,
            max_tokens=10
        )

        score = self._extract_score(response)
        return score

    def _extract_score(self, response: str) -> int:
        match = re.search(r'\b([1-5])\b', response)
        if match:
            return int(match.group(1))
        return 3

    def create_mlflow_metric(self, metric_name: str = "llm_judge_score"):
        examples = [
            EvaluationExample(
                input="What is the capital of Germany?",
                output="The capital of Germany is Berlin.",
                score=5,
                justification="Идеальный ответ."
            ),
            EvaluationExample(
                input="What is the capital of Germany?",
                output="I don't know.",
                score=1,
                justification="Нет ответа."
            )
        ]

        metric = make_genai_metric(
            name=metric_name,
            definition="Оценка качества ответа от 1 до 5.",
            grading_prompt="Оцените ответ от 1 до 5. Верните только число.",
            examples=examples,
            model=f"openai:/{self.config.vllm_model_name}",
            grading_context_columns=["question"],
            parameters={"temperature": 0.1, "max_tokens": 10},
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True
        )

        return metric
