import pandas as pd


def create_test_dataset() -> pd.DataFrame:
    """Единый датасет вопросов для всех экспериментов.

    Важно: genai-эксперимент и обычный эксперимент должны использовать
    один и тот же набор вопросов, чтобы метрики были сопоставимы.
    """

    return pd.DataFrame(
        {
            "question": [
                "What is the capital of Germany?",
                "What is 2 + 2?",
                "Who painted the Mona Lisa?",
                "Explain artificial intelligence in one sentence",
                "What is the largest planet in our solar system?",
                "Who wrote Romeo and Juliet?",
                "Explain quantum superposition and measurement problem",
                "What is the relationship between entropy and information theory?",
                "Describe the implications of general relativity for GPS satellites",
                "Can free will exist in a deterministic universe?",
                "What defines personal identity over time?",
                "How do you divide silence by infinity?",
                "What color is the smell of number 7?",
                "If a tree falls in a forest and speaks French, what is the square root of purple?",
            ]
        }
    )
