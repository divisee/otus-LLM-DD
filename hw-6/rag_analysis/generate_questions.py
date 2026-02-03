#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path
import os
import logging
import warnings

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Allow importing from hw-6 root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config_utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def _load_llm(config_path: Path, temperature: float) -> ChatOpenAI:
    config = load_config(config_path)
    openai_cfg = config.get("openai", {})
    api_key = openai_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        api_key=api_key,
        model=openai_cfg.get("model", "gpt-4o-mini"),
        base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
        temperature=temperature,
    )


def _build_prompt(title: str, description: str, difficulty: int) -> list:
    system = (
        "Веди себя как молодой человек, который не может вспомнить фильм, но помнит какие-то небольшие детали о нем."
        "Задай вопрос так, чтобы он звучал как просьба (пример): помоги вспомнить/найти фильм, в котором.../найти фильм, где было ..."
        "Тебе дадут название, описание и уровень сложности 1-5. "
        "Сформулируй один короткий вопрос о фильме из описания на русском языке. "
        "Уровень 1 — простой, маленькая цитата из описания. "
        "Уровень 5 — максимально косвенный, но издалека опирается на детали описания. "
        "Верни только одно предложение - один короткий вопрос (до 20 слов) без лишних деталей, без цитирования сюжета и без пояснений. "
        "Придумывай интересные творческие вопросы. Не используй многоточия и не обрывай фразу."
    )
    human = (
        f"Название: {title}\n"
        f"Описание: {description}\n"
        f"Уровень сложности: {difficulty}"
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _clean_question(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if cleaned and not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned


def _generate_question(
    llm: ChatOpenAI,
    title: str,
    description: str,
    difficulty: int,
    max_retries: int,
) -> str:
    title_lower = title.strip().lower()
    for attempt in range(1, max_retries + 1):
        messages = _build_prompt(title, description, difficulty)
        response = llm.invoke(messages)
        question = _clean_question(response.content or "")
        if not question:
            continue
        if title_lower and title_lower in question.lower():
            continue
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("difficulty=%s | llm=%s", difficulty, response.content)
        return question
    return _clean_question(f"О каком фильме идет речь: {description}")


def _heuristic_question(description: str, difficulty: int) -> str:
    prefix = {
        1: "О каком фильме идет речь, где",
        2: "Какой фильм описывает ситуацию, где",
        3: "Назовите фильм, в котором",
        4: "Какой фильм скрывается за историей, где",
        5: "Какой фильм можно узнать по намеку, где",
    }.get(difficulty, "О каком фильме идет речь, где")
    snippet = description.strip()
    if len(snippet) > 180:
        snippet = snippet[:177].rsplit(" ", 1)[0] + "..."
    question = f"{prefix} {snippet}"
    return _clean_question(question)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 100 hard/medium/easy questions for RAG evaluation."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "data_films.csv"),
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "rag_analysis" / "data" / "rag_questions.csv"),
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config with OpenAI settings.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per question if invalid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call LLM; generate heuristic questions.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs with LLM responses.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.debug else logging.INFO,
        format="%(message)s",
    )
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    random.seed(args.seed)

    df = pd.read_csv(args.input)
    if "Title" not in df.columns or "Description Kinopoisk" not in df.columns:
        raise ValueError("Dataset must include 'Title' and 'Description Kinopoisk' columns.")

    df = df[["Title", "Description Kinopoisk"]].copy()
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Description Kinopoisk"] = df["Description Kinopoisk"].fillna("").astype(str)
    df = df[df["Description Kinopoisk"].str.strip() != ""]

    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    total = len(sample)
    base, extra = divmod(total, 5)
    difficulties = [level for level in range(1, 6) for _ in range(base)]
    if extra:
        difficulties += random.sample(range(1, 6), extra)
    random.shuffle(difficulties)

    llm = None
    if not args.dry_run:
        llm = _load_llm(Path(args.config), args.temperature)

    rows = []
    for line_no, ((_, row), difficulty) in enumerate(zip(sample.iterrows(), difficulties), start=1):
        title = row["Title"].strip()
        description = row["Description Kinopoisk"].strip()

        if args.dry_run:
            question = _heuristic_question(description, difficulty)
        else:
            question = _generate_question(llm, title, description, difficulty, args.max_retries)

        logger.info(
            "line=%s\ndifficulty=%s\ntitle=%s\ndescription=%s\nquestion=%s\n",
            line_no,
            difficulty,
            title,
            description,
            question,
        )

        rows.append(
            {
                "title": title,
                "description": description,
                "question": question,
                "difficulty": difficulty,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
