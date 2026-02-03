#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def _load_text_series(csv_path: Path, column: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if column in df.columns:
        series = df[column]
    elif column == "text_ru" and "Description Kinopoisk" in df.columns:
        # Fallback for the current dataset schema.
        series = df["Description Kinopoisk"]
        column = "Description Kinopoisk"
    else:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {', '.join(df.columns)}"
        )
    return series.fillna("").astype(str), column, len(df)


def _series_lengths(texts: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    stripped = texts.str.strip()
    empty_count = int((stripped == "").sum())
    char_len = stripped.str.len()
    word_len = stripped.str.split().str.len()
    return char_len, word_len, empty_count


def _stats_block(values: pd.Series) -> dict[str, float]:
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    desc = values.describe(percentiles=percentiles)
    return {
        "mean": float(desc["mean"]),
        "std": float(desc["std"]),
        "min": float(desc["min"]),
        "p5": float(desc["5%"]),
        "p25": float(desc["25%"]),
        "median": float(desc["50%"]),
        "p75": float(desc["75%"]),
        "p95": float(desc["95%"]),
        "p99": float(desc["99%"]),
        "max": float(desc["max"]),
    }


def _format_stat(value: float, integer: bool = False) -> str:
    if integer:
        return str(int(round(value)))
    return f"{value:.2f}"


def build_report(csv_path: Path, column: str, output_path: Path) -> None:
    texts, resolved_column, total_rows = _load_text_series(csv_path, column)
    char_len, word_len, empty_count = _series_lengths(texts)

    non_empty_mask = texts.str.strip() != ""
    char_stats = _stats_block(char_len[non_empty_mask])
    word_stats = _stats_block(word_len[non_empty_mask])

    empty_share = (empty_count / total_rows) * 100 if total_rows else 0

    report_lines = [
        "# Анализ длины контекста для RAG",
        "",
        "## Источник",
        f"- Файл: `{csv_path}`",
        f"- Колонка: `{resolved_column}`",
        f"- Всего записей: {total_rows}",
        f"- Пустые значения: {empty_count} ({empty_share:.2f}%)",
        "",
        "## Статистика (только непустые значения)",
        "",
        "| Метрика | Длина в символах | Длина в словах |",
        "| --- | --- | --- |",
        f"| Среднее | {_format_stat(char_stats['mean'])} | {_format_stat(word_stats['mean'])} |",
        f"| Медиана | {_format_stat(char_stats['median'])} | {_format_stat(word_stats['median'])} |",
        f"| Стд. отклонение | {_format_stat(char_stats['std'])} | {_format_stat(word_stats['std'])} |",
        f"| Минимум | {_format_stat(char_stats['min'], integer=True)} | {_format_stat(word_stats['min'], integer=True)} |",
        f"| P5 | {_format_stat(char_stats['p5'])} | {_format_stat(word_stats['p5'])} |",
        f"| P25 | {_format_stat(char_stats['p25'])} | {_format_stat(word_stats['p25'])} |",
        f"| P50 | {_format_stat(char_stats['median'])} | {_format_stat(word_stats['median'])} |",
        f"| P75 | {_format_stat(char_stats['p75'])} | {_format_stat(word_stats['p75'])} |",
        f"| P95 | {_format_stat(char_stats['p95'])} | {_format_stat(word_stats['p95'])} |",
        f"| P99 | {_format_stat(char_stats['p99'])} | {_format_stat(word_stats['p99'])} |",
        f"| Максимум | {_format_stat(char_stats['max'], integer=True)} | {_format_stat(word_stats['max'], integer=True)} |",
        "",
        "## Примечания",
        "- Статистика рассчитана по непустым значениям, чтобы нули не занижали метрики.",
        "- Длина в словах рассчитана через разбиение по пробелам.",
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze text_ru length stats for RAG.")
    parser.add_argument(
        "--input",
        default="/Users/arasputina/PycharmProjects/otus-LLM-DD/hw-6/data/data_films.csv",
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--column",
        default="text_ru",
        help="Column name with Russian text (default: text_ru).",
    )
    parser.add_argument(
        "--output",
        default="/Users/arasputina/PycharmProjects/otus-LLM-DD/hw-6/rag_analysis/README.md",
        help="Path to markdown report.",
    )
    args = parser.parse_args()

    build_report(Path(args.input), args.column, Path(args.output))


if __name__ == "__main__":
    main()
