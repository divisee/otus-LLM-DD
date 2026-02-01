#!/usr/bin/env python3
"""
Тестовые запросы для проверки RAG-пайплайна.
Запуск: python tests/run_tests.py
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

TESTS = [
    # 1. Поиск в базе (RAG) - конкретный фильм по описанию
    {
        "name": "RAG: Фильм про мальчика с волшебными силами",
        "query": "Найди фильм про мальчика с волшебными силами который учится в школе магии",
        "expected": "rag",
    },
    # 2. Поиск в базе (RAG) - фильм по сюжету
    {
        "name": "RAG: Фильм про ограбление казино",
        "query": "Что за фильм где команда грабит три казино в Лас-Вегасе",
        "expected": "rag",
    },
    # 3. Веб-поиск - рекомендации
    {
        "name": "WEB: Топ комедий 2025",
        "query": "Посоветуй топ 5 лучших комедий 2025 года",
        "expected": "web",
    },
    # 4. Веб-поиск - подборка
    {
        "name": "WEB: Фильмы для семейного просмотра",
        "query": "Подбери фильмы для семейного просмотра на выходные",
        "expected": "web",
    },
    # 5. Комбинированный запрос
    {
        "name": "MIXED: Драмы про семью",
        "query": "Найди хорошие драмы про семейные отношения",
        "expected": "mixed",
    },
]


def run_test(query: str, test_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"ТЕСТ: {test_name}")
    print(f"ЗАПРОС: {query}")
    print("=" * 60)

    config_path = Path(__file__).parent.parent / "config.yaml"
    script_path = Path(__file__).parent.parent / "build_graph.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path), "--query", query],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
        timeout=300,
    )

    print("\n--- STDOUT ---")
    print(result.stdout if result.stdout else "(пусто)")

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr[:500])

    return {
        "name": test_name,
        "query": query,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def save_results(results: list) -> Path:
    """Сохраняет результаты тестов в JSON файл в папке tests."""
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "tests_count": len(results),
        "passed": sum(1 for r in results if r.get("returncode") == 0),
        "failed": sum(1 for r in results if r.get("returncode") != 0),
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Результаты сохранены в: {output_file}")
    return output_file


def main():
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ RAG-ПАЙПЛАЙНА")
    print("=" * 60)

    results = []
    for i, test in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {test['name']}")
        try:
            result = run_test(test["query"], test["name"])
            results.append(result)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: Тест {test['name']} превысил лимит времени")
            results.append({"name": test["name"], "returncode": -1, "error": "timeout"})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": test["name"], "returncode": -1, "error": str(e)})

    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    for r in results:
        status = "✅ OK" if r.get("returncode") == 0 else "❌ FAIL"
        print(f"{status} | {r['name']}")

    # Сохраняем результаты в файл
    save_results(results)


if __name__ == "__main__":
    main()
