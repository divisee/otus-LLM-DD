"""
LLM-as-a-Judge: модель-судья оценивает правильность ответов RAG.

Два теста:
  1. LLM Judge — gpt-4o-mini оценивает, назван ли правильный фильм.
  2. Strict title match — проверка вхождения названия фильма в ответ.

Подробные логи — в logs/.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pytest
from openai import OpenAI
from pydantic import BaseModel

JUDGE_MODEL = "gpt-4o-mini"
LLM_JUDGE_PASS_RATE = 0.7
STRICT_MATCH_PASS_RATE = 0.6

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
LOGS_DIR = PROJECT_DIR / "logs"

class JudgeVerdict(BaseModel):
    """Структурированный ответ LLM-судьи."""
    correct: bool
    explanation: str


JUDGE_SYSTEM = (
    "Ты — строгий судья качества ответов системы поиска фильмов. "
    "Оцени, правильно ли система определила фильм."
)

JUDGE_USER = (
    "Вопрос пользователя: {question}\n"
    "Ожидаемый фильм: {expected_title}\n"
    "Ответ системы: {answer}\n\n"
    "Критерии:\n"
    "- correct=true если в ответе упоминается правильный фильм "
    "(допускаются небольшие вариации в названии)\n"
    "- correct=false если фильм не угадан или назван другой"
)


class _Logger:
    """Пишет одновременно в stdout и в лог-файл."""

    def __init__(self, log_path: Path):
        LOGS_DIR.mkdir(exist_ok=True)
        self._f = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str, end: str = "\n", flush: bool = True):
        print(msg, end=end, flush=flush)
        self._f.write(msg + end)
        if flush:
            self._f.flush()

    def close(self):
        self._f.close()


def _truncate(text: str, max_len: int = 120) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


@pytest.fixture(scope="module")
def judge_results(rag_results):
    """Прогон LLM-судьи по всем результатам RAG."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = _Logger(LOGS_DIR / f"llm_judge_{ts}.log")

    log.log(f"LLM Judge evaluation started at {ts}")
    log.log(f"Samples: {len(rag_results)}, Model: {JUDGE_MODEL}")
    log.log(f"Pass rate threshold: {LLM_JUDGE_PASS_RATE:.0%}")

    client = OpenAI()
    judgements = []
    t_start = time.time()

    for i, r in enumerate(rag_results):
        step_t0 = time.time()
        log.log(f"\n{'='*80}")
        log.log(f"  SAMPLE {i+1}/{len(rag_results)}")
        log.log(f"  Question:       {r['question']}")
        log.log(f"  Expected title: {r['expected_title']}")
        log.log(f"  RAG answer:     {_truncate(r['answer'])}")
        log.log(f"  Ground truth:   {_truncate(r['ground_truth'])}")
        log.log(f"{'='*80}")

        user_msg = JUDGE_USER.format(
            question=r["question"],
            expected_title=r["expected_title"],
            answer=r["answer"],
        )
        log.log(f"  Calling LLM judge ...", end="")

        resp = client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format=JudgeVerdict,
            max_tokens=200,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is not None:
            judgement = parsed.model_dump()
        else:
            judgement = {"correct": False, "explanation": "Parse error"}

        elapsed = time.time() - step_t0
        verdict = "CORRECT" if judgement.get("correct") else "WRONG"
        log.log(f"  [{verdict}] ({elapsed:.1f}s)")
        log.log(f"  Judge says: {judgement.get('explanation', '—')}")

        judgements.append(
            {
                "question": r["question"],
                "expected_title": r["expected_title"],
                "answer": r["answer"],
                "judgement": judgement,
            }
        )

        elapsed_total = time.time() - t_start
        remaining = (elapsed_total / (i + 1)) * (len(rag_results) - i - 1)
        log.log(
            f"  >>> {i+1}/{len(rag_results)} done  "
            f"elapsed={elapsed_total:.0f}s  est. remaining={remaining:.0f}s"
        )

    correct_count = sum(1 for j in judgements if j["judgement"].get("correct"))
    pass_rate = correct_count / len(judgements) if judgements else 0

    RESULTS_DIR.mkdir(exist_ok=True)
    report = {
        "pass_rate": pass_rate,
        "correct_count": correct_count,
        "total": len(judgements),
        "details": judgements,
    }
    with open(RESULTS_DIR / "llm_judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.log(f"\n{'='*80}")
    log.log(f"=== LLM Judge Summary ===")
    log.log(f"  Pass rate: {pass_rate:.0%} ({correct_count}/{len(judgements)})")
    log.log(f"  Threshold: {LLM_JUDGE_PASS_RATE:.0%}")
    log.log(f"  Status: {'PASS' if pass_rate >= LLM_JUDGE_PASS_RATE else 'FAIL'}")
    log.log(f"  Total time: {time.time() - t_start:.0f}s")
    log.log(f"\nLog saved to {log._f.name}")
    log.close()

    return {"pass_rate": pass_rate, "correct_count": correct_count, "details": judgements}


def test_llm_judge_pass_rate(judge_results):
    """LLM Judge pass rate >= 70%."""
    rate = judge_results["pass_rate"]
    total = len(judge_results["details"])
    correct = judge_results["correct_count"]
    print(f"\nLLM Judge: {rate:.0%} ({correct}/{total})")
    assert rate >= LLM_JUDGE_PASS_RATE, (
        f"LLM Judge pass rate {rate:.0%} ниже порога {LLM_JUDGE_PASS_RATE:.0%}"
    )


def test_strict_title_match(rag_results):
    """Строгая проверка: название ожидаемого фильма содержится в ответе."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = _Logger(LOGS_DIR / f"strict_match_{ts}.log")

    log.log(f"Strict Title Match started at {ts}")
    log.log(f"Samples: {len(rag_results)}, Threshold: {STRICT_MATCH_PASS_RATE:.0%}")

    matches = 0
    details = []
    for i, r in enumerate(rag_results):
        found = r["expected_title"].lower() in r["answer"].lower()
        if found:
            matches += 1
        status = "MATCH" if found else "MISS"
        log.log(
            f"  [{i+1}/{len(rag_results)}] [{status}]  "
            f"expected=\"{r['expected_title']}\"  "
            f"answer={_truncate(r['answer'], 80)}"
        )
        details.append(
            {
                "expected": r["expected_title"],
                "found_in_answer": found,
            }
        )

    rate = matches / len(rag_results) if rag_results else 0

    log.log(f"\n=== Strict Match Summary ===")
    log.log(f"  Match rate: {rate:.0%} ({matches}/{len(rag_results)})")
    log.log(f"  Threshold: {STRICT_MATCH_PASS_RATE:.0%}")
    log.log(f"  Status: {'PASS' if rate >= STRICT_MATCH_PASS_RATE else 'FAIL'}")
    log.log(f"\nLog saved to {log._f.name}")
    log.close()

    print(f"\nStrict title match: {rate:.0%} ({matches}/{len(rag_results)})")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "strict_match_results.json", "w", encoding="utf-8") as f:
        json.dump({"match_rate": rate, "matches": matches, "details": details}, f, ensure_ascii=False, indent=2)

    assert rate >= STRICT_MATCH_PASS_RATE, (
        f"Strict match rate {rate:.0%} ниже порога {STRICT_MATCH_PASS_RATE:.0%}"
    )
