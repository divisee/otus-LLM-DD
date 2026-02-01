# HW-5 Langfuse Experiments

Набор демо-скриптов для работы с Langfuse: datasets, evaluator, observations, dashboards, LLM-as-a-judge и annotations queue.

## Требования

Заполните `hw-5/.env` или экспортируйте переменные:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL`

Дополнительно для retrieval-эксперимента:
- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `QDRANT_API_KEY` (если нужен)
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_API_KEY` (если нужен)

Для LLM-as-a-judge:
- `JUDGE_BASE_URL`
- `JUDGE_API_KEY`
- `JUDGE_MODEL` (опционально)

## Скрипты

- `hw-5/dataset_create.py` — создание dataset в Langfuse из `hw-6/data/data_films.csv`.
- `hw-5/experiment_retrieval.py` — retrieval experiment + custom evaluator (hit@k, MRR, title_match).
- `hw-5/llm_judge_demo.py` — LLM-as-a-judge для оценки соответствия.
- `hw-5/annotations_queue_demo.py` — создает trace ids для очереди разметки.
- `hw-5/observation_demo.py` — демонстрация разных типов observation.
- `hw-5/dashboard_demo.py` — генерирует метрики для дашборда.

## Запуск

```bash
python hw-5/dataset_create.py
python hw-5/experiment_retrieval.py
python hw-5/llm_judge_demo.py
python hw-5/annotations_queue_demo.py
python hw-5/observation_demo.py
python hw-5/dashboard_demo.py
```

## Выходные файлы

Скрипты пишут trace ids в `hw-5/out/*.json`.
