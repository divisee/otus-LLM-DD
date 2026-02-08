# RAG Retrieval Evaluation

Модуль для оценки качества retrieval в RAG-системе подбора фильмов.

## Быстрый старт

```bash
cd hw-6
python rag_analysis/run_retrieval_eval.py --config config.yaml
```

**Результаты**: см. [EXPERIMENTS.md](EXPERIMENTS.md)

## Структура модуля

```
hw-6/rag_analysis/
├── README.md               # Этот файл
├── EXPERIMENTS.md          # Подробный отчёт с результатами и анализом ошибок
├── run_retrieval_eval.py   # Основной скрипт оценки (4 эксперимента)
├── generate_questions.py   # Генерация вопросов через LLM
├── analyze_text_ru.py      # Статистика длины текстов
├── data/
│   └── rag_questions.csv   # 100 вопросов для оценки
└── out/                    # Результаты экспериментов
    ├── vector_rag/
    ├── bm25/
    ├── hybrid_rrf/
    └── hybrid_dbsf/
```

## Эксперименты

| Метод | Описание | Hit@10 | MRR@10 | Latency |
|-------|----------|--------|--------|---------|
| `vector_rag` | Qdrant cosine similarity | 93% | 0.806 | 172ms |
| `bm25` | Лексический BM25 | 79% | 0.649 | 4.5ms |
| `hybrid_rrf` | RRF fusion | 94% | 0.791 | 176ms |
| `hybrid_dbsf` | **DBSF fusion** | **94%** | **0.816** | 176ms |

**Лучший результат**: `hybrid_dbsf` с Hit@10=94% и MRR@10=0.816

## Метрики

- **Hit@k / Recall@k**: целевой фильм в top-k
- **MRR@k**: Mean Reciprocal Rank (1/позиция)
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **Coverage**: доля уникальных документов в выдаче
- **Latency**: время запроса (ms)

## DBSF vs RRF

**RRF** (Reciprocal Rank Fusion):
```
score = 1/(c + rank_vector) + 1/(c + rank_bm25)
```

**DBSF** (Distribution-Based Score Fusion):
```
norm = (score - min) / (max - min)
score = 0.5 × norm_vector + 0.5 × norm_bm25
```

DBSF учитывает "уверенность" каждого retriever'а через нормализацию скоров.

## Источник данных

- **Файл**: `hw-6/data/data_films.csv`
- **Корпус**: 477 фильмов (Kinopoisk/IMDb)
- **Embeddings**: Ollama embeddinggemma
- **Vector Store**: Qdrant

## Датасет вопросов

100 вопросов разной сложности (1-5), сгенерированных через LLM:

| Difficulty | Пример вопроса | Ожидаемый фильм |
|------------|----------------|-----------------|
| 1 | "Помоги вспомнить фильм, где бандиты ведут философские беседы" | Криминальное чтиво |
| 3 | "Фильм, где команда ученых сражается с существом в холодной пустыне" | Нечто |
| 5 | "Фильм, где любовь влияет на премьер-министра" | Реальная любовь |

## Запуск экспериментов

```bash
# Полный прогон на 100 вопросах
python rag_analysis/run_retrieval_eval.py \
  --config config.yaml \
  --questions rag_analysis/data/rag_questions.csv \
  --top-k 10

# Быстрый тест на 10 вопросах
python rag_analysis/run_retrieval_eval.py --limit 10
```

Результаты сохраняются в `out/<experiment>/<run>/`:
- `summary.json` — агрегированные метрики
- `per_query.jsonl` — результаты по каждому запросу

## Дальнейшие улучшения

1. **Query expansion** через LLM перефразирование
2. **Re-ranker** с cross-encoder
3. **Лемматизация** для BM25 (pymorphy3)
4. **Тюнинг весов DBSF** (α, β)

Подробнее см. [EXPERIMENTS.md](EXPERIMENTS.md)
