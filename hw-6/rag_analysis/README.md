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
├── run_retrieval_eval.py   # Оценка: vector, BM25, RRF, DBSF
├── ingest_chunked.py       # Индексация с рекурсивным чанкингом
├── eval_chunked.py         # Сравнение: без чанкинга vs с чанкингом
├── generate_questions.py   # Генерация вопросов через LLM
├── analyze_text_ru.py      # Статистика длины текстов
├── data/
│   └── rag_questions.csv   # 100 вопросов для оценки
└── out/                    # Результаты экспериментов
    ├── vector_rag/
    ├── bm25/
    ├── hybrid_rrf/
    ├── hybrid_dbsf/
    ├── vector_no_chunk/
    └── vector_chunked/
```

## Эксперименты

| Метод | Описание | Hit@10 | MRR@10 | Latency |
|-------|----------|--------|--------|---------|
| `vector_rag` | Qdrant cosine similarity | 93% | 0.806 | 308ms |
| `bm25` | Server-side BM25 (sparse vectors) | 67% | 0.551 | 7.1ms |
| `hybrid_rrf` | RRF fusion | 95% | 0.751 | 315ms |
| `hybrid_dbsf` | **DBSF fusion** | **95%** | **0.762** | 315ms |
| `vector_chunked` | **С чанкингом (500/100)** | **96%** | **0.827** | 177ms |

**Лучший результат**: `vector_chunked` с Hit@10=96% и MRR@10=0.827

### Эксперимент с чанкингом

Рекурсивный чанкинг улучшает результаты:

| Метрика | Без чанкинга | С чанкингом | Разница |
|---------|--------------|-------------|---------|
| Hit@10 | 93% | **96%** | +3% ✅ |
| Not Found | 6% | **4%** | -2% ✅ |

**Коллекции в Qdrant:**
- `movies`: 477 points (без чанкинга, полный текст описания)
- `movies_chunked`: 634 points (с чанкингом 500/100)

**Описание эксперимента:**
1. Создали коллекцию `movies_chunked` с рекурсивным чанкингом (chunk_size=500, overlap=100)
2. Для каждого фильма описание разбивается на чанки по 500 символов с перекрытием 100 символов
3. При поиске по chunked-коллекции применяется дедупликация — возвращается лучший чанк для каждого фильма
4. Сравнили метрики retrieval на 100 вопросах

Скрипты:
- `ingest_chunked.py` — индексация с чанкингом
- `eval_chunked.py` — сравнение коллекций

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

## Sparse vs Dense Vectors

В системе используются два типа векторов для retrieval:

**Dense Vectors** (плотные векторы):
- Генерируются локально через Ollama (embeddinggemma)
- Размерность: 1024
- Хранятся в Qdrant как `dense` vector
- Используются для семантического поиска по сходству (cosine similarity)
- Преимущества: высокая точность для семантических запросов
- Недостатки: высокая latency при генерации эмбеддингов

**Sparse Vectors** (разреженные векторы):
- Генерируются server-side в Qdrant с моделью `Qdrant/bm25`
- Представляют текст как sparse вектор (индексы слов + веса)
- Хранятся в Qdrant как `bm25` sparse vector
- Используются для лексического поиска (TF-IDF подобный)
- Преимущества: быстрая генерация, низкая latency
- Недостатки: менее точны для семантических запросов

**Гибридный поиск**: комбинация dense и sparse через RRF/DBSF для лучших результатов.

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
