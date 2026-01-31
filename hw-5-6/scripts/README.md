# Movies ingestion

Действия для импорта и веторизации в бд:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
python ingest_movies.py --config ../config.yaml --limit 100
```

Скрипт читает `data/data_films.csv`, строит эмбеддинги по полю `Description Kinopoisk` через Ollama и пересоздает коллекцию `movies` в Qdrant.
