# Movies ingestion

Действия для импорта и векторизации в бд:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
python ingest_movies.py --config ../config.yaml --limit 100
python ingest_movies.py --config ../config.yaml 
```

Скрипт читает `data/data_films.csv`, строит эмбеддинги по полю `Description Kinopoisk` через Ollama и пересоздает коллекцию `movies` в Qdrant.
