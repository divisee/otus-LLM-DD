import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_payload(row: pd.Series) -> dict:
    payload = {
        "title": row.get("Title"),
        "kinopoisk_id": row.get("kinopoiskId"),
        "imdb_id": row.get("imdbId"),
        "year": row.get("Year"),
        "rating_kinopoisk": row.get("Rating Kinopoisk"),
        "rating_imdb": row.get("Rating Imdb"),
        "age_Limit": row.get("Age Limit"),
        "genres": row.get("Genres"),
        "country": row.get("Country"),
        "director": row.get("Director"),
        "text_ru": row.get("Description Kinopoisk"),
        "text_en": row.get("Description Imdb"),
    }
    return {k: normalize_value(v) for k, v in payload.items()}


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if "embeddings" in data:
        return data["embeddings"]
    if "embedding" in data:
        return [data["embedding"]]
    raise ValueError("Unexpected Ollama embed response format")


def ingest(config_path: Path, limit: int | None) -> None:
    config = load_config(config_path)

    csv_path = Path(config["data"]["csv_path"])
    if not csv_path.is_absolute():
        csv_path = config_path.resolve().parent / csv_path
    text_column = config["data"].get("text_column", "Description Kinopoisk")
    batch_size = int(config["ingest"].get("batch_size", 64))
    distance = config["ingest"].get("distance", "cosine").lower()

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    df = df[df[text_column].notna()].copy()
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows with description found")

    qdrant_cfg = config["qdrant"]
    client = QdrantClient(url=qdrant_cfg["url"], api_key=qdrant_cfg.get("api_key") or None)

    ollama_cfg = config["ollama"]
    model = ollama_cfg["embedding_model"]
    base_url = ollama_cfg["base_url"]

    first_texts = df[text_column].iloc[:batch_size].tolist()
    first_embeddings = embed_texts(base_url, model, first_texts)
    vector_size = len(first_embeddings[0])

    distance_map = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclidean": Distance.EUCLID,
    }
    collection_name = qdrant_cfg["collection_name"]
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_map[distance]),
    )

    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        texts = df[text_column].iloc[start:end].tolist()
        embeddings = embed_texts(base_url, model, texts)

        points = []
        for offset, embedding in enumerate(embeddings):
            row = df.iloc[start + offset]
            payload = build_payload(row)
            points.append(
                PointStruct(
                    id=start + offset,
                    vector=embedding,
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)

        progress = int((end / total) * 100)
        print(f"\rProgress: {progress}% ({end}/{total})", end="", flush=True)

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Kinopoisk/IMDb movies into Qdrant")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for a test run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(args.config, args.limit)
