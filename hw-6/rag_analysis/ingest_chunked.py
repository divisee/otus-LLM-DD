#!/usr/bin/env python3
"""Ingest movies with recursive chunking into a new Qdrant collection.

Creates collection 'movies_chunked' with:
- Chunk size: 500 characters
- Overlap: 100 characters
- Recursive text splitter
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def recursive_split(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Recursive text splitter: split by paragraphs, then sentences, then characters."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try splitting by paragraphs first
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""

            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # Start new chunk with overlap from previous
                    if chunks and overlap > 0:
                        # Take last 'overlap' chars from previous chunk
                        prev = chunks[-1]
                        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                        current = overlap_text + sep + part
                    else:
                        current = part

            if current:
                chunks.append(current)

            if chunks:
                return chunks

    # Fallback: character-level split
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


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


def ingest_chunked(config_path: Path, chunk_size: int, overlap: int, collection_name: str) -> None:
    config = load_config(config_path)

    # Load data
    csv_path = Path(config["data"]["csv_path"])
    if not csv_path.is_absolute():
        csv_path = config_path.resolve().parent / csv_path
    text_column = config["data"].get("text_column", "Description Kinopoisk")

    df = pd.read_csv(csv_path)
    df = df[df[text_column].notna()].copy()
    df[text_column] = df[text_column].astype(str)
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} movies from {csv_path}")

    # Create chunks
    all_chunks = []
    for idx, row in df.iterrows():
        text = row[text_column]
        title = row.get("Title", "")
        chunks = recursive_split(text, chunk_size, overlap)

        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
                "movie_idx": idx,
                "chunk_idx": chunk_idx,
                "total_chunks": len(chunks),
                "title": title,
                "text": chunk,
                "kinopoisk_id": normalize_value(row.get("kinopoiskId")),
                "imdb_id": normalize_value(row.get("imdbId")),
                "year": normalize_value(row.get("Year")),
                "rating_kinopoisk": normalize_value(row.get("Rating Kinopoisk")),
                "rating_imdb": normalize_value(row.get("Rating Imdb")),
                "genres": normalize_value(row.get("Genres")),
                "country": normalize_value(row.get("Country")),
                "director": normalize_value(row.get("Director")),
                "text_ru": text,  # Full text for reference
            })

    print(f"Created {len(all_chunks)} chunks from {len(df)} movies")
    print(f"Average chunks per movie: {len(all_chunks)/len(df):.2f}")

    # Setup Qdrant
    qdrant_cfg = config["qdrant"]
    client = QdrantClient(url=qdrant_cfg["url"], api_key=qdrant_cfg.get("api_key") or None)

    ollama_cfg = config["ollama"]
    model = ollama_cfg["embedding_model"]
    base_url = ollama_cfg["base_url"]

    # Get vector size from first embedding
    first_emb = embed_texts(base_url, model, [all_chunks[0]["text"]])
    vector_size = len(first_emb[0])
    print(f"Vector size: {vector_size}")

    # Create collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection_name}'")

    # Ingest in batches
    batch_size = 32
    total = len(all_chunks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = all_chunks[start:end]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(base_url, model, texts)

        points = []
        for offset, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point_id = start + offset
            payload = {k: v for k, v in chunk.items() if k != "text"}
            payload["chunk_text"] = chunk["text"]  # Store chunk text
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        print(f"\rProgress: {end}/{total} ({100*end//total}%)", end="", flush=True)

    print(f"\nDone! Collection '{collection_name}' has {total} points")

    # Show stats
    info = client.get_collection(collection_name)
    print(f"Collection info: {info.points_count} points")


def main():
    parser = argparse.ArgumentParser(description="Ingest movies with chunking")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "config.yaml")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--collection", type=str, default="movies_chunked")
    args = parser.parse_args()

    print(f"Config: {args.config}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    print(f"Collection: {args.collection}")
    print()

    ingest_chunked(args.config, args.chunk_size, args.overlap, args.collection)


if __name__ == "__main__":
    main()


