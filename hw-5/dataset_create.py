import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client

DATASET_NAME = "hw-6/films-retrieval"
ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "hw-6" / "data" / "data_films.csv"
N_ITEMS = int(os.getenv("N_ITEMS", "200"))

ID_CANDIDATES = [
    "id", "film_id", "kp_id", "kinopoisk_id", "kinopoiskId", "movie_id", "title_id"
]
TITLE_CANDIDATES = [
    "title", "name", "name_ru", "nameRu", "name_en", "nameEn", "original_title"
]
YEAR_CANDIDATES = ["year", "release_year", "start_year", "releaseYear"]
DESC_CANDIDATES = ["description", "plot", "synopsis", "overview", "short_description", "text_ru"]
GENRE_CANDIDATES = ["genres", "genre"]


def require_env() -> None:
    load_dotenv()
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(
            "Missing env vars: "
            + ", ".join(missing)
        )


def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def make_query(row, title_col, year_col, desc_col, genre_col) -> str:
    title = str(row.get(title_col, "")).strip() if title_col else ""
    year = str(row.get(year_col, "")).strip() if year_col else ""
    genres = str(row.get(genre_col, "")).strip() if genre_col else ""
    desc = str(row.get(desc_col, "")).strip() if desc_col else ""

    if desc and len(desc) >= 80:
        snippet = desc[:220].replace("\n", " ").strip()
        base = f"Найди фильм по описанию: {snippet}"
        if genres:
            base += f" Жанры: {genres}."
        if year and year != "nan":
            base += f" Год примерно: {year}."
        return base

    q = f"Фильм {title}".strip()
    if year and year != "nan":
        q += f" {year}"
    if genres:
        q += f", жанры: {genres}"
    return q.strip()


def main() -> None:
    require_env()
    langfuse = get_client()

    df = read_csv(CSV_PATH)
    print("CSV columns:", list(df.columns)[:50])

    id_col = pick_col(df.columns, ID_CANDIDATES)
    title_col = pick_col(df.columns, TITLE_CANDIDATES)
    year_col = pick_col(df.columns, YEAR_CANDIDATES)
    desc_col = pick_col(df.columns, DESC_CANDIDATES)
    genre_col = pick_col(df.columns, GENRE_CANDIDATES)

    print(
        "Detected columns:",
        {"id": id_col, "title": title_col, "year": year_col, "desc": desc_col, "genre": genre_col},
    )

    try:
        langfuse.get_dataset(name=DATASET_NAME)
        print(f"Dataset exists: {DATASET_NAME}")
    except Exception:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="HW-6 films retrieval dataset from data_films.csv",
        )
        print(f"Created dataset: {DATASET_NAME}")

    added = 0
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    for idx, row in df.head(N_ITEMS).iterrows():
        point_id = str(row.get(id_col, idx)) if id_col else str(idx)
        title = str(row.get(title_col, "")).strip() if title_col else ""
        year = str(row.get(year_col, "")).strip() if year_col else ""

        query = make_query(row, title_col, year_col, desc_col, genre_col)

        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"query": query},
            expected_output={
                "expected_point_id": point_id,
                "title": title,
                "year": year,
            },
            metadata={
                "source": str(CSV_PATH),
                "row_index": int(idx),
                "batch": now,
            },
        )
        added += 1

    langfuse.flush()
    langfuse.shutdown()
    print(f"Done. Added items: {added}. Dataset: {DATASET_NAME}")


if __name__ == "__main__":
    main()
