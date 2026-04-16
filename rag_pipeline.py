
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from vector_store_manager import VectorStoreManager, SearchHit


def build_game_doc(game: Dict[str, Any]) -> str:
    """
    Convert a game JSON dict into a single document string for embedding + retrieval.
    Adjust the key names here to match your dataset schema.
    """
    title = game.get("title") or game.get("name") or "Unknown Title"
    developer = game.get("developer") or "Unknown"
    publisher = game.get("publisher") or "Unknown"
    release_date = game.get("release_date") or game.get("releaseDate") or "Unknown"
    platforms = game.get("platforms") or []
    genres = game.get("genres") or []
    description = game.get("description") or ""

    if isinstance(platforms, str):
        platforms = [platforms]
    if isinstance(genres, str):
        genres = [genres]

    return "\n".join(
        [
            f"Title: {title}",
            f"Developer: {developer}",
            f"Publisher: {publisher}",
            f"Release Date: {release_date}",
            f"Platforms: {', '.join(platforms) if platforms else 'Unknown'}",
            f"Genres: {', '.join(genres) if genres else 'Unknown'}",
            f"Description: {description}",
        ]
    )


def ingest_games_from_folder(vs: VectorStoreManager, folder: str) -> int:
    """
    Loads all *.json from folder and upserts into Chroma.
    Uses file name as fallback id.
    """
    paths = glob.glob(os.path.join(folder, "*.json"))
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    now = datetime.now(timezone.utc).isoformat()

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            game = json.load(f)

        game_id = game.get("game_id") or game.get("id") or os.path.splitext(os.path.basename(p))[0]
        doc = build_game_doc(game)

        meta = dict(game)
        meta["source_file"] = p
        meta["ingested_at_utc"] = now

        ids.append(str(game_id))
        docs.append(doc)
        metas.append(meta)

    if ids:
        vs.upsert(ids=ids, documents=docs, metadatas=metas)

    return len(ids)


def semantic_search(vs: VectorStoreManager, query: str, k: int = 5) -> List[SearchHit]:
    return vs.query(query_text=query, k=k)
