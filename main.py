from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from vector_store_manager import VectorStoreManager
from rag_pipeline import ingest_games_from_folder
from state_machine import run_agent


def load_env():
    """
    Your screenshot shows config.env at workspace root (same level as project/).
    This loader tries:
      - ./config.env
      - ../config.env (if running from project/)
      - .env and ../.env as fallbacks
    """
    candidates = [
        Path("config.env"),
        Path("../config.env"),
        Path(".env"),
        Path("../.env"),
    ]

    for p in candidates:
        if p.exists():
            load_dotenv(p)
            break

    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
    assert os.getenv("TAVILY_API_KEY"), "Missing TAVILY_API_KEY"


def pick_data_dir() -> str:
    """
    If your dataset is in project/starter/, we try common locations automatically.
    """
    candidates = [
        Path("./data"),
        Path("./starter/data"),
        Path("../project/data"),
        Path("../project/starter/data"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return str(p)
    # default even if not present yet
    return "./data"


def main():
    load_env()

    oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tv = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    vs = VectorStoreManager(
        persist_dir="./chroma_store",
        collection_name="games",
        embedding_model="text-embedding-3-small",
    )

    data_dir = pick_data_dir()
    n = ingest_games_from_folder(vs, folder=data_dir)
    print(f"Ingested {n} JSON files from: {data_dir}")

    questions = [
        "Who developed FIFA 21?",
        "When was God of War Ragnarok released?",
        "What platform was Pokémon Red launched on?",
        "What is Rockstar Games working on right now?",
    ]

    for q in questions:
        print("\n" + "=" * 90)
        report = run_agent(q, vs, oa, tv, k=5)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
