from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI


@dataclass
class SearchHit:
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float


class VectorStoreManager:
    """
    Reusable ChromaDB vector store manager:
      - persistent Chroma client + collection
      - embed with OpenAI
      - upsert + semantic query
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        collection_name: str = "games",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.oa = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.oa.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    def upsert(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        embeddings = self.embed(documents)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, k: int = 5) -> List[SearchHit]:
        q_emb = self.embed([query_text])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances", "ids"],
        )

        hits: List[SearchHit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(SearchHit(id=_id, document=doc, metadata=meta, distance=float(dist)))

        return hits
