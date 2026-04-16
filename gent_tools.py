from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from vector_store_manager import VectorStoreManager
from reporting import Source


# -----------------------
# TOOL 1: retrieve_game
# -----------------------
def retrieve_game(vs: VectorStoreManager, question: str, k: int = 5) -> Dict[str, Any]:
    hits = vs.query(question, k=k)
    return {
        "question": question,
        "hits": [
            {"id": h.id, "document": h.document, "metadata": h.metadata, "distance": h.distance}
            for h in hits
        ],
    }


# ---------------------------
# TOOL 2: evaluate_retrieval
# ---------------------------
def evaluate_retrieval(
    retrieval: Dict[str, Any],
    max_distance: float = 0.45,
    min_margin: float = 0.03,
) -> Dict[str, Any]:
    """
    Heuristic evaluator:
      - ok=False if no hits
      - ok=False if top distance too high (weak match)
      - ok=False if top2 too close (ambiguous)
    Returns:
      ok (bool), reason (str), score (0..1)
    """
    hits = retrieval.get("hits", [])
    if not hits:
        return {"ok": False, "reason": "no_hits", "score": 0.0}

    d0 = float(hits[0]["distance"])
    margin = float(hits[1]["distance"] - d0) if len(hits) > 1 else 1.0

    score = max(0.0, min(1.0, 1.0 - d0))

    if d0 > max_distance:
        return {"ok": False, "reason": f"distance_too_high(d0={d0:.3f})", "score": score}

    if margin < min_margin:
        return {"ok": False, "reason": f"ambiguous(margin={margin:.3f})", "score": score}

    return {"ok": True, "reason": "sufficient", "score": score}


# -------------------------
# TOOL 3: game_web_search
# -------------------------
def game_web_search(tavily: TavilyClient, question: str, max_results: int = 5) -> Dict[str, Any]:
    res = tavily.search(
        query=question,
        max_results=max_results,
        include_raw_content=True,
        include_images=False,
    )
    return {"question": question, "results": res.get("results", [])}


def build_source_list(internal_hits: List[Dict[str, Any]], web_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for h in internal_hits:
        sources.append(Source(kind="internal", id=h.get("id"), title=h.get("metadata", {}).get("title")).model_dump())
    for r in web_results:
        sources.append(Source(kind="web", url=r.get("url"), title=r.get("title")).model_dump())
    return sources
