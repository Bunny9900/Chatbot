from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


class Source(BaseModel):
    kind: str = Field(..., description="internal|web")
    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None


class AnswerReport(BaseModel):
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    fields: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Source] = Field(default_factory=list)
    notes: Optional[str] = None


def synthesize_answer(
    openai_client: OpenAI,
    question: str,
    internal_hits: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
    retrieval_score: float,
    used_web: bool,
) -> AnswerReport:
    """
    Produces a clean, structured answer.
    Uses internal docs first; if web_results exists, includes them too.
    """

    internal_context = "\n\n".join(
        [f"[INTERNAL:{h['id']}]\n{h['document']}" for h in internal_hits[:5]]
    )

    web_context = "\n\n".join(
        [
            f"[WEB:{r.get('url')}]\nTITLE: {r.get('title')}\nCONTENT:\n{(r.get('raw_content') or '')[:2000]}"
            for r in web_results[:3]
        ]
    )

    # Prompt template: SYSTEM / CONTEXT / OUTPUT
    system = (
        "You are UdaPlay, an AI research agent for video games.\n"
        "Use ONLY the provided context. If info is missing, say what's missing.\n"
        "Return a JSON object that matches the AnswerReport schema."
    )

    user = (
        f"SYSTEM: {question}\n"
        f"########################\n"
        f"CONTEXT:\n{internal_context}\n\n{web_context}\n"
        f"########################\n"
        f"OUTPUT:\n"
        f"Return JSON with keys:\n"
        f"- question (string)\n"
        f"- answer (string)\n"
        f"- confidence (0..1)\n"
        f"- fields (object; include developer/publisher/release_date/platforms/genres/description when available)\n"
        f"- sources (array of objects: kind=internal/web, id/url, title)\n"
        f"- notes (string, optional)\n"
        f"\nExtra guidance:\n"
        f"- Prefer internal hits when they contain the answer.\n"
        f"- If web is used, cite URLs.\n"
        f"- Set confidence higher when context directly states the answer.\n"
    )

    resp = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={"format": {"type": "json"}},
    )

    data = json.loads(resp.output_text)

    # If the model didn't include sources, we add basic ones:
    if not data.get("sources"):
        sources: List[Source] = []
        for h in internal_hits[:5]:
            sources.append(Source(kind="internal", id=h.get("id"), title=h.get("metadata", {}).get("title")))
        for r in web_results[:3]:
            sources.append(Source(kind="web", url=r.get("url"), title=r.get("title")))
        data["sources"] = [s.model_dump() for s in sources]

    # If model didn't include question, set it.
    data["question"] = data.get("question") or question

    # Optional: nudge confidence using retrieval score + web usage
    # (keeps it simple; model still controls final)
    if "confidence" not in data:
        base = retrieval_score if not used_web else max(0.4, retrieval_score)
        data["confidence"] = max(0.0, min(1.0, float(base)))

    return AnswerReport(**data)
