from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from openai import OpenAI
from tavily import TavilyClient

from vector_store_manager import VectorStoreManager
from gent_tools import retrieve_game, evaluate_retrieval, game_web_search
from reporting import synthesize_answer, AnswerReport


class AgentState(str, Enum):
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    WEB_SEARCH = "web_search"
    SYNTHESIZE = "synthesize"
    DONE = "done"


def run_agent(
    question: str,
    vs: VectorStoreManager,
    openai_client: OpenAI,
    tavily_client: TavilyClient,
    k: int = 5,
) -> Dict[str, Any]:
    """
    State machine:
      RETRIEVE -> EVALUATE -> (SYNTHESIZE or WEB_SEARCH) -> SYNTHESIZE -> DONE
    """
    state = AgentState.RETRIEVE

    retrieval: Dict[str, Any] = {}
    evaluation: Dict[str, Any] = {}
    web: Dict[str, Any] = {}
    tool_trace: List[Dict[str, Any]] = []

    internal_hits: List[Dict[str, Any]] = []
    web_results: List[Dict[str, Any]] = []

    while state != AgentState.DONE:
        if state == AgentState.RETRIEVE:
            retrieval = retrieve_game(vs, question, k=k)
            internal_hits = retrieval.get("hits", [])
            tool_trace.append({"tool": "retrieve_game", "hits": len(internal_hits)})
            state = AgentState.EVALUATE

        elif state == AgentState.EVALUATE:
            evaluation = evaluate_retrieval(retrieval)
            tool_trace.append({"tool": "evaluate_retrieval", "output": evaluation})
            state = AgentState.SYNTHESIZE if evaluation.get("ok") else AgentState.WEB_SEARCH

        elif state == AgentState.WEB_SEARCH:
            web = game_web_search(tavily_client, question, max_results=5)
            web_results = web.get("results", [])
            tool_trace.append({"tool": "game_web_search", "results": len(web_results)})
            state = AgentState.SYNTHESIZE

        elif state == AgentState.SYNTHESIZE:
            report: AnswerReport = synthesize_answer(
                openai_client=openai_client,
                question=question,
                internal_hits=internal_hits,
                web_results=web_results,
                retrieval_score=float(evaluation.get("score", 0.0)),
                used_web=not evaluation.get("ok", False),
            )
            out = report.model_dump()
            out["tool_trace"] = tool_trace
            out["retrieval_eval"] = evaluation
            state = AgentState.DONE
            return out

    return {"error": "unexpected_state", "tool_trace": tool_trace}
