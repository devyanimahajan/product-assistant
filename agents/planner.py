from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from agent_state import AgentState, RetrievalPlan, DataSourcePlan, AgentLog


def _should_use_live(user_query: str) -> bool:
    lower = user_query.lower()
    trigger_words = [
        "latest",
        "current",
        "now",
        "in stock",
        "availability",
        "today",
        "right now",
        "price today",
        "price",  # Any price query should check web for current pricing
        "cost",   # Similar to price
    ]
    should_use = any(word in lower for word in trigger_words)
    print(f"[PLANNER] Checking live search trigger for query: '{user_query}'")
    print(f"[PLANNER] Trigger words found: {[word for word in trigger_words if word in lower]}")
    print(f"[PLANNER] use_live = {should_use}")
    return should_use


def _build_filters(intent: Any | None) -> Dict[str, Any]:
    """Build filters that are supported by rag_tool.
    
    Note: Only brand filter is used. Category matching is unreliable due to
    nested category structures in the database (e.g., "Toys & Games | Hobbies | Paints").
    Semantic search via embeddings handles category/type matching better.
    """
    filters: Dict[str, Any] = {}
    if intent is None:
        return filters

    constraints = intent.constraints
    # Only use brand filter - it's more reliable for exact matches
    if constraints.brand:
        filters["brand"] = constraints.brand
    
    # Category and eco_friendly not used - semantic search handles these better
    return filters


def run_planner(state: AgentState) -> AgentState:
    """LangGraph node: decide data sources, filters, and retrieval plan."""

    user_query = state.get("user_query", "")
    intent = state.get("intent")

    use_live = _should_use_live(user_query)
    filters = _build_filters(intent)

    print(f"[PLANNER] Building plan with use_live={use_live}")

    data_sources = DataSourcePlan(
        use_private=True,
        use_live=use_live,
        notes="Use private Amazon 2020 slice by default; use web.search when user asks for current price or availability.",
    )

    plan = RetrievalPlan(
        data_sources=data_sources,
        max_results=5,
        rerank=True,
        reconcile_conflicts=True,
        filters=filters or {},
    )

    logs = state.get("agent_logs", [])
    logs.append(
        AgentLog(
            agent_name="planner",
            timestamp=datetime.utcnow().isoformat(),
            reasoning="Planner built retrieval plan from intent and user query.",
            decision=plan.to_dict(),
            confidence=0.9,
        )
    )

    state["retrieval_plan"] = plan
    state["agent_logs"] = logs
    return state
