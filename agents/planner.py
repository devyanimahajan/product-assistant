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
    ]
    return any(word in lower for word in trigger_words)


def _build_filters(intent: Any | None) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if intent is None:
        return filters

    constraints = intent.constraints
    if constraints.category:
        filters["category"] = constraints.category
    if constraints.brand:
        filters["brand"] = constraints.brand
    if constraints.eco_friendly is True:
        filters["eco_friendly"] = True
    return filters


def run_planner(state: AgentState) -> AgentState:
    """LangGraph node: decide data sources, filters, and retrieval plan."""

    user_query = state.get("user_query", "")
    intent = state.get("intent")

    use_live = _should_use_live(user_query)
    filters = _build_filters(intent)

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
