from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from agent_state import AgentState, RetrievalPlan, DataSourcePlan, AgentLog


class PlannerAgent:
    """
    Planner that decides which data sources to use and how many results to fetch.
    Simple rule-based logic plus optional model involvement if you extend it.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm
        self.name = "planner"

    def _should_use_live(self, user_query: str) -> bool:
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

    def _build_filters(self, intent: Any | None) -> Dict[str, Any]:
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

    def run(self, state: AgentState) -> AgentState:
        user_query = state.get("user_query", "")
        intent = state.get("intent")

        use_live = self._should_use_live(user_query)
        filters = self._build_filters(intent)

        data_sources = DataSourcePlan(
            use_private=True,
            use_live=use_live,
            notes="Use private Amazon 2020 slice by default; use web.search when user asks for current price or availability.",
        )

        retrieval_plan = RetrievalPlan(
            data_sources=data_sources,
            max_results=5,
            rerank=True,
            reconcile_conflicts=True,
            filters=filters,
        )

        logs = state.get("agent_logs", [])
        logs.append(
            AgentLog(
                agent_name=self.name,
                timestamp=datetime.utcnow().isoformat(),
                reasoning="Planner decided data sources, max_results, and filters based on intent and query.",
                decision=retrieval_plan.to_dict(),
                confidence=0.9,
            )
        )

        state["retrieval_plan"] = retrieval_plan
        state["agent_logs"] = logs
        return state
