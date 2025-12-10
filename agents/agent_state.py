from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, TypedDict

from langgraph.graph import StateGraph, END

from llm_factory import get_llm
from mcp_client import MCPHTTPClient

from router import run_router
from planner import run_planner
from retriever import run_retriever
from answerer import run_answerer


# ---------- Core data models ----------


@dataclass
class ProductConstraints:
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    materials_include: List[str] | None = None
    materials_exclude: List[str] | None = None
    eco_friendly: Optional[bool] = None
    size: Optional[str] = None
    query_language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyFlags:
    chemical_safety: bool = False
    child_safety: bool = False
    surface_safety: bool = False
    possible_misuse: bool = False
    needs_professional_advice: bool = False
    other_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Intent:
    intent_type: Literal["product_search", "comparison", "chitchat", "unsafe", "unknown"]
    constraints: ProductConstraints
    safety_flags: SafetyFlags
    raw_intent_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["constraints"] = self.constraints.to_dict()
        data["safety_flags"] = self.safety_flags.to_dict()
        return data


@dataclass
class DataSourcePlan:
    use_private: bool = True
    use_live: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalPlan:
    data_sources: DataSourcePlan
    max_results: int = 5
    rerank: bool = True
    reconcile_conflicts: bool = True
    filters: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_sources"] = self.data_sources.to_dict()
        return data


@dataclass
class ProductResult:
    source: Literal["private", "web"]
    sku: str
    title: str
    price: Optional[float] = None
    rating: Optional[float] = None
    brand: Optional[str] = None
    ingredients: Optional[str] = None
    features: Optional[List[str]] = None
    url: Optional[str] = None
    doc_id: Optional[str] = None
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    source: Literal["private", "web"]
    doc_id: str
    url: Optional[str] = None
    label: Optional[str] = None
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentLog:
    agent_name: str
    timestamp: str
    reasoning: str
    decision: Dict[str, Any]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallLog:
    agent_name: str
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: str
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- LangGraph state ----------


class AgentState(TypedDict, total=False):
    # Input
    user_query: str
    transcript: Optional[str]

    # Intent and planning
    intent: Optional[Intent]
    retrieval_plan: Optional[RetrievalPlan]

    # Retrieval results
    retrieved_products: List[ProductResult]

    # Final outputs
    final_response: Optional[str]
    tts_summary: Optional[str]

    # Citations and logs
    citations: List[Citation]
    agent_logs: List[AgentLog]
    tool_calls: List[ToolCallLog]


# ---------- LangGraph graph builder ----------


def build_assistant_graph():
    """
    Build the LangGraph StateGraph that orchestrates:
      router -> planner -> retriever -> answerer
    """

    llm = get_llm()
    mcp = MCPHTTPClient()

    # Node functions with deps injected via closures, like in the notebooks
    def router_node(state: AgentState) -> AgentState:
        return run_router(state, llm)

    def planner_node(state: AgentState) -> AgentState:
        return run_planner(state)

    def retriever_node(state: AgentState) -> AgentState:
        return run_retriever(state, llm, mcp)

    def answerer_node(state: AgentState) -> AgentState:
        return run_answerer(state, llm)

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)

    graph.set_entry_point("router")

    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_assistant_graph()
    initial_state: AgentState = {
        "user_query": "Recommend an eco friendly stainless steel cleaner under 15 dollars.",
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    final_state = app.invoke(initial_state)
    print("Final answer:", final_state.get("final_response"))
    print("TTS summary:", final_state.get("tts_summary"))
