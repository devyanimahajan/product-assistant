from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

from llm_factory import get_llm
from mcp_client import MCPHTTPClient

# Lazy imports to avoid circular dependency
# from router import run_router
# from planner import run_planner
# from retriever import run_retriever
# from answerer import run_answerer


# ---------- Core data models ----------


@dataclass
class ProductConstraints:
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    product_name: Optional[str] = None  # Extracted product name for web search
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
    purchase_url: Optional[str] = None  # Reconciled web purchase link for catalog products
    web_price: Optional[float] = None   # Current web price if different from catalog

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
      router -> [conditional: product_search/comparison -> planner -> retriever -> answerer]
                            [other intents -> simple_response -> END]
    """
    
    # Import here to avoid circular dependency
    from router import run_router
    from planner import run_planner
    from retriever import run_retriever
    from answerer import run_answerer

    llm = get_llm()
    mcp = MCPHTTPClient()

    # Node functions with deps injected via closures
    def router_node(state: AgentState) -> AgentState:
        return run_router(state, llm)

    def planner_node(state: AgentState) -> AgentState:
        return run_planner(state)

    def retriever_node(state: AgentState) -> AgentState:
        return run_retriever(state, llm, mcp)

    def answerer_node(state: AgentState) -> AgentState:
        return run_answerer(state, llm)
    
    def simple_response_node(state: AgentState) -> AgentState:
        """Handle non-product intents with friendly messages."""
        intent = state.get("intent")
        intent_type = intent.intent_type if intent else "unknown"
        
        if intent_type == "chitchat":
            message = "I'm a product recommendation assistant. I can help you find household products, compare options, and answer questions about product features. What product are you looking for?"
        elif intent_type == "unsafe":
            message = "I detected safety concerns in your query. For safety-related questions about chemicals or hazardous materials, please consult a professional or refer to product safety documentation."
        elif intent_type == "unknown":
            message = "I didn't understand your request. I help with product recommendations and comparisons for household items. Could you please rephrase your question about a specific product you're looking for?"
        else:
            message = "I can help you find and compare household products. What are you looking for?"
        
        state["final_response"] = message
        state["tts_summary"] = message
        state["retrieved_products"] = []
        state["citations"] = []
        
        return state
    
    def should_retrieve_products(state: AgentState) -> str:
        """Decide if we should retrieve products or give a simple response."""
        intent = state.get("intent")
        if not intent:
            return "simple_response"
        
        # Only do product retrieval for product_search and comparison intents
        if intent.intent_type in ["product_search", "comparison"]:
            return "planner"
        else:
            return "simple_response"

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answerer", answerer_node)
    graph.add_node("simple_response", simple_response_node)

    graph.set_entry_point("router")

    # Conditional routing after router
    graph.add_conditional_edges(
        "router",
        should_retrieve_products,
        {
            "planner": "planner",
            "simple_response": "simple_response"
        }
    )
    
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)
    graph.add_edge("simple_response", END)

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
