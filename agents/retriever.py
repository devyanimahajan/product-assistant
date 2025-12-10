from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from agent_state import (
    AgentState,
    ProductResult,
    AgentLog,
    ToolCallLog,
    RetrievalPlan,
)
from mcp_client import MCPHTTPClient


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Retriever: no JSON found in rerank output")
    return json.loads(match.group(0))


def _construct_web_query(query: str, constraints: Dict[str, Any]) -> str:
    parts = [query]
    if constraints.get("brand"):
        parts.append(str(constraints["brand"]))
    if constraints.get("category"):
        parts.append(str(constraints["category"]))
    return " ".join(parts)


def _extract_price(item: Dict[str, Any]) -> Optional[float]:
    price = item.get("price")
    if price is None:
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def _retrieve_private(
    query: str,
    constraints: Dict[str, Any],
    plan: Optional[RetrievalPlan],
    mcp_client: MCPHTTPClient,
) -> tuple[List[ProductResult], List[ToolCallLog]]:
    k = plan.max_results if plan else 5
    filters = plan.filters if plan else {}

    args = {
        "query": query,
        "constraints": constraints,
        "filters": filters,
        "k": k,
    }

    timestamp = datetime.utcnow().isoformat()
    raw_response: Dict[str, Any] | None = None
    results: List[ProductResult] = []
    logs: List[ToolCallLog] = []

    try:
        raw_response = mcp_client.call_tool("rag.search", args)
        items = raw_response.get("results", []) if isinstance(raw_response, dict) else raw_response

        for item in items or []:
            res = ProductResult(
                source="private",
                sku=item.get("sku") or item.get("doc_id") or "",
                title=item.get("title", ""),
                price=item.get("price"),
                rating=item.get("rating"),
                brand=item.get("brand"),
                ingredients=item.get("ingredients"),
                features=item.get("features") or [],
                url=None,
                doc_id=item.get("doc_id"),
                score=float(item.get("score", 0.0)),
            )
            results.append(res)
    except Exception as e:
        raw_response = {"error": str(e)}

    logs.append(
        ToolCallLog(
            agent_name="retriever",
            tool_name="rag.search",
            arguments=args,
            timestamp=timestamp,
            raw_response=raw_response,
        )
    )
    return results, logs


def _retrieve_web(
    query: str,
    constraints: Dict[str, Any],
    plan: Optional[RetrievalPlan],
    mcp_client: MCPHTTPClient,
) -> tuple[List[ProductResult], List[ToolCallLog]]:
    k = plan.max_results if plan else 5

    search_query = _construct_web_query(query, constraints)
    args = {"query": search_query, "k": k}

    timestamp = datetime.utcnow().isoformat()
    raw_response: Dict[str, Any] | None = None
    results: List[ProductResult] = []
    logs: List[ToolCallLog] = []

    try:
        raw_response = mcp_client.call_tool("web.search", args)
        items = raw_response.get("results", []) if isinstance(raw_response, dict) else raw_response

        for item in (items or [])[:k]:
            res = ProductResult(
                source="web",
                sku=item.get("sku") or item.get("url", ""),
                title=item.get("title", ""),
                price=_extract_price(item),
                rating=item.get("rating"),
                brand=item.get("brand"),
                ingredients=None,
                features=item.get("features") or [],
                url=item.get("url"),
                doc_id=item.get("url"),
                score=float(item.get("score", 0.0)),
            )
            results.append(res)
    except Exception as e:
        raw_response = {"error": str(e)}

    logs.append(
        ToolCallLog(
            agent_name="retriever",
            tool_name="web.search",
            arguments=args,
            timestamp=timestamp,
            raw_response=raw_response,
        )
    )
    return results, logs


def _rerank_results(
    query: str,
    results: List[ProductResult],
    llm: Any,
) -> List[ProductResult]:
    """Use LLM as reranker, still inside a LangGraph node."""

    items = [
        {
            "index": i,
            "source": r.source,
            "title": r.title,
            "price": r.price,
            "rating": r.rating,
            "brand": r.brand,
        }
        for i, r in enumerate(results)
    ]

    payload = {"query": query, "items": items}

    response = llm.invoke(
        [
            SystemMessage(
                content="Rank products by relevance. Return JSON {\"ranked_indices\": [int, ...]} only."
            ),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )

    try:
        data = _extract_json(response.content)
    except Exception:
        return results

    indices = data.get("ranked_indices") or []
    if not isinstance(indices, list):
        return results

    ordered: List[ProductResult] = []
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(results):
            ordered.append(results[idx])

    return ordered or results


def run_retriever(
    state: AgentState,
    llm: Any,
    mcp_client: MCPHTTPClient,
) -> AgentState:
    """LangGraph node: call rag.search and web.search via MCP, then rerank."""

    query = state.get("user_query", "")
    intent = state.get("intent")
    plan: RetrievalPlan | None = state.get("retrieval_plan")  # type: ignore[assignment]

    constraints = intent.constraints.to_dict() if intent else {}
    plan_dict = plan.to_dict() if plan else {}

    results: List[ProductResult] = []
    tool_calls: List[ToolCallLog] = state.get("tool_calls", [])

    private_results, private_logs = _retrieve_private(query, constraints, plan, mcp_client)
    results.extend(private_results)
    tool_calls.extend(private_logs)

    if plan and plan.data_sources.use_live:
        web_results, web_logs = _retrieve_web(query, constraints, plan, mcp_client)
        results.extend(web_results)
        tool_calls.extend(web_logs)

    if plan and plan.rerank and results:
        results = _rerank_results(query, results, llm)

    logs = state.get("agent_logs", [])
    logs.append(
        AgentLog(
            agent_name="retriever",
            timestamp=datetime.utcnow().isoformat(),
            reasoning=f"Retriever fetched {len(results)} products from private and optional web sources.",
            decision={
                "plan": plan_dict,
                "num_results": len(results),
                "sources": sorted({r.source for r in results}),
            },
            confidence=0.9,
        )
    )

    state["retrieved_products"] = results
    state["agent_logs"] = logs
    state["tool_calls"] = tool_calls
    return state
