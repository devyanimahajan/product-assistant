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


class RetrieverAgent:
    """
    Agentic RAG retriever that calls MCP tools:
      - rag.search for private Amazon 2020 catalog
      - web.search for live web comparisons
    Uses the LLM for reranking only.
    """

    def __init__(self, llm: Any, mcp_client: Optional[MCPHTTPClient] = None) -> None:
        self.llm = llm
        self.mcp_client = mcp_client or MCPHTTPClient()
        self.name = "retriever"

    # ---------- Public entrypoint ----------

    def run(self, state: AgentState) -> AgentState:
        query = state.get("user_query", "")
        intent = state.get("intent")
        plan: RetrievalPlan = state.get("retrieval_plan")  # type: ignore[assignment]

        constraints = intent.constraints.to_dict() if intent else {}
        plan_dict = plan.to_dict() if plan else {}

        results: List[ProductResult] = []
        tool_calls: List[ToolCallLog] = state.get("tool_calls", [])

        # 1. Private retrieval via rag.search
        private_results, private_call_logs = self._retrieve_private(query, constraints, plan)
        results.extend(private_results)
        tool_calls.extend(private_call_logs)

        # 2. Optional live retrieval via web.search
        if plan and plan.data_sources.use_live:
            web_results, web_call_logs = self._retrieve_web(query, constraints, plan)
            results.extend(web_results)
            tool_calls.extend(web_call_logs)

        # 3. Optional rerank using LLM
        if plan and plan.rerank and results:
            results = self._rerank_results(query, results)

        logs = state.get("agent_logs", [])
        logs.append(
            AgentLog(
                agent_name=self.name,
                timestamp=datetime.utcnow().isoformat(),
                reasoning=f"Retrieved {len(results)} products via rag.search and optional web.search.",
                decision={
                    "plan": plan_dict,
                    "num_results": len(results),
                    "sources": list(sorted({r.source for r in results})),
                },
                confidence=0.9,
            )
        )

        state["retrieved_products"] = results
        state["agent_logs"] = logs
        state["tool_calls"] = tool_calls
        return state

    # ---------- Private retrieval via rag.search ----------

    def _retrieve_private(
        self,
        query: str,
        constraints: Dict[str, Any],
        plan: Optional[RetrievalPlan],
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
            raw_response = self.mcp_client.call_tool("rag.search", args)
            items = raw_response.get("results", []) if isinstance(raw_response, dict) else raw_response

            for item in items or []:
                # rag.search contract: {sku, title, price, rating, brand?, ingredients?, doc_id, features?, score?}
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
                agent_name=self.name,
                tool_name="rag.search",
                arguments=args,
                timestamp=timestamp,
                raw_response=raw_response,
            )
        )

        return results, logs

    # ---------- Web retrieval via web.search ----------

    def _construct_web_query(self, query: str, constraints: Dict[str, Any]) -> str:
        parts = [query]
        if constraints.get("brand"):
            parts.append(str(constraints["brand"]))
        if constraints.get("category"):
            parts.append(str(constraints["category"]))
        return " ".join(parts)

    def _extract_price(self, item: Dict[str, Any]) -> Optional[float]:
        price = item.get("price")
        if price is None:
            return None
        try:
            return float(price)
        except (TypeError, ValueError):
            return None

    def _retrieve_web(
        self,
        query: str,
        constraints: Dict[str, Any],
        plan: Optional[RetrievalPlan],
    ) -> tuple[List[ProductResult], List[ToolCallLog]]:
        k = plan.max_results if plan else 5

        search_query = self._construct_web_query(query, constraints)
        args = {"query": search_query, "k": k}

        timestamp = datetime.utcnow().isoformat()
        raw_response: Dict[str, Any] | None = None
        results: List[ProductResult] = []
        logs: List[ToolCallLog] = []

        try:
            raw_response = self.mcp_client.call_tool("web.search", args)
            items = raw_response.get("results", []) if isinstance(raw_response, dict) else raw_response

            for item in (items or [])[:k]:
                # web.search contract: {title, url, snippet, price?, availability?, rating?, brand?, sku?, score?}
                res = ProductResult(
                    source="web",
                    sku=item.get("sku") or item.get("url", ""),
                    title=item.get("title", ""),
                    price=self._extract_price(item),
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
                agent_name=self.name,
                tool_name="web.search",
                arguments=args,
                timestamp=timestamp,
                raw_response=raw_response,
            )
        )

        return results, logs

    # ---------- Reranking with LLM ----------

    def _extract_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("RetrieverAgent: no JSON found in rerank output")
        return json.loads(match.group(0))

    def _rerank_results(
        self,
        query: str,
        results: List[ProductResult],
    ) -> List[ProductResult]:
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

        prompt_payload = {
            "query": query,
            "items": items,
        }

        response = self.llm.invoke(
            [
                SystemMessage(content="You rank products by relevance to the query and constraints. Return JSON with a field 'ranked_indices' as a list of integer indices."),
                HumanMessage(content=json.dumps(prompt_payload, indent=2)),
            ]
        )

        try:
            data = self._extract_json(response.content)
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
