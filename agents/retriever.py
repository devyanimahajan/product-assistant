from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
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


def _load_prompt(filename: str) -> str:
    """Load prompt from file to avoid sync issues."""
    prompt_file = Path(__file__).parent.parent / "prompts" / filename
    try:
        with open(prompt_file, 'r') as f:
            content = f.read()
            # Skip header lines and extract the actual prompt
            lines = content.split('\n')
            prompt_lines = []
            skip_header = True
            for line in lines:
                if skip_header:
                    # Skip until we find actual prompt content
                    if line.strip() and not line.startswith('=') and not any(line.startswith(h) for h in ['RERANKER', 'PRODUCT', 'Role:']):
                        skip_header = False
                        prompt_lines.append(line)
                else:
                    # Stop at Location: line
                    if line.startswith('Location:'):
                        break
                    prompt_lines.append(line)
            
            result = '\n'.join(prompt_lines).strip()
            if result:
                return result
            return content.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")


# Load prompts once at module level
RERANKER_PROMPT = _load_prompt("reranker_prompt.txt")
PRODUCT_PARAM_EXTRACTION_PROMPT = _load_prompt("product_param_extraction_prompt.txt")


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Retriever: no JSON found in rerank output")
    return json.loads(match.group(0))


def _construct_web_query(query: str, constraints: Dict[str, Any]) -> str:
    """
    Build optimized web search query using LLM-extracted constraints.
    
    Uses structured fields extracted by Router LLM:
    - brand: "Fisher-Price"
    - product_name: "Think and Learn toy"
    - category: "toys"
    
    Builds: "Fisher-Price Think and Learn toy toys price buy"
    """
    parts = []
    
    # Add brand first if available (helps with product disambiguation)
    if constraints.get("brand"):
        parts.append(str(constraints["brand"]))
    
    # Add LLM-extracted product name (preferred over regex)
    if constraints.get("product_name"):
        parts.append(str(constraints["product_name"]))
    
    # Add category if available and not already in product name
    if constraints.get("category"):
        category = str(constraints["category"])
        product_name = constraints.get("product_name", "").lower()
        if category.lower() not in product_name:
            parts.append(category)
    
    # Build final query
    search_query = " ".join(parts)
    
    # Add shopping keywords if not present
    if search_query and 'buy' not in search_query.lower() and 'price' not in search_query.lower():
        search_query += " price buy"
    
    print(f"[RETRIEVER] Web search query: '{search_query}' (from constraints)")
    
    return search_query


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
        "max_results": k,
        "filters": filters,
    }

    print(f"[RETRIEVER] Calling rag.search with query='{query}', max_results={k}, filters={filters}")

    timestamp = datetime.utcnow().isoformat()
    raw_response: Dict[str, Any] | None = None
    results: List[ProductResult] = []
    logs: List[ToolCallLog] = []

    try:
        raw_response = mcp_client.call_tool("rag.search", args)
        print(f"[RETRIEVER] MCP Response: success={raw_response.get('success')}, count={raw_response.get('count')}, has_error={'error' in raw_response}")
        
        # Check for errors in response
        if isinstance(raw_response, dict) and 'error' in raw_response:
            print(f"[RETRIEVER ERROR] {raw_response.get('error')}")
            raw_response = {"error": raw_response.get('error')}
        else:
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
            
            if results:
                print(f"[RETRIEVER] Retrieved {len(results)} products from private catalog")
    except Exception as e:
        print(f"[RETRIEVER ERROR] {type(e).__name__}: {e}")
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
    args = {"query": search_query, "max_results": k}

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
            SystemMessage(content=RERANKER_PROMPT),
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


def _extract_product_search_params(product: ProductResult, llm: Any) -> Dict[str, Any]:
    """
    Use LLM to extract structured search parameters from a catalog product.
    Returns constraints dict compatible with _construct_web_query().
    """
    # Format the prompt template with product details
    prompt = PRODUCT_PARAM_EXTRACTION_PROMPT.format(
        brand=product.brand or 'Unknown',
        title=product.title
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # Try to extract JSON from response
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        data = json.loads(content.strip())
        
        return {
            "brand": product.brand,
            "product_name": data.get("product_name", ""),
            "category": data.get("category", "")
        }
    except Exception as e:
        print(f"[RETRIEVER] LLM extraction failed: {e}, using fallback")
        # Fallback: use first 5 words of title
        words = product.title.split()[:5]
        return {
            "brand": product.brand,
            "product_name": " ".join(words),
            "category": ""
        }


def _find_best_web_match(
    catalog_product: ProductResult,
    web_results: List[Dict[str, Any]],
) -> tuple[Optional[str], Optional[float]]:
    """
    Simple heuristic: Pick result with most precise price (most decimal places).
    
    Rationale:
    - More sig figs = more specific product match (e.g., $12.99 > $100.0 > $20)
    - Trusts Brave API's relevance ranking (results already sorted by relevance)
    - If prices tie on precision, first result wins (highest Brave ranking)
    
    Returns (purchase_url, web_price) tuple.
    """
    if not web_results:
        return None, None
    
    best_match = None
    best_precision = -1
    
    for result in web_results:
        url = result.get("url")
        price = result.get("price")
        
        if not url or price is None:
            continue
        
        # Count decimal places as proxy for price precision
        price_str = str(float(price))
        if '.' in price_str:
            decimals = len(price_str.rstrip('0').split('.')[1]) if '.' in price_str.rstrip('0') else 0
        else:
            decimals = 0
        
        # Pick result with most precise price (or first if tie, trusting Brave ranking)
        if decimals > best_precision:
            best_precision = decimals
            best_match = (url, price)
            print(f"[RETRIEVER] Match candidate: {result.get('title')[:50]}... (price: ${price}, precision: {decimals} decimals)")
    
    return best_match if best_match else (None, None)


def run_retriever(
    state: AgentState,
    llm: Any,
    mcp_client: MCPHTTPClient,
) -> AgentState:
    """
    LangGraph node: TRUE RECONCILIATION APPROACH
    
    1. Get 3 products from private catalog (the recommendations)
    2. For EACH product, search web for live purchase link
    3. Attach best matching URL + price to each catalog product
    """

    query = state.get("user_query", "")
    intent = state.get("intent")
    plan: RetrievalPlan | None = state.get("retrieval_plan")  # type: ignore[assignment]

    constraints = intent.constraints.to_dict() if intent else {}
    plan_dict = plan.to_dict() if plan else {}

    results: List[ProductResult] = []
    tool_calls: List[ToolCallLog] = state.get("tool_calls", [])

    # Step 1: Get recommended products from private catalog (limit to 3 for reconciliation)
    max_catalog_results = 3 if (plan and plan.data_sources.use_live) else (plan.max_results if plan else 5)
    private_results, private_logs = _retrieve_private(query, constraints, plan, mcp_client)
    private_results = private_results[:max_catalog_results]  # Limit to 3 for reconciliation
    tool_calls.extend(private_logs)
    
    print(f"[RETRIEVER] Retrieved {len(private_results)} products from catalog")

    # Step 2: If use_live, reconcile each catalog product with web search
    if plan and plan.data_sources.use_live and private_results:
        print(f"[RETRIEVER] use_live=True, reconciling {len(private_results)} products with web search")
        
        for product in private_results:
            # Extract structured search params for this product using LLM
            print(f"[RETRIEVER] Extracting search params for: {product.title[:60]}...")
            product_params = _extract_product_search_params(product, llm)
            
            # Use the proven _construct_web_query function
            search_query = _construct_web_query("", product_params)
            
            print(f"[RETRIEVER] Web search query: '{search_query}'")
            
            # Call web search for this specific product (increased to 10 for better results)
            args = {"query": search_query, "max_results": 10}
            timestamp = datetime.utcnow().isoformat()
            
            try:
                raw_response = mcp_client.call_tool("web.search", args)
                web_items = raw_response.get("results", []) if isinstance(raw_response, dict) else []
                
                # Find best matching web result
                purchase_url, web_price = _find_best_web_match(product, web_items)
                
                if purchase_url:
                    product.purchase_url = purchase_url
                    product.web_price = web_price
                    print(f"[RETRIEVER] ✓ Found purchase link for {product.title}: {purchase_url}")
                else:
                    print(f"[RETRIEVER] ✗ No matching purchase link found for {product.title}")
                
                tool_calls.append(
                    ToolCallLog(
                        agent_name="retriever",
                        tool_name="web.search",
                        arguments=args,
                        timestamp=timestamp,
                        raw_response=raw_response,
                    )
                )
            except Exception as e:
                print(f"[RETRIEVER ERROR] Web search failed for {product.title}: {e}")
        
        results = private_results
    else:
        print(f"[RETRIEVER] Skipping reconciliation (use_live={plan.data_sources.use_live if plan else None})")
        results = private_results

    if plan and plan.rerank and results:
        results = _rerank_results(query, results, llm)

    logs = state.get("agent_logs", [])
    reconciled_count = sum(1 for r in results if r.purchase_url)
    logs.append(
        AgentLog(
            agent_name="retriever",
            timestamp=datetime.utcnow().isoformat(),
            reasoning=f"Retriever fetched {len(results)} products from catalog and reconciled {reconciled_count} with live purchase links.",
            decision={
                "plan": plan_dict,
                "num_results": len(results),
                "reconciled": reconciled_count,
                "sources": ["private_with_live_links" if reconciled_count > 0 else "private"],
            },
            confidence=0.9,
        )
    )

    state["retrieved_products"] = results
    state["agent_logs"] = logs
    state["tool_calls"] = tool_calls
    return state
