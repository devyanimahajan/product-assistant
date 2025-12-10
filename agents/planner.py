"""
Planner Agent - Retrieval Strategy and Source Selection
"""
import json
from datetime import datetime
from typing import Dict, Any
from agent_state import AgentState, RetrievalPlan, DataSource, Intent, AgentLog


class PlannerAgent:
    """
    Second agent in the pipeline. Takes the Intent and plans:
    1. Which data sources to use (private catalog, live web, or both)
    2. What fields to retrieve
    3. How to query (vector query optimization, metadata filters)
    4. Whether to reconcile conflicts between sources
    """
    
    def __init__(self, llm):
        """
        Args:
            llm: Language model instance
        """
        self.llm = llm
        self.name = "planner"
        
    def __call__(self, state: AgentState) -> AgentState:
        """
        Create retrieval plan based on intent
        """
        intent = state["intent"]
        
        if not intent:
            raise ValueError("No intent found in state. Router must run first.")
        
        # Build planning prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(intent)
        
        # Call LLM
        response = self._call_llm_with_structure(system_prompt, user_prompt)
        
        # Parse into RetrievalPlan
        plan = self._parse_plan(response, intent)
        
        # Log reasoning
        log = AgentLog(
            agent_name=self.name,
            timestamp=datetime.now().isoformat(),
            reasoning=response.get("reasoning", ""),
            decision={
                "use_private": plan.data_sources.use_private,
                "use_live": plan.data_sources.use_live,
                "will_compare_sources": plan.reconcile_conflicts,
                "max_results": plan.max_results
            },
            confidence=response.get("confidence", 0.9)
        )
        
        # Update state
        state["retrieval_plan"] = plan
        state["agent_logs"].append(log.dict())
        
        return state
    
    def _build_system_prompt(self) -> str:
        return """You are a Planner Agent for an e-commerce retrieval system. Your job is to create an optimal retrieval plan.

**Available Data Sources:**
1. **Private Catalog**: Amazon Product Dataset 2020 (vector DB with metadata)
   - Fields: title, price, rating, brand, ingredients, category, sku
   - Reliable but potentially outdated (2020 data)
   - Fast, no API costs

2. **Live Web Search**: Current online data via MCP tool
   - Fields: price, availability, current_url
   - Most up-to-date but may have rate limits
   - Use for price checks and availability

**Decision Rules:**
- **Use ONLY private** for: general product search, recommendations, comparisons where currency doesn't matter
- **Use private + live** for: price checks, availability checks, "is this still available?", "how much does X cost now?"
- **Reconcile conflicts** when using both sources (check if prices match)

**Query Optimization:**
- Convert user intent to effective vector search query
- Extract key product attributes for semantic search
- Apply metadata filters (price range, brand, category) to narrow results

**Field Selection:**
- Always retrieve: title, price, rating, brand
- Optional: ingredients (for cleaning/beauty), category, features
- From live: price, availability (only if needed)

**Comparison Handling:**
- If user wants comparison, retrieve max_results = 3-5 products
- Set comparison_criteria based on user request

Return JSON with this structure:
{
  "reasoning": "Explain your data source decisions",
  "confidence": 0.95,
  "use_private": true,
  "use_live": false,
  "private_fields": ["title", "price", "rating", "brand"],
  "live_fields": [],
  "vector_query": "eco-friendly stainless steel kitchen cleaner",
  "metadata_filters": {
    "price_max": 15.0,
    "materials": ["eco-friendly"]
  },
  "max_results": 5,
  "rerank": true,
  "reconcile_conflicts": false
}"""

    def _build_user_prompt(self, intent: Intent) -> str:
        intent_summary = f"""
Intent Analysis:
- Query Type: {intent.query_type}
- Product: {intent.product_description}
- Budget: ${intent.constraints.budget_max} (max) / ${intent.constraints.budget_min} (min)
- Materials: {intent.constraints.materials}
- Brands: {intent.constraints.brands}
- Categories: {intent.constraints.categories}
- Needs Comparison: {intent.needs_comparison}
- Comparison Criteria: {intent.comparison_criteria}

Create an optimal retrieval plan. Return ONLY valid JSON, no additional text."""
        return intent_summary

    def _call_llm_with_structure(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean markdown code blocks
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content: {content}")
            # Fallback
            return {
                "reasoning": "Failed to parse, using defaults",
                "confidence": 0.5,
                "use_private": True,
                "use_live": False,
                "private_fields": ["title", "price", "rating", "brand"],
                "live_fields": [],
                "vector_query": "product search",
                "metadata_filters": {},
                "max_results": 5,
                "rerank": True,
                "reconcile_conflicts": False
            }
    
    def _parse_plan(self, response: Dict[str, Any], intent: Intent) -> RetrievalPlan:
        """
        Convert LLM response to RetrievalPlan object
        """
        # Build DataSource
        data_sources = DataSource(
            use_private=response.get("use_private", True),
            use_live=response.get("use_live", False),
            private_fields=response.get("private_fields", ["title", "price", "rating", "brand"]),
            live_fields=response.get("live_fields", [])
        )
        
        # Build metadata filters from constraints
        metadata_filters = response.get("metadata_filters", {})
        
        # Add budget constraints if not already present
        if intent.constraints.budget_max and "price_max" not in metadata_filters:
            metadata_filters["price_max"] = intent.constraints.budget_max
        if intent.constraints.budget_min and "price_min" not in metadata_filters:
            metadata_filters["price_min"] = intent.constraints.budget_min
            
        # Add brand filters
        if intent.constraints.brands and "brands" not in metadata_filters:
            metadata_filters["brands"] = intent.constraints.brands
            
        # Add category filters
        if intent.constraints.categories and "categories" not in metadata_filters:
            metadata_filters["categories"] = intent.constraints.categories
        
        # Build RetrievalPlan
        plan = RetrievalPlan(
            data_sources=data_sources,
            vector_query=response.get("vector_query", intent.product_description),
            metadata_filters=metadata_filters,
            max_results=response.get("max_results", 5 if intent.needs_comparison else 3),
            rerank=response.get("rerank", True),
            reconcile_conflicts=response.get("reconcile_conflicts", data_sources.use_live)
        )
        
        return plan


# Example usage
if __name__ == "__main__":
    from agent_state import Intent, ProductConstraints, SafetyFlags
    
    # Mock LLM
    class MockLLM:
        def invoke(self, messages):
            class Response:
                content = json.dumps({
                    "reasoning": "User query is straightforward product search. Use private catalog only since no need for live pricing.",
                    "confidence": 0.95,
                    "use_private": True,
                    "use_live": False,
                    "private_fields": ["title", "price", "rating", "brand", "ingredients"],
                    "live_fields": [],
                    "vector_query": "eco-friendly stainless steel kitchen cleaner",
                    "metadata_filters": {
                        "price_max": 15.0,
                        "categories": ["cleaning"]
                    },
                    "max_results": 5,
                    "rerank": True,
                    "reconcile_conflicts": False
                })
            return Response()
    
    # Test
    planner = PlannerAgent(MockLLM())
    
    # Create test intent
    test_intent = Intent(
        query_type="product_search",
        product_description="eco-friendly stainless steel cleaner",
        constraints=ProductConstraints(
            budget_max=15.0,
            materials=["eco-friendly", "stainless-steel"],
            keywords=["cleaner"]
        ),
        safety=SafetyFlags(is_safe=True),
        needs_comparison=False
    )
    
    test_state: AgentState = {
        "user_query": "I need an eco-friendly stainless-steel cleaner under $15",
        "transcript_timestamp": None,
        "intent": test_intent,
        "retrieval_plan": None,
        "retrieved_products": [],
        "final_response": "",
        "citations": [],
        "agent_logs": [],
        "tool_calls": []
    }
    
    result = planner(test_state)
    print("Retrieval Plan created:")
    print(f"  Use private: {result['retrieval_plan'].data_sources.use_private}")
    print(f"  Use live: {result['retrieval_plan'].data_sources.use_live}")
    print(f"  Vector query: {result['retrieval_plan'].vector_query}")
    print(f"  Filters: {result['retrieval_plan'].metadata_filters}")
    print(f"  Max results: {result['retrieval_plan'].max_results}")
