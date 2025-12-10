"""
Router Agent - Intent Classification and Constraint Extraction
"""
import json
import re
from datetime import datetime
from typing import Dict, Any
from agent_state import AgentState, Intent, ProductConstraints, SafetyFlags, AgentLog


class RouterAgent:
    """
    First agent in the pipeline. Analyzes user query to:
    1. Extract task and constraints (budget, material, brand)
    2. Identify safety flags
    3. Classify query type
    """
    
    def __init__(self, llm):
        """
        Args:
            llm: Language model instance (e.g., ChatAnthropic, ChatOpenAI)
        """
        self.llm = llm
        self.name = "router"
        
    def __call__(self, state: AgentState) -> AgentState:
        """
        Process user query and extract intent
        """
        user_query = state["user_query"]
        
        # Create prompt for intent extraction
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_query)
        
        # Call LLM with structured output
        response = self._call_llm_with_structure(system_prompt, user_prompt)
        
        # Parse response into Intent object
        intent = self._parse_intent(response)
        
        # Log reasoning
        log = AgentLog(
            agent_name=self.name,
            timestamp=datetime.now().isoformat(),
            reasoning=response.get("reasoning", ""),
            decision={
                "query_type": intent.query_type,
                "has_budget": intent.constraints.budget_max is not None,
                "has_brand_pref": len(intent.constraints.brands) > 0,
                "is_safe": intent.safety.is_safe
            },
            confidence=response.get("confidence", 0.9)
        )
        
        # Update state
        state["intent"] = intent
        if "agent_logs" not in state:
            state["agent_logs"] = []
        state["agent_logs"].append(log.dict())
        
        return state
    
    def _build_system_prompt(self) -> str:
        return """You are a Router Agent for an e-commerce voice assistant. Your job is to analyze user queries and extract:

1. **Query Type**: What is the user trying to do?
   - product_search: Looking for a specific product
   - comparison: Wants to compare multiple options
   - recommendation: Wants suggestions
   - availability_check: Checking if something exists/is in stock

2. **Constraints**: Extract all constraints
   - Budget: max/min price (extract numbers, convert to float)
   - Materials: eco-friendly, stainless-steel, plastic-free, organic, etc.
   - Brands: specific brand names mentioned
   - Categories: cleaning, kitchen, beauty, etc.
   - Exclusions: things to avoid

3. **Safety Flags**: Check for:
   - Harmful products (weapons, dangerous chemicals)
   - Age-restricted items (alcohol, tobacco)
   - Inappropriate requests
   - Products that could be misused

Be precise with number extraction. "$15" → 15.0, "under twenty dollars" → 20.0

Return your analysis as JSON matching this structure:
{
  "reasoning": "Brief explanation of your analysis",
  "confidence": 0.95,
  "query_type": "product_search",
  "product_description": "Clear description of what user wants",
  "constraints": {
    "budget_max": 15.0,
    "materials": ["stainless-steel", "eco-friendly"],
    "brands": [],
    "keywords": ["cleaner", "kitchen"]
  },
  "safety": {
    "is_safe": true,
    "concerns": []
  },
  "needs_comparison": false,
  "comparison_criteria": []
}"""

    def _build_user_prompt(self, user_query: str) -> str:
        return f"""User query: "{user_query}"

Analyze this query and extract the intent, constraints, and safety flags.
Return ONLY valid JSON, no additional text."""

    def _call_llm_with_structure(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response
        """
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        response = self.llm.invoke(messages)
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON (handle markdown code blocks)
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
            # Fallback: try to extract JSON from text
            print(f"JSON parse error: {e}")
            print(f"Content: {content}")
            # Return minimal safe structure
            return {
                "reasoning": "Failed to parse LLM response",
                "confidence": 0.5,
                "query_type": "product_search",
                "product_description": user_query,
                "constraints": {},
                "safety": {"is_safe": True, "concerns": []},
                "needs_comparison": False,
                "comparison_criteria": []
            }
    
    def _parse_intent(self, response: Dict[str, Any]) -> Intent:
        """
        Convert LLM response to Intent object
        """
        # Extract constraints
        constraints_data = response.get("constraints", {})
        constraints = ProductConstraints(
            budget_max=constraints_data.get("budget_max"),
            budget_min=constraints_data.get("budget_min"),
            materials=constraints_data.get("materials", []),
            brands=constraints_data.get("brands", []),
            categories=constraints_data.get("categories", []),
            keywords=constraints_data.get("keywords", []),
            exclude_materials=constraints_data.get("exclude_materials", []),
            exclude_brands=constraints_data.get("exclude_brands", [])
        )
        
        # Extract safety
        safety_data = response.get("safety", {})
        safety = SafetyFlags(
            is_safe=safety_data.get("is_safe", True),
            concerns=safety_data.get("concerns", []),
            requires_age_verification=safety_data.get("requires_age_verification", False),
            potentially_harmful=safety_data.get("potentially_harmful", False)
        )
        
        # Build Intent
        intent = Intent(
            query_type=response.get("query_type", "product_search"),
            product_description=response.get("product_description", ""),
            constraints=constraints,
            safety=safety,
            needs_comparison=response.get("needs_comparison", False),
            comparison_criteria=response.get("comparison_criteria", [])
        )
        
        return intent


# Example usage
if __name__ == "__main__":
    # Mock LLM for testing
    class MockLLM:
        def invoke(self, messages):
            class Response:
                content = json.dumps({
                    "reasoning": "User wants eco-friendly cleaner under $15",
                    "confidence": 0.95,
                    "query_type": "product_search",
                    "product_description": "eco-friendly stainless steel cleaner",
                    "constraints": {
                        "budget_max": 15.0,
                        "materials": ["eco-friendly", "stainless-steel"],
                        "keywords": ["cleaner"]
                    },
                    "safety": {"is_safe": True, "concerns": []},
                    "needs_comparison": False,
                    "comparison_criteria": []
                })
            return Response()
    
    # Test
    router = RouterAgent(MockLLM())
    test_state: AgentState = {
        "user_query": "I need an eco-friendly stainless-steel cleaner under $15",
        "transcript_timestamp": None,
        "intent": None,
        "retrieval_plan": None,
        "retrieved_products": [],
        "final_response": "",
        "citations": [],
        "agent_logs": [],
        "tool_calls": []
    }
    
    result = router(test_state)
    print("Intent extracted:")
    print(f"  Query type: {result['intent'].query_type}")
    print(f"  Budget: ${result['intent'].constraints.budget_max}")
    print(f"  Materials: {result['intent'].constraints.materials}")
    print(f"  Safe: {result['intent'].safety.is_safe}")
