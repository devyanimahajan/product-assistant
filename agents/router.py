from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from agent_state import (
    AgentState,
    Intent,
    ProductConstraints,
    SafetyFlags,
    AgentLog,
)


def _load_prompt() -> str:
    """Load router prompt from file to avoid sync issues."""
    prompt_file = Path(__file__).parent.parent / "prompts" / "router_prompt.txt"
    try:
        with open(prompt_file, 'r') as f:
            content = f.read()
            # Skip header lines and extract the actual prompt
            lines = content.split('\n')
            # Find where the actual prompt starts (after header and separator)
            prompt_lines = []
            skip_header = True
            for line in lines:
                if skip_header:
                    # Skip until we find a line that's not a header or separator
                    if line.strip() and not line.startswith('=') and not line.startswith('ROUTER') and not line.startswith('Role:'):
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
            # If we got nothing, return the whole content
            return content.strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Router prompt file not found: {prompt_file}")


# Load prompt once at module level
SYSTEM_PROMPT = _load_prompt()


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Router: no JSON found in LLM output")
    
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print(f"[Router] JSON parsing error: {e}")
        print(f"[Router] Raw LLM output: {text}")
        # Return a safe default intent for malformed JSON
        return {
            "intent_type": "unknown",
            "constraints": {},
            "safety_flags": {
                "chemical_safety": False,
                "child_safety": False,
                "surface_safety": False,
                "possible_misuse": False,
                "needs_professional_advice": False,
                "other_notes": "Router failed to parse LLM response"
            }
        }


def _is_nonsense_query(query: str, constraints_data: Dict[str, Any]) -> bool:
    """
    Validate if a query classified as product_search is actually nonsense.
    
    Returns True if the query appears to be gibberish/nonsense.
    """
    query_lower = query.lower().strip()
    
    # Check if query is too short (less than 3 words)
    words = query_lower.split()
    if len(words) < 3:
        # Short queries like "blah" or "test test" are likely nonsense
        unique_words = set(words)
        if len(unique_words) <= 2:  # Repetitive
            return True
    
    # Check for repetitive nonsense patterns
    nonsense_patterns = [
        'blah', 'bla', 'test', 'asdf', 'qwer', 'xyz', 
        '123', '1005', 'hello world', 'foo', 'bar'
    ]
    
    # If query is just repetition of nonsense words
    if all(word in nonsense_patterns for word in words):
        return True
    
    # Check if all constraints are empty/null
    has_any_constraint = any([
        constraints_data.get("budget_min"),
        constraints_data.get("budget_max"),
        constraints_data.get("category"),
        constraints_data.get("brand"),
        constraints_data.get("materials_include"),
        constraints_data.get("materials_exclude"),
        constraints_data.get("eco_friendly") is not None,
        constraints_data.get("size")
    ])
    
    # If no constraints extracted and query looks repetitive/short, it's probably nonsense
    if not has_any_constraint:
        # Check for repetition (same word repeated)
        if len(set(words)) < len(words) / 2:  # More than half are duplicates
            return True
        
        # Check if query is just punctuation or numbers
        if query_lower.replace(' ', '').replace('.', '').replace(',', '').isdigit():
            return True
    
    return False


def run_router(state: AgentState, llm: Any) -> AgentState:
    """LangGraph node: classify intent and extract constraints and safety flags."""

    query = state.get("user_query") or state.get("transcript") or ""
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )
    raw = _extract_json(response.content)

    constraints_data = raw.get("constraints", {})
    safety_data = raw.get("safety_flags", {})
    
    # Validate product_search classification - override if it's nonsense
    if raw.get("intent_type") == "product_search":
        if _is_nonsense_query(query, constraints_data):
            print(f"[Router] Overriding product_search to unknown - query appears to be nonsense: '{query}'")
            raw["intent_type"] = "unknown"
            if safety_data.get("other_notes"):
                safety_data["other_notes"] += " | Query appears to be nonsense/gibberish"
            else:
                safety_data["other_notes"] = "Query appears to be nonsense/gibberish"

    constraints = ProductConstraints(
        budget_min=constraints_data.get("budget_min"),
        budget_max=constraints_data.get("budget_max"),
        category=constraints_data.get("category"),
        brand=constraints_data.get("brand"),
        materials_include=constraints_data.get("materials_include") or [],
        materials_exclude=constraints_data.get("materials_exclude") or [],
        eco_friendly=constraints_data.get("eco_friendly"),
        size=constraints_data.get("size"),
        query_language=constraints_data.get("query_language"),
    )

    safety_flags = SafetyFlags(
        chemical_safety=bool(safety_data.get("chemical_safety", False)),
        child_safety=bool(safety_data.get("child_safety", False)),
        surface_safety=bool(safety_data.get("surface_safety", False)),
        possible_misuse=bool(safety_data.get("possible_misuse", False)),
        needs_professional_advice=bool(safety_data.get("needs_professional_advice", False)),
        other_notes=safety_data.get("other_notes"),
    )

    intent = Intent(
        intent_type=raw.get("intent_type", "unknown"),
        constraints=constraints,
        safety_flags=safety_flags,
        raw_intent_json=raw,
    )

    logs = state.get("agent_logs", [])
    logs.append(
        AgentLog(
            agent_name="router",
            timestamp=datetime.utcnow().isoformat(),
            reasoning="Router classified intent and extracted constraints and safety flags.",
            decision=intent.to_dict(),
            confidence=0.85,
        )
    )

    state["intent"] = intent
    state["agent_logs"] = logs
    return state
