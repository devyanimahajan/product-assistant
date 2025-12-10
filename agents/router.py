from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from agent_state import (
    AgentState,
    ProductConstraints,
    SafetyFlags,
    Intent,
    AgentLog,
)


SYSTEM_PROMPT = """
You are an intent classifier and constraint extractor for a product discovery assistant.

Given a user query, you must output strict JSON with keys:
{
  "intent_type": "product_search" | "comparison" | "chitchat" | "unsafe" | "unknown",
  "constraints": {
    "budget_min": number or null,
    "budget_max": number or null,
    "category": string or null,
    "brand": string or null,
    "materials_include": [string],
    "materials_exclude": [string],
    "eco_friendly": boolean or null,
    "size": string or null,
    "query_language": string or null
  },
  "safety_flags": {
    "chemical_safety": boolean,
    "child_safety": boolean,
    "surface_safety": boolean,
    "possible_misuse": boolean,
    "needs_professional_advice": boolean,
    "other_notes": string or null
  }
}

Be conservative: mark "unsafe" when the user asks for advice that could damage surfaces, harm children or pets, or risk misuse of chemicals.
Only respond with JSON.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Router: no JSON found in LLM output")
    return json.loads(match.group(0))


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
