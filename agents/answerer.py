from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from agent_state import (
    AgentState,
    Citation,
    ProductResult,
    AgentLog,
)


class AnswererAgent:
    """
    Synthesizes a concise, grounded answer with citations and a short TTS summary.
    """

    SYSTEM_PROMPT = """
You are a product recommendation assistant for household and similar products.

You will receive:
- user_query: the original user request
- constraints: structured constraints (budget, eco_friendly, brand, etc.)
- products: a list of candidate products. Each has:
  - source: "private" or "web"
  - sku, title, price, rating, brand, ingredients, url, doc_id

Your tasks:
1. Select the top 1 to 3 products that best match the query and constraints.
2. Explain clearly why the top product is recommended, mentioning budget, material compatibility, eco friendliness, and rating if available.
3. Add light safety notes if needed (for example "avoid mixing with bleach" or "test on a small area first").
4. Provide citations that map back to:
   - private: use doc_id as private document id
   - web: use the url as the source link
5. Produce a brief tts_summary that can be spoken in about 10 to 15 seconds.

Respond with valid JSON only, with the following shape:
{
  "recommendation": string,
  "tts_summary": string,
  "citations": [
    {"source": "private" | "web", "doc_id": string, "url": string or null, "label": string or null, "snippet": string or null}
  ],
  "reasoning": string,
  "safety_warnings": [string],
  "confidence": number between 0 and 1
}
"""

    def __init__(self, llm: Any, enable_critic: bool = True) -> None:
        self.llm = llm
        self.enable_critic = enable_critic
        self.name = "answerer"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("AnswererAgent: no JSON found in model output")
        return json.loads(match.group(0))

    def _call_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.llm.invoke(
            [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, indent=2)),
            ]
        )
        return self._extract_json(response.content)

    def _build_payload(self, state: AgentState) -> Dict[str, Any]:
        products: List[ProductResult] = state.get("retrieved_products", [])  # type: ignore[assignment]
        intent = state.get("intent")
        constraints = intent.constraints.to_dict() if intent else {}

        return {
            "user_query": state.get("user_query", ""),
            "constraints": constraints,
            "products": [p.to_dict() for p in products],
        }

    def run(self, state: AgentState) -> AgentState:
        payload = self._build_payload(state)
        raw = self._call_llm(payload)

        recommendation = raw.get("recommendation", "")
        tts_summary = raw.get("tts_summary") or recommendation[:240]
        citations_raw = raw.get("citations", [])
        safety_warnings = raw.get("safety_warnings", [])
        reasoning = raw.get("reasoning", "")
        confidence = float(raw.get("confidence", 0.8))

        citations: List[Citation] = []
        for c in citations_raw:
            cite = Citation(
                source=c.get("source", "private"),
                doc_id=c.get("doc_id", ""),
                url=c.get("url"),
                label=c.get("label"),
                snippet=c.get("snippet"),
            )
            citations.append(cite)

        logs = state.get("agent_logs", [])
        logs.append(
            AgentLog(
                agent_name=self.name,
                timestamp=datetime.utcnow().isoformat(),
                reasoning=reasoning or "Answerer synthesized final response based on retrieved products.",
                decision={
                    "recommendation": recommendation,
                    "tts_summary": tts_summary,
                    "citations": [c.to_dict() for c in citations],
                    "safety_warnings": safety_warnings,
                },
                confidence=confidence,
            )
        )

        state["final_response"] = recommendation
        state["tts_summary"] = tts_summary
        state["citations"] = citations
        state["agent_logs"] = logs
        return state
