from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from agent_state import (
    AgentState,
    Citation,
    ProductResult,
    AgentLog,
)


def _load_prompt() -> str:
    """Load answerer prompt from file to avoid sync issues."""
    prompt_file = Path(__file__).parent.parent / "prompts" / "answerer_prompt.txt"
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
                    if line.strip() and not line.startswith('=') and not line.startswith('ANSWERER') and not line.startswith('Role:'):
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
        raise FileNotFoundError(f"Answerer prompt file not found: {prompt_file}")


# Load prompt once at module level
SYSTEM_PROMPT = _load_prompt()


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Answerer: no JSON found in LLM output")
    return json.loads(match.group(0))


def _build_payload(state: AgentState) -> Dict[str, Any]:
    products: List[ProductResult] = state.get("retrieved_products", [])  # type: ignore[assignment]
    intent = state.get("intent")
    constraints = intent.constraints.to_dict() if intent else {}

    return {
        "user_query": state.get("user_query", ""),
        "constraints": constraints,
        "products": [p.to_dict() for p in products],
    }


def run_answerer(state: AgentState, llm: Any) -> AgentState:
    """LangGraph node: synthesize grounded answer and TTS summary."""

    payload = _build_payload(state)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )
    raw = _extract_json(response.content)

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
            agent_name="answerer",
            timestamp=datetime.utcnow().isoformat(),
            reasoning=reasoning or "Answerer synthesized final response from retrieved products.",
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
