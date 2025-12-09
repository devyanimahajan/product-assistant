# agents/answerer_agent.py
"""
Answerer Agent for synthesizing grounded recommendations.

"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import json
import re


@dataclass
class Answer:
    """
    Structured answer with citations and safety information.

    """
    recommendation: str
    top_products: List[Dict[str, Any]]
    citations: List[str]
    reasoning: str
    safety_warnings: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class AnswererAgent:
    """
    Answerer/Critic Agent that synthesizes grounded recommendations.

    """

    SYSTEM_PROMPT = """You are a product recommendation assistant. Your role is to:

1. Provide concise, helpful recommendations based ONLY on the given context
2. Cite sources using [N] notation for EVERY factual claim
3. Compare products on relevant dimensions (price, rating, features)
4. Highlight the top recommendation with clear reasoning
5. Be honest about limitations or missing information

CRITICAL RULES:
- Every factual claim (price, rating, feature) must have a citation [N]
- Never invent product details not in the provided context
- If comparing prices from different sources, cite both and note discrepancy
- Keep response under 150 words for voice delivery compatibility
- Use natural, conversational language

Output format: JSON object with recommendation, top_products, reasoning, confidence"""

    def __init__(self, llm: ChatOpenAI, enable_critic: bool = True):
        """
        Initialize answerer agent.

        """
        self.llm = llm
        self.enable_critic = enable_critic

    def generate_answer(
        self,
        query: str,
        retrieval_results: List[Any],  # List[RetrievalResult]
        constraints: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Answer:
        """
        Generate grounded answer from retrieval results.

        """
        if not retrieval_results:
            return Answer(
                recommendation="I couldn't find products matching your criteria. Try adjusting your budget or requirements.",
                top_products=[],
                citations=[],
                reasoning="No results found",
                safety_warnings=[],
                confidence=0.0
            )

        # Step 1: Prepare context
        context = self._prepare_context(retrieval_results)

        # Step 2: Generate answer
        initial_answer = self._synthesize_answer(query, context, constraints, plan)

        # Step 3: Critic review
        if self.enable_critic and initial_answer.confidence > 0.5:
            final_answer = self._critic_review(initial_answer, context, query)
        else:
            final_answer = initial_answer

        # Step 4: Safety checks
        final_answer = self._apply_safety_checks(final_answer, retrieval_results)

        return final_answer

    def _prepare_context(self, results: List[Any]) -> str:
        """
        Prepare grounded context from retrieval results.

        Formats each result with citation marker [N] and all relevant info.
        """
        context_parts = []

        for i, result in enumerate(results[:5], 1):  # Top 5 results
            citation_id = f"[{i}]"

            # Format product information
            product_info = [
                f"{citation_id} {result.title}",
                f"- Source: {result.source}",
                f"- Price: ${result.price:.2f}",
                f"- Rating: {result.rating if result.rating else 'N/A'}★",
                f"- Brand: {result.brand or 'N/A'}"
            ]

            # Add features if available
            if result.features:
                features_str = ', '.join(result.features[:3])
                product_info.append(f"- Features: {features_str}")

            # Add ingredients if available
            if result.ingredients:
                product_info.append(f"- Ingredients: {result.ingredients}")

            # Add provenance
            if result.doc_id:
                product_info.append(f"- Doc ID: {result.doc_id}")
            if result.url:
                product_info.append(f"- URL: {result.url}")

            # Flag price discrepancies
            if hasattr(result, 'metadata') and result.metadata:
                if result.metadata.get('price_discrepancy'):
                    disc = result.metadata['price_discrepancy']
                    product_info.append(
                        f"- ⚠️ Price varies: ${disc['private']} (catalog) "
                        f"vs ${disc['web']} (web)"
                    )

            context_parts.append('\n'.join(product_info))

        return '\n\n'.join(context_parts)

    def _synthesize_answer(
        self,
        query: str,
        context: str,
        constraints: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Answer:
        """
        Synthesize answer using LLM with strict grounding requirements.
        """
        user_prompt = f"""Query: {query}

Constraints:
{json.dumps(constraints, indent=2)}

Available Products:
{context}

Task: Provide a recommendation that:
1. Identifies the top 1-2 products that best match the query
2. Explains why (features, price/value, rating)
3. Mentions alternatives if relevant
4. Cites EVERY claim using [N] notation

Return ONLY valid JSON:
{{
  "recommendation": "Natural language recommendation with [N] citations after each fact",
  "top_products": [
    {{"title": "Product Name", "citation": "[1]", "reason": "why recommended"}},
    ...
  ],
  "reasoning": "Brief explanation of selection criteria used",
  "confidence": 0.85
}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")

            answer_data = json.loads(json_match.group())

            # Extract citation numbers from recommendation text
            citations = re.findall(r'\[(\d+)\]', answer_data.get('recommendation', ''))

            return Answer(
                recommendation=answer_data.get('recommendation', ''),
                top_products=answer_data.get('top_products', []),
                citations=list(set(citations)),  # Deduplicate
                reasoning=answer_data.get('reasoning', ''),
                safety_warnings=[],
                confidence=float(answer_data.get('confidence', 0.7))
            )

        except Exception as e:
            print(f"Answer synthesis error: {e}")
            # Fallback answer
            return Answer(
                recommendation=f"I found {len(context.split('['))-1} products but had trouble creating a detailed recommendation. Please review the results.",
                top_products=[],
                citations=[],
                reasoning=f"Synthesis error: {str(e)}",
                safety_warnings=[],
                confidence=0.3
            )

    def _critic_review(
        self,
        answer: Answer,
        context: str,
        query: str
    ) -> Answer:
        """
        Critic agent reviews answer for grounding and quality.

        Checks:
        - All citations are valid
        - No hallucinated information
        - Recommendation addresses query
        - Reasoning is sound
        """
        critic_prompt = f"""Review this product recommendation for quality and grounding.

Original Query: {query}

Generated Answer: {answer.recommendation}

Available Context:
{context}

Check for:
1. All claims have valid citations [N] that reference actual products
2. Citations reference products that exist in the context
3. No hallucinated details (prices, features, ratings not in context)
4. Recommendation directly addresses the query
5. Reasoning is logical and sound

Return ONLY valid JSON:
{{
  "grounded": true/false,
  "issues": ["list any specific problems found"],
  "suggested_fixes": "corrections if needed",
  "approved": true/false
}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a fact-checker ensuring accurate, grounded recommendations."),
                HumanMessage(content=critic_prompt)
            ])

            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                review = json.loads(json_match.group())

                # If not approved, lower confidence and add note
                if not review.get('approved', True):
                    answer.confidence *= 0.7
                    issues = review.get('issues', [])
                    if issues:
                        answer.reasoning += f" [Critic notes: {'; '.join(issues[:2])}]"

                # If not grounded, flag heavily
                if not review.get('grounded', True):
                    answer.confidence *= 0.5
                    answer.safety_warnings.append(
                        "⚠️ Some information may not be fully verified"
                    )

        except Exception as e:
            print(f"Critic review error: {e}")

        return answer

    def _apply_safety_checks(
        self,
        answer: Answer,
        results: List[Any]
    ) -> Answer:
        """
        Apply safety guardrails and warnings.

        Checks for:
        - Harsh chemicals (bleach, ammonia, acids)
        - Mixing warnings
        - Unusual pricing
        """
        warnings = []

        # Check top 3 results for safety issues
        for result in results[:3]:
            title_lower = result.title.lower()
            features_text = ' '.join(result.features).lower() if result.features else ''
            combined_text = title_lower + ' ' + features_text

            # Harsh chemical warnings
            harsh_chemicals = {
                'bleach': 'Use bleach in well-ventilated areas. Never mix with ammonia or acids.',
                'ammonia': 'Use ammonia in well-ventilated areas. Never mix with bleach.',
                'acid': 'Handle acidic cleaners with care. Use proper ventilation.',
                'caustic': 'Caustic products can cause burns. Use protective equipment.'
            }

            for chemical, warning in harsh_chemicals.items():
                if chemical in combined_text:
                    warnings.append(f"⚠️ {warning}")

            # Generic mixing warning if multiple harsh chemicals
            harsh_found = [c for c in harsh_chemicals.keys() if c in combined_text]
            if len(harsh_found) > 1:
                warnings.append(
                    "⚠️ DANGER: Never mix different cleaning chemicals together - "
                    "this can create toxic fumes."
                )

        # Price anomaly detection
        if len(results) >= 3:
            prices = [r.price for r in results[:5] if r.price > 0]
            if prices:
                avg_price = sum(prices) / len(prices)
                for result in results[:2]:
                    if result.price > avg_price * 1.8:  # 80% more expensive
                        warnings.append(
                            f"💰 Note: {result.title} is significantly more expensive "
                            f"than alternatives (${result.price:.2f} vs avg ${avg_price:.2f})"
                        )

        # Deduplicate warnings
        answer.safety_warnings = list(set(warnings))[:3]  # Max 3 warnings

        return answer

    def format_for_voice(self, answer: Answer) -> str:
        """
        Format answer for TTS output (≤15 seconds, ~40-50 words).

        Strips citations and truncates to voice-appropriate length.
        """
        # Remove citation markers
        voice_text = re.sub(r'\[\d+\]', '', answer.recommendation)

        # Truncate to ~50 words
        words = voice_text.split()
        if len(words) > 50:
            voice_text = ' '.join(words[:47]) + "... Check screen for details."

        # Add critical safety note if present
        critical_warnings = [w for w in answer.safety_warnings
                           if 'DANGER' in w or 'toxic' in w.lower()]
        if critical_warnings:
            voice_text += " Important safety information on screen."

        return voice_text

    def format_for_display(
        self,
        answer: Answer,
        results: List[Any]
    ) -> Dict[str, Any]:
        """
        Format answer for UI display with full citations and comparison table.

        Returns:
            Dict with recommendation, citations map, comparison table, warnings
        """
        # Build citation reference map
        citation_map = {}
        for i, result in enumerate(results[:5], 1):
            citation_map[str(i)] = {
                'title': result.title,
                'price': result.price,
                'rating': result.rating,
                'brand': result.brand,
                'source': result.source,
                'doc_id': result.doc_id,
                'url': result.url,
                'features': result.features[:3] if result.features else []
            }

        # Build comparison table for top products
        comparison_table = []
        for prod in answer.top_products[:3]:
            # Extract citation number
            citation_match = re.search(r'\[(\d+)\]', prod.get('citation', ''))
            if citation_match:
                citation_num = citation_match.group(1)
                if citation_num in citation_map:
                    product_data = citation_map[citation_num].copy()
                    product_data['reason'] = prod.get('reason', '')
                    comparison_table.append(product_data)

        return {
            'recommendation': answer.recommendation,
            'reasoning': answer.reasoning,
            'confidence': answer.confidence,
            'top_products': comparison_table,
            'all_citations': citation_map,
            'safety_warnings': answer.safety_warnings,
            'sources_used': list(set(c['source'] for c in citation_map.values()))
        }


# LangGraph Node Wrapper
def answerer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for answerer agent.

    """
    answerer = state['answerer_agent']

    answer = answerer.generate_answer(
        query=state['query'],
        retrieval_results=state['retrieval_results'],
        constraints=state.get('constraints', {}),
        plan=state.get('plan', {})
    )

    # Format for different outputs
    voice_text = answerer.format_for_voice(answer)
    display_data = answerer.format_for_display(answer, state['retrieval_results'])

    return {
        'answer': answer,
        'voice_response': voice_text,
        'display_data': display_data,
        'safety_warnings': answer.safety_warnings
    }


# Standalone test function
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    # Mock retrieval result for testing
    class MockResult:
        def __init__(self, title, price, rating, source, doc_id, features=None):
            self.title = title
            self.price = price
            self.rating = rating
            self.brand = "TestBrand"
            self.source = source
            self.doc_id = doc_id
            self.url = None
            self.features = features or []
            self.ingredients = None
            self.metadata = {}

    # Initialize answerer
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    answerer = AnswererAgent(llm, enable_critic=True)

    # Test with mock data
    test_results = [
        MockResult(
            "EcoClean Steel Polish",
            12.49,
            4.6,
            "private",
            "prod_001",
            ["Plant-based", "Non-toxic", "Stainless steel safe"]
        ),
        MockResult(
            "GreenShine All-Purpose",
            8.99,
            4.2,
            "private",
            "prod_002",
            ["Eco-friendly", "Multi-surface"]
        )
    ]

    test_query = "eco-friendly stainless steel cleaner under $15"
    test_constraints = {'max_price': 15.0, 'eco_friendly': True}
    test_plan = {'comparison_needed': True}

    answer = answerer.generate_answer(
        test_query,
        test_results,
        test_constraints,
        test_plan
    )

    print("\n=== ANSWERER TEST ===")
    print(f"Query: {test_query}\n")
    print(f"Recommendation:\n{answer.recommendation}\n")
    print(f"Reasoning: {answer.reasoning}")
    print(f"Confidence: {answer.confidence:.2f}")
    print(f"\nVoice output:\n{answerer.format_for_voice(answer)}")
    print(f"\nSafety warnings: {answer.safety_warnings}")
