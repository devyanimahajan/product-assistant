"""
Example integration of TTS with the answerer agent.
This shows how to generate spoken responses for product recommendations.
"""

import os
from typing import TypedDict
from tts import TextToSpeech, get_tts_engine


class AnswererState(TypedDict):
    """State for the answerer agent."""
    query: str
    retrieved_products: list
    plan: dict
    final_answer: dict
    audio_path: str
    spoken_summary: str


def answerer_agent(state: AnswererState) -> AnswererState:
    """
    Answerer agent that synthesizes product recommendations
    and generates spoken responses.
    """
    print("\n=== ANSWERER AGENT ===\n")
    
    retrieved_products = state.get('retrieved_products', [])
    
    if not retrieved_products:
        # No products found
        response = {
            'message': 'No products found matching your criteria.',
            'products': [],
            'citations': []
        }
        spoken_text = 'I could not find any products matching your criteria. Please try adjusting your search.'
    else:
        # Get top recommendation
        top_product = retrieved_products[0]
        alternatives = retrieved_products[1:4]  # Up to 3 alternatives
        
        # Build response
        response = {
            'message': f"Found {len(retrieved_products)} products matching your criteria.",
            'top_recommendation': top_product,
            'alternatives': alternatives,
            'citations': [p.get('doc_id') for p in retrieved_products[:4]]
        }
        
        # Generate spoken summary
        spoken_text = generate_spoken_summary(top_product, len(alternatives))
    
    # Synthesize speech
    try:
        tts = get_tts_engine()
        audio_path = tts.synthesize_text(
            text=spoken_text,
            filename=f"response_{hash(state.get('query', ''))}.mp3"
        )
        print(f"[TTS] Generated audio: {audio_path}")
    except Exception as e:
        print(f"[TTS] Warning: Speech synthesis failed: {e}")
        audio_path = None
    
    # Update state
    state['final_answer'] = response
    state['audio_path'] = audio_path
    state['spoken_summary'] = spoken_text
    
    return state


def generate_spoken_summary(product: dict, alternatives_count: int) -> str:
    """
    Generate a concise spoken summary for TTS.
    Keeps it under 15 seconds (~150 characters).
    """
    title = product.get('title', 'this product')
    price = product.get('price', 0)
    brand = product.get('brand', '')
    rating = product.get('rating', 0)
    
    # Build summary parts
    parts = []
    
    # Main recommendation
    if brand:
        parts.append(f"I recommend {brand} {title}")
    else:
        parts.append(f"I recommend {title}")
    
    # Price
    parts.append(f"at ${price:.2f}")
    
    # Rating
    if rating > 0:
        parts.append(f"with {rating:.1f} stars")
    
    summary = ", ".join(parts) + "."
    
    # Alternatives
    if alternatives_count > 0:
        summary += f" I found {alternatives_count} more options on your screen."
    
    # Ensure it's concise (approx 15 seconds at normal speech rate)
    max_chars = 150
    if len(summary) > max_chars:
        summary = summary[:max_chars-3] + "..."
    
    return summary


def answerer_with_tts_detailed(state: AnswererState) -> AnswererState:
    """
    More detailed answerer that provides richer spoken context.
    """
    print("\n=== ANSWERER AGENT (Detailed) ===\n")
    
    retrieved_products = state.get('retrieved_products', [])
    query = state.get('query', '')
    
    if not retrieved_products:
        response = {
            'message': 'No matching products found.',
            'products': [],
            'citations': []
        }
        spoken_text = 'Sorry, I could not find any products matching your requirements.'
    else:
        top_product = retrieved_products[0]
        alternatives = retrieved_products[1:4]
        
        # Build detailed response
        response = {
            'query': query,
            'top_recommendation': top_product,
            'alternatives': alternatives,
            'total_found': len(retrieved_products),
            'citations': [p.get('doc_id') for p in retrieved_products[:4]],
            'sources': {
                'private_catalog': True,
                'live_web': state.get('plan', {}).get('use_web_search', False)
            }
        }
        
        # Generate detailed spoken summary
        spoken_text = generate_detailed_summary(
            top_product=top_product,
            alternatives_count=len(alternatives),
            query=query
        )
    
    # Synthesize
    try:
        tts = get_tts_engine()
        
        # You can customize TTS parameters here
        audio_path = tts.synthesize_text(
            text=spoken_text,
            filename=f"answer_{abs(hash(query))}.mp3"
        )
        print(f"[TTS] Audio generated: {audio_path}")
    except Exception as e:
        print(f"[TTS] Failed: {e}")
        audio_path = None
    
    state['final_answer'] = response
    state['audio_path'] = audio_path
    state['spoken_summary'] = spoken_text
    
    return state


def generate_detailed_summary(
    top_product: dict,
    alternatives_count: int,
    query: str
) -> str:
    """
    Generate a more detailed spoken summary.
    Still aims for ~15 seconds but with more context.
    """
    title = top_product.get('title', 'Product')
    price = top_product.get('price', 0)
    brand = top_product.get('brand', '')
    rating = top_product.get('rating', 0)
    features = top_product.get('features', [])
    
    # Start with recommendation
    if brand:
        summary = f"My top pick is {brand} {title}."
    else:
        summary = f"My top pick is {title}."
    
    # Add key feature if available
    if features and len(features) > 0:
        key_feature = features[0][:40]  # Limit feature length
        summary += f" It features {key_feature}."
    
    # Price and rating
    summary += f" It's ${price:.2f}"
    if rating > 0:
        summary += f" with {rating:.1f} stars."
    else:
        summary += "."
    
    # Alternatives
    if alternatives_count > 0:
        summary += f" I also found {alternatives_count} alternatives on your screen."
    
    return summary


# Example: Using with LangGraph
def create_answerer_node():
    """
    Create answerer node for LangGraph integration.
    """
    return answerer_agent


# Example: Standalone usage
if __name__ == "__main__":
    print("\n=== TESTING ANSWERER WITH TTS ===\n")
    
    # Mock state
    test_state = {
        'query': 'eco-friendly stainless steel cleaner under $15',
        'retrieved_products': [
            {
                'doc_id': 'prod_001',
                'title': 'Eco-Friendly Steel Cleaner',
                'brand': 'GreenClean',
                'price': 12.49,
                'rating': 4.6,
                'features': ['Plant-based surfactants', 'No harsh chemicals', 'Citrus scent']
            },
            {
                'doc_id': 'prod_002',
                'title': 'Natural Steel Polish',
                'brand': 'EcoHome',
                'price': 14.99,
                'rating': 4.5,
                'features': ['Biodegradable', 'Streak-free finish']
            },
            {
                'doc_id': 'prod_003',
                'title': 'Green Steel Shine',
                'brand': 'PureClean',
                'price': 11.99,
                'rating': 4.3,
                'features': ['Eco-certified', 'Safe for food surfaces']
            }
        ],
        'plan': {'use_web_search': False}
    }
    
    # Run answerer
    result = answerer_with_tts_detailed(test_state)
    
    print("\n=== RESULTS ===")
    print(f"Spoken summary: {result['spoken_summary']}")
    print(f"Audio path: {result.get('audio_path', 'Not generated')}")
    print(f"\nTop recommendation: {result['final_answer']['top_recommendation']['title']}")
    print(f"Citations: {result['final_answer']['citations']}")
