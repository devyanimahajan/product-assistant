"""
Integration example: Using Local TTS (SpeechT5) with LangGraph Answerer Agent

This shows how to integrate the notebook's TTS into your project.
"""

from typing import TypedDict
from local_tts import get_local_tts


class AnswererState(TypedDict):
    """State for the answerer agent."""
    query: str
    retrieved_products: list
    plan: dict
    final_answer: dict
    audio_path: str
    spoken_summary: str


def answerer_agent_with_local_tts(state: AnswererState) -> AnswererState:
    """
    Answerer agent using local SpeechT5 TTS.
    
    This is the FREE, open-source alternative.
    """
    print("\n=== ANSWERER AGENT (Local TTS) ===\n")
    
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
        alternatives = retrieved_products[1:4]
        
        # Build response
        response = {
            'message': f"Found {len(retrieved_products)} products matching your criteria.",
            'top_recommendation': top_product,
            'alternatives': alternatives,
            'citations': [p.get('doc_id') for p in retrieved_products[:4]]
        }
        
        # Generate spoken summary
        spoken_text = generate_spoken_summary(top_product, len(alternatives))
    
    # Synthesize speech using local TTS
    try:
        print("[Answerer] Initializing local TTS...")
        tts = get_local_tts()
        
        print("[Answerer] Generating speech...")
        audio_path = tts.synthesize_text(
            text=spoken_text,
            filename=f"response_{abs(hash(state.get('query', '')))}.wav"
        )
        print(f"[Answerer] Audio generated: {audio_path}")
        
    except Exception as e:
        print(f"[Answerer] WARNING: TTS failed: {e}")
        print("[Answerer] Continuing without audio...")
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
    
    # Build summary
    parts = []
    
    if brand:
        parts.append(f"I recommend {brand} {title}")
    else:
        parts.append(f"I recommend {title}")
    
    parts.append(f"at ${price:.2f}")
    
    if rating > 0:
        parts.append(f"with {rating:.1f} stars")
    
    summary = ", ".join(parts) + "."
    
    if alternatives_count > 0:
        summary += f" I found {alternatives_count} more options on your screen."
    
    # Ensure concise
    if len(summary) > 150:
        summary = summary[:147] + "..."
    
    return summary


def answerer_with_product_recommendation(state: AnswererState) -> AnswererState:
    """
    Alternative implementation using the recommendation method.
    """
    print("\n=== ANSWERER AGENT (Product Recommendation) ===\n")
    
    retrieved_products = state.get('retrieved_products', [])
    
    if not retrieved_products:
        response = {
            'message': 'No matching products found.',
            'products': [],
            'citations': []
        }
        spoken_text = 'Sorry, I could not find any products matching your requirements.'
        audio_path = None
    else:
        top_product = retrieved_products[0]
        alternatives = retrieved_products[1:4]
        
        response = {
            'top_recommendation': top_product,
            'alternatives': alternatives,
            'total_found': len(retrieved_products),
            'citations': [p.get('doc_id') for p in retrieved_products[:4]]
        }
        
        # Use the built-in recommendation synthesis
        try:
            tts = get_local_tts()
            
            # Prepare product dict for TTS
            product_for_tts = {
                'title': top_product.get('title', ''),
                'brand': top_product.get('brand', ''),
                'price': top_product.get('price', 0),
                'rating': top_product.get('rating', 0),
                'alternatives_count': len(alternatives)
            }
            
            spoken_text, audio_path = tts.synthesize_recommendation(
                product_for_tts,
                filename=f"recommendation_{abs(hash(state.get('query', '')))}.wav"
            )
            
        except Exception as e:
            print(f"[Answerer] TTS failed: {e}")
            spoken_text = "Product recommendation generated. Check your screen."
            audio_path = None
    
    state['final_answer'] = response
    state['audio_path'] = audio_path
    state['spoken_summary'] = spoken_text
    
    return state


# LangGraph node function
def create_answerer_node_local_tts():
    """
    Create answerer node for LangGraph with local TTS.
    
    Usage in LangGraph:
        from answerer_local_tts import create_answerer_node_local_tts
        
        graph = StateGraph(AgentState)
        graph.add_node("answerer", create_answerer_node_local_tts())
    """
    return answerer_agent_with_local_tts


# Example: Standalone test
if __name__ == "__main__":
    print("\n=== TESTING ANSWERER WITH LOCAL TTS ===\n")
    
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
                'features': ['Plant-based', 'No harsh chemicals', 'Citrus scent']
            },
            {
                'doc_id': 'prod_002',
                'title': 'Natural Steel Polish',
                'brand': 'EcoHome',
                'price': 14.99,
                'rating': 4.5,
                'features': ['Biodegradable', 'Streak-free']
            },
            {
                'doc_id': 'prod_003',
                'title': 'Green Steel Shine',
                'brand': 'PureClean',
                'price': 11.99,
                'rating': 4.3,
                'features': ['Eco-certified', 'Food-safe']
            }
        ],
        'plan': {'use_web_search': False}
    }
    
    print("Running answerer with local TTS...")
    print("(This will download models on first run - ~400MB)")
    print()
    
    try:
        # Run answerer
        result = answerer_with_product_recommendation(test_state)
        
        print("\n=== RESULTS ===")
        print(f"Spoken summary: {result['spoken_summary']}")
        print(f"Audio path: {result.get('audio_path', 'Not generated')}")
        print(f"\nTop recommendation: {result['final_answer']['top_recommendation']['title']}")
        print(f"Citations: {result['final_answer']['citations']}")
        
        if result.get('audio_path'):
            print(f"\nPlay audio with:")
            print(f"  ffplay {result['audio_path']}")
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nPlease install dependencies:")
        print("  pip install transformers datasets soundfile torch torchaudio")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
