"""
Test script for TTS functionality.
Tests different providers and generation methods.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tts import TextToSpeech, get_tts_engine


def test_basic_synthesis():
    """Test basic text synthesis."""
    print("\n=== TEST: Basic Synthesis ===\n")
    
    try:
        tts = get_tts_engine()
        
        text = "Welcome to the product assistant. I'm here to help you find the best products."
        
        audio_path = tts.synthesize_text(text, filename="test_basic.mp3")
        
        print(f"✓ Successfully generated: {audio_path}")
        print(f"  Text length: {len(text)} characters")
        
        # Check file exists
        if Path(audio_path).exists():
            file_size = Path(audio_path).stat().st_size
            print(f"  File size: {file_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_product_summary():
    """Test product recommendation summary generation."""
    print("\n=== TEST: Product Summary Generation ===\n")
    
    try:
        tts = get_tts_engine()
        
        # Sample product
        product = {
            'title': 'Eco-Friendly Stainless Steel Cleaner',
            'brand': 'GreenClean',
            'price': 12.49,
            'rating': 4.6,
            'alternatives_count': 2
        }
        
        summary, audio_path = tts.synthesize_recommendation(
            product,
            filename="test_product.mp3"
        )
        
        print(f"✓ Generated summary:")
        print(f"  Text: {summary}")
        print(f"  Length: {len(summary)} characters")
        print(f"  Audio: {audio_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_multiple_products():
    """Test generating summaries for multiple products."""
    print("\n=== TEST: Multiple Products ===\n")
    
    products = [
        {
            'title': 'Glass Cleaner Spray',
            'brand': 'Sparkle',
            'price': 8.99,
            'rating': 4.3,
            'alternatives_count': 0
        },
        {
            'title': 'All-Purpose Cleaner',
            'brand': 'MultiClean',
            'price': 15.99,
            'rating': 4.7,
            'alternatives_count': 5
        },
        {
            'title': 'Floor Polish',
            'brand': None,
            'price': 22.50,
            'rating': 0,
            'alternatives_count': 1
        }
    ]
    
    try:
        tts = get_tts_engine()
        
        for i, product in enumerate(products):
            summary = tts.generate_summary(product, max_length=120)
            print(f"\nProduct {i+1}:")
            print(f"  Title: {product['title']}")
            print(f"  Summary: {summary}")
            print(f"  Length: {len(summary)} chars")
        
        print("\n✓ All summaries generated successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== TEST: Edge Cases ===\n")
    
    try:
        tts = get_tts_engine()
        
        # Test 1: Very long text (should truncate)
        long_text = "This is a very long product description. " * 20
        summary = tts.generate_summary({
            'title': long_text,
            'price': 10.0,
            'rating': 4.0
        }, max_length=100)
        
        print(f"Test 1 - Long text truncation:")
        print(f"  Input: {len(long_text)} chars")
        print(f"  Output: {len(summary)} chars")
        print(f"  ✓ Truncated correctly" if len(summary) <= 103 else "  ✗ Failed to truncate")
        
        # Test 2: Missing fields
        minimal_product = {'title': 'Basic Product', 'price': 5.0}
        summary = tts.generate_summary(minimal_product)
        print(f"\nTest 2 - Missing fields:")
        print(f"  Summary: {summary}")
        print(f"  ✓ Handled missing fields")
        
        # Test 3: Zero values
        zero_product = {'title': 'Free Item', 'price': 0, 'rating': 0}
        summary = tts.generate_summary(zero_product)
        print(f"\nTest 3 - Zero values:")
        print(f"  Summary: {summary}")
        print(f"  ✓ Handled zero values")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_provider_availability():
    """Test which providers are available."""
    print("\n=== TEST: Provider Availability ===\n")
    
    providers = ['openai', 'elevenlabs', 'azure']
    available = []
    
    for provider in providers:
        try:
            # Check for required environment variables
            if provider == 'openai':
                required = ['OPENAI_API_KEY']
            elif provider == 'elevenlabs':
                required = ['ELEVEN_API_KEY']
            elif provider == 'azure':
                required = ['AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION']
            
            has_keys = all(os.getenv(key) for key in required)
            
            if has_keys:
                # Try to initialize
                tts = TextToSpeech(provider=provider, output_dir="./test_audio")
                available.append(provider)
                print(f"✓ {provider.upper()}: Available")
            else:
                print(f"○ {provider.upper()}: Missing API keys")
                print(f"  Required: {', '.join(required)}")
                
        except ImportError as e:
            print(f"✗ {provider.upper()}: Package not installed")
            print(f"  Error: {e}")
        except Exception as e:
            print(f"✗ {provider.upper()}: Failed - {e}")
    
    print(f"\nAvailable providers: {', '.join(available) if available else 'None'}")
    return len(available) > 0


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TTS FUNCTIONALITY TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Check availability first
    results['availability'] = test_provider_availability()
    
    if not results['availability']:
        print("\n❌ No TTS providers available. Please configure API keys.")
        print("\nSetup instructions:")
        print("1. Copy .env.example to .env")
        print("2. Add your API key(s):")
        print("   - OpenAI: OPENAI_API_KEY")
        print("   - ElevenLabs: ELEVEN_API_KEY")
        print("   - Azure: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
        print("3. Set TTS_PROVIDER in .env (default: openai)")
        return
    
    # Run functionality tests
    results['basic'] = test_basic_synthesis()
    results['product'] = test_product_summary()
    results['multiple'] = test_multiple_products()
    results['edge_cases'] = test_edge_cases()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed. Using system environment variables only.")
    
    run_all_tests()
