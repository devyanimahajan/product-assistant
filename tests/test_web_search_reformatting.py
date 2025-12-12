"""
Experimental Test: Web Search Query Reformatting
================================================

Test different strategies for reformatting conversational voice queries
into keyword-based product searches that work better with Brave API.

Goal: Find which reformatting strategy yields the most product results with prices.
"""

import sys
import os
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.web_tool import web_search_tool


def extract_nouns_simple(query: str) -> str:
    """
    Strategy A: Simple noun extraction - remove question words and verbs.
    
    Example: "What is the current price of Fisher Price toy?" 
          -> "current price Fisher Price toy"
    """
    # Remove common question words and filler
    stop_words = [
        'what', 'is', 'the', 'a', 'an', 'how', 'much', 'does', 'do',
        'can', 'you', 'find', 'me', 'for', 'of', 'to', 'in', 'on'
    ]
    
    words = query.lower().split()
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    
    return ' '.join(filtered)


def extract_product_name_regex(query: str) -> str:
    """
    Strategy B: Use regex patterns to extract product names.
    
    Patterns:
    - "price of X" -> X
    - "cost of X" -> X
    - "find X" -> X
    """
    patterns = [
        r'(?:price|cost|cost)\s+(?:of|for)\s+(?:the\s+)?(.+?)(?:\?|$)',
        r'(?:find|show|search)\s+(?:me\s+)?(?:the\s+)?(.+?)(?:\?|$)',
        r'(?:looking|search)\s+for\s+(.+?)(?:\?|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            product = match.group(1).strip()
            # Add shopping keywords
            return f"{product} price buy"
    
    # Fallback: use simple extraction
    return extract_nouns_simple(query) + " buy"


def extract_brand_product(query: str, brand: str = None, category: str = None) -> str:
    """
    Strategy C: Use extracted constraints (brand, category) to build query.
    
    Example: Query="price of toy", brand="Fisher-Price", category="toys"
          -> "Fisher-Price toys price buy"
    """
    # Start with simplified query
    base = extract_nouns_simple(query)
    
    parts = []
    if brand:
        parts.append(brand)
    
    # Add base query words
    parts.append(base)
    
    if category and category not in base.lower():
        parts.append(category)
    
    # Add shopping keywords if not present
    query_str = ' '.join(parts)
    if 'buy' not in query_str.lower() and 'price' not in query_str.lower():
        query_str += " price buy"
    
    return query_str


def test_reformatting_strategy(strategy_name: str, original_query: str, reformatted_query: str):
    """Test a single reformatting strategy."""
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy_name}")
    print(f"Original:    '{original_query}'")
    print(f"Reformatted: '{reformatted_query}'")
    print(f"{'='*70}")
    
    # Call web search
    results = web_search_tool(reformatted_query, max_results=5)
    
    # Analyze results
    success = results.get("success", False)
    result_list = results.get("results", [])
    count = len(result_list)
    prices_found = sum(1 for r in result_list if r.get("price") is not None)
    
    print(f"Success: {success}")
    print(f"Results returned: {count}")
    print(f"Results with prices: {prices_found}")
    
    if result_list:
        print(f"\nSample results:")
        for i, r in enumerate(result_list[:3], 1):
            title = r.get("title", "")[:60]
            price = r.get("price", "N/A")
            url = r.get("url", "")
            print(f"  {i}. {title}... | Price: ${price} | {url[:50]}...")
    
    if results.get("warning"):
        print(f"Warning: {results['warning']}")
    
    return {
        "strategy": strategy_name,
        "original": original_query,
        "reformatted": reformatted_query,
        "success": success,
        "count": count,
        "prices_found": prices_found,
        "score": prices_found * 10 + count  # Weighted score: prices matter more
    }


def run_comparative_test():
    """Run comparative test on multiple queries with different strategies."""
    
    print("\n" + "="*70)
    print("EXPERIMENTAL TEST: Query Reformatting for Product Search")
    print("="*70)
    
    # Test queries (conversational voice queries)
    test_cases = [
        {
            "query": "What is the current price of Fisher Price Think and Learn toy?",
            "brand": "Fisher-Price",
            "category": "toys"
        },
        {
            "query": "How much does the plastic plates cost today?",
            "brand": None,
            "category": "party supplies"
        },
        {
            "query": "Find me the cost of EcoClean Steel Cleaner",
            "brand": "EcoClean",
            "category": "cleaner"
        },
    ]
    
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        brand = test_case.get("brand")
        category = test_case.get("category")
        
        print(f"\n{'#'*70}")
        print(f"TEST CASE {i}")
        print(f"{'#'*70}")
        
        # Test Strategy A: Current (full conversational query)
        result_a = test_reformatting_strategy(
            "A - Current (Full Query)",
            query,
            query
        )
        all_results.append(result_a)
        
        # Test Strategy B: Simple noun extraction
        reformatted_b = extract_nouns_simple(query)
        result_b = test_reformatting_strategy(
            "B - Simple Noun Extraction",
            query,
            reformatted_b
        )
        all_results.append(result_b)
        
        # Test Strategy C: Regex pattern extraction
        reformatted_c = extract_product_name_regex(query)
        result_c = test_reformatting_strategy(
            "C - Regex Pattern Extraction",
            query,
            reformatted_c
        )
        all_results.append(result_c)
        
        # Test Strategy D: Brand + Product + Category
        reformatted_d = extract_brand_product(query, brand, category)
        result_d = test_reformatting_strategy(
            "D - Brand + Product + Keywords",
            query,
            reformatted_d
        )
        all_results.append(result_d)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'Avg Results':<15} {'Avg Prices':<15} {'Score':<10}")
    print(f"{'-'*70}")
    
    strategies = {}
    for result in all_results:
        strategy = result["strategy"]
        if strategy not in strategies:
            strategies[strategy] = {"count": [], "prices": [], "scores": []}
        
        strategies[strategy]["count"].append(result["count"])
        strategies[strategy]["prices"].append(result["prices_found"])
        strategies[strategy]["scores"].append(result["score"])
    
    for strategy, metrics in strategies.items():
        avg_count = sum(metrics["count"]) / len(metrics["count"])
        avg_prices = sum(metrics["prices"]) / len(metrics["prices"])
        avg_score = sum(metrics["scores"]) / len(metrics["scores"])
        
        print(f"{strategy:<35} {avg_count:<15.1f} {avg_prices:<15.1f} {avg_score:<10.1f}")
    
    # Recommendation
    print(f"\n{'='*70}")
    best_strategy = max(strategies.items(), key=lambda x: sum(x[1]["scores"]))
    print(f"RECOMMENDATION: {best_strategy[0]}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_comparative_test()
