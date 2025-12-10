"""
Comprehensive test of all MCP server components.
"""

import sys
import json
sys.path.insert(0, '../mcp_server')

from rag_tool import rag_search_tool
from web_tool import web_search_tool, clean_html
import time


def test_clean_html():
    """Test HTML cleaning function."""
    print("\n[TEST 1] HTML Cleaning")
    
    html_text = "<strong>$99.00</strong> Some product <div>description</div>"
    cleaned = clean_html(html_text, max_chars=10000)
    assert '<' not in cleaned and '>' not in cleaned, "HTML tags not removed!"
    
    long_text = "a" * 15000
    cleaned = clean_html(long_text, max_chars=10000)
    assert len(cleaned) <= 10000, "Character limit not enforced!"
    
    print("[PASS]")


def test_rag_search():
    """Test rag.search tool."""
    print("\n[TEST 2] rag.search")
    
    result = rag_search_tool(
        query="disposable plates",
        max_results=3,
        filters={"max_price": 10.0}
    )
    
    assert result['success'], "rag.search failed"
    assert result['count'] > 0, "No results found"
    
    print(f"\nFound {result['count']} products:")
    if result['results']:
        for i, r in enumerate(result['results'], 1):
            print(f"\n{i}. {r['title'][:50]}...")
            print(f"   Price: ${r['price']:.2f}")
            print(f"   Brand: {r.get('brand', 'N/A')}")
            print(f"   Doc ID: {r['doc_id'][:30]}...")
            assert r['price'] <= 10.0, f"Price filter failed: ${r['price']}"
    
    print("\n[PASS]")


def test_web_search():
    """Test web.search tool with improved query strategy."""
    print("\n[TEST 3] web.search + Improved Query Strategy")
    
    # Test with a query that should find product pages
    result = web_search_tool(
        query="disposable paper plates",
        max_results=2
    )
    
    assert result.get('success', False), f"web.search failed: {result.get('error', 'Unknown')}"
    
    print(f"\nQuery: 'disposable paper plates'")
    print(f"Found {result.get('count', 0)} results with prices")
    
    if result.get('warning'):
        print(f"Warning: {result['warning']}")
    
    if result.get('results'):
        for i, r in enumerate(result['results'], 1):
            print(f"\n{i}. {r['title'][:60]}...")
            print(f"   Price: ${r['price']:.2f}")
            print(f"   URL: {r['url'][:70]}...")
            # All results should have prices (that's the new behavior)
            assert isinstance(r['price'], (float, int)), "All returned results should have extracted prices"
    else:
        print("No results with prices found (category pages filtered out)")
    
    print("\n[PASS]")
    print("\n" + "="*70)
    print("FULL JSON OUTPUT (What Agent Receives)")
    print("="*70)
    print(json.dumps(result, indent=2))
    print("="*70)


def test_server_import():
    """Test that server can be imported."""
    print("\n[TEST 4] Server Import")
    
    from server import MCP_TOOLS
    
    assert len(MCP_TOOLS) == 2, "Expected 2 tools"
    assert any(t['name'] == 'rag.search' for t in MCP_TOOLS), "rag.search not found"
    assert any(t['name'] == 'web.search' for t in MCP_TOOLS), "web.search not found"
    
    print(f"[PASS] 2 tools registered")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MCP SERVER TESTS")
    print("="*70)
    
    try:
        test_clean_html()
        test_rag_search()
        test_web_search()
        test_server_import()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n[FAIL] {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
