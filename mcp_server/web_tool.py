"""
Web Search Tool - Search for current product prices and availability

Uses Brave Search API to get live product information.
Returns results with URLs for citations.
"""

import os
import re
import requests
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

# Simple cache for web results (TTL: 5 minutes)
_cache = {}
_cache_ttl = timedelta(minutes=5)


def clean_html(text: str, max_chars: int = 10000) -> str:
    """
    Strip HTML tags and limit to max characters.
    
    Args:
        text: Raw text potentially containing HTML
        max_chars: Maximum characters to return (default: 10000)
    
    Returns:
        Clean text without HTML, capped at max_chars
    """
    if not text:
        return ""
    
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    clean = ' '.join(clean.split())
    # Cap at max_chars
    return clean[:max_chars]


def extract_price(text: str):
    """
    Extract price from text with improved pattern matching.
    
    Examples:
        "$12.99" -> 12.99
        "Price: 15.50" -> 15.50
        "From $10.00" -> 10.0
        "$25.99 - $30.00" -> 25.99 (takes first price)
    
    Returns:
        float if price found, None if not found
    """
    if not text:
        return None
    
    patterns = [
        r'(?:Price|price):\s*\$\s*(\d+(?:,\d{3})*\.?\d{0,2})',  # Price: $12.99
        r'(?:From|from)\s+\$\s*(\d+(?:,\d{3})*\.?\d{0,2})',  # From $10.00
        r'\$\s*(\d+(?:,\d{3})*\.?\d{0,2})',  # $12.99 or $1,299.99
        r'(\d+(?:,\d{3})*\.?\d{0,2})\s*USD',  # 12.99 USD
        r'USD\s*(\d+(?:,\d{3})*\.?\d{0,2})',  # USD 12.99
        r'(?:^|\s)(\d+\.?\d{0,2})\s*(?:dollars?|usd)',  # 12.99 dollars
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                price_str = match.group(1).replace(',', '')
                price = float(price_str)
                # Sanity check: price should be reasonable (between $0.01 and $100,000)
                if 0.01 <= price <= 100000:
                    return price
            except ValueError:
                continue
    
    return None


def is_product_page(url: str) -> bool:
    """Check if URL is likely a product page (not category/listing)."""
    product_indicators = [
        '/dp/',      # Amazon product pages
        '/p/',       # Many retailers
        '/ip/',      # Walmart
        '/product/', # Generic
        '/item/',    # eBay, etc.
        '/pd/',      # Some retailers
    ]
    return any(indicator in url.lower() for indicator in product_indicators)


def improve_query(query: str) -> str:
    """Improve query to target product pages."""
    # Add "buy" if not present to bias toward product pages
    if 'buy' not in query.lower() and 'price' not in query.lower():
        query = f"{query} buy"
    return query


def web_search_tool(
    query: str,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search web for current product prices and availability.
    
    Uses Brave Search API to find live product listings.
    Filters to product pages and only returns results with extracted prices.
    
    Args:
        query: Product search query
        max_results: Maximum number of results (default: 5)
    
    Returns:
        {
            "success": bool,
            "results": [
                {
                    "title": str,
                    "url": str,  # For citations!
                    "snippet": str,
                    "price": float,  # Extracted price
                    "source": "web"
                }
            ],
            "warning": str (optional) - if no prices found
        }
    """
    cache_key = f"{query}_{max_results}"
    if cache_key in _cache:
        cached_time, cached_result = _cache[cache_key]
        if datetime.now() - cached_time < _cache_ttl:
            print(f"[web.search] Using cached result for: {query}")
            return cached_result
    
    try:
        api_key = os.getenv('BRAVE_API_KEY')
        if not api_key:
            return {
                "success": False,
                "error": "BRAVE_API_KEY not found in environment",
                "query": query,
                "results": []
            }
        
        url = "https://api.search.brave.com/res/v1/web/search"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        
        # Improve query to target product pages
        improved_query = improve_query(query)
        
        params = {
            "q": improved_query,
            "count": max_results * 2  # Request more to filter down
        }
        
        print(f"[web.search] Querying Brave API: {improved_query}")
        
        # Retry logic for rate limits
        max_retries = 2
        retry_delay = 1.2
        
        for attempt in range(max_retries + 1):
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                break
            elif response.status_code == 429:  # Rate limit
                if attempt < max_retries:
                    print(f"[web.search] Rate limited, waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return {
                        "success": False,
                        "error": "Brave API rate limit exceeded after retries",
                        "query": query,
                        "results": []
                    }
            else:
                return {
                    "success": False,
                    "error": f"Brave API error: {response.status_code} - {response.text}",
                    "query": query,
                    "results": []
                }
        
        data = response.json()
        
        all_results = []
        results_with_price = []
        web_results = data.get('web', {}).get('results', [])
        
        print(f"[web.search] Brave API returned {len(web_results)} raw results")
        
        for item in web_results:
            title = item.get('title', '')
            url_link = item.get('url', '')
            description = item.get('description', '')
            
            # Prefer product pages
            is_product = is_product_page(url_link)
            
            price = extract_price(title + ' ' + description)
            print(f"[web.search] Processing: {title[:50]}... | Price extracted: {price}")
            
            if price is not None:
                # Only include results with extracted prices
                result = {
                    "title": title,
                    "url": url_link,
                    "snippet": description[:200],
                    "price": price,
                    "source": "web",
                    "timestamp": datetime.now().isoformat(),
                    "is_product_page": is_product
                }
                results_with_price.append(result)
                
                # Stop once we have enough results with prices
                if len(results_with_price) >= max_results:
                    break
        
        # Prioritize product pages in final results
        results_with_price.sort(key=lambda x: x.get('is_product_page', False), reverse=True)
        
        # Remove is_product_page field before returning (internal use only)
        for r in results_with_price:
            r.pop('is_product_page', None)
        
        result_data = {
            "success": True,
            "query": query,
            "count": len(results_with_price),
            "results": results_with_price,
            "cached": False
        }
        
        # Add warning if no prices found
        if len(results_with_price) == 0:
            result_data["warning"] = (
                "Unable to find specific product pages with prices. "
                "Try narrowing your search to specific product names or brands instead of general categories."
            )
        
        _cache[cache_key] = (datetime.now(), result_data)
        
        return result_data
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Brave API request timed out",
            "query": query,
            "results": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Web search failed: {str(e)}",
            "query": query,
            "results": []
        }
