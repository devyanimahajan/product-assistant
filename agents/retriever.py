# agents/retriever_agent.py
"""
Retriever Agent for Agentic RAG

"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import json
import re


@dataclass
class RetrievalResult:
    """    
    This class holds all information about a retrieved product,
    including where it came from and how relevant it is.
    """
    source: str              
    product_id: str          
    title: str               
    price: float            
    rating: Optional[float] = None      
    brand: Optional[str] = None         
    features: List[str] = None         
    ingredients: Optional[str] = None   
    doc_id: Optional[str] = None        
    url: Optional[str] = None         
    relevance_score: float = 0.0       
    metadata: Dict[str, Any] = None     
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.features is None:
            self.features = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class RetrieverAgent:
    """
    Agentic RAG Retriever with hybrid search capabilities.

    """
    
    def __init__(
        self,
        vector_store: FAISS,
        embeddings: OpenAIEmbeddings,
        llm: ChatOpenAI,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize retriever agent.
        
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.mcp_client = mcp_client
        
    def retrieve(
        self,
        query: str,
        constraints: Dict[str, Any],
        plan: Dict[str, Any],
        k: int = 10
    ) -> List[RetrievalResult]:
        """
        Main retrieval orchestration method.
        
        This is the main entry point that coordinates all retrieval steps.
        
        """
        results = []
        
        # Step 1: Always search private catalog
        print(f"[Retriever] Searching private catalog for: {query}")
        private_results = self._retrieve_private(query, constraints, k)
        results.extend(private_results)
        print(f"[Retriever] Found {len(private_results)} results from private catalog")
        
        # Step 2: Optionally add live web search
        if plan.get('use_live_search', False) and self.mcp_client:
            print(f"[Retriever] Searching web for current info...")
            web_results = self._retrieve_web(query, constraints, k=5)
            results.extend(web_results)
            print(f"[Retriever] Found {len(web_results)} results from web")
        
        # Step 3: Reconcile duplicates and conflicts
        print(f"[Retriever] Reconciling {len(results)} total results...")
        results = self._reconcile_results(results)
        print(f"[Retriever] After reconciliation: {len(results)} unique results")
        
        # Step 4: Rerank by relevance
        print(f"[Retriever] Reranking results...")
        results = self._rerank_results(results, query, constraints)
        
        return results[:k]
    
    def _retrieve_private(
        self,
        query: str,
        constraints: Dict[str, Any],
        k: int
    ) -> List[RetrievalResult]:
        """
        Retrieve from private Amazon 2020 catalog using hybrid search.

        """
        # Build metadata filter for FAISS
        metadata_filter = {}
        if 'category' in constraints:
            metadata_filter['category'] = constraints['category']
        if 'brand' in constraints:
            metadata_filter['brand'] = constraints['brand']
        
        # Perform vector similarity search
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query,
            k=k * 3,  
            filter=metadata_filter if metadata_filter else None
        )
        
        results = []
        for doc, distance_score in docs_and_scores:
            metadata = doc.metadata
            
            # Apply constraint filters (budget, material, etc.)
            if not self._passes_constraints(metadata, constraints):
                continue  # Skip products that don't meet requirements
            
            # Create result object with all information
            result = RetrievalResult(
                source='private',
                product_id=metadata.get('id', ''),
                title=metadata.get('title', ''),
                price=float(metadata.get('price', 0)),
                rating=float(metadata.get('rating')) if metadata.get('rating') else None,
                brand=metadata.get('brand'),
                features=metadata.get('features', []),
                ingredients=metadata.get('ingredients'),
                doc_id=metadata.get('doc_id'),  # Important for citations!
                url=None,
                relevance_score=1.0 - distance_score,  # Convert distance to similarity
                metadata=metadata
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def _retrieve_web(
        self,
        query: str,
        constraints: Dict[str, Any],
        k: int
    ) -> List[RetrievalResult]:
        """
        Retrieve live results via MCP web.search tool.
        """
        if not self.mcp_client:
            return []
        
        try:
            # Construct optimized search query
            search_query = self._construct_web_query(query, constraints)
            
            # Call MCP tool
            response = self.mcp_client.call_tool(
                'web.search',
                {'query': search_query, 'max_results': k}
            )
            
            if 'error' in response:
                print(f"[Retriever] Web search error: {response['error']}")
                return []
            
            # Parse and normalize results
            results = []
            for item in response.get('results', [])[:k]:
                result = RetrievalResult(
                    source='web',
                    product_id=item.get('product_id', ''),
                    title=item.get('title', ''),
                    price=self._extract_price(item),
                    rating=item.get('rating'),
                    brand=item.get('brand'),
                    features=[],
                    ingredients=None,
                    doc_id=None,
                    url=item.get('url'),  # Web results get URLs for citations
                    relevance_score=item.get('relevance_score', 0.5),
                    metadata=item
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"[Retriever] Web search failed: {e}")
            return []
    
    def _reconcile_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Reconcile conflicts between private and web results.

        """
        reconciled = []
        seen_products = {}
        
        for result in results:
            # Normalize title for matching
            key = self._normalize_title(result.title)
            
            if key in seen_products:
                # We've seen this product before (duplicate)
                existing = seen_products[key]
                
                # If existing is private and new is web, enrich the private result
                if existing.source == 'private' and result.source == 'web':
                    # Check for price discrepancy
                    if result.price and abs(existing.price - result.price) > 2.0:
                        existing.metadata['price_discrepancy'] = {
                            'private': existing.price,
                            'web': result.price,
                            'web_url': result.url
                        }
                    # Add web URL for verification
                    if result.url:
                        existing.metadata['web_url'] = result.url
            else:
                # First time seeing this product
                seen_products[key] = result
                reconciled.append(result)
        
        return reconciled
    
    def _rerank_results(
        self,
        results: List[RetrievalResult],
        query: str,
        constraints: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """
        Rerank results using LLM for semantic relevance.

        """
        if not results:
            return []
        
        # Prepare candidates for reranking
        candidates = []
        for i, result in enumerate(results):
            candidates.append({
                'index': i,
                'title': result.title,
                'brand': result.brand,
                'price': result.price,
                'rating': result.rating,
                'features': result.features[:3] if result.features else []
            })
        
        # LLM reranking prompt
        rerank_prompt = f"""Rank these products by relevance to the user's query and constraints.

Query: {query}
Constraints: {json.dumps(constraints, indent=2)}

Products:
{json.dumps(candidates, indent=2)}

Ranking criteria (in order):
1. Semantic match to query intent
2. Constraint satisfaction (budget, material, brand)
3. Rating quality (prefer >4.0★)
4. Price-value ratio

Return ONLY a JSON object:
{{"ranked_indices": [most_relevant_index, second_most, ...]}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You rank products by relevance. Return only valid JSON."),
                HumanMessage(content=rerank_prompt)
            ])
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                ranked_data = json.loads(json_match.group())
                ranked_indices = ranked_data.get('ranked_indices', [])
                
                # Reorder results based on LLM ranking
                reranked = [results[i] for i in ranked_indices if i < len(results)]
                # Append any missing results
                remaining = [r for i, r in enumerate(results) if i not in ranked_indices]
                reranked.extend(remaining)
                return reranked
                
        except Exception as e:
            print(f"[Retriever] Reranking failed, using fallback: {e}")
        
        # Fallback: sort by relevance score and rating
        return sorted(
            results,
            key=lambda r: (r.relevance_score * 0.7 + (r.rating or 0) / 5.0 * 0.3),
            reverse=True
        )
    
    def _passes_constraints(
        self,
        metadata: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> bool:
        """
        Check if product satisfies all constraints.
        
        Acts as a "bouncer" - only products that pass ALL checks get through.
        """
        # Budget constraint
        if 'max_price' in constraints:
            price = float(metadata.get('price', 999))
            if price > constraints['max_price']:
                return False  # Too expensive
        
        # Minimum rating constraint
        if 'min_rating' in constraints:
            rating = metadata.get('rating')
            if not rating or float(rating) < constraints['min_rating']:
                return False  # Rating too low
        
        # Material constraint (check in title/features)
        if 'material' in constraints:
            material = constraints['material'].lower()
            title = metadata.get('title', '').lower()
            features = ' '.join(metadata.get('features', [])).lower()
            if material not in title and material not in features:
                return False  # Doesn't match material
        
        # Eco-friendly constraint
        if constraints.get('eco_friendly', False):
            text = (metadata.get('title', '') + ' ' + 
                   ' '.join(metadata.get('features', []))).lower()
            eco_keywords = ['eco', 'green', 'plant-based', 'biodegradable', 
                          'sustainable', 'natural']
            if not any(kw in text for kw in eco_keywords):
                return False  # Not eco-friendly
        
        # Brand constraint
        if 'brand' in constraints:
            brand = metadata.get('brand', '').lower()
            constraint_brand = constraints['brand'].lower()
            if constraint_brand not in brand and brand not in constraint_brand:
                return False  # Wrong brand
        
        return True  # Passes all constraints!
    
    def _construct_web_query(self, query: str, constraints: Dict[str, Any]) -> str:
        """Construct optimized web search query from constraints."""
        parts = [query]
        
        if 'brand' in constraints:
            parts.append(constraints['brand'])
        if 'material' in constraints:
            parts.append(constraints['material'])
        if 'category' in constraints:
            parts.append(constraints['category'])
        
        # Add "price" for product-focused results
        parts.append('price')
        
        return ' '.join(parts)
    
    def _extract_price(self, item: Dict[str, Any]) -> float:
        """Extract price from web result with various formats."""
        price = item.get('price')
        if price:
            if isinstance(price, (int, float)):
                return float(price)
            # Handle string prices like "$12.99"
            match = re.search(r'[\d.]+', str(price))
            if match:
                try:
                    return float(match.group())
                except:
                    pass
        return 0.0
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication matching."""
        # Remove special chars, lowercase, take first 50 chars
        normalized = re.sub(r'[^a-z0-9]', '', title.lower())
        return normalized[:50]


# LangGraph Node Wrapper
def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for retriever agent.
    
    This function wraps the retriever for use in a LangGraph workflow.
    
    """
    retriever = state['retriever_agent']
    
    results = retriever.retrieve(
        query=state['query'],
        constraints=state.get('constraints', {}),
        plan=state.get('plan', {}),
        k=state.get('top_k', 5)
    )
    
    return {
        'retrieval_results': results,
        'retrieval_count': len(results),
        'sources_used': list(set(r.source for r in results))
    }


# Standalone test function
if __name__ == "__main__":
    print("\n=== RETRIEVER AGENT TEST ===\n")
    
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    
    # Load your vector store
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✓ Vector store loaded\n")
    
    # Initialize retriever
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    retriever = RetrieverAgent(vector_store, embeddings, llm)
    print("✓ Retriever initialized\n")
    
    # Test retrieval
    test_query = "eco-friendly stainless steel cleaner under $15"
    test_constraints = {
        'max_price': 15.0,
        'material': 'stainless steel',
        'eco_friendly': True
    }
    test_plan = {
        'use_live_search': False,
        'top_k': 5
    }
    
    print(f"Query: {test_query}")
    print(f"Constraints: {test_constraints}")
    print(f"Plan: {test_plan}\n")
    
    results = retriever.retrieve(test_query, test_constraints, test_plan)
    
    print(f"\n=== RESULTS ===")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   Price: ${result.price:.2f} | Rating: {result.rating}★")
        print(f"   Source: {result.source} | Doc ID: {result.doc_id}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        if result.features:
            print(f"   Features: {', '.join(result.features[:3])}")
        print()