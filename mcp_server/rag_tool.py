"""
RAG Search Tool - Query private Amazon Product Dataset 2020

Searches the Chroma vector database with metadata filtering.
Returns products with doc_id for citations.
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path to import from data/
sys.path.append(str(Path(__file__).parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Global variables for lazy loading
_embedder = None
_vector_db = None


def get_vector_db():
    """Lazy load the vector database."""
    global _embedder, _vector_db
    
    if _vector_db is None:
        print("[rag.search] Loading vector database...")
        
        # Initialize embeddings
        _embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load Chroma index
        persist_dir = os.path.join(
            Path(__file__).parent.parent,
            "data",
            "chroma_product_catalog"
        )
        
        _vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=_embedder
        )
        
        print(f"[rag.search] Vector database loaded from {persist_dir}")
    
    return _vector_db


def rag_search_tool(
    query: str,
    max_results: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search private Amazon Product Dataset 2020.
    
    Args:
        query: Natural language search query
        max_results: Maximum number of results to return
        filters: Optional filters (max_price, min_rating, brand, category)
    
    Returns:
        {
            "success": bool,
            "results": [
                {
                    "doc_id": str,  # For citations!
                    "title": str,
                    "price": float,
                    "brand": str,
                    "category": str,
                    "features": list,
                    "relevance_score": float
                }
            ]
        }
    """
    try:
        db = get_vector_db()
        
        # Build metadata filter for Chroma
        where_filter = None
        if filters:
            where_conditions = []
            
            if 'max_price' in filters:
                where_conditions.append({
                    "price": {"$lte": filters['max_price']}
                })
            
            if 'brand' in filters and filters['brand']:
                where_conditions.append({
                    "brand": {"$eq": filters['brand']}
                })
            
            if 'category' in filters and filters['category']:
                where_conditions.append({
                    "category": {"$contains": filters['category']}
                })
            
            # Combine conditions with AND
            if where_conditions:
                if len(where_conditions) == 1:
                    where_filter = where_conditions[0]
                else:
                    where_filter = {"$and": where_conditions}
        
        docs_and_scores = db.similarity_search_with_score(
            query,
            k=max_results * 2,  # Get extra for filtering
            filter=where_filter
        )
        
        results = []
        for doc, distance_score in docs_and_scores[:max_results]:
            metadata = doc.metadata
            
            # Apply additional filters not supported by Chroma
            if filters:
                if 'min_rating' in filters and metadata.get('rating'):
                    if float(metadata['rating']) < filters['min_rating']:
                        continue
            
            features_str = metadata.get('features', '')
            features = features_str.split('|') if features_str else []
            
            result = {
                "doc_id": metadata.get('doc_id', ''),
                "title": metadata.get('title', ''),
                "price": float(metadata.get('price', 0)),
                "brand": metadata.get('brand', ''),
                "category": metadata.get('category', ''),
                "features": features,
                "relevance_score": 1.0 - distance_score,
                "source": "private"
            }
            results.append(result)
        
        return {
            "success": True,
            "query": query,
            "filters_applied": filters or {},
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"RAG search failed: {str(e)}",
            "query": query,
            "results": []
        }
