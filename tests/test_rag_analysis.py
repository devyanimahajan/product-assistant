"""
RAG Effectiveness Analysis for Product Discovery System

Tests retrieval quality at different corpus sizes with proper random sampling.
Fixes the flaw in the homework example by using random sampling instead of sequential.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from umap import UMAP


# Test queries from the spec
TEST_QUERIES = [
    "eco-friendly stainless steel cleaner under $15",
    "non-toxic paint for children under $10",
    "biodegradable cleaning supplies",
    "safe kitchen cleaner for stainless steel",
    "washable paint for kids art projects"
]

# Random seed for reproducibility
RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)


def load_full_database() -> Tuple[Chroma, HuggingFaceEmbeddings]:
    """Load the full ChromaDB with all products."""
    print("\n=== LOADING FULL DATABASE ===")
    
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    persist_dir = os.path.join(
        Path(__file__).parent.parent,
        "chroma_product_catalog"
    )
    
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder
    )
    
    # Get all documents to know corpus size
    collection = db._collection
    total_count = collection.count()
    
    print(f"Loaded database with {total_count} products")
    print(f"Persist directory: {persist_dir}")
    
    return db, embedder


def create_random_subset(
    db: Chroma, 
    embedder: HuggingFaceEmbeddings,
    fraction: float
) -> Chroma:
    """
    Create a random subset of the database.
    
    Args:
        db: Full ChromaDB instance
        embedder: Embedding function
        fraction: Fraction of corpus to sample (0.25, 0.5, 0.75)
    
    Returns:
        New ChromaDB with randomly sampled products
    """
    print(f"\n[SAMPLING] Creating {int(fraction*100)}% random subset...")
    
    # Get all documents
    collection = db._collection
    all_results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    total_count = len(all_results['ids'])
    sample_size = int(total_count * fraction)
    
    # RANDOM sampling (not sequential!)
    random_indices = np.random.choice(total_count, size=sample_size, replace=False)
    random_indices = sorted(random_indices)  # Sort for consistent behavior
    
    print(f"[SAMPLING] Selected {sample_size} random products from {total_count}")
    
    # Extract sampled data
    sampled_texts = [all_results['documents'][i] for i in random_indices]
    sampled_metadatas = [all_results['metadatas'][i] for i in random_indices]
    
    # Create new ChromaDB (in-memory)
    subset_db = Chroma.from_texts(
        texts=sampled_texts,
        embedding=embedder,
        metadatas=sampled_metadatas
    )
    
    print(f"[SAMPLING] Created subset database with {len(sampled_texts)} products")
    
    return subset_db


def test_query_at_size(
    db: Chroma,
    query: str,
    k: int = 5
) -> List[Tuple[Any, float]]:
    """
    Test a single query at a given corpus size.
    
    Returns:
        List of (document, score) tuples
    """
    results = db.similarity_search_with_score(query, k=k)
    return results


def analyze_retrieval_quality(
    databases: Dict[str, Chroma],
    query: str,
    k: int = 5
) -> Dict[str, Any]:
    """
    Analyze retrieval quality across different corpus sizes.
    
    Returns metrics for comparison.
    """
    results = {}
    
    for size_label, db in databases.items():
        docs_and_scores = test_query_at_size(db, query, k=k)
        
        if docs_and_scores:
            top_1_score = 1.0 - docs_and_scores[0][1]  # Convert distance to similarity
            avg_score = np.mean([1.0 - score for _, score in docs_and_scores])
            
            # Extract categories
            categories = []
            brands = []
            prices = []
            
            for doc, _ in docs_and_scores:
                meta = doc.metadata
                if 'category' in meta and meta['category']:
                    categories.append(meta['category'])
                if 'brand' in meta and meta['brand']:
                    brands.append(meta['brand'])
                if 'price' in meta and meta['price']:
                    try:
                        prices.append(float(meta['price']))
                    except:
                        pass
            
            results[size_label] = {
                'top_1_score': top_1_score,
                'avg_score': avg_score,
                'count': len(docs_and_scores),
                'categories': categories,
                'brands': brands,
                'prices': prices,
                'results': docs_and_scores
            }
        else:
            results[size_label] = {
                'top_1_score': 0.0,
                'avg_score': 0.0,
                'count': 0,
                'categories': [],
                'brands': [],
                'prices': [],
                'results': []
            }
    
    return results


def print_query_results(query: str, analysis: Dict[str, Any]):
    """Print formatted results for a single query."""
    print(f"\n{'='*80}")
    print(f"QUERY: \"{query}\"")
    print(f"{'='*80}\n")
    
    # Table header
    print(f"{'Corpus Size':<15} | {'Top-1 Score':<12} | {'Top-3 Avg':<12} | {'Count':<8}")
    print(f"{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    
    # Table rows
    for size_label in ['25%', '50%', '75%', '100%']:
        data = analysis[size_label]
        print(f"{size_label:<15} | {data['top_1_score']:<12.4f} | {data['avg_score']:<12.4f} | {data['count']:<8}")
    
    # Category distribution for 100%
    print(f"\n{'Category Distribution (100% corpus):'}")
    categories = analysis['100%']['categories']
    if categories:
        category_counts = defaultdict(int)
        for cat in categories:
            # Extract main category (before first |)
            main_cat = cat.split('|')[0].strip() if '|' in cat else cat
            category_counts[main_cat] += 1
        
        total = len(categories)
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}/{total} products")
    else:
        print("  - No categories found")
    
    # Price range
    prices = analysis['100%']['prices']
    if prices:
        print(f"\nPrice Range (100% corpus):")
        print(f"  - Min: ${min(prices):.2f}")
        print(f"  - Max: ${max(prices):.2f}")
        print(f"  - Avg: ${np.mean(prices):.2f}")
    
    # Show top 3 results from 100%
    print(f"\nTop 3 Results (100% corpus):")
    for i, (doc, distance) in enumerate(analysis['100%']['results'][:3], 1):
        similarity = 1.0 - distance
        meta = doc.metadata
        title = meta.get('title', 'Unknown')[:60]
        price = meta.get('price', 'N/A')
        print(f"  {i}. {title}... (score: {similarity:.4f}, price: ${price})")


def create_embedding_visualization(db: Chroma, output_path: str):
    """
    Create UMAP visualization of product embeddings.
    """
    print("\n=== CREATING EMBEDDING VISUALIZATION ===")
    
    # Get all embeddings and metadata
    collection = db._collection
    all_results = collection.get(include=['embeddings', 'metadatas'])
    
    embeddings = np.array(all_results['embeddings'])
    metadatas = all_results['metadatas']
    
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Extract categories for coloring
    categories = []
    for meta in metadatas:
        cat = meta.get('category', 'Unknown')
        # Get main category
        main_cat = cat.split('|')[0].strip() if '|' in cat else cat
        categories.append(main_cat)
    
    # Get top categories
    category_counts = defaultdict(int)
    for cat in categories:
        category_counts[cat] += 1
    
    top_categories = [cat for cat, _ in sorted(category_counts.items(), key=lambda x: -x[1])[:10]]
    
    # Map to colors
    category_colors = []
    for cat in categories:
        if cat in top_categories:
            category_colors.append(top_categories.index(cat))
        else:
            category_colors.append(len(top_categories))  # "Other" category
    
    # UMAP projection
    print("Projecting to 2D with UMAP...")
    reducer = UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=category_colors,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    
    plt.title("UMAP Projection of Product Embeddings", fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Create legend
    legend_labels = top_categories + ['Other']
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=plt.cm.tab10(i/10), markersize=8)
              for i in range(len(legend_labels))]
    plt.legend(handles, legend_labels, loc='best', framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    plt.close()


def print_summary(all_analyses: Dict[str, Dict[str, Any]]):
    """Print overall summary of analysis."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    # Calculate average improvements
    improvements_25_to_100 = []
    improvements_50_to_100 = []
    
    for query, analysis in all_analyses.items():
        score_25 = analysis['25%']['top_1_score']
        score_50 = analysis['50%']['top_1_score']
        score_100 = analysis['100%']['top_1_score']
        
        if score_25 > 0:
            improvements_25_to_100.append((score_100 - score_25) / score_25 * 100)
        if score_50 > 0:
            improvements_50_to_100.append((score_100 - score_50) / score_50 * 100)
    
    if improvements_25_to_100:
        avg_improvement = np.mean(improvements_25_to_100)
        print(f"\nRetrieval Quality:")
        print(f"  - Avg top-1 score improvement (25% -> 100%): {avg_improvement:+.1f}%")
    
    if improvements_50_to_100:
        avg_improvement = np.mean(improvements_50_to_100)
        print(f"  - Avg top-1 score improvement (50% -> 100%): {avg_improvement:+.1f}%")


def main():
    """Run complete RAG analysis."""
    print("\n" + "="*80)
    print("RAG ANALYSIS")
    print("="*80)
    
    # Load full database
    full_db, embedder = load_full_database()
    
    # Get corpus statistics
    collection = full_db._collection
    total_count = collection.count()
    
    print(f"\n{'='*80}")
    print("CORPUS SAMPLING STATISTICS")
    print(f"{'='*80}\n")
    print(f"Full corpus: {total_count} products")
    print(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    
    # Create random subsets
    databases = {
        '100%': full_db,
        '75%': create_random_subset(full_db, embedder, 0.75),
        '50%': create_random_subset(full_db, embedder, 0.50),
        '25%': create_random_subset(full_db, embedder, 0.25),
    }
    
    print(f"\n25% sample: {int(total_count * 0.25)} products (RANDOM)")
    print(f"50% sample: {int(total_count * 0.50)} products (RANDOM)")
    print(f"75% sample: {int(total_count * 0.75)} products (RANDOM)")
    
    # Test each query
    all_analyses = {}
    
    for query in TEST_QUERIES:
        analysis = analyze_retrieval_quality(databases, query, k=5)
        all_analyses[query] = analysis
        print_query_results(query, analysis)
    
    # Create UMAP visualization
    output_path = os.path.join(
        Path(__file__).parent,
        "outputs",
        "embedding_umap.png"
    )
    create_embedding_visualization(full_db, output_path)
    
    # Print summary
    print_summary(all_analyses)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
