import sys
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from load_data import load_amazon_products


def build_chroma_index(
    category_filter: str = "Household",
    max_products: int = None,
    persist_directory: str = "./chroma_product_catalog"
):
    """
    Build Chroma vector index for product search.
    
    Args:
        category_filter: Filter products by category (set to None for all products)
        max_products: Limit number of products (None = all)
        persist_directory: Where to save the index
    """
    print("\n=== BUILDING CHROMA INDEX ===\n")
    
    df = load_amazon_products(
        category_filter=category_filter,
        max_products=max_products
    )
    
    if len(df) == 0:
        print("ERROR: No products loaded. Check your filter settings.")
        return None
    
    texts = []
    metadatas = []
    
    print(f"\nPreparing {len(df)} products for indexing...")
    
    for idx, row in df.iterrows():
        # Text for embedding (title + features)
        texts.append(row['text_for_embedding'])
        
        # Metadata for filtering and citations
        metadatas.append({
            'doc_id': row['doc_id'],           # citations
            'title': row['title'],
            'category': row['category'],
            'price': float(row['price']),      # For budget filtering
            'brand': str(row['brand']) if row['brand'] else '',
            'features': '|'.join(row['features'][:5]),  # Store as string
            'url': row['url'] if 'url' in row else ''
        })
    
    print(f"Prepared {len(texts)} texts with metadata")
    
    print("\nInitializing embeddings model...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"\nBuilding Chroma index (saving to {persist_directory})...")
    db = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    print(f"\n[SUCCESS] Chroma index built successfully!")
    print(f"[SUCCESS] Indexed {len(texts)} products")
    print(f"[SUCCESS] Saved to: {persist_directory}")
    
    return db


def test_retrieval(db: Chroma):
    """Test the index with sample queries."""
    print("\n=== TESTING RETRIEVAL ===\n")
    
    test_queries = [
        "disposable plates and cups",
        "paper straws",
        "magnifier"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = db.similarity_search_with_score(query, k=3)
        
        print(f"Top {len(results)} results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. {doc.metadata['title'][:60]}...")
            print(f"Price: ${doc.metadata['price']:.2f}")
            print(f"Score: {score:.3f}")
            print(f"Doc ID: {doc.metadata['doc_id'][:20]}...")


if __name__ == "__main__":
    CATEGORY_FILTER = None  # Set to None for all products, any category string for subset
    MAX_PRODUCTS = None     # Set to number for testing, None for all
    PERSIST_DIR = "./chroma_product_catalog"
    
    print(f"Configuration:")
    print(f"Category Filter: {CATEGORY_FILTER or 'None (all products)'}")
    print(f"Max Products: {MAX_PRODUCTS or 'None (all matching)'}")
    print(f"Persist Directory: {PERSIST_DIR}")
    
    db = build_chroma_index(
        category_filter=CATEGORY_FILTER,
        max_products=MAX_PRODUCTS,
        persist_directory=PERSIST_DIR
    )
    
    if db:
        test_retrieval(db)
        
        print("\n=== INDEX LOADED ===")
