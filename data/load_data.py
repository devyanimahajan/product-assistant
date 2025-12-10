"""
Load Amazon Product Dataset 2020 from HuggingFace.
"""

import pandas as pd
from datasets import load_dataset
from typing import Optional
import re


def parse_price(price_str: str) -> float:
    """
    Parse price string to float.
    
    Examples:
        "$12.99" -> 12.99
        "12.99" -> 12.99
        "" -> 0.0
    """
    if not price_str or pd.isna(price_str):
        return 0.0
    
    # Extract numbers and decimal point
    match = re.search(r'[\d.]+', str(price_str))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0


def parse_features(features_str: str) -> list:
    """
    Parse features string (pipe-separated) into list.
    
    Example:
        "Feature 1 | Feature 2 | Feature 3" -> ["Feature 1", "Feature 2", "Feature 3"]
    """
    if not features_str or pd.isna(features_str):
        return []
    
    features = str(features_str).split('|')

    features = [f.strip() for f in features if f.strip()]
    return features


def load_amazon_products(
    category_filter: Optional[str] = None,
    max_products: Optional[int] = None
) -> pd.DataFrame:
    """
    Load Amazon Product Dataset 2020 from HuggingFace.
    
    Args:
        category_filter: Filter by category (e.g., "Household", "Kitchen")
        max_products: Limit number of products (for testing)
    
    Returns:
        DataFrame with processed product data
    """
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("calmgoose/amazon-product-data-2020")
    df = dataset['train'].to_pandas()
    
    print(f"Loaded {len(df)} total products")
    
    print(f"Available columns: {list(df.columns)[:10]}...")  # Show first 10
    
    columns_to_keep = [
        'Uniq Id',
        'Product Name',
        'Category',
        'Selling Price',
        'About Product',
        'Product Url'
    ]
    
    # Safety check to avoid columns that don't exist
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    print(f"Keeping columns: {columns_to_keep}")
    
    df = df[columns_to_keep].copy()
    
    rename_map = {
        'Uniq Id': 'doc_id',
        'Product Name': 'title',
        'Category': 'category',
        'Selling Price': 'price_raw',
        'About Product': 'features_raw',
        'Product Url': 'url'
    }

    # Similar safety check to guarantee existence
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)
    
    if 'brand' not in df.columns:
        df['brand'] = None
    
    if category_filter:
        print(f"Filtering by category: {category_filter}")
        df = df[df['category'].str.contains(category_filter, case=False, na=False)]
        print(f"After filter: {len(df)} products")
    
    print("Parsing prices...")
    df['price'] = df['price_raw'].apply(parse_price)
    
    print("Parsing features...")
    df['features'] = df['features_raw'].apply(parse_features)
    
    df = df[df['price'] > 0]
    df = df[df['title'].notna()]
    df = df[df['title'].str.strip() != '']
    
    print(f"After cleaning: {len(df)} valid products")
    
    # Limit if specified
    if max_products:
        df = df.head(max_products)
        print(f"Limited to {len(df)} products")
    
    # Create combined text for embedding
    df['text_for_embedding'] = df.apply(
        lambda row: f"{row['title']} {' '.join(row['features'][:5])}", 
        axis=1
    )
    
    return df


if __name__ == "__main__":
    print("\n=== TESTING DATA LOADER ===\n")
    
    df = load_amazon_products(category_filter="Household", max_products=10)
    
    print("\n=== SAMPLE PRODUCTS ===\n")
    for idx, row in df.head(3).iterrows():
        print(f"ID: {row['doc_id']}")
        print(f"Title: {row['title']}")
        print(f"Brand: {row['brand']}")
        print(f"Price: ${row['price']:.2f}")
        print(f"Category: {row['category']}")
        print(f"Features: {row['features'][:3]}")
        print(f"Text for embedding: {row['text_for_embedding'][:100]}...")
        print("-" * 80)
