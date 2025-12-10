"""
State definitions and schemas for the agentic system
"""
from typing import TypedDict, List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class ProductConstraints(BaseModel):
    """Extracted constraints from user query"""
    budget_max: Optional[float] = Field(None, description="Maximum price in USD")
    budget_min: Optional[float] = Field(None, description="Minimum price in USD")
    materials: List[str] = Field(default_factory=list, description="Required/preferred materials (e.g., stainless-steel, eco-friendly)")
    brands: List[str] = Field(default_factory=list, description="Preferred brands")
    categories: List[str] = Field(default_factory=list, description="Product categories (e.g., cleaning, kitchen)")
    keywords: List[str] = Field(default_factory=list, description="Key search terms")
    exclude_materials: List[str] = Field(default_factory=list, description="Materials to avoid")
    exclude_brands: List[str] = Field(default_factory=list, description="Brands to avoid")


class SafetyFlags(BaseModel):
    """Safety and content moderation flags"""
    is_safe: bool = Field(True, description="Overall safety check")
    concerns: List[str] = Field(default_factory=list, description="Specific safety concerns")
    requires_age_verification: bool = Field(False, description="Age-restricted product")
    potentially_harmful: bool = Field(False, description="Could be used harmfully")


class Intent(BaseModel):
    """Parsed user intent"""
    query_type: Literal["product_search", "comparison", "recommendation", "availability_check"] = Field(
        description="Type of query"
    )
    product_description: str = Field(description="What the user is looking for")
    constraints: ProductConstraints = Field(default_factory=ProductConstraints)
    safety: SafetyFlags = Field(default_factory=SafetyFlags)
    needs_comparison: bool = Field(False, description="User wants to compare multiple options")
    comparison_criteria: List[str] = Field(default_factory=list, description="What to compare (price, rating, ingredients)")


class DataSource(BaseModel):
    """Data source configuration"""
    use_private: bool = Field(True, description="Query private catalog")
    use_live: bool = Field(False, description="Query live web sources")
    private_fields: List[str] = Field(
        default_factory=lambda: ["title", "price", "rating", "brand"],
        description="Fields to retrieve from private DB"
    )
    live_fields: List[str] = Field(
        default_factory=lambda: ["price", "availability"],
        description="Fields to check from live sources"
    )


class RetrievalPlan(BaseModel):
    """Plan for data retrieval"""
    data_sources: DataSource = Field(default_factory=DataSource)
    vector_query: str = Field(description="Optimized query for vector search")
    metadata_filters: Dict = Field(default_factory=dict, description="Filters for private catalog")
    max_results: int = Field(5, description="Number of products to retrieve")
    rerank: bool = Field(True, description="Apply reranking after retrieval")
    reconcile_conflicts: bool = Field(False, description="Check for price/availability discrepancies")


class ProductResult(BaseModel):
    """Retrieved product information"""
    sku: Optional[str] = None
    title: str
    price: Optional[float] = None
    rating: Optional[float] = None
    brand: Optional[str] = None
    ingredients: Optional[str] = None
    category: Optional[str] = None
    source: Literal["private", "live", "both"] = "private"
    doc_id: Optional[str] = None  # For private catalog
    url: Optional[str] = None  # For live sources
    conflicts: List[str] = Field(default_factory=list, description="Discrepancies between sources")


class Citation(BaseModel):
    """Citation for grounding"""
    source_type: Literal["private_catalog", "web_search"]
    reference: str  # doc_id or URL
    snippet: str  # Supporting text
    field: str  # Which field this supports (price, rating, etc.)


class AgentState(TypedDict):
    """
    State passed between agents in the LangGraph workflow
    """
    # Input
    user_query: str
    transcript_timestamp: Optional[str]
    
    # Router outputs
    intent: Optional[Intent]
    
    # Planner outputs
    retrieval_plan: Optional[RetrievalPlan]
    
    # Retriever outputs
    retrieved_products: List[ProductResult]
    
    # Answerer outputs
    final_response: str
    citations: List[Citation]
    
    # Logging
    agent_logs: List[Dict]  # Each agent logs its reasoning
    tool_calls: List[Dict]  # Track all tool invocations


class AgentLog(BaseModel):
    """Structured log entry for transparency"""
    agent_name: str
    timestamp: str
    reasoning: str
    decision: Dict
    confidence: float = Field(ge=0.0, le=1.0)
