"""
Pydantic schemas for the Instacart recommendation API.

Request/response models for /recommend and /feedback endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    """Request body for POST /recommend. Provide user_context or user_id, plus top_k."""

    user_context: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="Full user context string, e.g. '[+7d w4h14] Organic Milk, Whole Wheat Bread.'",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier that can be resolved to a stored eval query (order_id) for demos.",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    exclude_product_ids: List[str] = Field(
        default_factory=list,
        description="Optional list of product_ids to exclude from the ranking (e.g. already in cart).",
    )


class RecommendationItem(BaseModel):
    """Single recommended product with ID, similarity score, and optional display text."""

    product_id: str
    score: float
    product_text: Optional[str] = None


class InferenceStatistics(BaseModel):
    """Per-request metrics returned with recommendations."""

    total_latency_ms: float
    query_embedding_time_ms: float
    similarity_compute_time_ms: float
    num_recommendations: int
    top_score: float
    avg_score: float
    timestamp: float


class RecommendationResponse(BaseModel):
    """Response from POST /recommend. Includes request_id for feedback correlation."""

    request_id: str
    recommendations: List[RecommendationItem]
    stats: Optional[InferenceStatistics] = None


EventType = Literal["impression", "click", "add_to_cart", "purchase"]


class FeedbackEvent(BaseModel):
    """Single feedback event (impression, click, add_to_cart, purchase) for POST /feedback."""

    request_id: str
    event_type: EventType
    product_id: str
    user_id: Optional[str] = None
    user_context_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class FeedbackBatchRequest(BaseModel):
    """Batch of feedback events for POST /feedback."""

    events: List[FeedbackEvent]


class HealthResponse(BaseModel):
    """Response for /health and /ready probes."""

    status: str = "ok"


class CorpusUploadRequest(BaseModel):
    """Request body for POST /admin/corpus. Corpus format: product_id -> product text."""

    corpus: Dict[str, str] = Field(
        ...,
        description="Map of product_id (string) to product text (string), same format as eval_corpus.json.",
    )

    @field_validator("corpus")
    @classmethod
    def corpus_non_empty(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not v:
            raise ValueError("corpus must be non-empty")
        return v


class CorpusUploadResponse(BaseModel):
    """Response from POST /admin/corpus."""

    status: str = "ok"
    n_products: int = Field(..., description="Number of products in the uploaded corpus.")
