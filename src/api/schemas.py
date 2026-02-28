from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
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
    request_id: str
    recommendations: List[RecommendationItem]
    stats: Optional[InferenceStatistics] = None


EventType = Literal["impression", "click", "add_to_cart", "purchase"]


class FeedbackEvent(BaseModel):
    request_id: str
    event_type: EventType
    product_id: str
    user_id: Optional[str] = None
    user_context_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class FeedbackBatchRequest(BaseModel):
    events: List[FeedbackEvent]


class HealthResponse(BaseModel):
    status: str = "ok"
