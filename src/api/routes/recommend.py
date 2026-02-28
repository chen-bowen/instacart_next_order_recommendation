"""
Recommendation endpoint: POST /recommend.

Returns top-k product recommendations for a user context. Instrumented with
Prometheus metrics for latency and request counts.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import verify_api_key
from src.api.metrics import (
    RECOMMENDATION_ENCODE_SECONDS,
    RECOMMENDATION_LATENCY_SECONDS,
    RECOMMENDATION_REQUESTS_TOTAL,
)
from src.api.schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    InferenceStatistics,
)
from src.inference.serve_recommendations import (
    MonitoredRecommender,
    Recommender,
    load_monitored_recommender,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_eval_queries(corpus_path: Path) -> dict[str, str]:
    """
    Load eval_queries.json from the same directory as the corpus, if present.

    This is primarily for demo / offline evaluation; in a production setting
    a dedicated user-context service would supply the query string.
    """
    queries_path = corpus_path.parent / "eval_queries.json"
    if not queries_path.exists():
        return {}
    try:
        with open(queries_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load eval_queries.json from %s", queries_path)
    return {}


def get_recommender(request: Request) -> Recommender:
    rec: Optional[Recommender] = getattr(request.app.state, "recommender", None)
    if rec is None:
        # As a fallback, try to load using default constants; readiness endpoint should catch this earlier.
        logger.warning("Recommender not preloaded; loading on-demand")
        rec = load_monitored_recommender()
        request.app.state.recommender = rec
    return rec


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
)
async def recommend_endpoint(
    payload: RecommendationRequest,
    request: Request,
    recommender: Recommender = Depends(get_recommender),
    _: None = Depends(verify_api_key),
) -> RecommendationResponse:
    """Get top-k product recommendations. Records Prometheus metrics on success/error."""
    start_time = time.perf_counter()
    try:
        # Resolve user context string
        context = payload.user_context
        if context is None and payload.user_id is not None:
            corpus_path: Path = getattr(request.app.state, "corpus_path", None) or recommender.corpus_path
            eval_queries = _load_eval_queries(Path(corpus_path))
            context = eval_queries.get(str(payload.user_id))
        if not context:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either user_context or a resolvable user_id must be provided.",
            )

        # Generate request_id that clients can reuse when sending feedback.
        request_id = str(uuid4())
        exclude_ids = set(payload.exclude_product_ids or [])
        user_id_str = str(payload.user_id) if payload.user_id is not None else None

        # MonitoredRecommender accepts user_id and sets _last_metrics; plain Recommender ignores user_id
        if isinstance(recommender, MonitoredRecommender):
            results = recommender.recommend(
                query=context,
                top_k=payload.top_k,
                user_id=user_id_str,
                exclude_product_ids=exclude_ids,
            )
        else:
            results = recommender.recommend(
                query=context,
                top_k=payload.top_k,
                exclude_product_ids=exclude_ids,
            )

        items = [
            RecommendationItem(
                product_id=pid,
                score=score,
                product_text=recommender.pid_to_text.get(pid),
            )
            for pid, score in results
        ]

        stats = None
        if isinstance(recommender, MonitoredRecommender) and recommender._last_metrics is not None:
            m = recommender._last_metrics
            stats = InferenceStatistics(
                total_latency_ms=m.total_latency_ms,
                query_embedding_time_ms=m.query_embedding_time_ms,
                similarity_compute_time_ms=m.similarity_compute_time_ms,
                num_recommendations=m.num_recommendations,
                top_score=m.top_score,
                avg_score=m.avg_score,
                timestamp=m.timestamp,
            )
            # Record encode time for Prometheus (model forward pass)
            RECOMMENDATION_ENCODE_SECONDS.observe(m.query_embedding_time_ms / 1000.0)

        elapsed = time.perf_counter() - start_time
        RECOMMENDATION_LATENCY_SECONDS.observe(elapsed)
        RECOMMENDATION_REQUESTS_TOTAL.labels(status="success").inc()

        logger.info(
            "recommendation_served request_id=%s top_k=%d",
            request_id,
            len(items),
        )
        return RecommendationResponse(request_id=request_id, recommendations=items, stats=stats)
    except HTTPException:
        # Client errors (e.g. 400) — count as error, re-raise for FastAPI
        RECOMMENDATION_REQUESTS_TOTAL.labels(status="error").inc()
        raise
    except Exception:
        # Server errors — count as error, re-raise
        RECOMMENDATION_REQUESTS_TOTAL.labels(status="error").inc()
        raise
