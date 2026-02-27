from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from src.inference.serve_recommendations import Recommender, load_recommender

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
        rec = load_recommender()
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
) -> RecommendationResponse:
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
    logger.info(
        "recommendation_served request_id=%s top_k=%d",
        request_id,
        len(items),
    )
    return RecommendationResponse(request_id=request_id, recommendations=items)

