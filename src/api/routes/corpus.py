"""
Corpus upload endpoint: POST /admin/corpus.

Accepts a JSON corpus (product_id -> text) and replaces the in-memory recommender,
enabling users to provide their own product catalog without mounting files.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import verify_api_key
from src.api.limiter import limiter
from src.api.schemas import CorpusUploadRequest, CorpusUploadResponse
from src.constants import DEFAULT_MODEL_DIR, MAX_CORPUS_UPLOAD_PRODUCTS
from src.inference.serve_recommendations import MonitoredRecommender

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_model_dir() -> Path | str:
    """Resolve model path from MODEL_DIR env or default."""
    value = os.getenv("MODEL_DIR")
    return Path(value) if value else DEFAULT_MODEL_DIR


def _get_max_corpus_products() -> int:
    """Max products allowed for corpus upload (env: MAX_CORPUS_UPLOAD_PRODUCTS)."""
    val = os.getenv("MAX_CORPUS_UPLOAD_PRODUCTS")
    if val is None:
        return MAX_CORPUS_UPLOAD_PRODUCTS
    try:
        return int(val)
    except ValueError:
        return MAX_CORPUS_UPLOAD_PRODUCTS


@router.post(
    "/admin/corpus",
    response_model=CorpusUploadResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.exempt
async def corpus_upload_endpoint(
    payload: CorpusUploadRequest,
    request: Request,
    _: None = Depends(verify_api_key),
) -> CorpusUploadResponse:
    """
    Upload a product corpus and replace the in-memory recommender.

    Corpus format: dict of product_id (string) to product text (string),
    same as eval_corpus.json. Subsequent /recommend requests use the new corpus.
    user_id lookup does not work with uploaded corpus; use user_context instead.
    """
    n = len(payload.corpus)
    max_allowed = _get_max_corpus_products()
    if n > max_allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corpus has {n} products; max allowed is {max_allowed}.",
        )

    model_dir = _resolve_model_dir()
    temp_path = Path(tempfile.gettempdir()) / f"uploaded_corpus_{uuid.uuid4().hex}.json"

    try:
        with open(temp_path, "w") as f:
            json.dump(payload.corpus, f, indent=0)
    except OSError as e:
        logger.exception("Failed to write temp corpus file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to write corpus to temporary file.",
        ) from e

    try:
        recommender = MonitoredRecommender(model_dir=model_dir, corpus_path=temp_path)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        logger.exception("Failed to load recommender with uploaded corpus: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load recommender: {e}",
        ) from e

    request.app.state.recommender = recommender
    request.app.state.corpus_path = temp_path
    request.app.state.ready = True

    logger.info(
        "corpus_uploaded n_products=%d model_dir=%s temp_path=%s",
        n,
        model_dir,
        temp_path,
    )
    return CorpusUploadResponse(status="ok", n_products=n)
