from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from src.api.feedback_store import init_db
from src.api.routes.feedback import router as feedback_router
from src.api.routes.recommend import router as recommend_router
from src.api.schemas import HealthResponse
from src.constants import DEFAULT_CORPUS_PATH, DEFAULT_MODEL_DIR
from src.inference.serve_recommendations import (
    MonitoredRecommender,
    load_monitored_recommender,
)

logger = logging.getLogger(__name__)


def _resolve_model_dir() -> Path:
    value = os.getenv("MODEL_DIR")
    return Path(value) if value else DEFAULT_MODEL_DIR


def _resolve_corpus_path() -> Path:
    value = os.getenv("CORPUS_PATH")
    return Path(value) if value else DEFAULT_CORPUS_PATH


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler.

    On startup:
      - Initialize the feedback SQLite database.
      - Load the recommender model and corpus into memory.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Starting Instacart recommendation API service")

    # Initialize database
    logger.info("Initializing database")
    init_db()

    # Load recommender model
    model_dir = _resolve_model_dir().resolve()
    corpus_path = _resolve_corpus_path().resolve()
    logger.info("Loading recommender model_dir=%s corpus=%s", model_dir, corpus_path)
    recommender: MonitoredRecommender = load_monitored_recommender(
        model_dir=model_dir, corpus_path=corpus_path
    )

    # Set state variables
    app.state.recommender = recommender
    app.state.corpus_path = corpus_path
    app.state.ready = True

    # Yield control to FastAPI
    try:
        yield
    finally:
        logger.info("Shutting down Instacart recommendation API service")


# FastAPI app
app = FastAPI(title="Instacart Next-Order Recommendation API", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Simple structured logging for incoming HTTP requests.
    """
    start = time.time()
    req_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = req_id
    try:
        response: Response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        logger.exception(
            "request_error path=%s method=%s request_id=%s latency_ms=%d",
            request.url.path,
            request.method,
            req_id,
            elapsed_ms,
        )
        raise exc
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "request path=%s method=%s status=%d request_id=%s latency_ms=%d",
        request.url.path,
        request.method,
        response.status_code,
        req_id,
        elapsed_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    ready_flag = bool(getattr(request.app.state, "ready", False))
    if not ready_flag or not getattr(request.app.state, "recommender", None):
        # FastAPI will still return 200 for the response model; status override could be added if needed.
        return HealthResponse(status="not_ready")
    return HealthResponse(status="ready")


app.include_router(recommend_router)
app.include_router(feedback_router)
