"""
Feedback endpoint: POST /feedback.

Ingests impression, click, add_to_cart, purchase events. Instrumented with
Prometheus metrics for event counts and ingest latency.
"""

from __future__ import annotations

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import verify_api_key
from src.api.feedback_store import FeedbackEventRecord, record_event, record_events
from src.api.metrics import FEEDBACK_EVENTS_TOTAL, FEEDBACK_INGEST_LATENCY_SECONDS
from src.api.schemas import FeedbackBatchRequest, FeedbackEvent

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
)
async def feedback_endpoint(
    payload: FeedbackBatchRequest | FeedbackEvent,
    _: None = Depends(verify_api_key),
) -> dict:
    """
    Ingest feedback events (impression, click, add_to_cart, purchase).

    Accepts either a single FeedbackEvent or a FeedbackBatchRequest with multiple events.
    """
    events: List[FeedbackEvent]
    if isinstance(payload, FeedbackBatchRequest):
        events = payload.events
    else:
        events = [payload]

    if not events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No feedback events provided.",
        )

    records = [
        FeedbackEventRecord(
            request_id=e.request_id,
            event_type=e.event_type,
            user_id=e.user_id,
            product_id=e.product_id,
            user_context_hash=e.user_context_hash,
            metadata=e.metadata,
            created_at=e.created_at,
        )
        for e in events
    ]

    start_time = time.perf_counter()
    if len(records) == 1:
        record_event(records[0])
    else:
        record_events(records)
    elapsed = time.perf_counter() - start_time

    # Record per-event-type counts for Prometheus
    for r in records:
        FEEDBACK_EVENTS_TOTAL.labels(event_type=r.event_type).inc()
    FEEDBACK_INGEST_LATENCY_SECONDS.observe(elapsed)

    logger.info(
        "feedback_ingested count=%d types=%s",
        len(records),
        {r.event_type for r in records},
    )
    return {"status": "accepted", "count": len(records)}

