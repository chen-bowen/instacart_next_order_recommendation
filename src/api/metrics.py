"""
Prometheus metrics for the Instacart recommendation API.

Exposes only API-relevant metrics (no Python GC, process, or platform metrics).
Use API_REGISTRY with generate_latest() to export.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Custom registry: API metrics only (no python_gc_*, python_info, process_*)
API_REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Counters: recommendation and feedback request counts
# ---------------------------------------------------------------------------

RECOMMENDATION_REQUESTS_TOTAL = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
    ["status"],
    registry=API_REGISTRY,
)

FEEDBACK_EVENTS_TOTAL = Counter(
    "feedback_events_total",
    "Total number of feedback events ingested",
    ["event_type"],
    registry=API_REGISTRY,
)

# ---------------------------------------------------------------------------
# Histograms: latency (Prometheus/Grafana compute p50, p95, p99 from buckets)
# ---------------------------------------------------------------------------

RECOMMENDATION_LATENCY_SECONDS = Histogram(
    "recommendation_latency_seconds",
    "End-to-end latency for recommendation requests in seconds",
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0),
    registry=API_REGISTRY,
)

RECOMMENDATION_ENCODE_SECONDS = Histogram(
    "recommendation_encode_seconds",
    "Query embedding time in seconds (model forward pass)",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    registry=API_REGISTRY,
)

FEEDBACK_INGEST_LATENCY_SECONDS = Histogram(
    "feedback_ingest_latency_seconds",
    "Time to ingest feedback events in seconds",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5),
    registry=API_REGISTRY,
)

# ---------------------------------------------------------------------------
# Gauges: model readiness
# ---------------------------------------------------------------------------

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the recommender model and corpus are loaded and ready, 0 otherwise",
    registry=API_REGISTRY,
)
