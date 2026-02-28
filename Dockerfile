# Instacart Next-Order Recommendation API
#
# Multi-stage build: install deps with uv, run FastAPI via uvicorn.
# Expect MODEL_DIR, CORPUS_PATH, FEEDBACK_DB_PATH at runtime (or mount volumes).
#
# Build:  docker build -t instacart-rec-api .
# Run:    docker run -p 8000:8000 \
#           -v $(pwd)/models:/app/models \
#           -v $(pwd)/processed:/app/processed \
#           -v $(pwd)/data:/app/data \
#           -e MODEL_DIR=/app/models/two_tower_sbert/final \
#           -e CORPUS_PATH=/app/processed/p5_mp20_ef0.1/eval_corpus.json \
#           instacart-rec-api

# ---------------------------------------------------------------------------
# Stage 1: Build dependencies
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies (excluding dev, jupyter, etc.)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src/ ./src/

# Install project in editable mode
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./

# Create directories for mounts (models, processed, data, feedback db)
RUN mkdir -p /app/models /app/processed /app/data && chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check: GET /health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run uvicorn with graceful shutdown
# Override MODEL_DIR, CORPUS_PATH, FEEDBACK_DB_PATH, INFERENCE_DEVICE via env
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
