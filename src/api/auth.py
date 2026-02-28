"""
API key authentication for the recommendation API.

When API_KEY env var is set, requests to /recommend and /feedback must include
the key via X-API-Key header or Authorization: Bearer <key>. Health and ready
endpoints remain unauthenticated for liveness/readiness probes.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status


def _get_expected_api_key() -> str | None:
    """Return the expected API key from env, or None if auth is disabled."""
    import os

    return os.getenv("API_KEY") or None


def _extract_api_key(x_api_key: str | None = None, authorization: str | None = None) -> str | None:
    """
    Extract API key from X-API-Key header or Authorization: Bearer header.

    Returns the key if present, None otherwise.
    """
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """
    Dependency that verifies API key when API_KEY env is set.

    When API_KEY is not set, all requests pass (auth disabled).
    When set, /recommend and /feedback must include valid key in X-API-Key or Authorization: Bearer.
    Raises 401 if key is missing or invalid.
    """
    expected = _get_expected_api_key()
    if not expected:
        # Auth disabled - no verification
        return

    provided = _extract_api_key(x_api_key, authorization)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header or Authorization: Bearer <key>.",
        )
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
