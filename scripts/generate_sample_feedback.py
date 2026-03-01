"""
Generate sample feedback by sending recommend + feedback requests to the API.

Populates the feedback DB (via the API) with realistic conversion funnels
(impression -> click -> add_to_cart -> purchase). Run feedback_analytics.py
separately for reports.

Usage:
  # Start the API first: uv run uvicorn src.api.main:app --port 8000
  uv run python scripts/generate_sample_feedback.py
  uv run python scripts/generate_sample_feedback.py --num-requests 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import httpx

from src.constants import DEFAULT_PROCESSED_DIR
from src.utils import resolve_processed_dir

# Sample user contexts when eval_queries.json is not available
SAMPLE_USER_CONTEXTS = [
    "[+7d w4h14] Organic Milk, Whole Wheat Bread.",
    "[+3d w1h9] Banana, Greek Yogurt, Honey.",
    "[+14d w6h18] Chicken Breast, Broccoli, Rice.",
    "[+1d w0h12] Coffee, Oat Milk, Granola.",
    "[+5d w3h20] Pasta, Tomato Sauce, Parmesan.",
]


def load_eval_user_ids(processed_dir: Path, limit: int = 50) -> list[str]:
    """
    Load user_ids (order_ids) from eval_queries.json for realistic queries.

    Args:
        processed_dir: Path to the processed directory.
        limit: Maximum number of user_ids to load.

    Returns:
        List of user_ids (up to limit), or empty list if file not found.
    """
    queries_path = processed_dir / "eval_queries.json"
    if not queries_path.exists():
        return []
    try:
        # eval_queries.json keys are order_ids (used as user/query identifiers)
        with open(queries_path) as f:
            data = json.load(f)
        ids = list(data.keys())[:limit]
        return [str(i) for i in ids]
    except (json.JSONDecodeError, OSError):
        return []


def post_recommend_request(
    client: httpx.Client,
    base_url: str,
    api_key: str | None,
    use_user_id: bool,
    user_id: str | None,
    user_context: str | None,
    top_k: int = 10,
) -> tuple[str | None, list[str]]:
    """
    POST /recommend and return (request_id, list of product_ids).

    Args:
        client: HTTP client.
        base_url: Base URL of the API.
        api_key: Optional API key for authenticated endpoints.
        use_user_id: Whether to use user_id (from eval) in the payload.
        user_id: User ID when use_user_id is True.
        user_context: User context string when not using user_id.
        top_k: Number of recommendations to return.

    Returns:
        Tuple of (request_id, list of product_ids).
    """
    # Optional API key for authenticated endpoints
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Either user_id (from eval) or user_context (sample string) for /recommend
    payload: dict = {"top_k": top_k}
    if use_user_id and user_id:
        payload["user_id"] = user_id
    else:
        payload["user_context"] = user_context or SAMPLE_USER_CONTEXTS[0]

    resp = client.post(f"{base_url}/recommend", json=payload, headers=headers or None)
    resp.raise_for_status()
    data = resp.json()
    request_id = data.get("request_id")
    recs = data.get("recommendations", [])
    product_ids = [r["product_id"] for r in recs]
    return request_id, product_ids


def get_feedbacks(
    client: httpx.Client,
    base_url: str,
    api_key: str | None,
    request_id: str,
    product_ids: list[str],
    click_rate: float = 0.15,
    atc_rate: float = 0.4,
    purchase_rate: float = 0.6,
) -> None:
    """
    Send feedback events: impression for each product, then random click/add_to_cart/purchase.

    Args:
        client: HTTP client.
        base_url: Base URL of the API.
        api_key: Optional API key for authenticated endpoints.
        request_id: Request ID from /recommend response.
        product_ids: List of recommended product IDs.
        click_rate: Fraction of impressions that become clicks.
        atc_rate: Fraction of clicks that become add-to-cart.
        purchase_rate: Fraction of add-to-cart that become purchases.
    """
    # Optional API key for authenticated endpoints
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # One impression per recommended product (position = rank)
    events = []
    for i, pid in enumerate(product_ids):
        events.append(
            {
                "request_id": request_id,
                "event_type": "impression",
                "product_id": pid,
                "user_id": None,
                "metadata": {"position": i + 1},
            }
        )

    # Probabilistic funnel: each product independently converts with given rate
    random.shuffle(product_ids)
    clicked = [p for p in product_ids if random.random() < click_rate]
    atc_pids = [p for p in clicked if random.random() < atc_rate]
    purch_pids = [p for p in atc_pids if random.random() < purchase_rate]

    # Append click / add_to_cart / purchase events for the simulated funnel
    for pid in clicked:
        events.append(
            {
                "request_id": request_id,
                "event_type": "click",
                "product_id": pid,
                "user_id": None,
            }
        )
    for pid in atc_pids:
        events.append(
            {
                "request_id": request_id,
                "event_type": "add_to_cart",
                "product_id": pid,
                "user_id": None,
            }
        )
    for pid in purch_pids:
        events.append(
            {
                "request_id": request_id,
                "event_type": "purchase",
                "product_id": pid,
                "user_id": None,
            }
        )

    # Send all events in one batch to /feedback
    resp = client.post(f"{base_url}/feedback", json={"events": events}, headers=headers or None)
    resp.raise_for_status()


def main() -> None:
    # CLI: URL, request count, API key, top-k, and funnel rates
    parser = argparse.ArgumentParser(description="Generate sample recommend + feedback requests to the API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="Number of recommend requests to send (default: 20)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key if API_KEY env is set on the server",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations per request (default: 10)",
    )
    parser.add_argument(
        "--click-rate",
        type=float,
        default=0.15,
        help="Fraction of impressions that become clicks (default: 0.15)",
    )
    parser.add_argument(
        "--atc-rate",
        type=float,
        default=0.4,
        help="Fraction of clicks that become add-to-cart (default: 0.4)",
    )
    parser.add_argument(
        "--purchase-rate",
        type=float,
        default=0.6,
        help="Fraction of add-to-cart that become purchases (default: 0.6)",
    )
    args = parser.parse_args()

    # Resolve processed dir and load eval user_ids so we can call /recommend with real queries
    processed_dir, _ = resolve_processed_dir(DEFAULT_PROCESSED_DIR, DEFAULT_PROCESSED_DIR)
    user_ids = load_eval_user_ids(processed_dir, limit=args.num_requests)
    use_user_id = len(user_ids) >= args.num_requests

    if not use_user_id:
        print("Note: eval_queries.json not found or too few; using sample user_context strings")

    # Prefer --api-key over API_KEY env
    api_key = args.api_key or __import__("os").environ.get("API_KEY")
    timeout = httpx.Timeout(60.0)

    # Check API is reachable before sending many requests
    try:
        with httpx.Client(timeout=5.0) as c:
            c.get(f"{args.url}/health")
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {args.url}. Start the API with: uv run uvicorn src.api.main:app --port 8000")
        sys.exit(1)

    success = 0
    failed = 0

    # For each request: /recommend then /feedback with simulated funnel
    with httpx.Client(timeout=timeout) as client:
        for i in range(args.num_requests):
            user_id = user_ids[i] if use_user_id and i < len(user_ids) else None
            user_context = None if use_user_id else random.choice(SAMPLE_USER_CONTEXTS)

            try:
                request_id, product_ids = post_recommend_request(client, args.url, api_key, use_user_id, user_id, user_context, args.top_k)
                if request_id and product_ids:
                    get_feedbacks(
                        client,
                        args.url,
                        api_key,
                        request_id,
                        product_ids,
                        click_rate=args.click_rate,
                        atc_rate=args.atc_rate,
                        purchase_rate=args.purchase_rate,
                    )
                    success += 1
                else:
                    failed += 1
            except httpx.HTTPError as e:
                print(f"Request {i + 1} failed: {e}")
                failed += 1

    # Summary: success = recommend + feedback both succeeded
    print(f"\nGenerated: {success} success, {failed} failed")
    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
