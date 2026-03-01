"""
Feedback analytics: compute CTR, add-to-cart rate, purchase rate from feedback_events.

Reads the SQLite feedback database and produces aggregate metrics plus per-request_id
conversion funnels. Useful for monitoring recommendation quality and tuning models.

Usage:
  uv run python scripts/feedback_analytics.py
  uv run python scripts/feedback_analytics.py --db-path data/feedback.db --since 2025-01-01
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.api.feedback_store import DEFAULT_FEEDBACK_DB_PATH, init_db


def _get_db_path() -> Path:
    """
    Resolve feedback DB path from env or default.

    Returns:
        Path to feedback.db (from FEEDBACK_DB_PATH or default).
    """
    import os

    value = os.getenv("FEEDBACK_DB_PATH")
    return Path(value) if value else DEFAULT_FEEDBACK_DB_PATH


def load_events(db_path: Path, since: str | None = None) -> list[tuple]:
    """
    Load feedback events from SQLite.

    Args:
        db_path: Path to feedback.db.
        since: Optional ISO date string; only load events created on or after this date.

    Returns:
        List of (request_id, event_type, product_id, user_id, created_at) tuples.
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        if since:
            cur.execute(
                """
                SELECT request_id, event_type, product_id, user_id, created_at
                FROM feedback_events
                WHERE created_at >= ?
                ORDER BY created_at
                """,
                (since,),
            )
        else:
            cur.execute(
                """
                SELECT request_id, event_type, product_id, user_id, created_at
                FROM feedback_events
                ORDER BY created_at
                """
            )
        rows = cur.fetchall()
        return [
            (
                r["request_id"],
                r["event_type"],
                r["product_id"],
                r["user_id"],
                r["created_at"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def compute_funnel_per_request(events: list[tuple]) -> dict[str, dict[str, set[str]]]:
    """
    Build per-request_id conversion funnel: which products had impression -> click -> add_to_cart -> purchase.

    Args:
        events: List of (request_id, event_type, product_id, user_id, created_at).

    Returns:
        Dict mapping request_id -> {event_type: set of product_ids}.
    """
    funnel: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for req_id, event_type, product_id, _, _ in events:
        if req_id:
            funnel[req_id][event_type].add(product_id)
    return dict(funnel)


def compute_aggregate_metrics(events: list[tuple]) -> dict[str, float]:
    """
    Compute aggregate CTR, add-to-cart rate, and purchase rate.

    Definitions:
      - CTR = unique (request_id, product_id) clicks / unique impressions
      - Add-to-cart rate = unique add_to_cart / unique impressions
      - Purchase rate = unique purchase / unique impressions

    Args:
        events: List of feedback events.

    Returns:
        Dict with impression_count, click_count, add_to_cart_count, purchase_count,
        ctr, add_to_cart_rate, purchase_rate.
    """
    impressions: set[tuple[str, str]] = set()
    clicks: set[tuple[str, str]] = set()
    add_to_carts: set[tuple[str, str]] = set()
    purchases: set[tuple[str, str]] = set()

    for req_id, event_type, product_id, _, _ in events:
        key = (req_id or "", product_id)
        if event_type == "impression":
            impressions.add(key)
        elif event_type == "click":
            clicks.add(key)
        elif event_type == "add_to_cart":
            add_to_carts.add(key)
        elif event_type == "purchase":
            purchases.add(key)

    n_imp = len(impressions)
    ctr = len(clicks) / n_imp if n_imp > 0 else 0.0
    atc_rate = len(add_to_carts) / n_imp if n_imp > 0 else 0.0
    purch_rate = len(purchases) / n_imp if n_imp > 0 else 0.0

    return {
        "impression_count": n_imp,
        "click_count": len(clicks),
        "add_to_cart_count": len(add_to_carts),
        "purchase_count": len(purchases),
        "ctr": ctr,
        "add_to_cart_rate": atc_rate,
        "purchase_rate": purch_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute feedback analytics: CTR, add-to-cart rate, purchase rate, per-request funnel"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to feedback.db (default: FEEDBACK_DB_PATH or data/feedback.db)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only include events on or after this date (ISO format, e.g. 2025-01-01)",
    )
    parser.add_argument(
        "--show-funnel-sample",
        type=int,
        default=3,
        help="Number of per-request funnels to print as sample (0 to disable)",
    )
    args = parser.parse_args()

    db_path = args.db_path or _get_db_path()

    # Ensure DB exists (init_db creates if missing)
    if args.db_path:
        import os

        os.environ["FEEDBACK_DB_PATH"] = str(args.db_path)
    init_db()
    db_path = args.db_path or _get_db_path()

    events = load_events(db_path, since=args.since)
    if not events:
        print(f"No feedback events found in {db_path}" + (f" since {args.since}" if args.since else ""))
        return

    metrics = compute_aggregate_metrics(events)
    print("\n--- Aggregate metrics ---")
    print(f"  Impressions (unique request+product): {metrics['impression_count']:,}")
    print(f"  Clicks: {metrics['click_count']:,}")
    print(f"  Add-to-cart: {metrics['add_to_cart_count']:,}")
    print(f"  Purchases: {metrics['purchase_count']:,}")
    print(f"  CTR (clicks/impressions): {metrics['ctr']:.4f}")
    print(f"  Add-to-cart rate: {metrics['add_to_cart_rate']:.4f}")
    print(f"  Purchase rate: {metrics['purchase_rate']:.4f}")

    funnel = compute_funnel_per_request(events)
    print(f"\n--- Per-request funnel ({len(funnel)} request_ids) ---")

    if args.show_funnel_sample > 0 and funnel:
        # Sort by purchase count descending so full-funnel conversions appear first
        def _funnel_depth(item: tuple) -> tuple:
            req_id, events_by_type = item
            purch = len(events_by_type.get("purchase", set()))
            atc = len(events_by_type.get("add_to_cart", set()))
            click = len(events_by_type.get("click", set()))
            return (-purch, -atc, -click, req_id or "")

        sorted_items = sorted(funnel.items(), key=_funnel_depth)
        for req_id, events_by_type in sorted_items[: args.show_funnel_sample]:
            imp = len(events_by_type.get("impression", set()))
            click = len(events_by_type.get("click", set()))
            atc = len(events_by_type.get("add_to_cart", set()))
            purch = len(events_by_type.get("purchase", set()))
            label = (req_id or "(no request_id)")[:20]
            print(f"  {label}: imp={imp} click={click} add_to_cart={atc} purchase={purch}")


if __name__ == "__main__":
    main()
