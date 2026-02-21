"""
Data preparation for Instacart two-tower SBERT training.

Builds (anchor, positive) pairs:
- anchor: user context text (prior order history + optional order pattern)
- positive: product text (name, aisle, department)

Uses only prior orders for user context (no leakage from train order).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

import pandas as pd
from datasets import Dataset

from src.constants import DEFAULT_DATA_DIR, DEFAULT_PROCESSED_DIR
from src.utils import setup_colored_logging

CHUNK_SIZE = 500_000  # for reading order_products__prior.csv


def load_product_text_map(
    products_path: Path, aisles_path: Path, departments_path: Path
) -> dict[int, str]:
    """
    Build a mapping from product_id to a single text string for the item tower.

    Joins products with aisle and department names, then formats each product as
    "Product: {name}. Aisle: {aisle}. Department: {department}." for use as
    the "positive" (item) side of (anchor, positive) pairs.

    Args:
        products_path: Path to products.csv (product_id, product_name, aisle_id, department_id).
        aisles_path: Path to aisles.csv (aisle_id, aisle).
        departments_path: Path to departments.csv (department_id, department).

    Returns:
        Dict mapping product_id (int) to the formatted product text (str).
    """
    # Load the three CSV tables into DataFrames.
    products = pd.read_csv(products_path)
    aisles = pd.read_csv(aisles_path)
    departments = pd.read_csv(departments_path)

    # Join products with aisle and department so each row has name, aisle name, department name.
    products = products.merge(aisles, on="aisle_id").merge(
        departments, on="department_id"
    )
    # Build one string per product: "Product: X. Aisle: Y. Department: Z."
    products["text"] = (
        "Product: "
        + products["product_name"].astype(str)
        + ". Aisle: "
        + products["aisle"].astype(str)
        + ". Department: "
        + products["department"].astype(str)
        + "."
    )
    # Return as dict: product_id -> text.
    return dict(zip(products["product_id"], products["text"]))


def load_orders(orders_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load orders.csv and split into train vs prior by eval_set.

    Target orders are the "next" order we predict for each user; history orders
    are that user's history used only to build context (no leakage).

    Args:
        orders_path: Path to orders.csv (order_id, user_id, eval_set, order_number, etc.).

    Returns:
        Tuple of (target_orders, history_orders), each a DataFrame with relevant columns.
    """
    # Load all orders.
    orders = pd.read_csv(orders_path)
    # Normalize hour column to 2-digit string if it was read as object (e.g. "08").
    if orders["order_hour_of_day"].dtype == object:
        orders["order_hour_of_day"] = (
            orders["order_hour_of_day"].astype(str).str.zfill(2)
        )
    # Rows where we have a "next order" to predict; keep columns needed for context and split.
    target_orders = orders[orders["eval_set"] == "train"][
        [
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ]
    ].copy()
    # Rows that are historical orders (used only to build user context).
    # Keep dow / hour / days_since_prior_order so temporal patterns between orders are available.
    history_orders = orders[orders["eval_set"] == "prior"][
        [
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ]
    ].copy()
    return target_orders, history_orders


def build_order_to_products(
    order_products_prior_path: Path,
    history_order_ids: set[int],
    chunk_size: int = CHUNK_SIZE,
) -> dict[int, list[int]]:
    """
    Build a mapping from each history order_id to the list of product_ids in that order.

    Reads order_products__prior.csv in chunks to avoid loading ~32M rows at once.
    Only rows whose order_id is in history_order_ids are kept (e.g. orders for users
    we care about when using max_train_orders).

    Args:
        order_products_prior_path: Path to order_products__prior.csv.
        history_order_ids: Set of order_ids to include (typically history orders for our users).
        chunk_size: Number of rows per chunk when reading the CSV.

    Returns:
        Dict mapping order_id (int) to list of product_id (int) in that order.
    """
    # Defaultdict so we can append without checking if key exists.
    order_to_products: dict[int, list[int]] = defaultdict(list)
    # Stream the CSV in chunks to limit memory.
    for chunk in pd.read_csv(order_products_prior_path, chunksize=chunk_size):
        # Keep only rows for orders we care about.
        chunk = chunk[chunk["order_id"].isin(history_order_ids)]
        # Append each (order_id, product_id) to the list for that order.
        for order_id, product_id in chunk[["order_id", "product_id"]].itertuples(
            index=False
        ):
            order_to_products[order_id].append(product_id)
    return dict(order_to_products)


def build_user_context_for_target_orders(
    target_orders: pd.DataFrame,
    history_orders: pd.DataFrame,
    order_to_products: dict[int, list[int]],
    product_text_map: dict[int, str],
    max_prior_orders: int = 5,
    max_product_names: int = 20,
) -> dict[int, str]:
    """
    For each target order, build one user-context string using only that user's order history.

    Context format (compact): "[w{dow}h{hour}] name1, name2; [+{days}d w{dow}h{hour}] ... Next: +{gap}d w{dow}h{hour}"
    Used as the "anchor" (query) side in (anchor, positive) pairs. No leakage: only
    history orders with order_number < this target order are used.

    Args:
        target_orders: DataFrame of orders we predict (order_id, user_id, order_number, order_dow, etc.).
        history_orders: DataFrame of past orders used only for context (order_id, user_id, order_number).
        order_to_products: Mapping from order_id to list of product_ids (for history orders).
        product_text_map: Mapping from product_id to full product text (used to get name only).
        max_prior_orders: Max number of history orders per user to consider.
        max_product_names: Max number of product names to include in the context string.

    Returns:
        Dict mapping target order_id (int) to the user context string (str).
    """
    # Sort so we can take "last N" history orders per user by order_number.
    history_orders = history_orders.sort_values(["user_id", "order_number"])
    order_id_to_context: dict[int, str] = {}

    for _, row in target_orders.iterrows():
        # Ensure order_id is always treated as an integer (avoids float keys like 3178496.0)
        order_id = int(row["order_id"])
        # Get the history orders for the user that happened before the target order.
        user_history = history_orders[
            (history_orders["user_id"] == row["user_id"])
            & (history_orders["order_number"] < row["order_number"])
        ].tail(max_prior_orders)

        segments: list[str] = []
        total_products = 0

        # Walk each prior order in chronological order and encode its time features + products.
        for _, h in user_history.iterrows():
            if total_products >= max_product_names:
                break
            oid = int(h["order_id"])
            order_products = []
            for pid in order_to_products.get(oid, []):
                if pid not in product_text_map:
                    continue
                if total_products >= max_product_names:
                    break
                name = product_text_map[pid].split("Product: ")[1].split(".")[0].strip()
                order_products.append(name)
                total_products += 1

            if not order_products:
                continue

            dow = int(h["order_dow"])
            hour = (
                h["order_hour_of_day"]
                if isinstance(h["order_hour_of_day"], str)
                else str(int(h["order_hour_of_day"]))
            )
            # Compact time prefix: w=weekday 0-6, h=hour; +Nd = N days after previous
            if pd.isna(h["days_since_prior_order"]):
                time_prefix = f"w{dow}h{hour}"
            else:
                days_gap = int(h["days_since_prior_order"])
                time_prefix = f"+{days_gap}d w{dow}h{hour}"

            seg = f"[{time_prefix}] " + ", ".join(order_products)
            segments.append(seg)

        products_str = "; ".join(segments) if segments else "(no prior orders)"
        row_dow = int(row["order_dow"])
        row_hour = (
            row["order_hour_of_day"]
            if isinstance(row["order_hour_of_day"], str)
            else str(int(row["order_hour_of_day"]))
        )
        if pd.isna(row["days_since_prior_order"]):
            next_clause = f"Next: w{row_dow}h{row_hour}"
        else:
            gap = int(row["days_since_prior_order"])
            next_clause = f"Next: +{gap}d w{row_dow}h{row_hour}"
        context = f"{products_str}. {next_clause}"
        order_id_to_context[order_id] = context

    return order_id_to_context


def strip_next_order_from_context(context: str) -> str:
    """
    Remove the ' Next: ...' clause from a user context string.
    Use this for eval/serve when we don't know the next order time (only past orders are known).
    """
    if " Next:" in context:
        return context.split(" Next:")[0].strip()
    return context


def build_anchor_positive_pairs(
    order_products_train_path: Path,
    order_id_to_context: dict[int, str],
    product_text_map: dict[int, str],
) -> tuple[list[str], list[str], list[int]]:
    """
    Build (anchor, positive) training pairs from order_products__train. No need to have
    Negative pairs as we are not using a contrastive loss.

    Each row in the train order-products table gives one positive pair: the user
    context for that order (anchor) and the product text (positive). Also returns
    order_id per row so we can split train/eval by order without leakage.

    Args:
        order_products_train_path: Path to order_products__train.csv (order_id, product_id, ...).
        order_id_to_context: Mapping from train order_id to user context string.
        product_text_map: Mapping from product_id to product text.

    Returns:
        Tuple of (anchors, positives, order_ids): parallel lists of same length.
    """
    # Load the table of (order_id, product_id) for train orders.
    train_op = pd.read_csv(order_products_train_path)
    anchors: list[str] = []
    positives: list[str] = []
    order_ids: list[int] = []

    for _, row in train_op.iterrows():
        order_id = row["order_id"]
        product_id = row["product_id"]
        # Skip if we have no context for this order or no text for this product.
        if order_id not in order_id_to_context or product_id not in product_text_map:
            continue
        # One positive pair: user context (anchor) and product text (positive).
        anchors.append(order_id_to_context[order_id])
        positives.append(product_text_map[product_id])
        order_ids.append(order_id)

    return anchors, positives, order_ids


def _params_subdir(
    max_prior_orders: int,
    max_product_names: int,
    eval_frac: float,
    eval_serve_time: bool,
    sample_frac: float | None,
    max_target_orders: int | None,
) -> str:
    """Build a short subdir name from data prep params so different settings get different folders."""
    parts = [
        f"p{max_prior_orders}",
        f"mp{max_product_names}",
        f"ef{eval_frac}",
    ]
    if not eval_serve_time:
        parts.append("no_serve")
    if sample_frac is not None:
        parts.append(f"sf{sample_frac}")
    if max_target_orders is not None:
        parts.append(f"mt{max_target_orders}")
    return "_".join(parts)


def run(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    max_prior_orders: int = 5,
    max_product_names: int = 20,
    sample_frac: float | None = None,
    eval_frac: float = 0.1,
    eval_serve_time: bool = True,
    max_target_orders: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None, dict, dict, dict]:
    """
    Run the full data preparation pipeline and save outputs to disk.

    Loads CSVs, builds (anchor, positive) pairs, splits by order into train/eval,
    optionally samples the train set, and writes HuggingFace datasets plus
    eval_queries, eval_corpus, and eval_relevant_docs for InformationRetrievalEvaluator.

    When eval_serve_time=True (default), eval_queries have " Next: ..." stripped
    so evaluation matches production (we don't know next order time at serve time).

    Returns:
        train_dataset: HuggingFace Dataset with columns "anchor", "positive".
        eval_dataset: Eval Dataset (same format) or None if no eval pairs.
        eval_queries: Dict qid (order_id as str) -> query text.
        eval_corpus: Dict cid (product_id as str) -> document text.
        eval_relevant_docs: Dict qid -> set of cid (product_ids in that eval order).
    """
    # Ensure paths are Path objects. Write to a param-based subdir so different settings don't overwrite.
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    subdir = _params_subdir(
        max_prior_orders,
        max_product_names,
        eval_frac,
        eval_serve_time,
        sample_frac,
        max_target_orders,
    )
    effective_output_dir = output_dir / subdir
    effective_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output subdir (params): %s -> %s", subdir, effective_output_dir)

    logger.info("[Step 1/7] Loading product text map...")
    # Paths to all input CSVs.
    products_path = data_dir / "products.csv"
    aisles_path = data_dir / "aisles.csv"
    departments_path = data_dir / "departments.csv"
    orders_path = data_dir / "orders.csv"
    order_products_prior_path = data_dir / "order_products__prior.csv"
    order_products_train_path = data_dir / "order_products__train.csv"

    # Step 1: product_id -> product text (for item tower).
    product_text_map = load_product_text_map(
        products_path, aisles_path, departments_path
    )
    logger.info("  -> %d products", len(product_text_map))

    # Step 2: Load orders; optionally limit to first max_target_orders for fast runs.
    logger.info("[Step 2/7] Loading orders...")
    target_orders, history_orders = load_orders(orders_path)
    if max_target_orders is not None:
        target_orders = target_orders.head(max_target_orders)
    # Restrict history orders to users we need (reduces memory when using max_target_orders).
    users_needed = set(target_orders["user_id"].tolist())
    history_orders = history_orders[history_orders["user_id"].isin(users_needed)]
    history_order_ids = set(history_orders["order_id"].tolist())
    logger.info(
        "  -> target: %d orders, history: %d orders",
        len(target_orders),
        len(history_order_ids),
    )

    # Step 3: For each history order, list of product_ids (chunked read).
    logger.info("[Step 3/7] Building order -> products mapping (chunked read)...")
    order_to_products = build_order_to_products(
        order_products_prior_path, history_order_ids
    )
    logger.info("  -> %d orders with products", len(order_to_products))

    # Step 4: For each target order, build user context string from history orders only.
    logger.info("[Step 4/7] Building user context for target orders...")
    order_id_to_context = build_user_context_for_target_orders(
        target_orders,
        history_orders,
        order_to_products,
        product_text_map,
        max_prior_orders=max_prior_orders,
        max_product_names=max_product_names,
    )
    logger.info("  -> %d order contexts", len(order_id_to_context))

    # Step 5: (anchor, positive) pairs and order_id per row for split.
    logger.info("[Step 5/7] Building anchor-positive pairs...")
    anchors, positives, order_ids = build_anchor_positive_pairs(
        order_products_train_path,
        order_id_to_context,
        product_text_map,
    )
    logger.info("  -> %d pairs", len(anchors))

    # Step 6: Split by order so all pairs from an order go to same split (no leakage).
    logger.info("[Step 6/7] Splitting train/eval by order...")
    train_order_ids_all = set(order_id_to_context.keys())
    order_list = sorted(train_order_ids_all)
    n_eval = max(1, int(len(order_list) * eval_frac))
    eval_order_ids = set(order_list[-n_eval:])

    train_anchors, train_positives = [], []
    eval_anchors, eval_positives = [], []
    for a, p, oid in zip(anchors, positives, order_ids):
        if oid in eval_order_ids:
            eval_anchors.append(a)
            eval_positives.append(p)
        else:
            train_anchors.append(a)
            train_positives.append(p)

    # Optionally subsample train pairs (e.g. for quick experiments).
    if sample_frac is not None and sample_frac < 1.0:
        train_df = pd.DataFrame({"anchor": train_anchors, "positive": train_positives})
        train_df = train_df.sample(frac=sample_frac, random_state=seed)
        train_anchors = train_df["anchor"].tolist()
        train_positives = train_df["positive"].tolist()

    train_dataset = Dataset.from_dict(
        {"anchor": train_anchors, "positive": train_positives}
    )

    # Build eval artifacts for InformationRetrievalEvaluator: queries, corpus, relevant_docs.
    # When eval_serve_time=True, strip " Next: ..." from queries so eval matches production (we don't know next order at serve time).
    if eval_serve_time:
        logger.info("  -> Eval queries: stripping 'Next:' clause (serve-time aligned)")
        eval_queries = {
            str(oid): strip_next_order_from_context(order_id_to_context[oid])
            for oid in eval_order_ids
            if oid in order_id_to_context
        }
    else:
        eval_queries = {
            str(oid): order_id_to_context[oid]
            for oid in eval_order_ids
            if oid in order_id_to_context
        }
    eval_relevant_docs = {str(oid): set() for oid in eval_order_ids}
    train_op = pd.read_csv(order_products_train_path)
    for _, row in train_op.iterrows():
        oid = int(row["order_id"])
        oid_str = str(oid)
        if oid_str in eval_relevant_docs:
            eval_relevant_docs[oid_str].add(str(int(row["product_id"])))
    eval_corpus = {str(pid): text for pid, text in product_text_map.items()}

    eval_dataset = (
        Dataset.from_dict({"anchor": eval_anchors, "positive": eval_positives})
        if eval_anchors and eval_positives
        else None
    )
    logger.info(
        "  -> train: %d pairs, eval: %d pairs", len(train_anchors), len(eval_anchors)
    )

    # Save all outputs to effective_output_dir (param-based subdir).
    logger.info("[Step 7/7] Saving outputs to %s...", effective_output_dir)
    train_dataset.save_to_disk(str(effective_output_dir / "train_dataset"))
    if eval_dataset is not None:
        eval_dataset.save_to_disk(str(effective_output_dir / "eval_dataset"))
    with open(effective_output_dir / "eval_queries.json", "w") as f:
        json.dump(eval_queries, f, indent=0)
    with open(effective_output_dir / "eval_corpus.json", "w") as f:
        json.dump(eval_corpus, f, indent=0)
    with open(effective_output_dir / "eval_relevant_docs.json", "w") as f:
        json.dump({k: list(v) for k, v in eval_relevant_docs.items()}, f, indent=0)

    data_prep_params = {
        "data_dir": str(data_dir),
        "output_dir": str(effective_output_dir),
        "max_prior_orders": max_prior_orders,
        "max_product_names": max_product_names,
        "sample_frac": sample_frac,
        "eval_frac": eval_frac,
        "eval_serve_time": eval_serve_time,
        "max_target_orders": max_target_orders,
        "seed": seed,
        "n_train_pairs": len(train_anchors),
        "n_eval_pairs": len(eval_anchors),
        "n_eval_queries": len(eval_queries),
        "n_corpus": len(eval_corpus),
    }
    with open(effective_output_dir / "data_prep_params.json", "w") as f:
        json.dump(data_prep_params, f, indent=2)

    logger.info("Done. Saved to %s", effective_output_dir)
    logger.info(
        "Train with: python -m src.train.train_sbert --processed-dir %s",
        effective_output_dir,
    )
    return train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs


def main() -> None:
    """
    CLI entrypoint: parse arguments, call run(), and print summary.
    """
    parser = argparse.ArgumentParser(
        description="Prepare Instacart data for two-tower SBERT training"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to data/ folder"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Path to save processed datasets",
    )
    parser.add_argument(
        "--max-prior-orders",
        type=int,
        default=5,
        help="Max prior orders per user context (default 5 for faster training)",
    )
    parser.add_argument(
        "--max-product-names",
        type=int,
        default=20,
        help="Max product names in user context (default 20 for faster training)",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of train pairs to keep (e.g. 0.2)",
    )
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=0.1,
        help="Fraction of orders for eval (Information retrieval evaluator)",
    )
    parser.add_argument(
        "--no-eval-serve-time",
        action="store_true",
        help="Keep 'Next: ...' in eval queries (matches training anchor; eval is optimistic). Default: strip it so eval matches production.",
    )
    parser.add_argument(
        "--max-target-orders",
        type=int,
        default=None,
        help="Limit target orders (for quick runs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_colored_logging(
        quiet_loggers=["httpx", "httpcore", "huggingface_hub", "urllib3", "datasets"],
    )

    train_ds, eval_ds, eq, ec, er = run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_prior_orders=args.max_prior_orders,
        max_product_names=args.max_product_names,
        sample_frac=args.sample_frac,
        eval_frac=args.eval_frac,
        eval_serve_time=not args.no_eval_serve_time,
        max_target_orders=args.max_target_orders,
        seed=args.seed,
    )
    logger.info("Train dataset size: %d", len(train_ds))
    if eval_ds is not None:
        logger.info("Eval dataset size: %d", len(eval_ds))
    logger.info("Eval queries: %d, corpus size: %d", len(eq), len(ec))
    logger.info(
        "Saved to %s/%s",
        args.output_dir,
        _params_subdir(
            args.max_prior_orders,
            args.max_product_names,
            args.eval_frac,
            not args.no_eval_serve_time,
            args.sample_frac,
            args.max_target_orders,
        ),
    )


if __name__ == "__main__":
    main()
