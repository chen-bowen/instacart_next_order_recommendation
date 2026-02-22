"""
Run content-based and collaborative filtering baselines and print IR metrics.

Use the same eval set and metrics as the SBERT model so you can compare.
Example: uv run python -m src.baselines.run_baselines --processed-dir processed/p5_mp20_ef0.1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.baselines.content_based import ContentBasedBaseline
from src.baselines.collaborative_filtering import ItemItemCFBaseline, load_eval_data
from src.baselines.metrics import compute_ir_metrics
from src.constants import DEFAULT_DATA_DIR, DEFAULT_PROCESSED_DIR, PROJECT_ROOT
from src.utils import resolve_processed_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n--- {name} ---")
    print(f"  Accuracy@1:   {metrics['accuracy_at_1']:.4f}")
    print(f"  Accuracy@3:   {metrics['accuracy_at_3']:.4f}")
    print(f"  Accuracy@5:   {metrics['accuracy_at_5']:.4f}")
    print(f"  Accuracy@10:  {metrics['accuracy_at_10']:.4f}")
    print(f"  Recall@10:    {metrics['recall_at_10']:.4f}")
    print(f"  MRR@10:       {metrics['mrr_at_10']:.4f}")
    print(f"  NDCG@10:      {metrics['ndcg_at_10']:.4f}")
    print(f"  MAP@100:      {metrics['map_at_100']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run content-based and CF baselines")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed dir (e.g. processed/p5_mp20_ef0.1)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Raw data dir (orders.csv, order_products__prior.csv)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model for content-based baseline (same as training base; untrained)",
    )
    parser.add_argument(
        "--content-only",
        action="store_true",
        help="Run only content-based baseline (faster; CF reads full order_products__prior)",
    )
    parser.add_argument(
        "--cf-only",
        action="store_true",
        help="Run only collaborative filtering baseline",
    )
    args = parser.parse_args()

    processed_dir, msg = resolve_processed_dir(args.processed_dir, DEFAULT_PROCESSED_DIR)
    if msg:
        logger.info("%s", msg)
    logger.info("Processed dir: %s", processed_dir)

    eval_queries, eval_corpus, eval_relevant_docs = load_eval_data(processed_dir)
    logger.info("Eval queries: %d, corpus size: %d", len(eval_queries), len(eval_corpus))

    if not args.cf_only:
        # Content-based (untrained SBERT: same model, no fine-tuning)
        logger.info("Building content-based (untrained SBERT) baseline...")
        cb = ContentBasedBaseline(eval_queries, eval_corpus, model_name=args.model_name)
        cb_rankings = cb.rank_all()
        cb_metrics = compute_ir_metrics(cb_rankings, eval_relevant_docs)
        print_metrics("Content-based (untrained SBERT)", cb_metrics)

    if not args.content_only:
        # Collaborative filtering (item-item) — reads full order_products__prior, can take 10+ min
        logger.info("Building collaborative filtering (item-item) baseline...")
        cf = ItemItemCFBaseline(args.data_dir, processed_dir)
        cf_rankings = cf.rank_all(eval_query_ids=list(eval_queries.keys()))
        cf_metrics = compute_ir_metrics(cf_rankings, eval_relevant_docs)
        print_metrics("Collaborative filtering (item-item)", cf_metrics)

    if not args.content_only and not args.cf_only:
        print("\n--- Compare with SBERT (see README) ---")
        print("  After 4–5 epochs SBERT typically reaches Accuracy@10 ~0.54, Recall@10 ~0.13, NDCG@10 ~0.15.")


if __name__ == "__main__":
    main()
