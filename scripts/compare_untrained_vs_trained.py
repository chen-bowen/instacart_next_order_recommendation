"""
Compare untrained (frozen pretrained) vs trained SBERT on the same eval set.

Useful to check for embedding collapse: if the trained model performs worse than
the untrained one, or if embedding diversity drops sharply, collapse may be an issue.

Usage:
  uv run python scripts/compare_untrained_vs_trained.py
  uv run python scripts/compare_untrained_vs_trained.py --processed-dir processed/p5_mp20_ef0.1 --model-dir models/two_tower_sbert/final
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.baselines.collaborative_filtering import load_eval_data
from src.baselines.metrics import compute_ir_metrics
from src.constants import DEFAULT_MODEL_DIR, DEFAULT_PROCESSED_DIR
from src.utils import resolve_processed_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _rank_all(
    model: SentenceTransformer,
    eval_queries: dict[str, str],
    eval_corpus: dict[str, str],
    batch_size: int = 64,
) -> dict[str, list[str]]:
    """Rank corpus by cosine similarity for each query; return query_id -> ranked product_ids."""
    product_ids = list(eval_corpus.keys())
    corpus_texts = [eval_corpus[pid] for pid in product_ids]

    corpus_emb = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_ids = list(eval_queries.keys())
    query_texts = [eval_queries[qid] for qid in query_ids]
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    sim = cos_sim(query_emb, corpus_emb)
    if hasattr(sim, "cpu"):
        sim = sim.cpu().numpy()
    else:
        sim = np.asarray(sim)

    out: dict[str, list[str]] = {}
    for i, qid in enumerate(query_ids):
        scores = np.asarray(sim[i]).flatten()
        order = np.argsort(scores)[::-1]
        out[qid] = [product_ids[j] for j in order]
    return out, query_emb, np.asarray(corpus_emb)


def _embedding_collapse_metrics(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    name: str,
    sample_pairs: int = 2000,
) -> dict[str, float]:
    """
    Simple collapse indicators: high mean pairwise cosine sim = less diversity (collapse).
    Also report per-dimension std (low = dimensions unused).
    """
    rng = random.Random(42)
    q_n, q_d = query_emb.shape
    c_n, c_d = corpus_emb.shape

    # Mean pairwise cosine sim (sample to keep it fast)
    def sample_mean_cos_sim(emb: np.ndarray, n: int) -> float:
        if emb.shape[0] < 2:
            return 0.0
        indices = list(range(emb.shape[0]))
        sims = []
        for _ in range(min(n, len(indices) * (len(indices) - 1) // 2)):
            i, j = rng.sample(indices, 2)
            if i == j:
                continue
            s = float(np.dot(emb[i], emb[j]))  # already normalized
            sims.append(s)
        return float(np.mean(sims)) if sims else 0.0

    query_mean_sim = sample_mean_cos_sim(query_emb, sample_pairs)
    corpus_mean_sim = sample_mean_cos_sim(corpus_emb, sample_pairs)

    # Per-dimension std over corpus (mean over dims): low = many dimensions underused
    corpus_std_per_dim = np.std(corpus_emb, axis=0)
    corpus_mean_std = float(np.mean(corpus_std_per_dim))

    return {
        f"{name}_query_mean_pairwise_cos_sim": query_mean_sim,
        f"{name}_corpus_mean_pairwise_cos_sim": corpus_mean_sim,
        f"{name}_corpus_mean_std_per_dim": corpus_mean_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare untrained vs trained SBERT; report IR metrics and collapse indicators")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed dir (e.g. processed/p5_mp20_ef0.1)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to trained model checkpoint (e.g. models/two_tower_sbert/final)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Pretrained model name (must match training base for fair comparison)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encode batch size",
    )
    parser.add_argument(
        "--sample-queries",
        type=int,
        default=None,
        help="If set, use a random subset of eval queries (faster run)",
    )
    args = parser.parse_args()

    processed_dir, msg = resolve_processed_dir(args.processed_dir, DEFAULT_PROCESSED_DIR)
    if msg:
        logger.info("%s", msg)
    logger.info("Processed dir: %s", processed_dir)

    eval_queries, eval_corpus, eval_relevant_docs = load_eval_data(processed_dir)
    logger.info("Eval queries: %d, corpus size: %d", len(eval_queries), len(eval_corpus))

    if args.sample_queries and args.sample_queries < len(eval_queries):
        rng = random.Random(123)
        qids = rng.sample(list(eval_queries.keys()), args.sample_queries)
        eval_queries = {q: eval_queries[q] for q in qids}
        eval_relevant_docs = {q: eval_relevant_docs[q] for q in qids if q in eval_relevant_docs}
        logger.info("Sampled to %d queries", len(eval_queries))

    # ---- Untrained (frozen pretrained) ----
    logger.info("Loading untrained model: %s", args.base_model)
    untrained_model = SentenceTransformer(args.base_model)
    logger.info("Ranking with untrained model...")
    untrained_rankings, q_emb_u, c_emb_u = _rank_all(untrained_model, eval_queries, eval_corpus, batch_size=args.batch_size)
    untrained_metrics = compute_ir_metrics(untrained_rankings, eval_relevant_docs)
    collapse_u = _embedding_collapse_metrics(q_emb_u, c_emb_u, "untrained")

    # ---- Trained (your checkpoint) ----
    model_path = Path(args.model_dir).resolve()
    if not model_path.exists():
        logger.error("Trained model dir not found: %s", model_path)
        return
    logger.info("Loading trained model: %s", model_path)

    trained_model = SentenceTransformer(str(model_path))
    logger.info("Ranking with trained model...")
    trained_rankings, q_emb_t, c_emb_t = _rank_all(trained_model, eval_queries, eval_corpus, batch_size=args.batch_size)
    trained_metrics = compute_ir_metrics(trained_rankings, eval_relevant_docs)
    collapse_t = _embedding_collapse_metrics(q_emb_t, c_emb_t, "trained")

    # ---- Report ----
    def print_metrics(label: str, m: dict[str, float]) -> None:
        print(f"\n--- {label} ---")
        print(f"  Accuracy@1:   {m['accuracy_at_1']:.4f}  |  Accuracy@10: {m['accuracy_at_10']:.4f}")
        print(f"  Recall@10:   {m['recall_at_10']:.4f}  |  MRR@10:      {m['mrr_at_10']:.4f}")
        print(f"  NDCG@10:     {m['ndcg_at_10']:.4f}  |  MAP@100:     {m['map_at_100']:.4f}")

    print_metrics("Untrained (frozen pretrained)", untrained_metrics)
    print_metrics("Trained (your checkpoint)", trained_metrics)

    print("\n--- Embedding collapse indicators ---")
    print("  (Higher mean pairwise cos_sim = less diversity, possible collapse.)")
    print(f"  Untrained  query mean pairwise cos_sim: {collapse_u['untrained_query_mean_pairwise_cos_sim']:.4f}")
    print(f"  Untrained  corpus mean pairwise cos_sim: {collapse_u['untrained_corpus_mean_pairwise_cos_sim']:.4f}")
    print(f"  Untrained  corpus mean std per dim:      {collapse_u['untrained_corpus_mean_std_per_dim']:.4f}")
    print(f"  Trained    query mean pairwise cos_sim: {collapse_t['trained_query_mean_pairwise_cos_sim']:.4f}")
    print(f"  Trained    corpus mean pairwise cos_sim: {collapse_t['trained_corpus_mean_pairwise_cos_sim']:.4f}")
    print(f"  Trained    corpus mean std per dim:      {collapse_t['trained_corpus_mean_std_per_dim']:.4f}")

    print("\n--- Summary ---")
    better = "Trained" if trained_metrics["accuracy_at_10"] >= untrained_metrics["accuracy_at_10"] else "Untrained"
    print(f"  Accuracy@10: {better} is better ({trained_metrics['accuracy_at_10']:.4f} vs {untrained_metrics['accuracy_at_10']:.4f})")
    if trained_metrics["accuracy_at_10"] < untrained_metrics["accuracy_at_10"]:
        print("  -> Trained model underperforming untrained may indicate overfitting or embedding collapse.")
    delta_ndcg = trained_metrics["ndcg_at_10"] - untrained_metrics["ndcg_at_10"]
    print(f"  NDCG@10 delta (trained - untrained): {delta_ndcg:+.4f}")


if __name__ == "__main__":
    main()
