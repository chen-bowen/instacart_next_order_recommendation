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
import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.baselines.collaborative_filtering import load_eval_data
from src.baselines.metrics import compute_ir_metrics
from src.constants import (
    DEFAULT_CONFIG_COMPARE,
    DEFAULT_MODEL_DIR,
    DEFAULT_PROCESSED_DIR,
    PROJECT_ROOT,
)
from src.utils import resolve_processed_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _rank_all(
    model: SentenceTransformer,
    eval_queries: dict[str, str],
    eval_corpus: dict[str, str],
    batch_size: int = 64,
) -> tuple[dict[str, list[str]], np.ndarray, np.ndarray]:
    """
    Rank corpus by cosine similarity for each query; return rankings and embeddings.

    Args:
        model: SentenceTransformer to encode query and corpus.
        eval_queries: Dict mapping query_id to query text.
        eval_corpus: Dict mapping product_id to document text.
        batch_size: Encode batch size.

    Returns:
        Tuple of (query_id -> ranked product_ids, query_embeddings, corpus_embeddings).
    """
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

    Also reports per-dimension std (low = dimensions unused).

    Args:
        query_emb: Query embeddings (n_queries, dim).
        corpus_emb: Corpus embeddings (n_corpus, dim).
        name: Prefix for returned metric keys (e.g. "untrained").
        sample_pairs: Number of random pairs to sample for mean cosine sim.

    Returns:
        Dict with keys {name}_query_mean_pairwise_cos_sim, {name}_corpus_*.
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


def load_config(config_path: Path | None = None) -> dict:
    """Load compare config from YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_COMPARE
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {
        "processed_dir": raw.get("processed_dir", str(DEFAULT_PROCESSED_DIR)),
        "model_dir": raw.get("model_dir", str(DEFAULT_MODEL_DIR)),
        "base_model": str(raw.get("base_model", "sentence-transformers/all-MiniLM-L6-v2")),
        "batch_size": int(raw.get("batch_size", 64)),
        "sample_queries": raw.get("sample_queries"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare untrained vs trained SBERT; report IR metrics and collapse indicators")
    parser.add_argument("--config", type=Path, default=None, help=f"Path to YAML config (default: {DEFAULT_CONFIG_COMPARE.relative_to(PROJECT_ROOT)})")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir, msg = resolve_processed_dir(Path(cfg["processed_dir"]), DEFAULT_PROCESSED_DIR)
    if msg:
        logger.info("%s", msg)
    logger.info("Processed dir: %s", processed_dir)

    eval_queries, eval_corpus, eval_relevant_docs = load_eval_data(processed_dir)
    logger.info("Eval queries: %d, corpus size: %d", len(eval_queries), len(eval_corpus))

    if cfg["sample_queries"] and cfg["sample_queries"] < len(eval_queries):
        rng = random.Random(123)
        qids = rng.sample(list(eval_queries.keys()), cfg["sample_queries"])
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
    model_path = Path(cfg["model_dir"]).resolve()
    if not model_path.exists():
        logger.error("Trained model dir not found: %s", model_path)
        return
    logger.info("Loading trained model: %s", model_path)

    trained_model = SentenceTransformer(str(model_path))
    logger.info("Ranking with trained model...")
    trained_rankings, q_emb_t, c_emb_t = _rank_all(trained_model, eval_queries, eval_corpus, batch_size=cfg["batch_size"])
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
