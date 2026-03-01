"""
Serve product recommendations using the trained two-tower SBERT model.

Inference is embedding-based (no text generation):
  - We encode the user context string and each product's text with the same SentenceTransformer.
  - Recommendations are the top-k products by cosine similarity between query and product embeddings.
  - One shared encoder is used for both "towers" (query and document); there is no model.generate().

Usage:
  # Run inference: python -m src.inference
  python -m src.inference --model-dir models/two_tower_sbert/final --corpus processed/p5_mp20_ef0.1/eval_corpus.json
  # Or use as a module:
  from src.inference.serve_recommendations import load_recommender, recommend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.constants import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_MODEL_DIR,
    EMBEDDINGS_FILENAME,
    INDEX_SUBDIR,
    MANIFEST_FILENAME,
    PRODUCT_IDS_FILENAME,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env")


def get_inference_device() -> str:
    """
    Return the best available device for inference: cuda > mps > cpu.

    Override with INFERENCE_DEVICE env var (e.g. INFERENCE_DEVICE=cpu).

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    override = os.getenv("INFERENCE_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends.mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def embedding_index_dir(corpus_path: Path, model_dir: Path | str) -> Path:
    """
    Directory for cached product embeddings; one cache per (model_dir, corpus_path) pair.

    Args:
        corpus_path: Path to the corpus JSON.
        model_dir: Model directory or Hugging Face model ID.

    Returns:
        Path to the index subdirectory under the corpus parent.
    """
    canonical = f"{model_dir!s}|{corpus_path.resolve()!s}"
    name = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return Path(corpus_path).parent / INDEX_SUBDIR / name


def load_embedding_index(
    index_dir: Path, corpus_path: Path, model_dir: Path | str
) -> tuple[np.ndarray, list[str]] | None:
    """
    Load cached product embeddings if manifest matches current corpus and model.

    Args:
        index_dir: Directory containing the cached index.
        corpus_path: Path to the corpus JSON.
        model_dir: Model directory or Hugging Face model ID.

    Returns:
        (embeddings, product_ids) on cache hit, None on miss or invalid cache.
    """
    manifest_path = index_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    corpus_resolved = str(Path(corpus_path).resolve())
    model_resolved = str(model_dir)
    try:
        if meta.get("corpus_path") != corpus_resolved or meta.get("model_dir") != model_resolved:
            return None
        if meta.get("corpus_mtime") != Path(corpus_path).stat().st_mtime:
            return None
    except OSError:
        return None
    emb_path = index_dir / EMBEDDINGS_FILENAME
    ids_path = index_dir / PRODUCT_IDS_FILENAME
    if not emb_path.exists() or not ids_path.exists():
        return None
    try:
        embeddings = np.load(emb_path)
        with open(ids_path, "r") as f:
            product_ids = json.load(f)
    except (OSError, ValueError):
        return None
    if len(product_ids) != len(embeddings):
        return None
    return embeddings, product_ids


def save_embedding_index(
    index_dir: Path,
    corpus_path: Path,
    model_dir: Path | str,
    product_ids: list[str],
    embeddings: np.ndarray,
) -> None:
    """
    Write manifest, embeddings, and product_ids to disk for later loading.

    Args:
        index_dir: Directory to write the index into.
        corpus_path: Path to the corpus JSON (for manifest).
        model_dir: Model directory or Hugging Face model ID (for manifest).
        product_ids: List of product IDs in same order as embeddings.
        embeddings: Product embeddings array.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtime = Path(corpus_path).stat().st_mtime
    except OSError:
        mtime = 0
    manifest = {
        "corpus_path": str(Path(corpus_path).resolve()),
        "model_dir": str(model_dir),
        "corpus_mtime": mtime,
        "n_products": len(product_ids),
    }
    with open(index_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, indent=2)
    np.save(index_dir / EMBEDDINGS_FILENAME, embeddings.astype(np.float32))
    with open(index_dir / PRODUCT_IDS_FILENAME, "w") as f:
        json.dump(product_ids, f)
    logger.info("Saved embedding index to %s (%d products)", index_dir, len(product_ids))


def normalize_model_dir(model_dir: Path | str) -> Path | str:
    """
    Resolve local paths for stable cache keys; leave Hugging Face model IDs unchanged.

    Args:
        model_dir: Local path or Hub model ID (e.g. username/repo-name).

    Returns:
        Resolved Path if local path exists, otherwise unchanged (Path or str) for Hub ID.
    """
    p = Path(model_dir)
    if p.exists():
        return p.resolve()
    return p


@dataclass
class RecommendationMetrics:
    """Metrics logged per recommendation request."""

    user_id: str
    query_embedding_time_ms: float
    similarity_compute_time_ms: float
    total_latency_ms: float
    num_recommendations: int
    top_score: float
    avg_score: float
    timestamp: float


def load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    """
    Load product corpus from JSON: {product_id: product_text}.

    Args:
        corpus_path: Path to the corpus JSON file.

    Returns:
        Tuple of (product_ids, product_texts) in matching order.
    """
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    ids = list(corpus.keys())
    texts = [corpus[pid] for pid in ids]
    return ids, texts


class Recommender:
    """
    Two-tower recommender: encodes user context and products, returns top-k by cosine similarity.

    Product embeddings are cached on disk (by corpus + model) so they are not recomputed every startup.
    """

    def __init__(
        self,
        model_dir: Path | str,
        corpus_path: Path,
        batch_size: int = 64,
        use_index: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.corpus_path = Path(corpus_path)
        self.product_ids, self.product_texts = load_corpus(self.corpus_path)
        self.pid_to_text = dict(zip(self.product_ids, self.product_texts))

        device = get_inference_device()
        logger.info("Using inference device: %s", device)
        self.model = SentenceTransformer(str(self.model_dir), device=device)
        index_dir = embedding_index_dir(self.corpus_path, self.model_dir)
        cached = load_embedding_index(index_dir, self.corpus_path, self.model_dir) if use_index else None
        if cached is not None:
            embeddings, ids = cached
            if ids == self.product_ids:
                self.product_embeddings = embeddings
                logger.info(
                    "Loaded model from %s, corpus %d products from %s (product embeddings from index)",
                    model_dir,
                    len(self.product_ids),
                    corpus_path,
                )
            else:
                cached = None
        if cached is None:
            self.product_embeddings = self.model.encode(
                self.product_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            if use_index:
                save_embedding_index(
                    index_dir,
                    self.corpus_path,
                    self.model_dir,
                    self.product_ids,
                    self.product_embeddings,
                )
            logger.info(
                "Loaded model from %s, corpus %d products from %s",
                model_dir,
                len(self.product_ids),
                corpus_path,
            )

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        exclude_product_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return top-k product recommendations for the given user context.

        Args:
            query: User context string (same format as anchor in training).
            top_k: Number of recommendations to return.
            exclude_product_ids: Optional set of product IDs to exclude (e.g. already in cart).

        Returns:
            List of (product_id, score) sorted by score descending. Score is cosine similarity.
        """
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        scores = cos_sim(query_emb, self.product_embeddings)[0]
        indices = scores.argsort(descending=True)

        results: list[tuple[str, float]] = []
        excluded = exclude_product_ids or set()
        for idx in indices:
            pid = self.product_ids[idx]
            if pid in excluded:
                continue
            results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results


class MonitoredRecommender(Recommender):
    """Recommender with built-in timing and metrics logging. Sets last_metrics after each recommend()."""

    def __init__(self, *args, metrics_logger: Optional[logging.Logger] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metrics_logger = metrics_logger or logging.getLogger("recommender.metrics")
        self.last_metrics: Optional[RecommendationMetrics] = None

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        exclude_product_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Recommend with instrumentation; sets self.last_metrics for API stats.

        Args:
            query: User context string.
            top_k: Number of recommendations to return.
            user_id: Optional user ID for logging.
            exclude_product_ids: Optional set of product IDs to exclude.

        Returns:
            List of (product_id, score) sorted by score descending.
        """
        start_time = time.time()

        encode_start = time.time()
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        encode_time_ms = (time.time() - encode_start) * 1000

        sim_start = time.time()
        scores = cos_sim(query_emb, self.product_embeddings)[0]
        indices = scores.argsort(descending=True)
        sim_time_ms = (time.time() - sim_start) * 1000

        results: list[tuple[str, float]] = []
        excluded = exclude_product_ids or set()
        for idx in indices:
            pid = self.product_ids[idx]
            if pid in excluded:
                continue
            results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break

        total_time_ms = (time.time() - start_time) * 1000
        top_score = results[0][1] if results else 0.0
        avg_score = sum(s for _, s in results) / len(results) if results else 0.0

        self.last_metrics = RecommendationMetrics(
            user_id=user_id or "anonymous",
            query_embedding_time_ms=encode_time_ms,
            similarity_compute_time_ms=sim_time_ms,
            total_latency_ms=total_time_ms,
            num_recommendations=len(results),
            top_score=top_score,
            avg_score=avg_score,
            timestamp=time.time(),
        )
        self.log_metrics(self.last_metrics)

        return results

    def log_metrics(self, metrics: RecommendationMetrics) -> None:
        """
        Log metrics in structured format for aggregation (ELK, Datadog, CloudWatch).

        Args:
            metrics: RecommendationMetrics from the last recommend() call.
        """
        self.metrics_logger.info(
            "recommendation_served",
            extra={
                "user_id": metrics.user_id,
                "latency_ms": metrics.total_latency_ms,
                "encode_time_ms": metrics.query_embedding_time_ms,
                "similarity_time_ms": metrics.similarity_compute_time_ms,
                "num_results": metrics.num_recommendations,
                "top_score": metrics.top_score,
                "avg_score": metrics.avg_score,
            },
        )


recommender_cache: dict[tuple[Path | str, Path, bool], Recommender] = {}
monitored_recommender_cache: dict[tuple[Path | str, Path, bool], MonitoredRecommender] = {}


def load_recommender(
    model_dir: Path = DEFAULT_MODEL_DIR,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    batch_size: int = 64,
    use_index: bool = True,
) -> Recommender:
    """
    Load model and corpus, return a Recommender instance (for repeated recommend() calls).

    If the same (model_dir, corpus_path, use_index) was already loaded this session,
    returns that cached instance (no reload).

    Args:
        model_dir: Local path to trained model or Hugging Face model ID (e.g. username/repo-name).
        corpus_path: Path to eval_corpus.json.
        batch_size: Batch size for encoding products.
        use_index: Whether to load/save product embedding cache.

    Returns:
        Recommender instance.
    """
    model_dir = normalize_model_dir(model_dir)
    corpus_path = Path(corpus_path).resolve()
    key = (model_dir, corpus_path, use_index)
    if key in recommender_cache:
        logger.info(
            "Reusing recommender from session cache (model %s, corpus %s)",
            model_dir,
            corpus_path.name,
        )
        return recommender_cache[key]
    rec = Recommender(
        model_dir=model_dir,
        corpus_path=corpus_path,
        batch_size=batch_size,
        use_index=use_index,
    )
    recommender_cache[key] = rec
    return rec


def load_monitored_recommender(
    model_dir: Path = DEFAULT_MODEL_DIR,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    batch_size: int = 64,
    use_index: bool = True,
    metrics_logger: Optional[logging.Logger] = None,
) -> MonitoredRecommender:
    """
    Load model and corpus, return a MonitoredRecommender (timing + metrics logging).

    Same caching behavior as load_recommender; model_dir can be local path or Hugging Face model ID.

    Args:
        model_dir: Local path to trained model or Hugging Face model ID.
        corpus_path: Path to eval_corpus.json.
        batch_size: Batch size for encoding products.
        use_index: Whether to load/save product embedding cache.
        metrics_logger: Optional logger for recommendation_served events.

    Returns:
        MonitoredRecommender instance.
    """
    model_dir = normalize_model_dir(model_dir)
    corpus_path = Path(corpus_path).resolve()
    key = (model_dir, corpus_path, use_index)
    if key in monitored_recommender_cache:
        logger.info(
            "Reusing monitored recommender from session cache (model %s, corpus %s)",
            model_dir,
            corpus_path.name,
        )
        return monitored_recommender_cache[key]
    rec = MonitoredRecommender(
        model_dir=model_dir,
        corpus_path=corpus_path,
        batch_size=batch_size,
        use_index=use_index,
        metrics_logger=metrics_logger,
    )
    monitored_recommender_cache[key] = rec
    return rec


def recommend(
    query: str,
    top_k: int = 10,
    model_dir: Path = DEFAULT_MODEL_DIR,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
) -> list[tuple[str, float]]:
    """
    One-shot: load recommender and return top-k for the query.

    For repeated calls, use load_recommender() once and call .recommend() on it.

    Args:
        query: User context string.
        top_k: Number of recommendations to return.
        model_dir: Local path or Hugging Face model ID.
        corpus_path: Path to eval_corpus.json.

    Returns:
        List of (product_id, score) sorted by score descending.
    """
    rec = load_recommender(model_dir=model_dir, corpus_path=corpus_path)
    return rec.recommend(query=query, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve product recommendations")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to trained model or Hugging Face model ID (e.g. username/repo-name)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to eval_corpus.json",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Disable loading/saving product embedding index (recompute embeddings every time)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="User context string for recommendations",
    )
    parser.add_argument(
        "--eval-query-id",
        type=str,
        default=None,
        help="Use query from eval_queries.json by ID",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rec = load_recommender(
        model_dir=args.model_dir,
        corpus_path=args.corpus,
        use_index=not args.no_index,
    )

    if args.eval_query_id:
        queries_path = args.corpus.parent / "eval_queries.json"
        with open(queries_path, "r") as f:
            eval_queries = json.load(f)
        if args.eval_query_id not in eval_queries:
            raise KeyError(f"eval_query_id {args.eval_query_id} not in {queries_path}")
        query = eval_queries[args.eval_query_id]
        print(f"Query (eval_id={args.eval_query_id}):\n  {query[:200]}...\n")
    elif args.query:
        query = args.query
        print(f"Query:\n  {query}\n")
    else:
        query = "[+7d w4h14] Organic Milk, Whole Wheat Bread."
        print("No --query or --eval-query-id given. Using demo query (past orders only):\n")
        print(f"  {query}\n")

    results = rec.recommend(query=query, top_k=args.top_k)
    print(f"Top-{args.top_k} recommendations:")
    for i, (pid, score) in enumerate(results, 1):
        text = rec.pid_to_text[pid]
        print(f"  {i}. product_id={pid} (score={score:.4f}) {text}")


if __name__ == "__main__":
    main()
