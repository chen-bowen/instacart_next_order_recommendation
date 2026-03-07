"""
Serve product recommendations using the trained two-tower SBERT model.

Inference is embedding-based:
  - We encode the user context string and each product's text with the same SentenceTransformer.
  - Recommendations are the top-k products by cosine similarity between query and product embeddings.
  - One shared encoder is used for both "towers" (query and document); there is no model.generate().

Usage:
  rec = Recommender(model_dir=..., corpus_path=...)
  results = rec.recommend(query="...", top_k=10)
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

import yaml

from src.constants import (
    DEFAULT_CONFIG_INFERENCE,
    DEFAULT_CORPUS_PATH,
    DEFAULT_DOTENV_PATH,
    DEFAULT_MODEL_DIR,
    EMBEDDINGS_FILENAME,
    EVAL_QUERIES_FILENAME,
    INDEX_SUBDIR,
    MANIFEST_FILENAME,
    PRODUCT_IDS_FILENAME,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)
load_dotenv(DEFAULT_DOTENV_PATH)


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


class EmbeddingIndex:
    """Manages cached product embeddings on disk. Cache keyed by corpus_path + model_dir + corpus mtime."""

    def __init__(self, corpus_path: Path, model_dir: Path | str):
        """Initialize index; index dir is under corpus parent / .embedding_index / <hash>."""
        self.corpus_path = Path(corpus_path).resolve()
        self.model_dir = model_dir
        self._dir = self._index_dir()

    def _index_dir(self) -> Path:
        """Compute cache directory from hash of model_dir and corpus_path."""
        canonical = f"{self.model_dir!s}|{self.corpus_path!s}"
        name = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return self.corpus_path.parent / INDEX_SUBDIR / name

    def load(self, product_ids: list[str]) -> np.ndarray | None:
        """Load cached embeddings if manifest matches. Returns None on miss."""
        manifest_path = self._dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if meta.get("corpus_path") != str(self.corpus_path) or meta.get("model_dir") != str(self.model_dir):
            return None
        try:
            if meta.get("corpus_mtime") != self.corpus_path.stat().st_mtime:
                return None
        except OSError:
            return None
        emb_path = self._dir / EMBEDDINGS_FILENAME
        ids_path = self._dir / PRODUCT_IDS_FILENAME
        if not emb_path.exists() or not ids_path.exists():
            return None
        try:
            embeddings = np.load(emb_path)
            with open(ids_path) as f:
                cached_ids = json.load(f)
        except (OSError, ValueError):
            return None
        if cached_ids != product_ids or len(embeddings) != len(product_ids):
            return None
        return embeddings

    def save(self, product_ids: list[str], embeddings: np.ndarray) -> None:
        """Write manifest, embeddings, and product_ids to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            mtime = self.corpus_path.stat().st_mtime
        except OSError:
            mtime = 0
        manifest = {
            "corpus_path": str(self.corpus_path),
            "model_dir": str(self.model_dir),
            "corpus_mtime": mtime,
            "n_products": len(product_ids),
        }
        with open(self._dir / MANIFEST_FILENAME, "w") as f:
            json.dump(manifest, f, indent=2)
        np.save(self._dir / EMBEDDINGS_FILENAME, embeddings.astype(np.float32))
        with open(self._dir / PRODUCT_IDS_FILENAME, "w") as f:
            json.dump(product_ids, f)
        logger.info("Saved embedding index to %s (%d products)", self._dir, len(product_ids))


class Recommender:
    """
    Two-tower recommender: encodes user context and products, returns top-k by cosine similarity.
    Product embeddings are cached on disk via EmbeddingIndex.
    """

    def __init__(
        self,
        model_dir: Path | str,
        corpus_path: Path,
        batch_size: int = 64,
        use_index: bool = True,
    ):
        self.model_dir = self._resolve_model_dir(model_dir)
        self.corpus_path = Path(corpus_path).resolve()
        self.product_ids, self.product_texts = self._load_corpus()
        self.pid_to_text = dict(zip(self.product_ids, self.product_texts))
        self.model = self._load_model()
        self.product_embeddings = self._load_or_build_embeddings(batch_size, use_index)

    def _resolve_model_dir(self, model_dir: Path | str) -> Path | str:
        """Resolve to absolute Path if local dir exists; else return as-is (Hugging Face ID)."""
        p = Path(model_dir)
        return p.resolve() if p.exists() else model_dir

    def _load_corpus(self) -> tuple[list[str], list[str]]:
        """Load eval_corpus.json; return (product_ids, product_texts) preserving key order."""
        with open(self.corpus_path) as f:
            corpus = json.load(f)
        ids = list(corpus.keys())
        texts = [corpus[pid] for pid in ids]
        return ids, texts

    def _load_model(self) -> SentenceTransformer:
        """Load SentenceTransformer from model_dir (local or Hugging Face)."""
        device = self._inference_device()
        logger.info("Using inference device: %s", device)
        return SentenceTransformer(str(self.model_dir), device=device)

    def _inference_device(self) -> str:
        """Return cuda, mps, or cpu based on INFERENCE_DEVICE env or auto-detect."""
        override = os.getenv("INFERENCE_DEVICE")
        if override:
            return override
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends.mps, "is_available", lambda: False)():
            return "mps"
        return "cpu"

    def _load_or_build_embeddings(self, batch_size: int, use_index: bool) -> np.ndarray:
        """Load from EmbeddingIndex cache if valid; else encode corpus and save to cache."""
        index = EmbeddingIndex(self.corpus_path, self.model_dir)
        if use_index:
            cached = index.load(self.product_ids)
            if cached is not None:
                logger.info(
                    "Loaded model from %s, corpus %d products (embeddings from index)",
                    self.model_dir,
                    len(self.product_ids),
                )
                return cached
        embeddings = self.model.encode(
            self.product_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        if use_index:
            index.save(self.product_ids, embeddings)
        logger.info("Loaded model from %s, corpus %d products", self.model_dir, len(self.product_ids))
        return embeddings

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        exclude_product_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k (product_id, score) sorted by cosine similarity."""
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = cos_sim(query_emb, self.product_embeddings)[0]
        indices = scores.argsort(descending=True)
        excluded = exclude_product_ids or set()
        results: list[tuple[str, float]] = []
        for idx in indices:
            pid = self.product_ids[idx]
            if pid in excluded:
                continue
            results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results


class MonitoredRecommender(Recommender):
    """Recommender with timing and metrics. Sets last_metrics after each recommend()."""

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
        """Recommend with instrumentation; sets self.last_metrics."""
        start = time.time()
        encode_start = time.time()
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        encode_ms = (time.time() - encode_start) * 1000

        sim_start = time.time()
        scores = cos_sim(query_emb, self.product_embeddings)[0]
        indices = scores.argsort(descending=True)
        sim_ms = (time.time() - sim_start) * 1000

        excluded = exclude_product_ids or set()
        results: list[tuple[str, float]] = []
        for idx in indices:
            pid = self.product_ids[idx]
            if pid in excluded:
                continue
            results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break

        total_ms = (time.time() - start) * 1000
        top_score = results[0][1] if results else 0.0
        avg_score = sum(s for _, s in results) / len(results) if results else 0.0

        self.last_metrics = RecommendationMetrics(
            user_id=user_id or "anonymous",
            query_embedding_time_ms=encode_ms,
            similarity_compute_time_ms=sim_ms,
            total_latency_ms=total_ms,
            num_recommendations=len(results),
            top_score=top_score,
            avg_score=avg_score,
            timestamp=time.time(),
        )
        self._log_metrics(self.last_metrics)
        return results

    def _log_metrics(self, m: RecommendationMetrics) -> None:
        self.metrics_logger.info(
            "recommendation_served",
            extra={
                "user_id": m.user_id,
                "latency_ms": m.total_latency_ms,
                "encode_time_ms": m.query_embedding_time_ms,
                "similarity_time_ms": m.similarity_compute_time_ms,
                "num_results": m.num_recommendations,
                "top_score": m.top_score,
                "avg_score": m.avg_score,
            },
        )


class InferenceConfig:
    """Loads inference config from YAML. Attributes: model_dir, corpus, use_index, query, eval_query_id, top_k."""

    def __init__(self, raw: dict):
        """Parse raw YAML dict into typed config attributes."""
        self.model_dir = self._resolve_model_dir(raw.get("model_dir", str(DEFAULT_MODEL_DIR)))
        self.corpus = self._resolve_path(raw.get("corpus"), DEFAULT_CORPUS_PATH)
        self.use_index = bool(raw.get("use_index", True))
        self.query = raw.get("query")
        self.eval_query_id = raw.get("eval_query_id")
        self.top_k = int(raw.get("top_k", 10))

    def _resolve_model_dir(self, model_dir: str) -> Path | str:
        p = Path(model_dir)
        if not p.is_absolute():
            local = PROJECT_ROOT / model_dir
            return local if local.exists() else model_dir
        return p

    def _resolve_path(self, p: str | None, default: Path) -> Path:
        """Resolve path string to Path; relative paths are under PROJECT_ROOT."""
        if not p:
            return default
        path = Path(p)
        return PROJECT_ROOT / p if not path.is_absolute() else path

    @classmethod
    def load(cls, config_path: Path | None = None) -> InferenceConfig:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_INFERENCE
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw)


def main() -> None:
    """CLI entrypoint: load config, create Recommender, run demo query and print top-k."""
    parser = argparse.ArgumentParser(description="Serve product recommendations")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_INFERENCE.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    cfg = InferenceConfig.load(args.config)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_dir = cfg.model_dir if isinstance(cfg.model_dir, Path) else Path(str(cfg.model_dir))
    rec = Recommender(model_dir=model_dir, corpus_path=cfg.corpus, use_index=cfg.use_index)

    if cfg.eval_query_id:
        queries_path = cfg.corpus.parent / EVAL_QUERIES_FILENAME
        with open(queries_path) as f:
            eval_queries = json.load(f)
        if cfg.eval_query_id not in eval_queries:
            raise KeyError(f"eval_query_id {cfg.eval_query_id} not in {queries_path}")
        query = eval_queries[cfg.eval_query_id]
        print(f"Query (eval_id={cfg.eval_query_id}):\n  {query[:200]}...\n")
    elif cfg.query:
        query = cfg.query
        print(f"Query:\n  {query}\n")
    else:
        query = "[+7d w4h14] Organic Milk, Whole Wheat Bread."
        print("No query or eval_query_id in config. Using demo query:\n")
        print(f"  {query}\n")

    results = rec.recommend(query=query, top_k=cfg.top_k)
    print(f"Top-{cfg.top_k} recommendations:")
    for i, (pid, score) in enumerate(results, 1):
        print(f"  {i}. product_id={pid} (score={score:.4f}) {rec.pid_to_text[pid]}")


if __name__ == "__main__":
    main()
