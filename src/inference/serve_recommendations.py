"""
Serve product recommendations using the trained two-tower SBERT model.

Usage:
  # Run inference: python -m src.inference
  python -m src.inference --model-dir models/two_tower_sbert/final --corpus processed/p5_mp20_ef0.1/eval_corpus.json
  # Or use as a module:
  from src.inference.serve_recommendations import load_recommender, recommend
"""

from __future__ import annotations

import argparse  # Parse CLI flags like --query, --top-k
import json  # Load eval_corpus.json and eval_queries.json
import logging  # Log model/corpus load
from pathlib import Path  # Path handling for model/corpus dirs

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer  # Load trained model and encode text
from sentence_transformers.util import cos_sim  # Compute cosine similarity between query and product embeddings

from src.constants import DEFAULT_CORPUS_PATH, DEFAULT_MODEL_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)  # Module-level logger

# Load .env from project root so HF_TOKEN is set before loading models from Hub
load_dotenv(PROJECT_ROOT / ".env")


def load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    """
    Load product corpus from JSON: {product_id: product_text}.

    Returns:
        product_ids: List of product IDs (strings), same order as texts.
        product_texts: List of product text strings.
    """
    with open(corpus_path, "r") as f:
        corpus = json.load(f)  # Dict: product_id -> "Product: X. Aisle: Y. Department: Z."
    ids = list(corpus.keys())  # Preserve order for alignment with embeddings
    texts = [corpus[pid] for pid in ids]  # Parallel list of product text strings
    return ids, texts


class Recommender:
    """
    Two-tower recommender: encodes user context and products, returns top-k by cosine similarity.
    """

    def __init__(
        self,
        model_dir: Path,
        corpus_path: Path,
        batch_size: int = 64,
    ):
        self.model_dir = Path(model_dir)
        self.corpus_path = Path(corpus_path)
        self.model = SentenceTransformer(str(self.model_dir))  # Load trained two-tower encoder
        self.product_ids, self.product_texts = load_corpus(self.corpus_path)  # Parallel lists
        self.pid_to_text = dict(zip(self.product_ids, self.product_texts))  # O(1) lookup for printing
        # Precompute embeddings for all products (done once at startup)
        self.product_embeddings = self.model.encode(
            self.product_texts,
            batch_size=batch_size,  # Encode in batches to control memory
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
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
        # Encode user context to embedding (same space as products)
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[
            0
        ]  # [0] extracts single embedding from batch of 1
        # Cosine similarity between query and each product (both L2-normalized -> dot product = cosine)
        scores = cos_sim(query_emb, self.product_embeddings)[0]
        # Indices that sort scores from highest to lowest
        indices = scores.argsort(descending=True)

        results: list[tuple[str, float]] = []
        excluded = exclude_product_ids or set()
        # Walk products by score descending, skip excluded, stop at top_k
        for idx in indices:
            pid = self.product_ids[idx]
            if pid in excluded:
                continue
            results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results


def load_recommender(
    model_dir: Path = DEFAULT_MODEL_DIR,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    batch_size: int = 64,
) -> Recommender:
    """Load model and corpus, return a Recommender instance (for repeated recommend() calls)."""
    return Recommender(model_dir=model_dir, corpus_path=corpus_path, batch_size=batch_size)


def recommend(
    query: str,
    top_k: int = 10,
    model_dir: Path = DEFAULT_MODEL_DIR,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
) -> list[tuple[str, float]]:
    """
    One-shot: load recommender and return top-k for the query.
    For repeated calls, use load_recommender() once and call .recommend() on it.
    """
    rec = load_recommender(model_dir=model_dir, corpus_path=corpus_path)  # Loads model + encodes corpus
    return rec.recommend(query=query, top_k=top_k)  # Encode query, rank, return top-k


def main() -> None:
    # Define CLI flags
    parser = argparse.ArgumentParser(description="Serve product recommendations")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to trained model")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH, help="Path to eval_corpus.json")
    parser.add_argument("--query", type=str, default=None, help="User context string for recommendations")
    parser.add_argument("--eval-query-id", type=str, default=None, help="Use query from eval_queries.json by ID")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")  # Simple log format

    # Load model and precompute product embeddings (slow on first run)
    rec = load_recommender(model_dir=args.model_dir, corpus_path=args.corpus)

    # Resolve query: --eval-query-id > --query > demo
    if args.eval_query_id:
        queries_path = args.corpus.parent / "eval_queries.json"
        with open(queries_path, "r") as f:
            eval_queries = json.load(f)  # order_id -> user context text
        if args.eval_query_id not in eval_queries:
            raise KeyError(f"eval_query_id {args.eval_query_id} not in {queries_path}")
        query = eval_queries[args.eval_query_id]
        print(f"Query (eval_id={args.eval_query_id}):\n  {query[:200]}...\n")
    elif args.query:
        query = args.query
        print(f"Query:\n  {query}\n")
    else:
        # Demo: use only info available at serve time (past orders + timing). We do NOT know
        # when the user's next order will be, so we omit "Next order: ..." used in training.
        query = "[+7d w4h14] Organic Milk, Whole Wheat Bread."
        print("No --query or --eval-query-id given. Using demo query (past orders only):\n")
        print(f"  {query}\n")

    # Encode query, rank products by similarity, return top-k
    results = rec.recommend(query=query, top_k=args.top_k)
    # Print each (product_id, score) with product text preview
    print(f"Top-{args.top_k} recommendations:")
    for i, (pid, score) in enumerate(results, 1):
        text = rec.pid_to_text[pid]
        print(f"  {i}. product_id={pid} (score={score:.4f}) {text[:60]}...")


if __name__ == "__main__":
    main()  # Entry point when run as script
