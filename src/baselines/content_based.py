"""
Content-based baseline: same SBERT model but untrained (frozen pretrained weights).

Encodes query and product text with the base SentenceTransformer (e.g. all-MiniLM-L6-v2)
with no fine-tuning on Instacart data. Rank by cosine similarity. This isolates the
gain from training the two-tower model on (anchor, positive) pairs.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class ContentBasedBaseline:
    """Recommend products by cosine similarity of pretrained (untrained) SBERT embeddings."""

    def __init__(
        self,
        eval_queries: dict[str, str],
        eval_corpus: dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.eval_queries = eval_queries
        self.eval_corpus = eval_corpus
        self.product_ids = list(eval_corpus.keys())
        self.corpus_texts = [eval_corpus[pid] for pid in self.product_ids]
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = self.model.encode(
            self.corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def rank_all(self) -> dict[str, list[str]]:
        """
        Rank corpus for each query by cosine similarity (descending).

        Returns:
            Dict mapping query_id to list of product_id ranked by score descending.
        """
        query_ids = list(self.eval_queries.keys())
        query_texts = [self.eval_queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        # query_embeddings (n_queries, dim), corpus_embeddings (n_corpus, dim)
        sim = cos_sim(query_embeddings, self.corpus_embeddings)
        if hasattr(sim, "cpu"):
            sim = sim.cpu().numpy()
        elif hasattr(sim, "numpy"):
            sim = sim.numpy()
        out: dict[str, list[str]] = {}
        for i, qid in enumerate(query_ids):
            scores = np.asarray(sim[i]).flatten()
            order = np.argsort(scores)[::-1]
            out[qid] = [self.product_ids[j] for j in order]
        return out
