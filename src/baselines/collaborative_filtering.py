"""
Collaborative filtering baseline: item-item co-occurrence (products bought together).

For each eval order we have the user's prior basket (products in prior orders). We score
each candidate product by the sum of co-occurrence counts with those prior products
(products that appeared in the same order in the training data). Rank by score.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_eval_data(processed_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Load eval_queries, eval_corpus, eval_relevant_docs from a processed dir."""
    with open(processed_dir / "eval_queries.json") as f:
        eval_queries = json.load(f)
    with open(processed_dir / "eval_corpus.json") as f:
        eval_corpus = json.load(f)
    with open(processed_dir / "eval_relevant_docs.json") as f:
        raw = json.load(f)
    eval_relevant_docs = {k: set(v) for k, v in raw.items()}
    return eval_queries, eval_corpus, eval_relevant_docs


class ItemItemCFBaseline:
    """
    Item-item CF: score(candidate) = sum over prior products h of co_occur(candidate, h).
    Co-occurrence = number of orders containing both products (from order_products__prior).
    """

    def __init__(
        self,
        data_dir: Path,
        processed_dir: Path,
        order_products_chunk_size: int = 500_000,
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.order_products_chunk_size = order_products_chunk_size
        self.co_occur: dict[tuple[str, str], int] = defaultdict(int)
        self.order_to_products: dict[int, list[str]] = {}
        self.eval_order_to_history: dict[str, set[str]] = {}
        self.product_ids: set[str] = set()
        self._build()

    def _build(self) -> None:
        # Load orders: train orders (eval targets) and prior orders
        orders = pd.read_csv(self.data_dir / "orders.csv")
        train_orders = orders[orders["eval_set"] == "train"][["order_id", "user_id", "order_number"]]
        prior_orders = orders[orders["eval_set"] == "prior"][["order_id", "user_id", "order_number"]]

        # Eval query IDs = train order_ids we have in eval_queries
        with open(self.processed_dir / "eval_queries.json") as f:
            eval_q = json.load(f)
        eval_order_ids = {int(oid) for oid in eval_q.keys()}

        # Restrict to users that have an eval order
        train_eval = train_orders[train_orders["order_id"].isin(eval_order_ids)]
        users_eval = set(train_eval["user_id"].tolist())
        prior_orders = prior_orders[prior_orders["user_id"].isin(users_eval)]
        prior_order_ids = set(prior_orders["order_id"].tolist())

        # order_id -> (user_id, order_number) for prior
        prior_order_info = dict(zip(prior_orders["order_id"], zip(prior_orders["user_id"], prior_orders["order_number"])))
        train_order_info = dict(zip(train_orders["order_id"], zip(train_orders["user_id"], train_orders["order_number"])))

        # Load order_products__prior in chunks; keep only prior_order_ids
        chunk_iter = pd.read_csv(
            self.data_dir / "order_products__prior.csv", chunksize=self.order_products_chunk_size
        )
        for chunk in tqdm(chunk_iter, desc="Loading order_products__prior", unit="chunk"):
            chunk = chunk[chunk["order_id"].isin(prior_order_ids)]
            for _, row in chunk.iterrows():
                oid = int(row["order_id"])
                pid = str(int(row["product_id"]))
                self.order_to_products.setdefault(oid, []).append(pid)
                self.product_ids.add(pid)

        # Build co-occurrence from order_to_products
        for oid, pids in tqdm(
            self.order_to_products.items(), desc="Building co-occurrence", unit="order"
        ):
            unique = list(dict.fromkeys(pids))
            for i, a in enumerate(unique):
                for b in unique[i:]:
                    if a == b:
                        self.co_occur[(a, b)] = self.co_occur.get((a, b), 0) + 1
                    else:
                        self.co_occur[(a, b)] = self.co_occur.get((a, b), 0) + 1
                        self.co_occur[(b, a)] = self.co_occur.get((b, a), 0) + 1

        # For each eval order: user's prior product set (from prior orders with order_number < this order's order_number)
        for order_id in tqdm(eval_order_ids, desc="Building eval history", unit="order"):
            if order_id not in train_order_info:
                continue
            user_id, order_num = train_order_info[order_id]
            history = set()
            for oid, (uid, onum) in prior_order_info.items():
                if uid == user_id and onum < order_num:
                    history.update(self.order_to_products.get(oid, []))
            self.eval_order_to_history[str(order_id)] = history
        # Ensure every eval query has an entry (empty history if not in train_order_info)
        for qid in eval_q:
            if qid not in self.eval_order_to_history:
                self.eval_order_to_history[qid] = set()

        # Product list for ranking (from eval_corpus so we only rank known products)
        with open(self.processed_dir / "eval_corpus.json") as f:
            corpus = json.load(f)
        self.corpus_ids = list(corpus.keys())
        self.product_ids.update(self.corpus_ids)

    def rank_all(self, eval_query_ids: list[str] | None = None) -> dict[str, list[str]]:
        """Return query_id (order_id) -> list of product_id ranked by CF score (desc)."""
        if eval_query_ids is None:
            eval_query_ids = list(self.eval_order_to_history.keys())
        out: dict[str, list[str]] = {}
        for qid in tqdm(eval_query_ids, desc="CF ranking", unit="query"):
            history = self.eval_order_to_history.get(qid, set())
            scores: dict[str, float] = {}
            for pid in self.corpus_ids:
                if pid in history:
                    continue
                score = sum(self.co_occur.get((pid, h), 0) for h in history)
                scores[pid] = score
            ranked = sorted(scores.keys(), key=lambda x: -scores[x])
            out[qid] = ranked
        return out
