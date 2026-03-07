"""
Data preparation for Instacart two-tower SBERT training.

Class-based: use InstacartDataLoader directly. No wrapper functions.

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

import pandas as pd
import yaml
from datasets import Dataset

from src.constants import (
    AISLES_CSV,
    DATA_PREP_PARAMS_FILENAME,
    DEFAULT_CONFIG_DATA_PREP,
    DEFAULT_DATA_DIR,
    DEFAULT_PROCESSED_DIR,
    DEPARTMENTS_CSV,
    EVAL_CORPUS_FILENAME,
    EVAL_DATASET_SUBDIR,
    EVAL_QUERIES_FILENAME,
    EVAL_RELEVANT_DOCS_FILENAME,
    EVAL_SET_PRIOR,
    EVAL_SET_TRAIN,
    ORDER_PRODUCTS_CHUNK_SIZE,
    ORDER_PRODUCTS_PRIOR_CSV,
    ORDER_PRODUCTS_TRAIN_CSV,
    ORDERS_CSV,
    PRODUCTS_CSV,
    PROJECT_ROOT,
    TRAIN_DATASET_SUBDIR,
)
from src.utils import setup_colored_logging

logger = logging.getLogger(__name__)


def _strip_next_order_from_context(context: str) -> str:
    """Remove the ' Next: ...' clause from a user context string."""
    if " Next:" in context:
        return context.split(" Next:")[0].strip()
    return context


class DataPrepConfig:
    """Loads data prep config from YAML. Maps config keys to paths and numeric params."""

    def __init__(self, raw: dict):
        """Parse raw YAML dict into typed config attributes."""
        data_dir = raw.get("data_dir", str(DEFAULT_DATA_DIR))
        output_dir = raw.get("output_dir", str(DEFAULT_PROCESSED_DIR))
        self.data_dir = PROJECT_ROOT / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
        self.output_dir = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        self.max_prior_orders = int(raw.get("max_prior_orders", 5))
        self.max_product_names = int(raw.get("max_product_names", 20))
        self.sample_frac = float(raw["sample_frac"]) if raw.get("sample_frac") is not None else None
        self.eval_frac = float(raw.get("eval_frac", 0.1))
        self.eval_serve_time = bool(raw.get("eval_serve_time", True))
        self.max_target_orders = int(raw["max_target_orders"]) if raw.get("max_target_orders") is not None else None
        self.seed = int(raw.get("seed", 42))

    @classmethod
    def load(cls, config_path: Path | None = None) -> DataPrepConfig:
        """Load config from YAML file. Uses default path if config_path is None."""
        path = Path(config_path) if config_path else DEFAULT_CONFIG_DATA_PREP
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw)


class InstacartDataLoader:
    """
    Prepares Instacart data for two-tower SBERT training.

    Loads CSVs, builds (anchor, positive) pairs, splits by order into train/eval,
    and writes HuggingFace datasets plus eval_queries, eval_corpus, eval_relevant_docs.
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        output_dir: Path = DEFAULT_PROCESSED_DIR,
        max_prior_orders: int = 5,
        max_product_names: int = 20,
        sample_frac: float | None = None,
        eval_frac: float = 0.1,
        eval_serve_time: bool = True,
        max_target_orders: int | None = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_prior_orders = max_prior_orders
        self.max_product_names = max_product_names
        self.sample_frac = sample_frac
        self.eval_frac = eval_frac
        self.eval_serve_time = eval_serve_time
        self.max_target_orders = max_target_orders
        self.seed = seed

    def prepare(self) -> tuple[Dataset, Dataset | None, dict, dict, dict]:
        """Run the full pipeline and save outputs. Returns (train_ds, eval_ds, eval_queries, eval_corpus, eval_relevant_docs)."""
        effective_dir = self._effective_output_dir()
        effective_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output subdir: %s", effective_dir)

        product_text_map = self._load_product_text_map()
        logger.info("[Step 1/7] Loaded %d products", len(product_text_map))

        target_orders, history_orders = self._load_orders()
        if self.max_target_orders is not None:
            target_orders = target_orders.head(self.max_target_orders)
        users_needed = set(target_orders["user_id"].tolist())
        history_orders = history_orders[history_orders["user_id"].isin(users_needed)]
        history_order_ids = set(history_orders["order_id"].tolist())
        logger.info("[Step 2/7] target: %d orders, history: %d orders", len(target_orders), len(history_order_ids))

        order_to_products = self._build_order_to_products(history_order_ids)
        logger.info("[Step 3/7] %d orders with products", len(order_to_products))

        order_id_to_context = self._build_user_context(target_orders, history_orders, order_to_products, product_text_map)
        logger.info("[Step 4/7] %d order contexts", len(order_id_to_context))

        anchors, positives, order_ids = self._build_anchor_positive_pairs(order_id_to_context, product_text_map)
        logger.info("[Step 5/7] %d pairs", len(anchors))

        train_anchors, train_positives, eval_anchors, eval_positives, eval_order_ids = self._split_train_eval(
            anchors, positives, order_ids, order_id_to_context
        )
        if self.sample_frac is not None and self.sample_frac < 1.0:
            train_df = pd.DataFrame({"anchor": train_anchors, "positive": train_positives})
            train_df = train_df.sample(frac=self.sample_frac, random_state=self.seed)
            train_anchors = train_df["anchor"].tolist()
            train_positives = train_df["positive"].tolist()

        train_dataset = Dataset.from_dict({"anchor": train_anchors, "positive": train_positives})
        logger.info("[Step 6/7] train: %d pairs, eval: %d pairs", len(train_anchors), len(eval_anchors))

        eval_queries, eval_corpus, eval_relevant_docs = self._build_eval_artifacts(
            eval_order_ids, order_id_to_context, product_text_map
        )
        eval_dataset = (
            Dataset.from_dict({"anchor": eval_anchors, "positive": eval_positives})
            if eval_anchors and eval_positives
            else None
        )

        self._save_outputs(effective_dir, train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs)
        logger.info("[Step 7/7] Saved to %s", effective_dir)

        return train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs

    def _effective_output_dir(self) -> Path:
        """Build param-based subdir name (e.g. p5_mp20_ef0.1) for output isolation."""
        parts = [f"p{self.max_prior_orders}", f"mp{self.max_product_names}", f"ef{self.eval_frac}"]
        if not self.eval_serve_time:
            parts.append("no_serve")
        if self.sample_frac is not None:
            parts.append(f"sf{self.sample_frac}")
        if self.max_target_orders is not None:
            parts.append(f"mt{self.max_target_orders}")
        return self.output_dir / "_".join(parts)

    def _load_product_text_map(self) -> dict[int, str]:
        """Load products, aisles, departments; build product_id -> 'Product: X. Aisle: Y. Department: Z.' map."""
        products = pd.read_csv(self.data_dir / PRODUCTS_CSV)
        aisles = pd.read_csv(self.data_dir / AISLES_CSV)
        departments = pd.read_csv(self.data_dir / DEPARTMENTS_CSV)
        products = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
        products["text"] = (
            "Product: "
            + products["product_name"].astype(str)
            + ". Aisle: "
            + products["aisle"].astype(str)
            + ". Department: "
            + products["department"].astype(str)
            + "."
        )
        return dict(zip(products["product_id"], products["text"]))

    def _load_orders(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load orders; split into target (eval_set=train) and history (eval_set=prior)."""
        orders = pd.read_csv(self.data_dir / ORDERS_CSV)
        if orders["order_hour_of_day"].dtype == object:
            orders["order_hour_of_day"] = orders["order_hour_of_day"].astype(str).str.zfill(2)
        cols = ["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
        target = orders[orders["eval_set"] == EVAL_SET_TRAIN][cols].copy()
        history = orders[orders["eval_set"] == EVAL_SET_PRIOR][cols].copy()
        return target, history

    def _build_order_to_products(self, history_order_ids: set[int]) -> dict[int, list[int]]:
        """Build order_id -> [product_ids] from order_products__prior (chunked)."""
        order_to_products: dict[int, list[int]] = defaultdict(list)
        path = self.data_dir / ORDER_PRODUCTS_PRIOR_CSV
        for chunk in pd.read_csv(path, chunksize=ORDER_PRODUCTS_CHUNK_SIZE):
            chunk = chunk[chunk["order_id"].isin(history_order_ids)]
            for order_id, product_id in chunk[["order_id", "product_id"]].itertuples(index=False):
                order_to_products[order_id].append(product_id)
        return dict(order_to_products)

    def _build_user_context(
        self,
        target_orders: pd.DataFrame,
        history_orders: pd.DataFrame,
        order_to_products: dict[int, list[int]],
        product_text_map: dict[int, str],
    ) -> dict[int, str]:
        """Build order_id -> user context string (prior orders + Next: clause)."""
        history_orders = history_orders.sort_values(["user_id", "order_number"])
        order_id_to_context: dict[int, str] = {}

        for _, row in target_orders.iterrows():
            order_id = int(row["order_id"])
            user_history = history_orders[
                (history_orders["user_id"] == row["user_id"]) & (history_orders["order_number"] < row["order_number"])
            ].tail(self.max_prior_orders)

            segments: list[str] = []
            total_products = 0

            for _, h in user_history.iterrows():
                if total_products >= self.max_product_names:
                    break
                oid = int(h["order_id"])
                order_products = []
                for pid in order_to_products.get(oid, []):
                    if pid not in product_text_map:
                        continue
                    if total_products >= self.max_product_names:
                        break
                    name = product_text_map[pid].split("Product: ")[1].split(".")[0].strip()
                    order_products.append(name)
                    total_products += 1

                if not order_products:
                    continue

                dow = int(h["order_dow"])
                hour = h["order_hour_of_day"] if isinstance(h["order_hour_of_day"], str) else str(int(h["order_hour_of_day"]))
                time_prefix = f"w{dow}h{hour}" if pd.isna(h["days_since_prior_order"]) else f"+{int(h['days_since_prior_order'])}d w{dow}h{hour}"
                segments.append(f"[{time_prefix}] " + ", ".join(order_products))

            products_str = "; ".join(segments) if segments else "(no prior orders)"
            row_dow = int(row["order_dow"])
            row_hour = row["order_hour_of_day"] if isinstance(row["order_hour_of_day"], str) else str(int(row["order_hour_of_day"]))
            next_clause = f"Next: w{row_dow}h{row_hour}" if pd.isna(row["days_since_prior_order"]) else f"Next: +{int(row['days_since_prior_order'])}d w{row_dow}h{row_hour}"
            order_id_to_context[order_id] = f"{products_str}. {next_clause}"

        return order_id_to_context

    def _build_anchor_positive_pairs(
        self, order_id_to_context: dict[int, str], product_text_map: dict[int, str]
    ) -> tuple[list[str], list[str], list[int]]:
        """Build (anchor, positive) pairs from order_products__train; anchor=context, positive=product text."""
        train_op = pd.read_csv(self.data_dir / ORDER_PRODUCTS_TRAIN_CSV)
        anchors, positives, order_ids = [], [], []
        for _, row in train_op.iterrows():
            order_id, product_id = row["order_id"], row["product_id"]
            if order_id not in order_id_to_context or product_id not in product_text_map:
                continue
            anchors.append(order_id_to_context[order_id])
            positives.append(product_text_map[product_id])
            order_ids.append(order_id)
        return anchors, positives, order_ids

    def _split_train_eval(
        self,
        anchors: list[str],
        positives: list[str],
        order_ids: list[int],
        order_id_to_context: dict[int, str],
    ) -> tuple[list[str], list[str], list[str], list[str], set[int]]:
        """Split pairs by order: last eval_frac of orders go to eval; rest to train."""
        order_list = sorted(set(order_id_to_context.keys()))
        n_eval = max(1, int(len(order_list) * self.eval_frac))
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
        return train_anchors, train_positives, eval_anchors, eval_positives, eval_order_ids

    def _build_eval_artifacts(
        self,
        eval_order_ids: set[int],
        order_id_to_context: dict[int, str],
        product_text_map: dict[int, str],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
        """Build eval_queries, eval_corpus, eval_relevant_docs for IR evaluator."""
        if self.eval_serve_time:
            eval_queries = {
                str(oid): _strip_next_order_from_context(order_id_to_context[oid])
                for oid in eval_order_ids
                if oid in order_id_to_context
            }
        else:
            eval_queries = {str(oid): order_id_to_context[oid] for oid in eval_order_ids if oid in order_id_to_context}

        eval_relevant_docs = {str(oid): [] for oid in eval_order_ids}
        train_op = pd.read_csv(self.data_dir / ORDER_PRODUCTS_TRAIN_CSV)
        for _, row in train_op.iterrows():
            oid_str = str(int(row["order_id"]))
            if oid_str in eval_relevant_docs:
                eval_relevant_docs[oid_str].append(str(int(row["product_id"])))

        eval_corpus = {str(pid): text for pid, text in product_text_map.items()}
        return eval_queries, eval_corpus, eval_relevant_docs

    def _save_outputs(
        self,
        effective_dir: Path,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        eval_queries: dict,
        eval_corpus: dict,
        eval_relevant_docs: dict,
    ) -> None:
        """Write train/eval datasets, eval_queries.json, eval_corpus.json, eval_relevant_docs.json, data_prep_params.json."""
        train_dataset.save_to_disk(str(effective_dir / TRAIN_DATASET_SUBDIR))
        if eval_dataset is not None:
            eval_dataset.save_to_disk(str(effective_dir / EVAL_DATASET_SUBDIR))
        with open(effective_dir / EVAL_QUERIES_FILENAME, "w") as f:
            json.dump(eval_queries, f, indent=0)
        with open(effective_dir / EVAL_CORPUS_FILENAME, "w") as f:
            json.dump(eval_corpus, f, indent=0)
        with open(effective_dir / EVAL_RELEVANT_DOCS_FILENAME, "w") as f:
            json.dump(eval_relevant_docs, f, indent=0)

        params = {
            "data_dir": str(self.data_dir),
            "output_dir": str(effective_dir),
            "max_prior_orders": self.max_prior_orders,
            "max_product_names": self.max_product_names,
            "sample_frac": self.sample_frac,
            "eval_frac": self.eval_frac,
            "eval_serve_time": self.eval_serve_time,
            "max_target_orders": self.max_target_orders,
            "seed": self.seed,
            "n_train_pairs": len(train_dataset),
            "n_eval_pairs": len(eval_dataset) if eval_dataset else 0,
            "n_eval_queries": len(eval_queries),
            "n_corpus": len(eval_corpus),
        }
        with open(effective_dir / DATA_PREP_PARAMS_FILENAME, "w") as f:
            json.dump(params, f, indent=2)


def main() -> None:
    """CLI entrypoint: load config from YAML, create InstacartDataLoader, call prepare()."""
    parser = argparse.ArgumentParser(description="Prepare Instacart data for two-tower SBERT training")
    parser.add_argument(
        "--config", type=Path, default=None, help=f"Path to YAML config (default: {DEFAULT_CONFIG_DATA_PREP.relative_to(PROJECT_ROOT)})"
    )
    args = parser.parse_args()

    cfg = DataPrepConfig.load(args.config)
    setup_colored_logging(quiet_loggers=["httpx", "httpcore", "huggingface_hub", "urllib3", "datasets"])

    loader = InstacartDataLoader(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        max_prior_orders=cfg.max_prior_orders,
        max_product_names=cfg.max_product_names,
        sample_frac=cfg.sample_frac,
        eval_frac=cfg.eval_frac,
        eval_serve_time=cfg.eval_serve_time,
        max_target_orders=cfg.max_target_orders,
        seed=cfg.seed,
    )
    train_ds, eval_ds, eq, ec, er = loader.prepare()

    logger.info("Train dataset size: %d", len(train_ds))
    if eval_ds is not None:
        logger.info("Eval dataset size: %d", len(eval_ds))
    logger.info("Eval queries: %d, corpus size: %d", len(eq), len(ec))
    logger.info("Saved to %s", loader._effective_output_dir())


if __name__ == "__main__":
    main()
