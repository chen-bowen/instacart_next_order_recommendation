"""
Training script for the Instacart two-tower SBERT model for recommendations.

Class-based: use SBERTTrainer directly. No wrapper functions.

Uses the processed (anchor, positive) datasets and IR artifacts produced by
`src.data.prepare_instacart_sbert` to train a SentenceTransformers bi-encoder
with MultipleNegativesRankingLoss and evaluate with InformationRetrievalEvaluator.

Run: python -m src.training
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from src.constants import (
    DATA_PREP_PARAMS_FILENAME,
    DEFAULT_CONFIG_TRAIN,
    DEFAULT_DOTENV_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DIR,
    EVAL_CORPUS_FILENAME,
    EVAL_DATASET_SUBDIR,
    EVAL_QUERIES_FILENAME,
    EVAL_RELEVANT_DOCS_FILENAME,
    PROJECT_ROOT,
    TRAIN_DATASET_SUBDIR,
)
from src.utils import resolve_processed_dir, setup_colored_logging

load_dotenv(DEFAULT_DOTENV_PATH)
logger = logging.getLogger(__name__)


class TrainConfig:
    """Loads training config from YAML. Maps config keys to paths and hyperparameters."""

    def __init__(self, raw: dict):
        """Parse raw YAML dict into typed config attributes."""
        def resolve(p: str) -> Path:
            return PROJECT_ROOT / p if p and not Path(p).is_absolute() else Path(p)

        self.processed_dir = resolve(raw.get("processed_dir", str(DEFAULT_PROCESSED_DIR)))
        self.output_dir = resolve(raw.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
        self.model_name = str(raw.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        self.max_seq_length = int(raw.get("max_seq_length", 256))
        self.epochs = int(raw.get("epochs", 5))
        self.train_batch_size = int(raw.get("train_batch_size", 64))
        self.eval_batch_size = int(raw.get("eval_batch_size", 64))
        self.gradient_accumulation_steps = int(raw.get("gradient_accumulation_steps", 1))
        self.learning_rate = float(raw.get("learning_rate", 5e-5))
        self.loss_scale = float(raw.get("loss_scale", 30.0))
        self.dataloader_num_workers = int(raw.get("dataloader_num_workers", 0))
        self.run_information_retrieval_evaluator = bool(raw.get("run_information_retrieval_evaluator", True))

    @classmethod
    def load(cls, config_path: Path | None = None) -> TrainConfig:
        """Load config from YAML file. Uses default path if config_path is None."""
        path = Path(config_path) if config_path else DEFAULT_CONFIG_TRAIN
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw)


class SBERTTrainer:
    """
    Trains a two-tower SBERT model on Instacart (anchor, positive) pairs.

    Loads processed data, builds model and loss, optionally runs IR evaluator,
    and trains with SentenceTransformerTrainer.
    """

    def __init__(
        self,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int | None = 256,
        num_train_epochs: int = 3,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        loss_scale: float | None = 30.0,
        dataloader_num_workers: int = 4,
        run_information_retrieval_evaluator: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.loss_scale = loss_scale
        self.dataloader_num_workers = dataloader_num_workers
        self.run_information_retrieval_evaluator = run_information_retrieval_evaluator

    def train(self) -> None:
        """Run the full training pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs = self._load_processed_data()
        model = self._build_model()
        loss = self._build_loss(model)
        evaluator = self._build_evaluator(eval_queries, eval_corpus, eval_relevant_docs)
        args = self._build_training_args(len(train_dataset))

        self._log_params()
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        logger.info("[Step 5/5] Training started...")
        trainer.train()

        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        logger.info("Done. Model saved to %s", final_dir)

    def _load_processed_data(
        self,
    ) -> tuple[Dataset, Dataset | None, dict[str, str], dict[str, str], dict[str, set[str]]]:
        logger.info("[Step 1/5] Loading processed data...")
        self.processed_dir, subdir_msg = resolve_processed_dir(self.processed_dir, DEFAULT_PROCESSED_DIR)
        if subdir_msg:
            logger.info("%s", subdir_msg)

        train_dataset = load_from_disk(str(self.processed_dir / TRAIN_DATASET_SUBDIR))
        eval_dataset = None
        if (self.processed_dir / EVAL_DATASET_SUBDIR).exists():
            eval_dataset = load_from_disk(str(self.processed_dir / EVAL_DATASET_SUBDIR))

        with open(self.processed_dir / EVAL_QUERIES_FILENAME) as f:
            eval_queries = json.load(f)
        with open(self.processed_dir / EVAL_CORPUS_FILENAME) as f:
            eval_corpus = json.load(f)
        with open(self.processed_dir / EVAL_RELEVANT_DOCS_FILENAME) as f:
            raw = json.load(f)
            eval_relevant_docs = {k: set(v) for k, v in raw.items()}

        logger.info(
            "  -> train: %d, eval: %d, queries: %d, corpus: %d",
            len(train_dataset),
            len(eval_dataset) if eval_dataset else 0,
            len(eval_queries),
            len(eval_corpus),
        )
        return train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs

    def _build_model(self) -> SentenceTransformer:
        """Load base SentenceTransformer and set max_seq_length."""
        logger.info("[Step 2/5] Building model: %s", self.model_name)
        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
        return model

    def _build_loss(self, model: SentenceTransformer) -> MultipleNegativesRankingLoss:
        """Build MultipleNegativesRankingLoss (in-batch negatives) with optional scale."""
        kwargs = {} if self.loss_scale is None else {"scale": self.loss_scale}
        return MultipleNegativesRankingLoss(model, **kwargs)

    def _build_evaluator(
        self,
        eval_queries: dict[str, str],
        eval_corpus: dict[str, str],
        eval_relevant_docs: dict[str, set[str]],
    ) -> InformationRetrievalEvaluator | None:
        logger.info("[Step 3/5] Setting up evaluator...")
        if not self.run_information_retrieval_evaluator:
            logger.info("  -> IR evaluator disabled")
            return None
        return InformationRetrievalEvaluator(
            queries=eval_queries,
            corpus=eval_corpus,
            relevant_docs=eval_relevant_docs,
            name="order-recommendation",
        )

    def _build_training_args(self, n_train: int) -> SentenceTransformerTrainingArguments:
        """Build training args; disables fp16 and uses gradient checkpointing on MPS for stability."""
        logger.info("[Step 4/5] Configuring training...")
        use_mps = getattr(torch.backends.mps, "is_available", lambda: False)()
        num_workers = 0 if use_mps else self.dataloader_num_workers
        pin_memory = not use_mps
        use_fp16 = not use_mps
        use_gradient_checkpointing = use_mps

        num_devices = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        effective_batch = self.train_batch_size * self.gradient_accumulation_steps * num_devices
        steps_per_epoch = math.ceil(n_train / effective_batch)
        total_steps = self.num_train_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)

        load_best = self.run_information_retrieval_evaluator
        metric_for_best = "eval_order-recommendation_cosine_ndcg@10" if load_best else None

        return SentenceTransformerTrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=use_gradient_checkpointing,
            learning_rate=self.learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            fp16=use_fp16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=100,
            load_best_model_at_end=load_best,
            metric_for_best_model=metric_for_best,
            greater_is_better=True,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=pin_memory,
            dataloader_drop_last=True,
        )

    def _log_params(self) -> None:
        """Log data prep params and training config to stdout."""
        params_path = self.processed_dir / DATA_PREP_PARAMS_FILENAME
        data_prep_params = json.load(open(params_path)) if params_path.exists() else None

        logger.info("=" * 60)
        logger.info("Params")
        logger.info("-" * 60)
        if data_prep_params:
            for k, v in data_prep_params.items():
                logger.info("  %s: %s", k, v)
        logger.info("-" * 60)
        logger.info("  processed_dir: %s", self.processed_dir)
        logger.info("  output_dir: %s", self.output_dir)
        logger.info("  model_name: %s", self.model_name)
        logger.info("  num_train_epochs: %d", self.num_train_epochs)
        logger.info("  run_information_retrieval_evaluator: %s", self.run_information_retrieval_evaluator)
        logger.info("=" * 60)


def main() -> None:
    """CLI entrypoint: load config, create SBERTTrainer, call train()."""
    parser = argparse.ArgumentParser(description="Train two-tower SBERT model on Instacart data")
    parser.add_argument(
        "--config", type=Path, default=None, help=f"Path to YAML config (default: {DEFAULT_CONFIG_TRAIN.relative_to(PROJECT_ROOT)})"
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    processed_dir, _ = resolve_processed_dir(cfg.processed_dir, DEFAULT_PROCESSED_DIR)

    setup_colored_logging(
        quiet_loggers=["httpx", "httpcore", "huggingface_hub", "urllib3", "sentence_transformers"],
    )

    trainer = SBERTTrainer(
        processed_dir=processed_dir,
        output_dir=cfg.output_dir,
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        num_train_epochs=cfg.epochs,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        loss_scale=cfg.loss_scale,
        dataloader_num_workers=cfg.dataloader_num_workers,
        run_information_retrieval_evaluator=cfg.run_information_retrieval_evaluator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
