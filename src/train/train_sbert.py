"""
Training script for the Instacart two-tower SBERT model for recommendations.

Uses the processed (anchor, positive) datasets and Information retrieval artifacts produced by
`src.data.prepare_instacart_sbert` to train a SentenceTransformers bi-encoder
with MultipleNegativesRankingLoss and evaluate it with InformationRetrievalEvaluator.

Run: python -m src.train
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from src.constants import DEFAULT_PROCESSED_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from src.utils import resolve_processed_dir, setup_colored_logging

# Load .env from project root so HF_TOKEN (and others) are set before any Hugging Face calls
load_dotenv(PROJECT_ROOT / ".env")

import torch

from datasets import Dataset, load_from_disk  # Hugging Face Datasets for train/eval data
from sentence_transformers import (  # Core Sentence Transformers training primitives
    SentenceTransformer,  # bi-encoder model
    SentenceTransformerTrainer,  # high-level Trainer
    SentenceTransformerTrainingArguments,  # Trainer config wrapper
)
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import InformationRetrievalEvaluator  # Information Retrieval metrics (MRR, Recall@k, etc.)
from sentence_transformers.losses import MultipleNegativesRankingLoss  # in-batch negatives loss
from sentence_transformers.training_args import BatchSamplers  # batch samplers (NO_DUPLICATES, etc.)


def load_processed_data(processed_dir: Path) -> tuple[Dataset, Dataset | None, dict, dict, dict]:
    """
    Load train/eval datasets and Information retrieval artifacts from the processed directory.

    Args:
        processed_dir: Directory containing train_dataset/, eval_dataset/ (optional),
            and eval_queries.json, eval_corpus.json, eval_relevant_docs.json.

    Returns:
        train_dataset: HF Dataset with columns "anchor", "positive".
        eval_dataset: HF Dataset or None if eval_dataset/ is missing.
        eval_queries: Dict qid -> query text.
        eval_corpus: Dict cid -> document text.
        eval_relevant_docs: Dict qid -> set of cid.
    """
    processed_dir = Path(processed_dir)  # allow string/Path interchangeably
    logger.info("[Step 1/5] Loading processed data from disk...")

    processed_dir, subdir_msg = resolve_processed_dir(processed_dir, DEFAULT_PROCESSED_DIR)
    if subdir_msg:
        logger.info("%s", subdir_msg)

    train_path = processed_dir / "train_dataset"
    eval_path = processed_dir / "eval_dataset"

    # Load train dataset with columns: anchor, positive
    train_dataset = load_from_disk(str(train_path))

    # Eval dataset is optional (used for validation loss, not Information retrieval metrics)
    eval_dataset: Dataset | None = None
    if eval_path.exists():
        eval_dataset = load_from_disk(str(eval_path))

    # Information Retrieval artifacts for InformationRetrievalEvaluator
    with open(processed_dir / "eval_queries.json", "r") as f:
        eval_queries: dict[str, str] = json.load(f)
    with open(processed_dir / "eval_corpus.json", "r") as f:
        eval_corpus: dict[str, str] = json.load(f)
    with open(processed_dir / "eval_relevant_docs.json", "r") as f:
        raw_relevant = json.load(f)
        # Convert JSON lists to sets for the evaluator
        eval_relevant_docs: dict[str, set[str]] = {k: set(v) for k, v in raw_relevant.items()}

    logger.info(
        "  -> train: %d pairs, eval: %d pairs, queries: %d, corpus: %d",
        len(train_dataset),
        len(eval_dataset) if eval_dataset else 0,
        len(eval_queries),
        len(eval_corpus),
    )
    return train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs


def build_model(model_name: str, max_seq_length: int | None = None) -> SentenceTransformer:
    """
    Instantiate a SentenceTransformer model for the two-tower setup.

    Args:
        model_name: Name or path of the base SBERT model.
        max_seq_length: Optional max sequence length for inputs.

    Returns:
        A SentenceTransformer instance.
    """
    # Load base encoder (e.g. all-MiniLM-L6-v2)
    logger.info("  -> Loading base model: %s", model_name)
    model = SentenceTransformer(model_name)
    # Optionally enforce a max sequence length to avoid OOMs
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model


def build_information_retrieval_evaluator(
    eval_queries: dict[str, str],
    eval_corpus: dict[str, str],
    eval_relevant_docs: dict[str, set[str]],
    name: str = "order-recommendation",
) -> InformationRetrievalEvaluator:
    """
    Construct an InformationRetrievalEvaluator from prepared artifacts.

    This is used to measure Information retrieval metrics (Recall@k, MRR, NDCG) on the eval split.
    """
    return InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        name=name,
    )


def train_two_tower_sbert(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_length: int | None = 256,
    num_train_epochs: int = 3,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    learning_rate: float = 1e-4,
    loss_scale: float | None = 30.0,
    dataloader_num_workers: int = 4,
    run_information_retrieval_evaluator: bool = False,
) -> None:
    """
    Run training for the two-tower SBERT model using preprocessed data.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data prep params if available (saved by prepare_instacart_sbert)
    data_prep_params_path = processed_dir / "data_prep_params.json"
    if data_prep_params_path.exists():
        with open(data_prep_params_path) as f:
            data_prep_params = json.load(f)
    else:
        data_prep_params = None

    # 1. Load datasets and Information retrieval artifacts from disk
    train_dataset, eval_dataset, eval_queries, eval_corpus, eval_relevant_docs = load_processed_data(processed_dir)

    # 2. Build SentenceTransformer model and in-batch negatives loss
    logger.info("[Step 2/5] Building model and loss...")
    model = build_model(model_name, max_seq_length=max_seq_length)
    loss_kwargs = {} if loss_scale is None else {"scale": loss_scale}
    loss = MultipleNegativesRankingLoss(model, **loss_kwargs)

    # 3. Set up Information retrieval evaluator for Recall@k / MRR / NDCG feedback (optional, can be slow)
    logger.info("[Step 3/5] Setting up evaluator and training config...")
    information_retrieval_evaluator = (
        build_information_retrieval_evaluator(eval_queries, eval_corpus, eval_relevant_docs)
        if run_information_retrieval_evaluator
        else None
    )
    logger.info(
        "  -> Information retrieval evaluator: %s",
        "enabled" if information_retrieval_evaluator else "disabled",
    )

    # 4. Configure training loop (epochs, batch size, LR, eval cadence, etc.)
    # On MPS (Apple Silicon), multiprocessing and pin_memory cause RuntimeError; fp16 causes NaN gradients.
    use_mps = getattr(torch.backends.mps, "is_available", lambda: False)()
    num_workers = 0 if use_mps else dataloader_num_workers
    pin_memory = False if use_mps else True
    use_fp16 = not use_mps  # MPS + fp16 often leads to nan in contrastive loss (softmax/log)
    if use_mps:
        logger.info("  -> MPS detected: dataloader_num_workers=0, pin_memory=False, fp16=False (stability on Apple Silicon)")

    num_devices = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    steps_per_epoch = math.ceil(len(train_dataset) / (train_batch_size * num_devices))
    total_steps = num_train_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% warmup (equivalent to previous warmup_ratio=0.1)

    # When IR evaluator is on, save/load best checkpoint by NDCG@10
    ir_eval_name = "order-recommendation"
    load_best = run_information_retrieval_evaluator
    metric_for_best = f"eval_{ir_eval_name}_cosine_ndcg@10" if load_best else None

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",  # Use cosine decay instead of linear (gentler, helps avoid plateaus)
        fp16=use_fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # good for MultipleNegativesRankingLoss
        eval_strategy="epoch",  # evaluate once at end of each epoch
        save_strategy="epoch",  # save each epoch so we can load best
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        # Performance optimizations (disabled on MPS to avoid _share_filename_ RuntimeError):
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        # Fixed batch size every step (no partial batches) → avoids MPS recompilation from varying batch dim.
        dataloader_drop_last=True,
    )

    # 5. Wrap everything into a SentenceTransformerTrainer
    logger.info("[Step 4/5] Creating trainer...")
    logger.info("=" * 60)
    logger.info("Params")
    logger.info("-" * 60)
    if data_prep_params:
        logger.info("Data prep:")
        for k, v in data_prep_params.items():
            logger.info("  %s: %s", k, v)
        logger.info("-" * 60)
    logger.info("Training:")
    logger.info("  processed_dir: %s", processed_dir)
    logger.info("  output_dir: %s", output_dir)
    logger.info("  model_name: %s", model_name)
    logger.info("  max_seq_length: %s", max_seq_length)
    logger.info("  num_train_epochs: %d", num_train_epochs)
    logger.info("  train_batch_size: %d", train_batch_size)
    logger.info("  eval_batch_size: %d", eval_batch_size)
    logger.info("  learning_rate: %g", learning_rate)
    logger.info("  loss_scale: %s", loss_scale)
    logger.info("  dataloader_num_workers: %d", dataloader_num_workers)
    logger.info("  dataloader_drop_last: True (fixed batch size for MPS)")
    logger.info("  run_information_retrieval_evaluator: %s", run_information_retrieval_evaluator)
    logger.info("=" * 60)

    # Dynamic padding: pad to longest in batch. Transformer module already truncates to max_seq_length.
    # Batch shape will vary (may trigger MPS recompilation on shape change).
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss, evaluator=information_retrieval_evaluator
    )

    # Run the full training loop
    logger.info("[Step 5/5] Training started (loss and metrics will appear below)...")
    trainer.train()

    # Save final finetuned model
    logger.info("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    logger.info("Done. Model saved to %s", final_dir)


def main() -> None:
    """
    CLI entrypoint: parse arguments, call train_two_tower_sbert, and exit.
    """
    parser = argparse.ArgumentParser(description="Train two-tower SBERT model on Instacart data")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed datasets (train_dataset/, eval_dataset/, eval_*.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write trained models/checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model to finetune",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum sequence length for inputs (user context + product text). Default 256 for faster training with max_prior_orders=5, max_product_names=30.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=64, help="Per-device train batch size (64 or 128 recommended)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Per-device eval batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=30.0,
        metavar="S",
        help="Scale for MultipleNegativesRankingLoss (default 30.0). Higher = sharper softmax.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,  # mps only supports single-threaded data loading
        help="Number of worker processes for data loading (0 = single-threaded, 4-8 recommended)",
    )
    parser.add_argument(
        "--no-information-retrieval-evaluator",
        action="store_true",
        help="Disable Information retrieval evaluator during training (much faster, only validation loss will be tracked)",
    )

    args = parser.parse_args()

    setup_colored_logging(
        quiet_loggers=["httpx", "httpcore", "huggingface_hub", "urllib3", "sentence_transformers"],
    )

    train_two_tower_sbert(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        loss_scale=args.loss_scale,
        dataloader_num_workers=args.dataloader_num_workers,
        run_information_retrieval_evaluator=not args.no_information_retrieval_evaluator,
    )


if __name__ == "__main__":
    main()
