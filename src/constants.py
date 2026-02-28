"""Shared path and other constants for the Instacart personalization project."""

from __future__ import annotations

from pathlib import Path

# Repository root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data prep: input data and output (param-based subdirs go under DEFAULT_PROCESSED_DIR)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "processed"

# Training: where to read processed data and where to write checkpoints
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "two_tower_sbert"

# Serve: trained model and product corpus (data prep writes to param subdir, e.g. p5_mp20_ef0.1)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "two_tower_sbert" / "final"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "processed" / "p5_mp20_ef0.1" / "eval_corpus.json"

# Embedding index cache (under corpus parent dir)
INDEX_SUBDIR = ".embedding_index"
MANIFEST_FILENAME = "manifest.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
PRODUCT_IDS_FILENAME = "product_ids.json"
