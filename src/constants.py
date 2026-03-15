"""
Shared path and other constants for the Instacart personalization project.

Centralizes file paths, config locations, and magic strings used across
data prep, training, inference, baselines, and API.
"""

from __future__ import annotations

from pathlib import Path

# Repository root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Environment: .env file and feedback DB path override
DOTENV_FILENAME = ".env"
DEFAULT_DOTENV_PATH = PROJECT_ROOT / DOTENV_FILENAME
ENV_FEEDBACK_DB_PATH = "FEEDBACK_DB_PATH"

# Config files (YAML)
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_DATA_PREP = CONFIG_DIR / "data_prep.yaml"
DEFAULT_CONFIG_TRAIN = CONFIG_DIR / "train.yaml"
DEFAULT_CONFIG_INFERENCE = CONFIG_DIR / "inference.yaml"
DEFAULT_CONFIG_BASELINES = CONFIG_DIR / "baselines.yaml"
DEFAULT_CONFIG_COMPARE = CONFIG_DIR / "compare_untrained_vs_trained.yaml"
DEFAULT_CONFIG_FEEDBACK_ANALYTICS = CONFIG_DIR / "feedback_analytics.yaml"
DEFAULT_CONFIG_UPLOAD_MODEL = CONFIG_DIR / "upload_model.yaml"
DEFAULT_CONFIG_UPLOAD_CORPUS = CONFIG_DIR / "upload_corpus.yaml"
DEFAULT_CONFIG_GENERATE_SAMPLE_FEEDBACK = CONFIG_DIR / "generate_sample_feedback.yaml"

# Data prep: input CSVs under data_dir; output goes to processed/<param_subdir>/
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "processed"

# Raw data filenames (under data_dir)
PRODUCTS_CSV = "products.csv"
AISLES_CSV = "aisles.csv"
DEPARTMENTS_CSV = "departments.csv"
ORDERS_CSV = "orders.csv"
ORDER_PRODUCTS_PRIOR_CSV = "order_products__prior.csv"
ORDER_PRODUCTS_TRAIN_CSV = "order_products__train.csv"

# Chunk size for reading order_products__prior.csv (avoids loading ~32M rows at once)
ORDER_PRODUCTS_CHUNK_SIZE = 500_000

# orders.csv eval_set column values
EVAL_SET_TRAIN = "train"
EVAL_SET_PRIOR = "prior"

# Processed output filenames (under processed subdir)
EVAL_QUERIES_FILENAME = "eval_queries.json"
EVAL_CORPUS_FILENAME = "eval_corpus.json"
EVAL_RELEVANT_DOCS_FILENAME = "eval_relevant_docs.json"
DATA_PREP_PARAMS_FILENAME = "data_prep_params.json"
TRAIN_DATASET_SUBDIR = "train_dataset"
EVAL_DATASET_SUBDIR = "eval_dataset"

# Sample user contexts when eval_queries.json is not available (generate_sample_feedback)
SAMPLE_USER_CONTEXTS = [
    "[+7d w4h14] Organic Milk, Whole Wheat Bread.",
    "[+3d w1h9] Banana, Greek Yogurt, Honey.",
    "[+14d w6h18] Chicken Breast, Broccoli, Rice.",
    "[+1d w0h12] Coffee, Oat Milk, Granola.",
    "[+5d w3h20] Pasta, Tomato Sauce, Parmesan.",
]

# Training: where to read processed data and where to write checkpoints
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "two_tower_sbert"

# Serve: trained model and product corpus (data prep writes to param subdir, e.g. p5_mp20_ef0.1)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "two_tower_sbert" / "final"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "processed" / "p5_mp20_ef0.1" / EVAL_CORPUS_FILENAME

# Hugging Face fallback when eval_corpus.json not found locally (env: CORPUS_HF_REPO, CORPUS_HF_REPO_TYPE)
DEFAULT_CORPUS_HF_REPO = "chenbowen184/instacart-eval-corpus"
DEFAULT_CORPUS_HF_REPO_TYPE = "dataset"

# Corpus upload: max products allowed via POST /admin/corpus (env: MAX_CORPUS_UPLOAD_PRODUCTS)
MAX_CORPUS_UPLOAD_PRODUCTS = 100_000

# Feedback
DEFAULT_FEEDBACK_DB_PATH = PROJECT_ROOT / "data" / "feedback.db"

# Embedding index cache (under corpus parent dir)
INDEX_SUBDIR = ".embedding_index"
MANIFEST_FILENAME = "manifest.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
PRODUCT_IDS_FILENAME = "product_ids.json"
