#!/usr/bin/env python3
"""
Upload the trained two-tower SBERT model (models/two_tower_sbert/final/) to the Hugging Face Hub.

Usage:
  uv run python scripts/upload_model_to_hf.py --repo-id YOUR_USERNAME/instacart-two-tower-sbert

Authenticate with: huggingface-cli login or HF_TOKEN in .env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Project root and .env so HF_TOKEN is available for private repos / upload
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Default path written by train_sbert (best checkpoint by NDCG@10 when IR eval is on)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "two_tower_sbert" / "final"


def main() -> None:
    # CLI: repo id (required), optional model path and private flag
    parser = argparse.ArgumentParser(
        description="Upload the trained two-tower SBERT model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. username/instacart-two-tower-sbert",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local directory containing the saved model (default: models/two_tower_sbert/final)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private",
    )
    args = parser.parse_args()

    # Resolve path and ensure the trained model directory exists
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(
            f"Model directory not found: {model_dir}. Train first with:\n"
            "  uv run python -m src.train.train_sbert --lr 1e-4"
        )

    # Create repo if needed, then upload all files under model_dir
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded {model_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
