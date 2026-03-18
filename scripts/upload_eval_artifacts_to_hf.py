#!/usr/bin/env python3
"""
Upload eval_corpus.json and eval_queries.json to a Hugging Face dataset repo
with descriptive, versioned names based on the processed subdir.

Examples of remote filenames for processed/p5_mp20_ef0.1:
  instacart_eval_corpus_p5_mp20_ef0.1.json
  instacart_eval_queries_p5_mp20_ef0.1.json

Usage:
  uv run python scripts/upload_eval_artifacts_to_hf.py \
    --repo-id YOUR_USERNAME/instacart-eval-artifacts \
    --processed-dir processed/p5_mp20_ef0.1

Authenticate with: huggingface-cli login or HF_TOKEN in .env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.constants import DEFAULT_DOTENV_PATH, PROJECT_ROOT


load_dotenv(DEFAULT_DOTENV_PATH)


def _resolve(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload eval_corpus.json and eval_queries.json to a Hugging Face dataset repo " "with descriptive, versioned filenames."
        )
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo id, e.g. YOUR_USERNAME/instacart-eval-artifacts",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed/p5_mp20_ef0.1"),
        help=("Processed dir containing eval_corpus.json and eval_queries.json " "(default: processed/p5_mp20_ef0.1)"),
    )
    args = parser.parse_args()

    processed_dir = _resolve(args.processed_dir)
    version = processed_dir.name  # e.g. p5_mp20_ef0.1

    corpus_path = processed_dir / "eval_corpus.json"
    queries_path = processed_dir / "eval_queries.json"

    if not corpus_path.is_file():
        raise SystemExit(f"eval_corpus.json not found at {corpus_path}")
    if not queries_path.is_file():
        raise SystemExit(f"eval_queries.json not found at {queries_path}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    remote_corpus_name = f"product_catalog_corpus_{version}.json"
    remote_queries_name = f"product_queries_{version}.json"

    print(f"Uploading {corpus_path} to datasets/{args.repo_id} as {remote_corpus_name} ...")
    api.upload_file(
        path_or_fileobj=str(corpus_path),
        path_in_repo=remote_corpus_name,
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    print(f"Uploading {queries_path} to datasets/{args.repo_id} as {remote_queries_name} ...")
    api.upload_file(
        path_or_fileobj=str(queries_path),
        path_in_repo=remote_queries_name,
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    base = f"https://huggingface.co/datasets/{args.repo_id}"
    print("Done.")
    print(f"  - {base}/blob/main/{remote_corpus_name}")
    print(f"  - {base}/blob/main/{remote_queries_name}")


if __name__ == "__main__":
    main()
