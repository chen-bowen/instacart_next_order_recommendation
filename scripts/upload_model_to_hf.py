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

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.constants import DEFAULT_CONFIG_UPLOAD_MODEL, DEFAULT_DOTENV_PATH, DEFAULT_MODEL_DIR, PROJECT_ROOT

load_dotenv(DEFAULT_DOTENV_PATH)


def load_config(config_path: Path | None = None) -> dict:
    """Load upload config from YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_UPLOAD_MODEL
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    model_dir = raw.get("model_dir", str(DEFAULT_MODEL_DIR))
    if model_dir and not Path(str(model_dir)).is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    return {
        "repo_id": raw.get("repo_id"),
        "model_dir": Path(model_dir) if model_dir else DEFAULT_MODEL_DIR,
        "private": bool(raw.get("private", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the trained two-tower SBERT model to the Hugging Face Hub.")
    parser.add_argument("--config", type=Path, default=None, help=f"Path to YAML config (default: {DEFAULT_CONFIG_UPLOAD_MODEL.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--repo-id", type=str, default=None, help="Override repo_id from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg["repo_id"]
    if not repo_id:
        raise SystemExit(f"repo_id is required. Set it in {DEFAULT_CONFIG_UPLOAD_MODEL.relative_to(PROJECT_ROOT)} or pass --repo-id YOUR_USERNAME/instacart-two-tower-sbert")

    model_dir = Path(cfg["model_dir"]).resolve()
    if not model_dir.is_dir():
        raise SystemExit(
            f"Model directory not found: {model_dir}. Train first with:\n"
            "  uv run python -m src.train"
        )

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=cfg["private"], exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {model_dir} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
