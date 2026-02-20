"""Shared utilities for the Instacart personalization project."""

from __future__ import annotations

import logging
from pathlib import Path


_GRAY = "\033[90m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_LEVEL_COLORS = {
    "DEBUG": _GRAY,
    "INFO": _GREEN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
}


class ColoredFormatter(logging.Formatter):
    """Formatter that colorizes level name and logger name, message-only output."""

    def format(self, record: logging.LogRecord) -> str:
        c = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{c}{record.levelname:5}{_RESET}"
        record.name = f"{_GRAY}{record.name}{_RESET}"
        return super().format(record)


def setup_colored_logging(
    level: int = logging.INFO,
    fmt: str = "%(message)s",
    quiet_loggers: list[str] | None = None,
) -> None:
    """Configure root logger with colored, compact output.

    Args:
        level: Root log level (default INFO).
        fmt: Log record format (default message only).
        quiet_loggers: Logger names to set to WARNING (e.g. httpx, urllib3).
    """
    if quiet_loggers:
        for name in quiet_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(fmt))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def resolve_processed_dir(processed_dir: Path, default_processed_dir: Path) -> tuple[Path, str | None]:
    """Resolve processed dir, auto-selecting a param subdir when using the default and no train_dataset at top level.

    When processed_dir is the default (e.g. processed/) and has no train_dataset, looks for subdirs
    that contain train_dataset (e.g. processed/p5_mp20_ef0.1) and returns the single or latest one.

    Returns:
        (resolved_path, log_message): log_message is non-None when a subdir was auto-selected (caller may log it).

    Raises:
        FileNotFoundError: If train_dataset is not found at processed_dir or any subdir.
    """
    processed_dir = Path(processed_dir)
    train_path = processed_dir / "train_dataset"

    if not train_path.exists() and processed_dir == default_processed_dir:
        subdirs_with_data = [
            d for d in processed_dir.iterdir()
            if d.is_dir() and (d / "train_dataset").exists()
        ]
        if len(subdirs_with_data) == 1:
            resolved = subdirs_with_data[0]
            return resolved, f"  -> Using param subdir: {resolved.name}"
        if len(subdirs_with_data) > 1:
            resolved = max(subdirs_with_data, key=lambda d: (d / "train_dataset").stat().st_mtime)
            return resolved, f"  -> Multiple subdirs found, using latest: {resolved.name}"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train dataset not found at {train_path}. Run data prep first or pass --processed-dir (e.g. processed/p5_mp20_ef0.1)."
        )
    return processed_dir, None
