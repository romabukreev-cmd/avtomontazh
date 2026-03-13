"""utils.py — вспомогательные функции."""

import logging
import sys
from pathlib import Path

import config


def setup_logging() -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-20s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOGS_DIR / "automontazh.log", encoding="utf-8"),
        ],
    )


def ensure_dirs() -> None:
    for d in [config.INPUT_DIR, config.OUTPUT_DIR, config.TEMP_DIR, config.LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    """1834.0 → '30:34'"""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"
