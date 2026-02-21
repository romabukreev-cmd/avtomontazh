"""
utils.py — вспомогательные функции, используемые во всех модулях.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List


def setup_logging(logs_dir: Path, level: str = "INFO") -> None:
    """
    Настраивает логирование:
      - В консоль: вывод с временем и уровнем
      - В файл:    logs/automontazh.log с ротацией (10 МБ × 7 файлов)
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Файл с автоматической ротацией
    log_file = logs_dir / "automontazh.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 МБ на файл
        backupCount=7,               # хранить 7 старых файлов
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Не добавляем хендлеры повторно при перезапуске
    if not root.handlers:
        root.addHandler(console_handler)
        root.addHandler(file_handler)


def ensure_dirs(dirs: List[Path]) -> None:
    """Создаёт папки если они не существуют."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    """Форматирует секунды в читаемый вид: 3661 → '1:01:01', 125 → '2:05'."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
