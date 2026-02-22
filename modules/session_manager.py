"""
session_manager.py — управление сессиями записи.

Сессия = папка в input/ содержащая пронумерованные файлы:
  input/2024-01-15_logo-design/
    screen_001.mp4
    screen_002.mp4
    webcam_001.mp4
    webcam_002.mp4

Модуль умеет:
  - Найти все доступные сессии (папки с нужными файлами)
  - Определить какие сессии ещё не обработаны
  - Склеить несколько пронумерованных файлов в один через FFmpeg concat
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import config

log = logging.getLogger(__name__)


@dataclass
class Session:
    """Описание одной сессии записи."""
    name: str                    # имя папки, например "2024-01-15_logo-design"
    path: Path                   # полный путь к папке сессии
    screen_files: List[Path]     # файлы экрана, отсортированные по номеру
    webcam_files: List[Path]     # файлы вебки, отсортированные по номеру

    @property
    def file_count(self) -> int:
        return len(self.screen_files)

    def __str__(self) -> str:
        return f"{self.name} ({self.file_count} файл{'а' if self.file_count in (2,3,4) else 'ов'})"


class SessionManager:

    def scan_sessions(self) -> List[Session]:
        """
        Сканирует input/ и возвращает список сессий, готовых к обработке.
        Готова = есть файлы экрана и вебки, количество совпадает, сессия не обработана.
        """
        sessions = []

        if not config.INPUT_DIR.exists():
            log.warning(f"Папка input/ не существует: {config.INPUT_DIR}")
            return sessions

        for session_dir in sorted(config.INPUT_DIR.iterdir()):
            if not session_dir.is_dir():
                continue

            screen_files = self._find_sorted_files(session_dir, config.SCREEN_FILE_PATTERN)
            webcam_files = self._find_sorted_files(session_dir, config.WEBCAM_FILE_PATTERN)

            if not screen_files:
                log.debug(f"Пропуск {session_dir.name}: нет файлов экрана")
                continue
            if not webcam_files:
                log.debug(f"Пропуск {session_dir.name}: нет файлов вебки")
                continue
            if len(screen_files) != len(webcam_files):
                log.warning(
                    f"Пропуск {session_dir.name}: "
                    f"количество файлов не совпадает "
                    f"(экран: {len(screen_files)}, вебка: {len(webcam_files)})"
                )
                continue

            if self.is_processed(session_dir.name):
                log.debug(f"Пропуск {session_dir.name}: уже обработана")
                continue

            sessions.append(Session(
                name=session_dir.name,
                path=session_dir,
                screen_files=screen_files,
                webcam_files=webcam_files,
            ))

        log.info(f"Найдено {len(sessions)} сессий для обработки")
        return sessions

    def is_processed(self, session_name: str) -> bool:
        """Сессия считается обработанной если в output/session_name/ есть хотя бы один .mp4."""
        output_dir = config.OUTPUT_DIR / session_name
        if not output_dir.exists():
            return False
        return any(output_dir.glob("*.mp4"))

    def concat_files(self, session: Session) -> Tuple[Path, Path]:
        """
        Склеивает все файлы сессии в два объединённых файла (экран и вебка).

        Если файл один — возвращает его путь без конкатенации.
        Если несколько — создаёт temp/{session_name}_screen_full.mp4 и _webcam_full.mp4.

        Returns:
            (screen_full_path, webcam_full_path)
        """
        if session.file_count == 1:
            # Один файл — конкатенация не нужна
            return session.screen_files[0], session.webcam_files[0]

        log.info(f"Конкатенация {session.file_count} файлов для сессии {session.name}")
        screen_out = self._concat(session.screen_files, f"{session.name}_screen_full")
        webcam_out = self._concat(session.webcam_files, f"{session.name}_webcam_full")
        return screen_out, webcam_out

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _find_sorted_files(self, directory: Path, pattern: str) -> List[Path]:
        """
        Находит файлы по glob-паттерну и сортирует по числовому суффиксу.
        screen_001.mp4 < screen_002.mp4 < screen_010.mp4 (числовая, не лексикографическая)
        """
        files = [
            f for f in directory.glob(pattern)
            if f.suffix.lower() in config.VIDEO_EXTENSIONS
        ]
        files.sort(key=lambda f: self._extract_number(f.stem))
        return files

    def _extract_number(self, stem: str) -> int:
        """Извлекает номер из имени файла: 'screen_003' → 3."""
        match = re.search(r"(\d+)$", stem)
        return int(match.group(1)) if match else 0

    def _concat(self, files: List[Path], output_stem: str) -> Path:
        """
        Склеивает список видеофайлов в один через FFmpeg concat demuxer.
        Сохраняет результат в temp/.
        """
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.TEMP_DIR / f"{output_stem}.mp4"

        # Создаём временный список файлов для FFmpeg
        concat_list_path = config.TEMP_DIR / f"{output_stem}_list.txt"
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for file in files:
                # FFmpeg требует абсолютные пути с экранированием апострофов
                escaped = str(file.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",          # копирование без перекодирования — быстро
            str(output_path),
        ]

        log.info(f"FFmpeg concat: {len(files)} файлов → {output_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat завершился с ошибкой:\n{result.stderr[-2000:]}"
            )

        # Удаляем временный список
        concat_list_path.unlink(missing_ok=True)
        log.info(f"Конкатенация завершена: {output_path.name}")
        return output_path
