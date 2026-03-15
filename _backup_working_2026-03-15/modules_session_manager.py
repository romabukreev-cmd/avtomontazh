"""
session_manager.py — управление сессиями записи.

Сессия = папка в input/ с файлами экрана и вебки:
  input/2024-01-15_logo-design/
    screen_001.mp4
    screen_002.mp4
    webcam_001.mp4
    webcam_002.mp4
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import config

log = logging.getLogger(__name__)


@dataclass
class Session:
    name:         str        # имя папки, например "2024-01-15_logo-design"
    path:         Path       # полный путь к папке
    screen_files: List[Path] # файлы экрана, отсортированные по номеру
    webcam_files: List[Path] # файлы вебки, отсортированные по номеру

    @property
    def file_count(self) -> int:
        return len(self.screen_files)

    def __str__(self) -> str:
        n = self.file_count
        return f"{self.name} ({n} файл{'а' if n in (2, 3, 4) else 'ов'})"


class SessionManager:

    def scan_sessions(self) -> List[Session]:
        """Возвращает сессии готовые к обработке (не обработанные, экран + вебка)."""
        sessions = []
        if not config.INPUT_DIR.exists():
            log.warning(f"Папка input/ не существует: {config.INPUT_DIR}")
            return sessions

        for session_dir in sorted(config.INPUT_DIR.iterdir()):
            if not session_dir.is_dir():
                continue

            screen = self._find_sorted_files(session_dir, config.SCREEN_FILE_PATTERN)
            webcam = self._find_sorted_files(session_dir, config.WEBCAM_FILE_PATTERN)

            if not screen:
                log.debug(f"Пропуск {session_dir.name}: нет файлов экрана")
                continue
            if not webcam:
                log.debug(f"Пропуск {session_dir.name}: нет файлов вебки")
                continue
            if len(screen) != len(webcam):
                log.warning(
                    f"Пропуск {session_dir.name}: "
                    f"несовпадение файлов (экран: {len(screen)}, вебка: {len(webcam)})"
                )
                continue
            if self.is_processed(session_dir.name):
                log.debug(f"Пропуск {session_dir.name}: уже обработана")
                continue

            sessions.append(Session(
                name=session_dir.name,
                path=session_dir,
                screen_files=screen,
                webcam_files=webcam,
            ))

        log.info(f"Найдено {len(sessions)} сессий для обработки")
        return sessions

    def is_processed(self, session_name: str) -> bool:
        """Сессия обработана — в output/name/ есть хотя бы один .mp4."""
        out = config.OUTPUT_DIR / session_name
        return out.exists() and any(out.glob("*.mp4"))

    def concat_files(self, session: Session) -> Tuple[Path, Path]:
        """
        Склеивает части сессии в один файл экрана и один файл вебки.
        Если часть одна — возвращает её напрямую без конкатенации.
        """
        if session.file_count == 1:
            return session.screen_files[0], session.webcam_files[0]

        log.info(f"Конкатенация {session.file_count} частей: {session.name}")
        screen_out = self._concat(session.screen_files, f"{session.name}_screen_full")
        webcam_out = self._concat(session.webcam_files, f"{session.name}_webcam_full")
        return screen_out, webcam_out

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _find_sorted_files(self, directory: Path, pattern: str) -> List[Path]:
        files = [
            f for f in directory.glob(pattern)
            if f.suffix.lower() in config.VIDEO_EXTENSIONS
        ]
        files.sort(key=lambda f: self._extract_number(f.stem))
        return files

    def _extract_number(self, stem: str) -> int:
        m = re.search(r"(\d+)$", stem)
        return int(m.group(1)) if m else 0

    def _concat(self, files: List[Path], output_stem: str) -> Path:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_path    = config.TEMP_DIR / f"{output_stem}.mp4"
        list_path      = config.TEMP_DIR / f"{output_stem}_list.txt"

        with open(list_path, "w", encoding="utf-8") as f:
            for file in files:
                escaped = str(file.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        log.info(f"FFmpeg concat: {len(files)} файлов → {output_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        list_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat завершился с ошибкой:\n{result.stderr[-2000:]}")

        log.info(f"Конкатенация готова: {output_path.name}")
        return output_path
