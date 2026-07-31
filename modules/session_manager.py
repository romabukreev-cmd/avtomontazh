"""
session_manager.py — управление сессиями записи.

Сессия = папка в input/ с файлами вида:
  input/2024-01-15_logo-design/
    screen_001.mp4    # запись экрана
    screen_002.mp4
    webcam_001.mp4    # запись вебки
    webcam_002.mp4

Допускаются сессии только с экраном или только с вебкой.
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import config

log = logging.getLogger(__name__)


@dataclass
class Session:
    name:         str
    path:         Path
    screen_files: List[Path] = field(default_factory=list)
    webcam_files: List[Path] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.screen_files) + len(self.webcam_files)

    @property
    def has_screen(self) -> bool:
        return bool(self.screen_files)

    @property
    def has_webcam(self) -> bool:
        return bool(self.webcam_files)

    def __str__(self) -> str:
        parts = []
        if self.screen_files:
            n = len(self.screen_files)
            parts.append(f"{n} 🖥")
        if self.webcam_files:
            n = len(self.webcam_files)
            parts.append(f"{n} 🎥")
        return f"{self.name} ({' + '.join(parts)})"


class SessionManager:

    def scan_sessions(self) -> List[Session]:
        """Возвращает сессии готовые к обработке."""
        sessions = []
        if not config.INPUT_DIR.exists():
            log.warning(f"Папка input/ не существует: {config.INPUT_DIR}")
            return sessions

        for session_dir in sorted(config.INPUT_DIR.iterdir()):
            if not session_dir.is_dir():
                continue

            screen_files = self._find_files(session_dir, "screen")
            webcam_files = self._find_files(session_dir, "webcam")

            if not screen_files and not webcam_files:
                log.debug(f"Пропуск {session_dir.name}: нет screen_*/webcam_* файлов")
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
        """Сессия считается обработанной если есть хоть один готовый итоговый файл."""
        out = config.OUTPUT_DIR / session_name
        if not out.exists():
            return False
        for fname in ("vertical_9min.mp4", "horizontal_9min.mp4"):
            f = out / fname
            if f.exists() and f.is_file() and f.stat().st_size > 0:
                return True
        return False

    def concat_files(self, session: Session) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Склеивает части сессии.
        Возвращает (screen_path, webcam_path) — любое из двух может быть None.
        Если частей одна — возвращает её напрямую (без создания temp-файла).
        """
        screen_out = self._prepare(session.screen_files, f"{session.name}_screen_full")
        webcam_out = self._prepare(session.webcam_files, f"{session.name}_webcam_full")
        return screen_out, webcam_out

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _find_files(self, directory: Path, prefix: str) -> List[Path]:
        files = [
            f for f in directory.iterdir()
            if f.is_file()
            and f.suffix.lower() in config.VIDEO_EXTENSIONS
            and f.stem.lower().startswith(prefix)
        ]
        files.sort(key=lambda f: self._extract_number(f.stem))
        return files

    def _extract_number(self, stem: str) -> int:
        m = re.search(r"(\d+)$", stem)
        return int(m.group(1)) if m else 0

    def _prepare(self, files: List[Path], output_stem: str) -> Optional[Path]:
        if not files:
            return None
        if len(files) == 1:
            return files[0]
        log.info(f"Конкатенация {len(files)} файлов: {output_stem}")
        return self._concat(files, output_stem)

    def _probe_params(self, file: Path) -> tuple:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "default=noprint_wrappers=1",
            str(file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return tuple(sorted(result.stdout.strip().splitlines()))

    def _concat(self, files: List[Path], output_stem: str) -> Path:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.TEMP_DIR / f"{output_stem}.mp4"
        list_path   = config.TEMP_DIR / f"{output_stem}_list.txt"

        with open(list_path, "w", encoding="utf-8") as f:
            for file in files:
                escaped = str(file.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        params    = {self._probe_params(f) for f in files}
        safe_copy = len(params) == 1

        if safe_copy:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(output_path),
            ]
            log.info(f"FFmpeg concat (-c copy): {len(files)} файлов → {output_path.name}")
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-fps_mode", "cfr", "-r", str(config.CONCAT_FPS),
                "-c:v", "libx264", "-preset", config.CONCAT_PRESET, "-crf", str(config.CONCAT_CRF),
                "-pix_fmt", config.CONCAT_PIX_FMT,
                "-c:a", "aac", "-ar", str(config.CONCAT_AUDIO_RATE), "-b:a", "192k",
                str(output_path),
            ]
            log.warning(f"FFmpeg concat (перекодирование): {len(files)} файлов → {output_path.name}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        list_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat ошибка:\n{result.stderr[-2000:]}")

        log.info(f"Конкатенация готова: {output_path.name}")
        return output_path
