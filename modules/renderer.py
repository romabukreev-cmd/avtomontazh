"""
renderer.py — сборка финального видео через FFmpeg.

Нарезка выполняется через concat demuxer:
  1. Создаём список отрезков (inpoint/outpoint) для каждого kept-блока
  2. FFmpeg читает список и склеивает отрезки в один поток
  3. Поверх применяем filtergraph (кроп, масштаб, vstack)

Формат 1: вертикальный 1080×1920 (split-screen, экран сверху + вебка снизу).
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config

log = logging.getLogger(__name__)


class VideoRenderer:

    def __init__(self, screen_file: Path, webcam_file: Path, session_name: str):
        self.screen_file  = screen_file
        self.webcam_file  = webcam_file
        self.session_name = session_name
        self._temp_files: List[Path] = []

    def render_vertical(
        self,
        timeline: List[Dict],
        output_dir: Path,
        output_filename: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Рендерит вертикальное видео 1080×1920 (экран сверху, вебка снизу).

        timeline — список блоков с полями start и end.
        """
        if not timeline:
            raise ValueError("Пустой таймлайн — нечего рендерить")

        output_file = output_dir / output_filename
        screen_list = self._write_concat_list(self.screen_file, timeline, "screen")
        webcam_list = self._write_concat_list(self.webcam_file, timeline, "webcam")

        # Экран может быть 1920×1200 — кропаем по Y чтобы получить 1920×1080
        cy = config.SCREEN_CROP_Y  # 60 для 1920×1200, 0 для 1920×1080

        # Filtergraph:
        #   [0:v] экран  → crop центр 1215×1080 → scale 1080×960 → [top]
        #   [1:v] вебка  → нормализуем → crop центр 1215×1080 → scale 1080×960 → [bot]
        #   [top][bot]   → vstack → [vout]
        #   [0:a]        → afade in/out → [aout]
        total = sum(s["end"] - s["start"] for s in timeline)
        fade_out_start = max(0.0, total - 0.5)

        filtergraph = (
            f"[0:v]crop=1215:1080:352:{cy},scale=1080:960[top];"
            "[1:v]scale=1920:1080:flags=lanczos,setsar=1,crop=1215:1080:352:0,scale=1080:960[bot];"
            "[top][bot]vstack[vout];"
            f"[0:a]afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_start:.3f}:d=0.5[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(screen_list),
            "-f", "concat", "-safe", "0", "-i", str(webcam_list),
            "-filter_complex", filtergraph,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", config.VIDEO_CODEC,
            "-crf", str(config.VIDEO_CRF),
            "-preset", config.VIDEO_PRESET,
            "-c:a", config.AUDIO_CODEC,
            "-b:a", config.AUDIO_BITRATE,
            "-threads", str(config.FFMPEG_THREADS),
            str(output_file),
        ]

        log.info(f"Рендер: {output_filename} ({len(timeline)} блоков, {total:.1f}с)")
        self._run_ffmpeg(cmd, total, progress_callback)
        return output_file

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _write_concat_list(self, source_file: Path, timeline: List[Dict], tag: str) -> Path:
        """Создаёт временный .txt файл со списком inpoint/outpoint для FFmpeg concat."""
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        list_path = config.TEMP_DIR / f"{self.session_name}_{tag}_list.txt"

        with open(list_path, "w", encoding="utf-8") as f:
            for seg in timeline:
                escaped = str(source_file.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
                f.write(f"inpoint {seg['start']:.3f}\n")
                f.write(f"outpoint {seg['end']:.3f}\n")

        self._temp_files.append(list_path)
        return list_path

    def _run_ffmpeg(
        self,
        cmd: List[str],
        total_duration: float,
        progress_callback: Optional[Callable[[float], None]],
    ) -> None:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stderr_lines = []
        last_pct = -1

        for line in process.stderr:
            stderr_lines.append(line)
            match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if match and total_duration > 0 and progress_callback:
                h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
                current = h * 3600 + m * 60 + s
                pct = min(current / total_duration * 100, 99)
                if pct - last_pct >= 5:
                    last_pct = pct
                    try:
                        progress_callback(pct)
                    except Exception:
                        pass

        process.wait()

        if process.returncode != 0:
            error_tail = "".join(stderr_lines[-30:])
            raise RuntimeError(f"FFmpeg завершился с ошибкой:\n{error_tail}")

        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass

    def cleanup_temp(self) -> None:
        """Удаляет все временные файлы созданные при рендере."""
        for f in self._temp_files:
            f.unlink(missing_ok=True)
        self._temp_files.clear()
