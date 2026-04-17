"""
renderer.py — рендер видео через FFmpeg.

Старый монолитный подход (один filter_complex с 3×N trim-нод + 3 concat=n=N)
ловил swap death на длинных видео: все trim-ветки буферизуются в памяти пока
concat не получит данные со всех входов → OOM. Заменено на chunked-рендер:
каждый отрезок timeline рендерится отдельной короткой ffmpeg-командой
(≤100 MB памяти на процесс), затем все chunk_*.mp4 склеиваются через
`concat demuxer -c copy` без перекодирования. Память O(1) по длине видео.

Для безболезненной склейки через `-c copy` все чанки должны быть одинаковыми
по кодеку/fps/разрешению/audio rate — параметры жёстко фиксируются.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config

log = logging.getLogger(__name__)

_TARGET_FPS         = 30
_TARGET_AUDIO_RATE  = 48000
_TARGET_PIX_FMT     = "yuv420p"
_KEYFRAME_INTERVAL  = 60   # GOP=60 кадров ≈ 2с при 30fps


class VideoRenderer:

    def render_vertical(
        self,
        timeline:    List[Dict],
        output_dir:  Path,
        screen_file: Path,
        webcam_file: Path,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Path:
        """Вертикальный split-screen 1080×1920: сверху экран, снизу вебка."""
        return self._render(
            timeline, output_dir, screen_file, webcam_file,
            output_name="vertical_9min.mp4",
            filter_complex=_vertical_filter(config.SCREEN_CROP_Y),
            on_progress=on_progress,
        )

    def render_horizontal(
        self,
        timeline:    List[Dict],
        output_dir:  Path,
        screen_file: Path,
        webcam_file: Path,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Path:
        """Горизонтальный PiP 1920×1080: экран + вебка квадратом в углу."""
        return self._render(
            timeline, output_dir, screen_file, webcam_file,
            output_name="horizontal_9min.mp4",
            filter_complex=_horizontal_filter(
                config.SCREEN_CROP_Y,
                config.PIP_WIDTH, config.PIP_HEIGHT,
                config.PIP_MARGIN_RIGHT, config.PIP_MARGIN_BOTTOM,
            ),
            on_progress=on_progress,
        )

    # ── Общая логика ──────────────────────────────────────────────────────────

    def _render(
        self,
        timeline:       List[Dict],
        output_dir:     Path,
        screen_file:    Path,
        webcam_file:    Path,
        output_name:    str,
        filter_complex: str,
        on_progress:    Optional[Callable[[int], None]],
    ) -> Path:
        if not timeline:
            raise RuntimeError("Пустой timeline — нечего рендерить.")

        output_path = output_dir / output_name
        chunks_dir  = config.TEMP_DIR / f"chunks_{output_dir.name}_{output_name.split('.')[0]}"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Рендер {output_name}: {len(timeline)} отрезков → chunks_dir={chunks_dir}")

        chunk_paths: List[Path] = []
        try:
            for i, seg in enumerate(timeline):
                chunk_path = chunks_dir / f"chunk_{i:04d}.mp4"
                self._render_chunk(seg, screen_file, webcam_file, chunk_path, filter_complex)
                chunk_paths.append(chunk_path)

                if on_progress:
                    pct = min(99, int((i + 1) / len(timeline) * 100))
                    on_progress(pct)

            list_path = chunks_dir / "list.txt"
            with open(list_path, "w", encoding="utf-8") as f:
                for p in chunk_paths:
                    escaped = str(p.resolve()).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Concat-demuxer упал (код {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )

            if on_progress:
                on_progress(100)
        finally:
            shutil.rmtree(chunks_dir, ignore_errors=True)

        return output_path

    def _render_chunk(
        self,
        seg:            Dict,
        screen_file:    Path,
        webcam_file:    Path,
        chunk_path:     Path,
        filter_complex: str,
    ) -> None:
        start = seg["start"]
        dur   = seg["end"] - start

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
            "-i", str(screen_file),
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
            "-i", str(webcam_file),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", config.VIDEO_CODEC,
            "-crf", str(config.VIDEO_CRF),
            "-preset", config.VIDEO_PRESET,
            "-pix_fmt", _TARGET_PIX_FMT,
            "-r", str(_TARGET_FPS),
            "-g", str(_KEYFRAME_INTERVAL),
            "-force_key_frames", "expr:eq(n,0)",
            "-c:a", config.AUDIO_CODEC,
            "-b:a", config.AUDIO_BITRATE,
            "-ar", str(_TARGET_AUDIO_RATE),
            "-threads", str(config.FFMPEG_THREADS),
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Рендер чанка {chunk_path.name} упал "
                f"(seg {start:.2f}→{seg['end']:.2f}, код {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )


# ── Фильтры компоновки (один отрезок, два входа) ──────────────────────────────

def _vertical_filter(crop_y: int) -> str:
    return (
        f"[0:v]crop=1215:1080:352:{crop_y},scale=1080:960[top];"
        f"[1:v]scale=1920:1080:flags=lanczos,setsar=1,crop=1215:1080:352:0,scale=1080:960[bot];"
        f"[top][bot]vstack[vout];"
        f"[0:a]anull[aout]"
    )


def _horizontal_filter(crop_y: int, pip_w: int, pip_h: int, pip_mr: int, pip_mb: int) -> str:
    return (
        f"[0:v]crop=1920:1080:0:{crop_y}[screen];"
        f"[1:v]crop=1080:1080:420:0,scale={pip_w}:{pip_h}[pip];"
        f"[screen][pip]overlay=W-w-{pip_mr}:H-h-{pip_mb}[vout];"
        f"[0:a]anull[aout]"
    )
