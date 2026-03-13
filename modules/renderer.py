"""
renderer.py — рендер видео через FFmpeg.

Принимает timeline (список под-сегментов с start/end) и два видеофайла (экран, вебка).
Создаёт два файла: vertical_9min.mp4 (1080×1920) и horizontal_9min.mp4 (1920×1080).

Используется FFmpeg concat demuxer с inpoint/outpoint для точной нарезки без перемотки.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config

log = logging.getLogger(__name__)

_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


class VideoRenderer:

    def render_vertical(
        self,
        timeline:    List[Dict],
        output_dir:  Path,
        screen_file: Path,
        webcam_file: Path,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Path:
        """
        Вертикальный split-screen 1080×1920:
          сверху — экран (crop→scale 1080×960)
          снизу  — вебка (normalize→crop→scale 1080×960)
        """
        output_path = output_dir / "vertical_9min.mp4"
        total_dur   = sum(s["end"] - s["start"] for s in timeline)
        fade_out    = max(0.0, total_dur - 0.5)
        crop_y      = config.SCREEN_CROP_Y

        filtergraph = (
            f"[0:v]crop=1215:1080:352:{crop_y},scale=1080:960[top];"
            f"[1:v]scale=1920:1080:flags=lanczos,setsar=1,crop=1215:1080:352:0,scale=1080:960[bot];"
            f"[top][bot]vstack[vout];"
            f"[0:a]afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out:.3f}:d=0.5[aout]"
        )

        cmd = self._build_cmd(
            timeline, screen_file, webcam_file, output_path, filtergraph, total_dur
        )
        log.info(f"Рендер vertical: {output_path.name}")
        self._run_ffmpeg(cmd, total_dur, on_progress)
        return output_path

    def render_horizontal(
        self,
        timeline:    List[Dict],
        output_dir:  Path,
        screen_file: Path,
        webcam_file: Path,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Path:
        """
        Горизонтальный PiP 1920×1080:
          фон  — экран (full crop 1920×1080)
          PiP  — вебка квадрат 350×350, правый нижний угол
        """
        output_path = output_dir / "horizontal_9min.mp4"
        total_dur   = sum(s["end"] - s["start"] for s in timeline)
        fade_out    = max(0.0, total_dur - 0.5)
        crop_y      = config.SCREEN_CROP_Y
        pw          = config.PIP_WIDTH
        ph          = config.PIP_HEIGHT
        mr          = config.PIP_MARGIN_RIGHT
        mb          = config.PIP_MARGIN_BOTTOM

        filtergraph = (
            f"[0:v]crop=1920:1080:0:{crop_y}[screen];"
            f"[1:v]crop=1080:1080:420:0,scale={pw}:{ph}[pip];"
            f"[screen][pip]overlay=W-w-{mr}:H-h-{mb}[vout];"
            f"[0:a]afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out:.3f}:d=0.5[aout]"
        )

        cmd = self._build_cmd(
            timeline, screen_file, webcam_file, output_path, filtergraph, total_dur
        )
        log.info(f"Рендер horizontal: {output_path.name}")
        self._run_ffmpeg(cmd, total_dur, on_progress)
        return output_path

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _build_cmd(
        self,
        timeline:    List[Dict],
        screen_file: Path,
        webcam_file: Path,
        output_path: Path,
        filtergraph: str,
        total_dur:   float,
    ) -> List[str]:
        """Строит команду FFmpeg с двумя concat-списками."""
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem

        screen_list = self._write_concat_list(
            timeline, screen_file, config.TEMP_DIR / f"{stem}_screen.txt"
        )
        webcam_list = self._write_concat_list(
            timeline, webcam_file, config.TEMP_DIR / f"{stem}_webcam.txt"
        )

        return [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(screen_list),
            "-f", "concat", "-safe", "0", "-i", str(webcam_list),
            "-filter_complex", filtergraph,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", config.VIDEO_CODEC,
            "-crf", str(config.VIDEO_CRF),
            "-preset", config.VIDEO_PRESET,
            "-c:a", config.AUDIO_CODEC,
            "-b:a", config.AUDIO_BITRATE,
            "-threads", str(config.FFMPEG_THREADS),
            str(output_path),
        ]

    def _write_concat_list(
        self, timeline: List[Dict], video_file: Path, list_path: Path
    ) -> Path:
        escaped = str(video_file.resolve()).replace("'", "'\\''")
        with open(list_path, "w", encoding="utf-8") as f:
            for seg in timeline:
                f.write(f"file '{escaped}'\n")
                f.write(f"inpoint {seg['start']:.3f}\n")
                f.write(f"outpoint {seg['end']:.3f}\n")
        return list_path

    def _run_ffmpeg(
        self,
        cmd:         List[str],
        total_dur:   float,
        on_progress: Optional[Callable[[int], None]],
    ) -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        last_pct = -1
        for line in proc.stderr:
            m = _TIME_RE.search(line)
            if m and total_dur > 0 and on_progress:
                t   = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                pct = min(99, int(t / total_dur * 100))
                if pct != last_pct:
                    on_progress(pct)
                    last_pct = pct

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg завершился с ошибкой (код {proc.returncode})")

        if on_progress:
            on_progress(100)

        # Удаляем временные concat-списки
        for arg in cmd:
            if arg.endswith("_screen.txt") or arg.endswith("_webcam.txt"):
                try:
                    Path(arg).unlink(missing_ok=True)
                except Exception:
                    pass
