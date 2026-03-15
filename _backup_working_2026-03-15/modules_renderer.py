"""
renderer.py — рендер видео через FFmpeg.

Принимает timeline (список под-сегментов с start/end) и два видеофайла (экран, вебка).
Создаёт два файла: vertical_9min.mp4 (1080×1920) и horizontal_9min.mp4 (1920×1080).

Используется filter_complex с trim/atrim — побайтово точная нарезка без артефактов.
Concat demuxer с inpoint/outpoint НЕ используется: он даёт аудио-артефакты на срезах.
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

        fc = _build_filter_complex(timeline, mode="vertical",
                                   crop_y=crop_y, fade_out=fade_out,
                                   pip_w=0, pip_h=0, pip_mr=0, pip_mb=0)
        cmd = _build_cmd(screen_file, webcam_file, fc, output_path)

        log.info(f"Рендер vertical: {output_path.name} ({len(timeline)} сегментов)")
        _run_ffmpeg(cmd, total_dur, on_progress)
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

        fc = _build_filter_complex(timeline, mode="horizontal",
                                   crop_y=crop_y, fade_out=fade_out,
                                   pip_w=config.PIP_WIDTH, pip_h=config.PIP_HEIGHT,
                                   pip_mr=config.PIP_MARGIN_RIGHT,
                                   pip_mb=config.PIP_MARGIN_BOTTOM)
        cmd = _build_cmd(screen_file, webcam_file, fc, output_path)

        log.info(f"Рендер horizontal: {output_path.name} ({len(timeline)} сегментов)")
        _run_ffmpeg(cmd, total_dur, on_progress)
        return output_path


# ── Построение filter_complex ─────────────────────────────────────────────────

def _build_filter_complex(
    timeline: List[Dict],
    mode:     str,       # "vertical" или "horizontal"
    crop_y:   int,
    fade_out: float,
    pip_w:    int,
    pip_h:    int,
    pip_mr:   int,
    pip_mb:   int,
) -> str:
    """
    Строит filter_complex для N сегментов.

    Для каждого сегмента i:
        [0:v]trim=start=S:end=E,setpts=PTS-STARTPTS[sv{i}]
        [1:v]trim=start=S:end=E,setpts=PTS-STARTPTS[wv{i}]
        [0:a]atrim=start=S:end=E,asetpts=PTS-STARTPTS[sa{i}]

    Затем concat всех видео- и аудио-потоков.
    Затем компоновка (vstack или overlay) и fade аудио.
    """
    parts       = []
    sv_labels   = []   # screen video per segment
    wv_labels   = []   # webcam video per segment
    sa_labels   = []   # screen audio per segment
    n           = len(timeline)

    for i, seg in enumerate(timeline):
        s = f"{seg['start']:.3f}"
        e = f"{seg['end']:.3f}"

        parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[sv{i}]")
        parts.append(f"[1:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[wv{i}]")
        parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[sa{i}]")

        sv_labels.append(f"[sv{i}]")
        wv_labels.append(f"[wv{i}]")
        sa_labels.append(f"[sa{i}]")

    # Concat всех сегментов
    parts.append(f"{''.join(sv_labels)}concat=n={n}:v=1:a=0[screen_concat]")
    parts.append(f"{''.join(wv_labels)}concat=n={n}:v=1:a=0[webcam_concat]")
    parts.append(f"{''.join(sa_labels)}concat=n={n}:v=0:a=1[audio_concat]")

    # Компоновка
    if mode == "vertical":
        parts.append(f"[screen_concat]crop=1215:1080:352:{crop_y},scale=1080:960[top]")
        parts.append(f"[webcam_concat]scale=1920:1080:flags=lanczos,setsar=1,crop=1215:1080:352:0,scale=1080:960[bot]")
        parts.append("[top][bot]vstack[vout]")
    else:  # horizontal
        parts.append(f"[screen_concat]crop=1920:1080:0:{crop_y}[screen]")
        parts.append(f"[webcam_concat]crop=1080:1080:420:0,scale={pip_w}:{pip_h}[pip]")
        parts.append(f"[screen][pip]overlay=W-w-{pip_mr}:H-h-{pip_mb}[vout]")

    # Аудио fade
    parts.append(
        f"[audio_concat]afade=t=in:st=0:d=0.5,"
        f"afade=t=out:st={fade_out:.3f}:d=0.5[aout]"
    )

    return ";".join(parts)


def _build_cmd(
    screen_file: Path,
    webcam_file: Path,
    filter_complex: str,
    output_path: Path,
) -> List[str]:
    return [
        "ffmpeg", "-y",
        "-i", str(screen_file),
        "-i", str(webcam_file),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", config.VIDEO_CODEC,
        "-crf", str(config.VIDEO_CRF),
        "-preset", config.VIDEO_PRESET,
        "-c:a", config.AUDIO_CODEC,
        "-b:a", config.AUDIO_BITRATE,
        "-threads", str(config.FFMPEG_THREADS),
        str(output_path),
    ]


def _run_ffmpeg(
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
