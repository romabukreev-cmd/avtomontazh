"""
renderer.py — рендер видео через FFmpeg.

Два выходных формата:
  vertical_9min.mp4   — 1080×1920, экран (70%) сверху + вебка (30%) снизу
  horizontal_9min.mp4 — 1920×1080, три ветки:
    • экран + вебка  → экран на весь кадр, вебка PiP в углу
    • только экран   → экран на весь кадр
    • только вебка   → вебка на весь кадр

Каждый отрезок timeline рендерится отдельным ffmpeg-процессом (O(1) памяти),
чанки склеиваются через concat demuxer -c copy без перекодирования.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config

log = logging.getLogger(__name__)

# ── Параметры вывода ──────────────────────────────────────────────────────────
_V_W          = 1080
_V_H          = 1920
_V_SCREEN_H   = 1350   # 70% высоты — экран
_V_WEBCAM_H   = 570    # 30% высоты — вебка

_H_W          = 1920
_H_H          = 1080
_PIP_W        = 320    # размер вебки-PiP в горизонтальном режиме
_PIP_H        = 180
_PIP_MARGIN   = 20

_TARGET_FPS         = 30
_TARGET_AUDIO_RATE  = 48000
_TARGET_PIX_FMT     = "yuv420p"
_KEYFRAME_INTERVAL  = 60
_PRE_SEEK_BUFFER    = 2.0


class VideoRenderer:

    def render_vertical(
        self,
        timeline:    List[Dict],
        output_dir:  Path,
        screen_file: Optional[Path],
        webcam_file: Optional[Path],
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Path:
        """Вертикальный 9:16 — экран сверху, вебка снизу. Оба файла обязательны."""
        if not screen_file or not webcam_file:
            raise RuntimeError("Вертикальный формат требует оба файла: экран и вебку.")
        output_path = output_dir / "vertical_9min.mp4"
        crop_y = config.SCREEN_CROP_Y
        self._render(
            timeline, output_path,
            inputs=[screen_file, webcam_file],
            filter_fn=lambda s, e: _vertical_filter(crop_y, s, e),
            on_progress=on_progress,
        )
        return output_path

    def render_horizontal(
        self,
        timeline:    List[Dict],
        output_dir:  Path,
        screen_file: Optional[Path] = None,
        webcam_file: Optional[Path] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Path:
        """Горизонтальный 16:9. Режим зависит от наличия файлов."""
        if not screen_file and not webcam_file:
            raise RuntimeError("Нет файлов для горизонтального рендера.")
        output_path = output_dir / "horizontal_9min.mp4"
        crop_y = config.SCREEN_CROP_Y

        if screen_file and webcam_file:
            self._render(
                timeline, output_path,
                inputs=[screen_file, webcam_file],
                filter_fn=lambda s, e: _horizontal_pip_filter(crop_y, s, e),
                on_progress=on_progress,
            )
        elif screen_file:
            self._render(
                timeline, output_path,
                inputs=[screen_file],
                filter_fn=lambda s, e: _screen_only_filter(crop_y, s, e),
                on_progress=on_progress,
            )
        else:
            self._render(
                timeline, output_path,
                inputs=[webcam_file],
                filter_fn=_webcam_only_filter,
                on_progress=on_progress,
            )
        return output_path

    # ── Основной рендер-цикл ──────────────────────────────────────────────────

    def _render(
        self,
        timeline:    List[Dict],
        output_path: Path,
        inputs:      List[Path],
        filter_fn:   Callable[[float, float], str],
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        if not timeline:
            raise RuntimeError("Пустой timeline — нечего рендерить.")

        chunks_dir = config.TEMP_DIR / f"chunks_{output_path.stem}"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Рендер {output_path.name}: {len(timeline)} отрезков, входов: {len(inputs)}")

        chunk_paths: List[Path] = []
        try:
            for i, seg in enumerate(timeline):
                chunk_path = chunks_dir / f"chunk_{i:04d}.mp4"
                self._render_chunk(seg, inputs, filter_fn, chunk_path)
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
                    f"Concat-demuxer ошибка (код {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )
            if on_progress:
                on_progress(100)
        finally:
            shutil.rmtree(chunks_dir, ignore_errors=True)

    def _render_chunk(
        self,
        seg:       Dict,
        inputs:    List[Path],
        filter_fn: Callable[[float, float], str],
        out_path:  Path,
    ) -> None:
        start = seg["start"]
        end   = seg["end"]
        pre_seek    = max(0.0, start - _PRE_SEEK_BUFFER)
        local_start = start - pre_seek
        local_end   = end   - pre_seek

        filter_complex = filter_fn(local_start, local_end)

        cmd = ["ffmpeg", "-y"]
        for inp in inputs:
            cmd += ["-ss", f"{pre_seek:.3f}", "-i", str(inp)]
        cmd += [
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
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Рендер чанка {out_path.name} упал "
                f"(seg {start:.2f}→{end:.2f}, код {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )


# ── FFmpeg filter_complex строки ──────────────────────────────────────────────

def _fill_crop(input_label: str, out_label: str, w: int, h: int) -> str:
    """scale-fill → crop-center к размеру w×h."""
    return (
        f"{input_label}scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h}:(iw-{w})/2:(ih-{h})/2{out_label}"
    )


def _vertical_filter(screen_crop_y: int, s: float, e: float) -> str:
    """[0]=экран, [1]=вебка → 1080×1920."""
    sw, sh = _V_W, _V_SCREEN_H
    ww, wh = _V_W, _V_WEBCAM_H
    crop = f"crop=iw:ih-{screen_crop_y}:0:{screen_crop_y}," if screen_crop_y > 0 else ""
    screen_part = (
        f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,{crop}"
        + _fill_crop("", "[sv]", sw, sh)
    )
    webcam_part = (
        f"[1:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,"
        + _fill_crop("", "[wv]", ww, wh)
    )
    return (
        f"{screen_part};"
        f"{webcam_part};"
        f"[sv][wv]vstack=inputs=2[vout];"
        f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[aout]"
    )


def _horizontal_pip_filter(screen_crop_y: int, s: float, e: float) -> str:
    """[0]=экран, [1]=вебка → 1920×1080, вебка PiP в углу."""
    crop = f"crop=iw:ih-{screen_crop_y}:0:{screen_crop_y}," if screen_crop_y > 0 else ""
    x = _H_W - _PIP_W - _PIP_MARGIN
    y = _H_H - _PIP_H - _PIP_MARGIN
    screen_part = (
        f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,{crop}"
        + _fill_crop("", "[sf]", _H_W, _H_H)
    )
    webcam_part = (
        f"[1:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,"
        + _fill_crop("", "[wf]", _PIP_W, _PIP_H)
    )
    return (
        f"{screen_part};"
        f"{webcam_part};"
        f"[sf][wf]overlay=x={x}:y={y}[vout];"
        f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[aout]"
    )


def _screen_only_filter(screen_crop_y: int, s: float, e: float) -> str:
    """[0]=экран → 1920×1080."""
    crop = f"crop=iw:ih-{screen_crop_y}:0:{screen_crop_y}," if screen_crop_y > 0 else ""
    return (
        f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,{crop}"
        + _fill_crop("", "[vout]", _H_W, _H_H) + ";"
        f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[aout]"
    )


def _webcam_only_filter(s: float, e: float) -> str:
    """[0]=вебка → 1920×1080."""
    return (
        f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,"
        + _fill_crop("", "[vout]", _H_W, _H_H) + ";"
        f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[aout]"
    )
