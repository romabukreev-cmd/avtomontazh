"""
standard_pipeline.py — пайплайн для сценария "Стандартный".

Назначение: из папки с дублями Reels вырезать последний удачный дубль
каждой части сценария и склеить в черновой ролик 1080×1920.

Входные данные:
  - StandardSession: папка с видеофайлами (один поток, уже 9:16)
  - scenario.txt в папке (опционально — если нет, LLM работает без сценария)

Выход:
  - output/{session_name}/{session_name}.mp4 — черновой монтаж
"""

import logging
import subprocess
from pathlib import Path
from typing import Callable, List, Dict

import config
from modules.llm_analyzer import LLMAnalyzer
from modules.session_manager import SessionManager, StandardSession
from modules.timeline import build_blocks, total_duration
from modules.transcriber import Transcriber
from modules.utils import format_duration

log = logging.getLogger(__name__)


def process_standard_session(
    session: StandardSession,
    progress: Callable[[str], None],
    transcriber: Transcriber,
) -> None:
    """
    Полный цикл обработки одной стандартной сессии.

    Args:
        session:     описание сессии (видеофайлы + сценарий)
        progress:    функция отправки статуса в Telegram
        transcriber: переиспользуемый объект транскрибатора
    """
    log = logging.getLogger(f"standard.{session.name}")
    log.info(f"▶  Стандартный: {session.name} ({session.file_count} файлов)")

    # ── Шаг 0: Склеиваем файлы ──────────────────────────────────────────────
    progress(
        f"▶ <b>{session.name}</b> [Стандартный]\n\n"
        f"⏳ Подготовка файлов ({session.file_count} частей)..."
    )
    session_manager = SessionManager()
    video_file = session_manager.concat_standard_files(session)
    log.info(f"Файл готов: {video_file.name}")

    # ── Шаг 1: Читаем сценарий ──────────────────────────────────────────────
    scenario_text = ""
    if session.has_scenario:
        scenario_text = session.scenario_file.read_text(encoding="utf-8").strip()
        log.info(f"Сценарий загружен ({len(scenario_text)} символов)")
    else:
        log.warning("scenario.txt не найден — LLM будет работать без сценария")

    # ── Шаг 2: Транскрипция ──────────────────────────────────────────────────
    progress(
        f"▶ <b>{session.name}</b> [Стандартный]\n\n"
        "✅ Файлы готовы\n"
        "⏳ Транскрибирую через Whisper...\n"
        "<i>(несколько минут)</i>"
    )
    segments = transcriber.transcribe(video_file)

    # ── Шаг 3: Речевые блоки ────────────────────────────────────────────────
    blocks = build_blocks(segments)
    total_speech = total_duration(blocks)
    log.info(f"Блоков: {len(blocks)}, суммарная речь: {format_duration(total_speech)}")

    # ── Шаг 4: LLM-анализ ───────────────────────────────────────────────────
    progress(
        f"▶ <b>{session.name}</b> [Стандартный]\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(blocks)} блоков ({format_duration(total_speech)})\n"
        "⏳ AI анализирует дубли..."
    )
    analyzer = LLMAnalyzer()
    if scenario_text:
        scored_blocks = analyzer.analyze_with_scenario(blocks, scenario_text)
    else:
        scored_blocks = analyzer.analyze(blocks)

    kept = [b for b in scored_blocks if b["keep"]]
    kept_duration = total_duration(kept)
    log.info(f"После LLM: {len(kept)}/{len(scored_blocks)} блоков ({format_duration(kept_duration)})")

    # ── Шаг 5: Рендер ───────────────────────────────────────────────────────
    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{session.name}.mp4"

    def render_progress(pct: float) -> None:
        bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
        progress(
            f"▶ <b>{session.name}</b> [Стандартный]\n\n"
            "✅ Файлы готовы\n"
            f"✅ Транскрипция: {len(blocks)} блоков ({format_duration(total_speech)})\n"
            f"✅ AI: {len(kept)}/{len(scored_blocks)} блоков ({format_duration(kept_duration)})\n"
            f"⏳ Рендер: [{bar}] {pct:.0f}%"
        )

    _render_standard(video_file, kept, output_file, render_progress)

    progress(
        f"✅ <b>{session.name}</b> [Стандартный]\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(blocks)} блоков ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(scored_blocks)} блоков ({format_duration(kept_duration)})\n"
        "✅ Рендер завершён"
    )
    log.info(f"✅ Готово: {session.name}")

    # ── Шаг 6: Уборка ───────────────────────────────────────────────────────
    if config.CLEANUP_TEMP_ON_SUCCESS:
        if config.TEMP_DIR in video_file.parents:
            video_file.unlink(missing_ok=True)


# ── Рендер одного видео 1080×1920 ──────────────────────────────────────────────

def _render_standard(
    source_file: Path,
    timeline: List[Dict],
    output_file: Path,
    progress_callback: Callable[[float], None],
) -> None:
    """
    Рендерит черновой монтаж из одного источника.

    - Нарезает по kept-блокам с буфером STANDARD_CLIP_GAP_SEC
    - Масштабирует/кадрирует до 1080×1920
    - Между клипами — тишина (пустых кадров нет, просто прямой concat)
    """
    buf = config.STANDARD_CLIP_GAP_SEC
    buffered = []
    for b in timeline:
        buffered.append({
            "start": max(0.0, b["start"] - buf),
            "end":   b["end"] + buf,
        })

    # Пишем concat-список
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_file.stem
    list_path = config.TEMP_DIR / f"{stem}_list.txt"

    with open(list_path, "w", encoding="utf-8") as f:
        for seg in buffered:
            escaped = str(source_file.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write(f"inpoint {seg['start']:.3f}\n")
            f.write(f"outpoint {seg['end']:.3f}\n")

    total = sum(s["end"] - s["start"] for s in buffered)
    fade_out_start = max(0.0, total - 0.5)

    # scale=1080:1920:force_original_aspect_ratio=decrease — вписываем в кадр,
    # pad — добавляем чёрные полосы если нужно
    filtergraph = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2[vout];"
        f"[0:a]afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_start:.3f}:d=0.5[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
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

    log.info(f"Рендер: {output_file.name} ({len(timeline)} блоков, {total:.1f}с)")
    import re
    process = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
    )

    last_pct = -1
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
        match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
        if match and total > 0:
            h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
            current = h * 3600 + m * 60 + s
            pct = min(current / total * 100, 99)
            if pct - last_pct >= 5:
                last_pct = pct
                try:
                    progress_callback(pct)
                except Exception:
                    pass

    process.wait()
    list_path.unlink(missing_ok=True)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg завершился с ошибкой:\n{''.join(stderr_lines[-30:])}")

    try:
        progress_callback(100)
    except Exception:
        pass
