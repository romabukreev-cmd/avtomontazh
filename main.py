"""
main.py — точка входа системы Автомонтаж.

Пайплайн:
  1. concat_files()    — склеить части сессии
  2. transcribe()      — Whisper → sentence-level сегменты с word timestamps
  3. analyze()         — LLM чанками → kept_segments (удалены смысловые повторы)
  4. _build_timeline()  — удалить паузы внутри kept-сегментов → timeline
  5. render_vertical()  — FFmpeg → vertical_9min.mp4
  6. render_horizontal() — FFmpeg → horizontal_9min.mp4
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List

import config
from modules.bot import AutomontazhBot
from modules.llm_analyzer import LLMAnalyzer
from modules.renderer import VideoRenderer
from modules.session_manager import Session, SessionManager
from modules.transcriber import Transcriber
from modules.utils import ensure_dirs, format_duration, setup_logging

log = logging.getLogger("main")


# ── Глобальные объекты (создаются один раз при запуске) ───────────────────────

session_manager = SessionManager()
transcriber     = Transcriber()
analyzer        = LLMAnalyzer()
renderer        = VideoRenderer()


# ── Разбивка по паузам ────────────────────────────────────────────────────────

def _build_timeline(segments: List[Dict]) -> List[Dict]:
    """
    Строит timeline из kept-сегментов:
    1. Режет паузы >= PAUSE_CUT_SEC между словами
    2. Добавляет маленький буфер вокруг каждого блока слов
    3. Сортирует по времени и мёрджит пересечения — гарантирует корректный порядок
    """
    intervals: List[List[float]] = []

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            intervals.append([seg["start"], seg["end"]])
            continue

        # Группируем слова по паузам
        g_start = words[0]["start"]
        g_end   = words[0]["end"]
        for w in words[1:]:
            if w["start"] - g_end >= config.PAUSE_CUT_SEC:
                intervals.append([g_start, g_end])
                g_start = w["start"]
            g_end = w["end"]
        intervals.append([g_start, g_end])

    if not intervals:
        return []

    # Маленький буфер вокруг каждого блока слов
    padded = [
        [
            max(0.0, s - config.TIMELINE_START_PAD_SEC),
            e + config.TIMELINE_END_PAD_SEC,
        ]
        for s, e in intervals
    ]

    # Сортировка + merge пересекающихся интервалов
    padded.sort(key=lambda x: x[0])
    merged = [padded[0][:]]
    for s, e in padded[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [{"start": round(s, 3), "end": round(e, 3)} for s, e in merged]


# ── Пайплайн ──────────────────────────────────────────────────────────────────

def _pipeline(session: Session, progress: Callable[[str], None]) -> None:
    name      = session.name
    completed = []

    def done(line: str) -> None:
        completed.append(line)
        progress(f"▶ <b>{name}</b>\n\n" + "\n".join(completed))

    def working(line: str) -> None:
        progress(f"▶ <b>{name}</b>\n\n" + "\n".join(completed + [line]))

    # 1. Транскрипция по исходным частям (с offset-ами) — без лишнего склея/резки
    working("⏳ Транскрипция (Whisper)...")
    segments   = transcriber.transcribe(session.screen_files)
    speech_dur = sum(s["end"] - s["start"] for s in segments)
    done(f"✅ Транскрипция: {len(segments)} сегментов ({format_duration(speech_dur)})")

    # 2. Конкатенация (для рендера — один screen.mp4 + один webcam.mp4)
    working("⏳ Склеиваю файлы...")
    screen_file, webcam_file = session_manager.concat_files(session)
    done("✅ Файлы готовы")

    # 3. LLM анализ
    def on_llm_progress(msg: str) -> None:
        working(f"⏳ AI анализ: {msg}")

    working("⏳ AI анализ повторов...")
    kept      = analyzer.analyze(segments, on_progress=on_llm_progress)
    kept_dur  = sum(s["end"] - s["start"] for s in kept)
    done(f"✅ AI: {len(kept)}/{len(segments)} сегментов ({format_duration(kept_dur)})")

    # 4. Удаление пауз
    timeline = _build_timeline(kept)
    tl_dur   = sum(s["end"] - s["start"] for s in timeline)
    done(f"✅ Паузы вырезаны: {format_duration(tl_dur)}")

    # 5. Рендер
    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)

    def bar(pct: int) -> str:
        filled = pct // 10
        return "[" + "▓" * filled + "░" * (10 - filled) + "]"

    working("⏳ Рендер 9:16...")
    renderer.render_vertical(
        timeline, output_dir, screen_file, webcam_file,
        on_progress=lambda pct: working(f"⏳ Рендер 9:16: {bar(pct)} {pct}%"),
    )
    done("✅ Рендер 9:16 готов")

    # Горизонтальный рендер временно отключён — нужна только вертикалка.
    # working("⏳ Рендер 16:9...")
    # renderer.render_horizontal(
    #     timeline, output_dir, screen_file, webcam_file,
    #     on_progress=lambda pct: working(f"⏳ Рендер 16:9: {bar(pct)} {pct}%"),
    # )
    # done("✅ Рендер 16:9 готов")

    # Очистка temp
    if config.CLEANUP_TEMP_ON_SUCCESS:
        for f in config.TEMP_DIR.glob(f"{session.name}_*"):
            try:
                f.unlink()
            except Exception:
                pass


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    ensure_dirs()

    log.info("Автомонтаж — запуск")
    log.info(f"Входные файлы:  {config.INPUT_DIR}")
    log.info(f"Выходные файлы: {config.OUTPUT_DIR}")
    log.info("═" * 55)

    bot = AutomontazhBot(pipeline_fn=_pipeline)
    bot.run()


if __name__ == "__main__":
    main()
