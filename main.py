"""
main.py — точка входа системы Автомонтаж.

Пайплайн:
  1. concat_files()    — склеить части сессии
  2. transcribe()      — Whisper → sentence-level сегменты с word timestamps
  3. analyze()         — LLM чанками → kept_segments (удалены смысловые повторы)
  4. _detect_silence() — silencedetect на оригинале → карта тишины
  5. _build_timeline() — удалить паузы + вычесть тихие зоны → timeline
  6. render_vertical() — FFmpeg → vertical_9min.mp4
"""

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config
from modules.bot import AutomontazhBot
from modules.llm_analyzer import LLMAnalyzer
from modules.renderer import VideoRenderer
from modules.session_manager import Session, SessionManager
from modules.transcriber import Transcriber
from modules.utils import ensure_dirs, format_duration, setup_logging

log = logging.getLogger("main")

# Параметры silencedetect
_SILENCE_NOISE_DB = -30       # порог тишины (dBFS)
_SILENCE_MIN_DUR  = 1.0       # минимальная длительность тишины (сек)
_MIN_SPEECH_CHUNK = 0.3       # минимальный остаток после вычитания (сек)


# ── Глобальные объекты (создаются один раз при запуске) ───────────────────────

session_manager = SessionManager()
transcriber     = Transcriber()
analyzer        = LLMAnalyzer()
renderer        = VideoRenderer()


# ── Определение тишины ────────────────────────────────────────────────────────

def _detect_silence(video_path: Path) -> List[List[float]]:
    """
    Запускает ffmpeg silencedetect на аудиодорожке видео.
    Возвращает список [start, end] тихих зон.
    """
    cmd = [
        "ffmpeg", "-i", str(video_path), "-af",
        f"silencedetect=noise={_SILENCE_NOISE_DB}dB:d={_SILENCE_MIN_DUR}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    regions: List[List[float]] = []
    for line in result.stderr.split("\n"):
        m_start = re.search(r"silence_start: ([\d.]+)", line)
        m_end = re.search(r"silence_end: ([\d.]+)", line)
        if m_start:
            regions.append([float(m_start.group(1)), 0.0])
        elif m_end and regions and regions[-1][1] == 0.0:
            regions[-1][1] = float(m_end.group(1))

    # Убрать незакрытые (тишина до конца файла)
    regions = [r for r in regions if r[1] > 0.0]
    log.info(f"Silencedetect: {len(regions)} тихих зон "
             f"(noise={_SILENCE_NOISE_DB}dB, min_dur={_SILENCE_MIN_DUR}с)")
    return regions


def _subtract_silence(
    start: float,
    end: float,
    silence_regions: List[List[float]],
) -> List[List[float]]:
    """
    Вычитает тихие зоны из интервала [start, end].
    Возвращает список речевых кусков >= _MIN_SPEECH_CHUNK.
    """
    parts = [[start, end]]
    for s_start, s_end in silence_regions:
        if not parts:
            break
        # Быстрый skip: тишина полностью до или после всех частей
        if s_end <= parts[0][0]:
            continue
        if s_start >= parts[-1][1]:
            break
        new_parts: List[List[float]] = []
        for p_start, p_end in parts:
            if s_end <= p_start or s_start >= p_end:
                # Нет пересечения
                new_parts.append([p_start, p_end])
            else:
                # Часть до тишины
                if s_start > p_start:
                    new_parts.append([p_start, s_start])
                # Часть после тишины
                if s_end < p_end:
                    new_parts.append([s_end, p_end])
        parts = new_parts

    return [p for p in parts if p[1] - p[0] >= _MIN_SPEECH_CHUNK]


# ── Разбивка по паузам ────────────────────────────────────────────────────────

def _build_timeline(
    segments: List[Dict],
    silence_regions: Optional[List[List[float]]] = None,
) -> List[Dict]:
    """
    Строит timeline из kept-сегментов:
    1. Режет паузы >= PAUSE_CUT_SEC между словами
    2. Добавляет маленький буфер вокруг каждого блока слов
    3. Сортирует по времени и мёрджит пересечения
    4. Вычитает тихие зоны (silencedetect) — убирает галлюцинации Whisper
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

    # Сортировка + merge пересекающихся интервалов (без паддинга)
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Вычесть тихие зоны ДО паддинга — иначе паддинг срезается тишиной
    if silence_regions:
        before_count = len(merged)
        before_dur = sum(e - s for s, e in merged)
        filtered: List[List[float]] = []
        for s, e in merged:
            speech_parts = _subtract_silence(s, e, silence_regions)
            filtered.extend(speech_parts)
        if filtered:
            merged = filtered
            after_dur = sum(e - s for s, e in merged)
            log.info(f"Silence filter: {before_count}→{len(merged)} интервалов, "
                     f"{before_dur:.0f}с→{after_dur:.0f}с "
                     f"(вырезано {before_dur - after_dur:.0f}с тишины)")
        else:
            log.warning("Silence filter убрал ВСЕ интервалы — оставляем без фильтра")

    # Паддинг ПОСЛЕ вычитания тишины — буфер гарантированно сохраняется
    padded = [
        [
            max(0.0, s - config.TIMELINE_START_PAD_SEC),
            e + config.TIMELINE_END_PAD_SEC,
        ]
        for s, e in merged
    ]

    # Re-merge после паддинга (паддинг может создать пересечения)
    padded.sort(key=lambda x: x[0])
    final: List[List[float]] = [padded[0][:]]
    for s, e in padded[1:]:
        if s <= final[-1][1]:
            final[-1][1] = max(final[-1][1], e)
        else:
            final.append([s, e])

    result = [{"start": round(s, 3), "end": round(e, 3)} for s, e in final]

    # Диагностика
    if result:
        total_dur = sum(r["end"] - r["start"] for r in result)
        max_dur = max(r["end"] - r["start"] for r in result)
        log.info(f"Timeline: {len(result)} интервалов, "
                 f"total={round(total_dur, 1)}с, max={round(max_dur, 2)}с")

    return result


# ── Пайплайн ──────────────────────────────────────────────────────────────────

def _pipeline(
    session: Session,
    progress: Callable[[str], None],
    output_format: str = "vertical",
) -> None:
    name      = session.name
    fmt_label = "9:16" if output_format == "vertical" else "16:9"
    completed = []

    def done(line: str) -> None:
        completed.append(line)
        progress(f"▶ <b>{name}</b> [{fmt_label}]\n\n" + "\n".join(completed))

    def working(line: str) -> None:
        progress(f"▶ <b>{name}</b> [{fmt_label}]\n\n" + "\n".join(completed + [line]))

    # 1. Конкатенация
    working("⏳ Склеиваю файлы...")
    screen_file, webcam_file = session_manager.concat_files(session)
    done("✅ Файлы готовы")

    # 2. Транскрипция
    working("⏳ Транскрипция (Whisper)...")
    segments   = transcriber.transcribe(screen_file)
    speech_dur = sum(s["end"] - s["start"] for s in segments)
    done(f"✅ Транскрипция: {len(segments)} сегментов ({format_duration(speech_dur)})")

    # 3. LLM анализ
    def on_llm_progress(msg: str) -> None:
        working(f"⏳ AI анализ: {msg}")

    working("⏳ AI анализ повторов...")
    kept      = analyzer.analyze(segments, on_progress=on_llm_progress)
    kept_dur  = sum(s["end"] - s["start"] for s in kept)
    done(f"✅ AI: {len(kept)}/{len(segments)} сегментов ({format_duration(kept_dur)})")

    # 4. Определение тишины + удаление пауз
    working("⏳ Анализ тихих зон...")
    silence_regions = _detect_silence(screen_file)
    timeline = _build_timeline(kept, silence_regions=silence_regions)
    tl_dur   = sum(s["end"] - s["start"] for s in timeline)
    done(f"✅ Паузы вырезаны: {format_duration(tl_dur)}")

    # 5. Рендер
    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)

    def bar(pct: int) -> str:
        filled = pct // 10
        return "[" + "▓" * filled + "░" * (10 - filled) + "]"

    if output_format == "vertical":
        working("⏳ Рендер 9:16...")
        renderer.render_vertical(
            timeline, output_dir, screen_file, webcam_file,
            on_progress=lambda pct: working(f"⏳ Рендер 9:16: {bar(pct)} {pct}%"),
        )
        done("✅ Рендер 9:16 готов")
    else:
        working("⏳ Рендер 16:9...")
        renderer.render_horizontal(
            timeline, output_dir, screen_file, webcam_file,
            on_progress=lambda pct: working(f"⏳ Рендер 16:9: {bar(pct)} {pct}%"),
        )
        done("✅ Рендер 16:9 готов")

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
