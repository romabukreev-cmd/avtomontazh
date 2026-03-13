"""
main.py — точка входа системы Автомонтаж.

Пайплайн:
  1. concat_files()    — склеить части сессии
  2. transcribe()      — Whisper → sentence-level сегменты с word timestamps
  3. analyze()         — LLM чанками → kept_segments (удалены смысловые повторы)
  4. _split_on_pauses() — удалить тишину внутри kept-сегментов → timeline
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

def _split_on_pauses(segments: List[Dict]) -> List[Dict]:
    """
    Разбивает kept-сегменты на под-сегменты, удаляя паузы >= PAUSE_CUT_SEC.

    Для каждого сегмента берём word timestamps, ищем межсловные паузы.
    Если пауза >= порога — это граница нового под-сегмента.

    Границы под-сегмента:
        start = first_word.start - SEG_BUF_START
        end   = min(last_word.end, last_word.start + MAX_WORD_DUR) + SEG_BUF_END

    Финальный клиппинг: следующий под-сегмент не должен перекрывать предыдущий.
    Если clip попадает внутрь последнего слова — откатываем до начала этого слова.
    """
    result = []

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # Нет word timestamps — оставляем сегмент как есть
            result.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg.get("text", ""),
                "words": [],
            })
            continue

        # Группируем слова по паузам
        groups: List[List[Dict]] = []
        current_group = [words[0]]
        for w in words[1:]:
            gap = w["start"] - current_group[-1]["end"]
            if gap >= config.PAUSE_CUT_SEC:
                groups.append(current_group)
                current_group = [w]
            else:
                current_group.append(w)
        groups.append(current_group)

        # Создаём под-сегмент для каждой группы
        for group in groups:
            first = group[0]
            last  = group[-1]
            capped_end = min(last["end"], last["start"] + config.MAX_WORD_DUR)
            sub_start  = max(0.0, first["start"] - config.SEG_BUF_START)
            sub_end    = capped_end + config.SEG_BUF_END
            text       = " ".join(w["word"].strip() for w in group)
            result.append({
                "start": round(sub_start, 3),
                "end":   round(sub_end,   3),
                "text":  text,
                "words": group,
            })

    # Клиппинг: под-сегмент N не должен перекрывать под-сегмент N+1
    for i in range(len(result) - 1):
        if result[i]["end"] > result[i + 1]["start"]:
            clip    = result[i + 1]["start"]
            words_i = result[i].get("words", [])

            if words_i:
                last = words_i[-1]
                # Если clip попадает внутрь последнего слова — откатываем до его начала
                if last["start"] <= clip <= last["end"]:
                    clip = max(0.0, last["start"] - 0.01)
                    result[i]["words"] = words_i[:-1]

            result[i]["end"] = max(result[i]["start"], round(clip, 3))

    # Убираем под-сегменты нулевой или отрицательной длины
    result = [s for s in result if s["end"] > s["start"]]

    return result


# ── Пайплайн ──────────────────────────────────────────────────────────────────

def _pipeline(session: Session, progress: Callable[[str], None]) -> None:
    name      = session.name
    completed = []

    def done(line: str) -> None:
        completed.append(line)
        progress(f"▶ <b>{name}</b>\n\n" + "\n".join(completed))

    def working(line: str) -> None:
        progress(f"▶ <b>{name}</b>\n\n" + "\n".join(completed + [line]))

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

    # 4. Удаление пауз
    timeline = _split_on_pauses(kept)
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
