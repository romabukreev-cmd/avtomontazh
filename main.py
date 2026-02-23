"""
main.py — точка входа системы Автомонтаж.

Запуск:
    python main.py

Система запускает Telegram-бота который:
  - Принимает команды от пользователя (/sync, /sessions, /status)
  - По команде запускает полный пайплайн обработки видео
  - Присылает прогресс и уведомление о завершении
"""

import logging
import sys
from typing import Callable

import config
from modules.bot             import AutomontazhBot
from modules.session_manager import Session, SessionManager
from modules.transcriber     import Transcriber
from modules.llm_analyzer    import LLMAnalyzer
from modules.timeline        import TimelineBuilder
from modules.renderer        import VideoRenderer
from modules.utils           import setup_logging, ensure_dirs, format_duration


# ══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНЫЙ ПАЙПЛАЙН
#  Вызывается ботом для каждой выбранной пользователем сессии
# ══════════════════════════════════════════════════════════════════════════════

def process_session(session: Session, progress: Callable) -> None:
    """
    Полный цикл обработки одной сессии.
    Принимает Session с набором файлов — создаёт три готовых видео.

    progress — синхронная функция для отправки сообщений в Telegram.
    Thread-safe обёртка создаётся в bot.py перед запуском потока.
    """
    log = logging.getLogger(f"session.{session.name}")
    log.info(f"▶  Начало обработки: {session.name} ({session.file_count} файлов)")

    # ── Шаг 0: Конкатенация файлов (если несколько) ───────────────────────────
    # Если OBS создал несколько файлов (паузы) — склеиваем их в один поток.
    # session_manager делает это через FFmpeg concat без перекодирования.

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        f"⏳ Подготовка файлов ({session.file_count} частей)..."
    )
    session_manager = SessionManager()
    screen_file, webcam_file = session_manager.concat_files(session)
    log.info(f"Файлы готовы: {screen_file.name}, {webcam_file.name}")


    # ── Шаг 1: Транскрипция ───────────────────────────────────────────────────
    # Whisper транскрибирует аудио записи экрана.
    # Возвращает список сегментов речи с таймстемпами (паузы > 1.5с удалены).

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        "⏳ Транскрибирую аудио через Whisper...\n"
        "<i>(это может занять несколько минут)</i>"
    )
    transcriber = Transcriber()
    speech_segments = transcriber.transcribe_and_cut_pauses(screen_file)
    total_speech = sum(s["end"] - s["start"] for s in speech_segments)
    log.info(f"Транскрипция: {len(speech_segments)} сегментов, {format_duration(total_speech)} речи")


    # ── Шаг 2: LLM-анализ контента ───────────────────────────────────────────
    # Транскрипция отправляется в LLM (OpenRouter).
    # LLM оценивает каждый сегмент и помечает: оставить или вырезать.

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(speech_segments)} сегментов ({format_duration(total_speech)})\n"
        "⏳ AI анализирует контент..."
    )
    analyzer = LLMAnalyzer()
    scored_segments = analyzer.analyze(speech_segments)
    kept = [s for s in scored_segments if s["keep"]]
    kept_duration = sum(s["end"] - s["start"] for s in kept)
    log.info(f"LLM-анализ: оставлено {len(kept)}/{len(scored_segments)} ({format_duration(kept_duration)})")


    # ── Шаг 3: Построение таймлайна ───────────────────────────────────────────
    # TimelineBuilder собирает два списка отрезков:
    #   - длинный (все отобранные сегменты, до 10 минут)
    #   - хайлайты (второй LLM-проход по длинным сегментам, до 3 минут)

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(speech_segments)} сег. ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(scored_segments)} сег. ({format_duration(kept_duration)})\n"
        "⏳ Строю таймлайн..."
    )

    # Длительность исходного видео (для корректного padding в timeline builder)
    import subprocess as _sp
    _probe = _sp.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(screen_file)],
        capture_output=True, text=True,
    )
    try:
        video_duration = float(_probe.stdout.strip())
    except ValueError:
        video_duration = None

    builder = TimelineBuilder()
    timeline_long = builder.build_long(
        kept,
        max_sec=config.FORMAT_1["max_duration_sec"],
        video_duration=video_duration,
    )
    dur_long = builder.total_duration(timeline_long)
    log.info(f"Длинный таймлайн: {format_duration(dur_long)}")

    # Второй LLM-проход: выбираем хайлайты из уже отобранных сегментов длинного таймлайна
    # Для этого берём scored_segments чьи временные отрезки вошли в timeline_long
    long_start_ends = {(round(s["start"], 1), round(s["end"], 1)) for s in timeline_long}
    long_segments = [
        s for s in kept
        if any(
            s["start"] >= ts - 0.6 and s["end"] <= te + 0.6
            for ts, te in long_start_ends
        )
    ]
    if not long_segments:
        long_segments = kept  # fallback

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(speech_segments)} сег. ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(scored_segments)} сег. ({format_duration(kept_duration)})\n"
        f"✅ Длинный таймлайн: {format_duration(dur_long)}\n"
        "⏳ AI выбирает хайлайты (3 мин)..."
    )
    highlights_scored = analyzer.analyze_highlights(
        long_segments,
        max_sec=config.FORMAT_2["max_duration_sec"],
    )
    timeline_highlight = builder.build_highlights(
        highlights_scored,
        video_duration=video_duration,
    )
    dur_hl = builder.total_duration(timeline_highlight)
    log.info(f"Хайлайты: {format_duration(dur_hl)}")


    # ── Шаг 4: Рендер трёх форматов ───────────────────────────────────────────
    # VideoRenderer создаёт три видеофайла через FFmpeg.
    # progress_callback вызывается с процентом завершения рендера.

    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = VideoRenderer(screen_file, webcam_file, session.name)

    def make_render_progress(format_num: int, format_label: str):
        """Фабрика колбэка прогресса рендера для каждого формата."""
        def cb(pct: float) -> None:
            bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
            progress(
                f"▶ <b>{session.name}</b>\n\n"
                "✅ Файлы готовы\n"
                f"✅ Транскрипция: {len(speech_segments)} сег. ({format_duration(total_speech)})\n"
                f"✅ AI: {len(kept)}/{len(scored_segments)} сег. ({format_duration(kept_duration)})\n"
                f"✅ Таймлайн: {format_duration(dur_long)} / хайлайты {format_duration(dur_hl)}\n"
                f"⏳ Рендер {format_num}/3 ({format_label}): [{bar}] {pct:.0f}%"
            )
        return cb

    # Формат 1 — вертикальный 9:16, 10 минут
    renderer.render_vertical(
        timeline_long, output_dir,
        output_filename="vertical_10min.mp4",
        progress_callback=make_render_progress(1, "верт. 10мин"),
    )

    # Формат 2 — вертикальный 9:16, 3 минуты (хайлайты)
    renderer.render_vertical(
        timeline_highlight, output_dir,
        output_filename="vertical_3min.mp4",
        progress_callback=make_render_progress(2, "верт. 3мин"),
    )

    # Формат 3 — горизонтальный 16:9, 10 минут (PiP)
    renderer.render_horizontal(
        timeline_long, output_dir,
        output_filename="horizontal_10min.mp4",
        progress_callback=make_render_progress(3, "гориз. 10мин"),
    )

    log.info(f"✅ Обработка завершена: {session.name}")

    # ── Шаг 5: Уборка временных файлов ───────────────────────────────────────

    if config.CLEANUP_TEMP_ON_SUCCESS:
        renderer.cleanup_temp()
        # Удаляем склеенные temp-файлы если они создавались
        for temp_file in [screen_file, webcam_file]:
            if config.TEMP_DIR in temp_file.parents:
                temp_file.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging(config.LOGS_DIR, level=config.LOG_LEVEL)
    log = logging.getLogger("main")

    ensure_dirs([config.INPUT_DIR, config.OUTPUT_DIR, config.TEMP_DIR, config.LOGS_DIR])

    # Проверка обязательных настроек
    errors = []
    if not config.OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY не задан в .env")
    if not config.TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN не задан в .env")
    if not config.TELEGRAM_ALLOWED_CHAT_ID:
        errors.append("TELEGRAM_CHAT_ID не задан в .env")

    if errors:
        for e in errors:
            log.error(f"❌ {e}")
        sys.exit(1)

    log.info("═" * 55)
    log.info("  Автомонтаж — запуск")
    log.info(f"  Входные файлы: {config.INPUT_DIR}")
    log.info(f"  Выходные файлы: {config.OUTPUT_DIR}")
    log.info("═" * 55)

    bot = AutomontazhBot(pipeline_fn=process_session)
    bot.run()  # run_polling() управляет event loop сам


if __name__ == "__main__":
    main()
