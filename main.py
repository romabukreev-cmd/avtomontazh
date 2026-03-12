"""
main.py — точка входа системы Автомонтаж.

Пайплайн:
  1. Concat файлов сессии (если несколько частей)
  2. Whisper транскрипция → сегменты с пословными таймстемпами
  3. LLMAnalyzer.analyze() → убирает повторы, возвращает kept_segments
  4. FFmpeg рендер → вертикальное 1080×1920 + горизонтальное 1920×1080
"""

import logging
import sys
from typing import Callable

import config
from modules.bot             import AutomontazhBot
from modules.session_manager import Session, SessionManager
from modules.transcriber     import Transcriber
from modules.llm_analyzer    import LLMAnalyzer
from modules.renderer        import VideoRenderer
from modules.utils           import setup_logging, ensure_dirs, format_duration


# ══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНЫЙ ПАЙПЛАЙН
# ══════════════════════════════════════════════════════════════════════════════

def process_session(session: Session, progress: Callable, transcriber: Transcriber) -> None:
    """
    Полный цикл обработки одной сессии.
    transcriber — переиспользуется между сессиями (загружается один раз).
    progress    — функция для отправки сообщений в Telegram.
    """
    log = logging.getLogger(f"session.{session.name}")
    log.info(f"▶  Начало: {session.name} ({session.file_count} файлов)")

    # ── Шаг 0: Склеиваем файлы ──────────────────────────────────────────────
    progress(
        f"▶ <b>{session.name}</b>\n\n"
        f"⏳ Подготовка файлов ({session.file_count} частей)..."
    )
    session_manager = SessionManager()
    screen_file, webcam_file = session_manager.concat_files(session)
    log.info(f"Файлы готовы: {screen_file.name}, {webcam_file.name}")

    # ── Шаг 1: Транскрипция ──────────────────────────────────────────────────
    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        "⏳ Транскрибирую через Whisper...\n"
        "<i>(несколько минут)</i>"
    )
    segments = transcriber.transcribe(screen_file)
    total_speech = sum(s["end"] - s["start"] for s in segments)
    log.info(f"Сегментов: {len(segments)}, суммарная речь: {format_duration(total_speech)}")

    # ── Шаг 2: LLM-анализ ───────────────────────────────────────────────────
    # Claude получает сегменты чанками по 30 (~4 минуты).
    # Находит и удаляет смысловые повторы по всей длине видео.
    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(segments)} сегментов ({format_duration(total_speech)})\n"
        "⏳ AI анализирует повторы..."
    )
    analyzer = LLMAnalyzer()
    kept = analyzer.analyze(segments)
    kept_duration = sum(s["end"] - s["start"] for s in kept)
    log.info(f"После LLM: {len(kept)}/{len(segments)} сегментов ({format_duration(kept_duration)})")

    # ── Шаг 3: Рендер ───────────────────────────────────────────────────────
    # kept — список сегментов с полями start/end — это и есть финальный таймлайн.
    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = VideoRenderer(screen_file, webcam_file, session.name)

    def render_progress(label: str) -> Callable[[float], None]:
        def _cb(pct: float) -> None:
            bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
            progress(
                f"▶ <b>{session.name}</b>\n\n"
                "✅ Файлы готовы\n"
                f"✅ Транскрипция: {len(segments)} сегментов ({format_duration(total_speech)})\n"
                f"✅ AI: {len(kept)}/{len(segments)} сегментов ({format_duration(kept_duration)})\n"
                f"⏳ Рендер {label}: [{bar}] {pct:.0f}%"
            )
        return _cb

    renderer.render_vertical(
        kept, output_dir,
        output_filename="vertical_9min.mp4",
        progress_callback=render_progress("9:16"),
    )
    renderer.render_horizontal(
        kept, output_dir,
        output_filename="horizontal_9min.mp4",
        progress_callback=render_progress("16:9"),
    )

    progress(
        f"✅ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(segments)} сегментов ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(segments)} сегментов ({format_duration(kept_duration)})\n"
        "✅ Рендер завершён (9:16 + 16:9)"
    )
    log.info(f"✅ Готово: {session.name}")

    # ── Шаг 4: Уборка временных файлов ──────────────────────────────────────
    if config.CLEANUP_TEMP_ON_SUCCESS:
        renderer.cleanup_temp()
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
    log.info(f"  Входные файлы:  {config.INPUT_DIR}")
    log.info(f"  Выходные файлы: {config.OUTPUT_DIR}")
    log.info("═" * 55)

    transcriber = Transcriber()

    def _pipeline(session: Session, progress: Callable) -> None:
        process_session(session, progress, transcriber=transcriber)

    bot = AutomontazhBot(pipeline_fn=_pipeline)
    bot.run()


if __name__ == "__main__":
    main()
