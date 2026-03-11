"""
main.py — точка входа системы Автомонтаж.

Пайплайн:
  1. Concat файлов сессии (если несколько частей)
  2. Whisper транскрипция → слова с пословными таймстемпами
  3. build_blocks() → речевые блоки (паузы >= 0.6с = границы блоков)
  4. LLMAnalyzer.analyze() → маркирует повторы как keep=False
  5. FFmpeg рендер → вертикальное видео 1080×1920
"""

import logging
import sys
from typing import Callable

import config
from modules.bot               import AutomontazhBot
from modules.session_manager   import Session, SessionManager, StandardSession
from modules.transcriber       import Transcriber
from modules.llm_analyzer      import LLMAnalyzer
from modules.timeline          import build_blocks, total_duration
from modules.renderer          import VideoRenderer
from modules.standard_pipeline import process_standard_session
from modules.utils             import setup_logging, ensure_dirs, format_duration


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

    # ── Шаг 2: Речевые блоки ────────────────────────────────────────────────
    # Разбиваем слова на блоки по паузам >= PAUSE_CUT_SEC.
    # Блок = непрерывная речь. Паузы между блоками вырезаются автоматически.
    blocks = build_blocks(segments)
    total_speech = total_duration(blocks)
    log.info(f"Блоков: {len(blocks)}, суммарная речь: {format_duration(total_speech)}")

    # ── Шаг 3: LLM-анализ ───────────────────────────────────────────────────
    # Отправляем все блоки в Claude одним запросом.
    # Claude видит полный текст и находит смысловые повторы.
    # Возвращает {"delete": [список индексов]} — блоки для удаления.
    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(blocks)} блоков ({format_duration(total_speech)})\n"
        "⏳ AI анализирует повторы..."
    )
    analyzer = LLMAnalyzer()
    scored_blocks = analyzer.analyze(blocks, segments)
    kept = [b for b in scored_blocks if b["keep"]]
    kept_duration = total_duration(kept)
    log.info(f"После LLM: {len(kept)}/{len(scored_blocks)} блоков ({format_duration(kept_duration)})")

    # ── Шаг 4: Рендер ───────────────────────────────────────────────────────
    # kept — список блоков с полями start/end.
    # Это и есть финальный таймлайн: порядок соответствует оригиналу,
    # паузы между блоками вырезаны, повторы удалены.
    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = VideoRenderer(screen_file, webcam_file, session.name)

    def render_progress_fmt(label: str) -> Callable[[float], None]:
        def _cb(pct: float) -> None:
            bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
            progress(
                f"▶ <b>{session.name}</b>\n\n"
                "✅ Файлы готовы\n"
                f"✅ Транскрипция: {len(blocks)} блоков ({format_duration(total_speech)})\n"
                f"✅ AI: {len(kept)}/{len(scored_blocks)} блоков ({format_duration(kept_duration)})\n"
                f"⏳ Рендер {label}: [{bar}] {pct:.0f}%"
            )
        return _cb

    renderer.render_vertical(
        kept, output_dir,
        output_filename="vertical_9min.mp4",
        progress_callback=render_progress_fmt("9:16"),
    )
    renderer.render_horizontal(
        kept, output_dir,
        output_filename="horizontal_9min.mp4",
        progress_callback=render_progress_fmt("16:9"),
    )

    progress(
        f"✅ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(blocks)} блоков ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(scored_blocks)} блоков ({format_duration(kept_duration)})\n"
        "✅ Рендер завершён (9:16 + 16:9)"
    )
    log.info(f"✅ Готово: {session.name}")

    # ── Шаг 5: Уборка временных файлов ──────────────────────────────────────
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

    def _auto_pipeline(session: Session, progress: Callable) -> None:
        process_session(session, progress, transcriber=transcriber)

    def _standard_pipeline(session: StandardSession, progress: Callable) -> None:
        process_standard_session(session, progress, transcriber=transcriber)

    bot = AutomontazhBot(
        auto_pipeline_fn=_auto_pipeline,
        standard_pipeline_fn=_standard_pipeline,
    )
    bot.run()


if __name__ == "__main__":
    main()
