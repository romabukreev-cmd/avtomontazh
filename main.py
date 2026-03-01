"""
main.py — точка входа системы Автомонтаж.

Пайплайн:
  1. Concat файлов сессии
  2. Whisper транскрипция → сегменты с пословными timestamps
  3. LLM (Claude Sonnet) → keep/remove для каждого сегмента
  4. Timeline → нарезка пауз > 0.6с с буфером 0.2с
  5. FFmpeg рендер → вертикальное видео 1080×1920
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

def process_session(session: Session, progress: Callable, transcriber: Transcriber) -> None:
    """
    Полный цикл обработки одной сессии.
    Принимает Session с набором файлов — создаёт три готовых видео.

    transcriber — переиспользуется между сессиями (модель загружается один раз в main).
    progress    — синхронная функция для отправки сообщений в Telegram.
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

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        "⏳ Транскрибирую аудио через Whisper...\n"
        "<i>(это может занять несколько минут)</i>"
    )
    segments = transcriber.transcribe(screen_file)
    total_speech = sum(s["end"] - s["start"] for s in segments)
    log.info(f"Транскрипция: {len(segments)} сегментов, {format_duration(total_speech)} речи")


    # ── Шаг 2: LLM-анализ ────────────────────────────────────────────────────

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(segments)} сегментов ({format_duration(total_speech)})\n"
        "⏳ AI анализирует контент..."
    )
    analyzer = LLMAnalyzer()
    scored_segments = analyzer.analyze(segments)
    kept = [s for s in scored_segments if s["keep"]]
    kept_duration = sum(s["end"] - s["start"] for s in kept)
    log.info(f"LLM-анализ: оставлено {len(kept)}/{len(scored_segments)} ({format_duration(kept_duration)})")


    # ── Шаг 3: Таймлайн ──────────────────────────────────────────────────────

    progress(
        f"▶ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(segments)} сег. ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(scored_segments)} сег. ({format_duration(kept_duration)})\n"
        "⏳ Строю таймлайн..."
    )

    builder = TimelineBuilder()
    timeline_vertical = builder.build(kept)
    dur_vert = builder.total_duration(timeline_vertical)
    log.info(f"Таймлайн: {format_duration(dur_vert)}")


    # ── Шаг 4: Рендер ────────────────────────────────────────────────────────

    output_dir = config.OUTPUT_DIR / session.name
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = VideoRenderer(screen_file, webcam_file, session.name)

    def render_progress(pct: float) -> None:
        bar = "▓" * int(pct / 10) + "░" * (10 - int(pct / 10))
        progress(
            f"▶ <b>{session.name}</b>\n\n"
            "✅ Файлы готовы\n"
            f"✅ Транскрипция: {len(segments)} сег. ({format_duration(total_speech)})\n"
            f"✅ AI: {len(kept)}/{len(scored_segments)} сег. ({format_duration(kept_duration)})\n"
            f"✅ Таймлайн: {format_duration(dur_vert)}\n"
            f"⏳ Рендер: [{bar}] {pct:.0f}%"
        )

    renderer.render_vertical(
        timeline_vertical, output_dir,
        output_filename="vertical_9min.mp4",
        progress_callback=render_progress,
    )

    progress(
        f"✅ <b>{session.name}</b>\n\n"
        "✅ Файлы готовы\n"
        f"✅ Транскрипция: {len(segments)} сег. ({format_duration(total_speech)})\n"
        f"✅ AI: {len(kept)}/{len(scored_segments)} сег. ({format_duration(kept_duration)})\n"
        f"✅ Таймлайн: {format_duration(dur_vert)}\n"
        "✅ Рендер завершён"
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

    # Загружаем модель Whisper один раз — переиспользуется для всех сессий
    transcriber = Transcriber()

    def _pipeline(session: Session, progress: Callable) -> None:
        process_session(session, progress, transcriber=transcriber)

    bot = AutomontazhBot(pipeline_fn=_pipeline)
    bot.run()  # run_polling() управляет event loop сам


if __name__ == "__main__":
    main()
