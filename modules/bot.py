"""
bot.py — Telegram-бот для управления системой Автомонтаж.

Команды:
  /start    — приветствие и список команд
  /sync     — скачать новые файлы с Яндекс Диска
  /sessions — показать список сессий, готовых к обработке
  /status   — текущий статус (обрабатывается / свободно)

Безопасность:
  Бот отвечает ТОЛЬКО пользователю с TELEGRAM_ALLOWED_CHAT_ID из config.py.
  Все остальные сообщения игнорируются.

Зависимость: python-telegram-bot >= 20.0 (async API)
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

import config
from modules.session_manager import Session, SessionManager
from modules.utils import format_duration

log = logging.getLogger(__name__)


class AutomontazhBot:

    def __init__(self, pipeline_fn: Callable):
        """
        pipeline_fn — функция из main.py которая запускает полный пайплайн обработки.
        Сигнатура: pipeline_fn(session, progress_callback) → None
        """
        self.pipeline_fn     = pipeline_fn
        self.session_manager = SessionManager()
        self.is_processing   = False        # флаг: идёт ли сейчас обработка
        self.current_session: Optional[str] = None

    # ── Запуск бота ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Запускает бота в режиме polling. Блокирует до Ctrl+C."""
        app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .build()
        )

        # Регистрируем обработчики команд
        app.add_handler(CommandHandler("start",    self._cmd_start))
        app.add_handler(CommandHandler("sync",     self._cmd_sync))
        app.add_handler(CommandHandler("sessions", self._cmd_sessions))
        app.add_handler(CommandHandler("status",   self._cmd_status))

        # Обработчик нажатий на inline-кнопки (выбор сессии)
        app.add_handler(CallbackQueryHandler(self._on_session_selected, pattern="^process:"))

        log.info("Telegram-бот запущен. Жду команды...")
        await app.run_polling(allowed_updates=Update.ALL_TYPES)

    # ── Команды ───────────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        text = (
            "Привет! Я Автомонтаж — система автоматического монтажа видео.\n\n"
            "Команды:\n"
            "/sync — скачать новые файлы с Яндекс Диска\n"
            "/sessions — показать сессии для обработки\n"
            "/status — статус обработки\n\n"
            "Перед первым запуском выполни /sync чтобы скачать файлы."
        )
        await update.message.reply_text(text)

    async def _cmd_sync(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        msg = await update.message.reply_text("⏳ Синхронизирую с Яндекс Диском...")

        try:
            # Считаем сессии до синхронизации
            before = {s.name for s in self.session_manager.scan_sessions()}

            # Запускаем rclone sync
            await asyncio.to_thread(self._run_rclone_sync)

            # Считаем сессии после
            after_sessions = self.session_manager.scan_sessions()
            after  = {s.name for s in after_sessions}
            new    = after - before

            if new:
                names = "\n".join(f"  • {n}" for n in sorted(new))
                reply = f"✅ Синхронизация завершена.\n\nНовые сессии ({len(new)}):\n{names}\n\nИспользуй /sessions чтобы запустить обработку."
            else:
                reply = f"✅ Синхронизация завершена. Новых файлов нет.\nДоступно сессий: {len(after_sessions)}"

        except Exception as e:
            log.error(f"Ошибка rclone sync: {e}")
            reply = f"❌ Ошибка синхронизации:\n<code>{e}</code>"

        await msg.edit_text(reply, parse_mode="HTML")

    async def _cmd_sessions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        sessions = self.session_manager.scan_sessions()

        if not sessions:
            await update.message.reply_text(
                "Нет сессий для обработки.\n"
                "Загрузи файлы на Яндекс Диск и выполни /sync"
            )
            return

        if self.is_processing:
            await update.message.reply_text(
                f"⏳ Сейчас обрабатывается: <b>{self.current_session}</b>\n"
                "Дождись завершения перед запуском новой сессии.",
                parse_mode="HTML"
            )
            return

        # Формируем inline-клавиатуру с кнопками сессий
        keyboard = []
        for s in sessions:
            label = f"{s.name}  ({s.file_count} файл{'а' if s.file_count in (2,3,4) else 'ов'})"
            keyboard.append([InlineKeyboardButton(label, callback_data=f"process:{s.name}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"Доступно сессий: {len(sessions)}\nВыбери сессию для обработки:",
            reply_markup=reply_markup
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        if self.is_processing:
            await update.message.reply_text(
                f"⏳ Обрабатывается: <b>{self.current_session}</b>",
                parse_mode="HTML"
            )
        else:
            sessions = self.session_manager.scan_sessions()
            await update.message.reply_text(
                f"✅ Свободно. Ожидает обработки: {len(sessions)} сессий."
            )

    # ── Callback: выбор сессии ────────────────────────────────────────────────

    async def _on_session_selected(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Вызывается когда пользователь нажимает кнопку с именем сессии."""
        query = update.callback_query
        await query.answer()

        if not self._is_allowed(update):
            return

        if self.is_processing:
            await query.edit_message_text(
                f"⏳ Уже обрабатывается: <b>{self.current_session}</b>\n"
                "Дождись завершения.",
                parse_mode="HTML"
            )
            return

        session_name = query.data.replace("process:", "")

        # Находим сессию по имени
        sessions = self.session_manager.scan_sessions()
        session  = next((s for s in sessions if s.name == session_name), None)

        if session is None:
            await query.edit_message_text(f"❌ Сессия '{session_name}' не найдена или уже обработана.")
            return

        # Запускаем обработку в фоне (чтобы не блокировать бота)
        self.is_processing   = True
        self.current_session = session_name

        status_msg = await query.edit_message_text(
            f"▶ Начинаю обработку: <b>{session_name}</b>\n"
            f"Файлов: {session.file_count} экран + {session.file_count} вебка",
            parse_mode="HTML"
        )

        async def progress(text: str) -> None:
            """Обновляет сообщение статуса в боте."""
            try:
                await status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass  # сообщение могло не измениться (Telegram возвращает ошибку)

        # Запускаем тяжёлый пайплайн в отдельном потоке (не блокируем event loop)
        try:
            await asyncio.to_thread(self.pipeline_fn, session, progress)

            # Загружаем результаты на Яндекс Диск
            await progress("📤 Загружаю результаты на Яндекс Диск...")
            await asyncio.to_thread(self._upload_output, session_name)

            await progress(
                f"✅ <b>Готово!</b>\n\n"
                f"Сессия: <b>{session_name}</b>\n"
                f"Видео загружены на Яндекс Диск:\n"
                f"<code>Автомонтаж/output/{session_name}/</code>"
            )
        except Exception as e:
            log.error(f"Ошибка при обработке {session_name}: {e}", exc_info=True)
            await progress(
                f"❌ <b>Ошибка при обработке {session_name}</b>\n\n"
                f"<code>{str(e)[:500]}</code>\n\n"
                "Подробности в логах на сервере."
            )
        finally:
            self.is_processing   = False
            self.current_session = None

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _is_allowed(self, update: Update) -> bool:
        """Проверяет что сообщение от разрешённого пользователя."""
        chat_id = (
            update.effective_chat.id
            if update.effective_chat
            else None
        )
        if chat_id != config.TELEGRAM_ALLOWED_CHAT_ID:
            log.warning(f"Отклонён запрос от неизвестного chat_id: {chat_id}")
            return False
        return True

    def _run_rclone_sync(self) -> None:
        """Запускает rclone sync: Яндекс Диск → локальная папка input/."""
        remote_path = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_INPUT_PATH}"
        local_path  = str(config.INPUT_DIR)

        cmd = [
            "rclone", "sync",
            remote_path, local_path,
            "--min-age", "30s",          # не копировать файлы изменявшиеся последние 30с
            "--progress",
        ]

        log.info(f"rclone sync: {remote_path} → {local_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"rclone завершился с ошибкой:\n{result.stderr[-1000:]}")

    def _upload_output(self, session_name: str) -> None:
        """Загружает папку output/session_name/ на Яндекс Диск."""
        local_path  = str(config.OUTPUT_DIR / session_name)
        remote_path = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_OUTPUT_PATH}/{session_name}"

        cmd = ["rclone", "copy", local_path, remote_path, "--progress"]

        log.info(f"rclone upload: {local_path} → {remote_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"rclone upload завершился с ошибкой:\n{result.stderr[-1000:]}")
