"""
bot.py — Telegram-бот для управления системой Автомонтаж.

Команды:
  /start    — приветствие и список команд
  /sync     — скачать новые файлы с Google Drive
  /sessions — показать список сессий, готовых к обработке
  /status   — текущий статус (обрабатывается / свободно / очередь)
  /cancel   — остановить текущую обработку и очистить очередь
  /reset    — сбросить незавершённую сессию (удалить частичный output)

Безопасность:
  Бот отвечает ТОЛЬКО пользователю с TELEGRAM_ALLOWED_CHAT_ID из config.py.
  Все остальные сообщения игнорируются.

Зависимость: python-telegram-bot >= 20.0 (async API)
"""

import asyncio
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from telegram import (
    BotCommand,
    InlineKeyboardButton, InlineKeyboardMarkup,
    KeyboardButton, ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from modules.session_manager import Session, SessionManager
from modules.session_manager import Session, SessionManager, StandardSession
from modules.utils import format_duration

log = logging.getLogger(__name__)

# ── Клавиатуры ────────────────────────────────────────────────────────────────

def _make_keyboard(mode: str) -> ReplyKeyboardMarkup:
    """Постоянная клавиатура с кнопкой Назад."""
    label = "🎬 Автоформат" if mode == "auto" else "📝 Стандартный"
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("📥 Sync"), KeyboardButton("📋 Сессии"), KeyboardButton("📊 Статус")],
            [KeyboardButton("⛔ Отмена"), KeyboardButton("🔄 Сброс"), KeyboardButton("← Назад")],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )

# Текст кнопки → метод (без учёта регистра)
_BUTTON_COMMANDS = {
    "📥 sync":      "_cmd_sync",
    "📋 сессии":    "_cmd_sessions",
    "📊 статус":    "_cmd_status",
    "⛔ отмена":    "_cmd_cancel",
    "🔄 сброс":     "_cmd_reset",
    "← назад":     "_cmd_start",
}

# Inline-кнопки выбора сценария
_MODE_KEYBOARD = InlineKeyboardMarkup([[
    InlineKeyboardButton("🎬 Автоформат",   callback_data="mode:auto"),
    InlineKeyboardButton("📝 Стандартный",  callback_data="mode:standard"),
]])


class AutomontazhBot:

    def __init__(self, auto_pipeline_fn: Callable, standard_pipeline_fn: Callable):
        """
        auto_pipeline_fn     — пайплайн Автоформат (session, progress) → None
        standard_pipeline_fn — пайплайн Стандартный (standard_session, progress) → None
        """
        self.auto_pipeline_fn     = auto_pipeline_fn
        self.standard_pipeline_fn = standard_pipeline_fn
        self.session_manager = SessionManager()
        self.is_processing   = False
        self.current_session: Optional[str] = None
        self._queue: list = []         # очередь (Session или StandardSession)
        self._mode: str   = "auto"     # "auto" или "standard"
        self._app = None

    # ── Запуск бота ───────────────────────────────────────────────────────────

    async def _post_init(self, app: Application) -> None:
        """Регистрирует команды в меню Telegram (кнопка "/" у поля ввода)."""
        await app.bot.set_my_commands([
            BotCommand("sync",     "скачать файлы с Google Drive"),
            BotCommand("sessions", "показать сессии для обработки"),
            BotCommand("status",   "статус обработки и очередь"),
            BotCommand("cancel",   "остановить обработку"),
            BotCommand("reset",    "сбросить незавершённую сессию"),
        ])

    def run(self) -> None:
        """Запускает бота в режиме polling. Блокирует до Ctrl+C."""
        app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .post_init(self._post_init)
            .build()
        )
        self._app = app  # сохраняем для отправки сообщений из очереди

        # Регистрируем обработчики команд
        app.add_handler(CommandHandler("start",    self._cmd_start))
        app.add_handler(CommandHandler("sync",     self._cmd_sync))
        app.add_handler(CommandHandler("sessions", self._cmd_sessions))
        app.add_handler(CommandHandler("status",   self._cmd_status))
        app.add_handler(CommandHandler("cancel",   self._cmd_cancel))
        app.add_handler(CommandHandler("reset",    self._cmd_reset))

        # Обработчик нажатий на inline-кнопки
        app.add_handler(CallbackQueryHandler(self._on_mode_selected,     pattern="^mode:"))
        app.add_handler(CallbackQueryHandler(self._on_session_selected,  pattern="^process:"))
        app.add_handler(CallbackQueryHandler(self._on_reset_selected,    pattern="^reset:"))
        app.add_handler(CallbackQueryHandler(self._on_all_sessions,      pattern="^all_sessions$"))

        # Обработчик кнопок постоянной клавиатуры
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_keyboard_button))

        log.info("Telegram-бот запущен. Жду команды...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)

    # ── Команды ───────────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return
        await update.message.reply_text(
            "Выбери сценарий обработки:",
            reply_markup=_MODE_KEYBOARD,
        )

    async def _on_mode_selected(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка выбора сценария."""
        query = update.callback_query
        await query.answer()
        if not self._is_allowed(update):
            return

        self._mode = query.data.replace("mode:", "")
        label = "🎬 Автоформат" if self._mode == "auto" else "📝 Стандартный"
        await query.edit_message_text(f"Режим: <b>{label}</b>", parse_mode="HTML")
        await self._app.bot.send_message(
            chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
            text=(
                f"Режим <b>{label}</b> активен.\n\n"
                "Используй кнопки ниже или команды:\n"
                "/sync — скачать файлы с Google Drive\n"
                "/sessions — сессии для обработки\n"
                "/status — статус и очередь"
            ),
            reply_markup=_make_keyboard(self._mode),
            parse_mode="HTML",
        )

    async def _on_keyboard_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает нажатия на кнопки постоянной клавиатуры."""
        if not self._is_allowed(update):
            return
        text = (update.message.text or "").lower()
        method_name = _BUTTON_COMMANDS.get(text)
        if method_name:
            await getattr(self, method_name)(update, ctx)

    async def _cmd_sync(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        msg = await update.message.reply_text("⏳ Синхронизирую с Google Drive...")

        try:
            before = {s.name for s in self.session_manager.scan_sessions()}
            await asyncio.to_thread(self._run_rclone_sync)
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

        if self._mode == "standard":
            sessions = self.session_manager.scan_standard_sessions()
        else:
            sessions = self.session_manager.scan_sessions()

        if not sessions:
            mode_label = "Стандартный" if self._mode == "standard" else "Автоформат"
            await update.message.reply_text(
                f"Нет сессий для обработки ({mode_label}).\n"
                "Загрузи файлы на Google Drive и выполни /sync"
            )
            return

        queued_names = {s.name for s in self._queue}
        status_line = f"⏳ Сейчас: <b>{self.current_session}</b>\n" if self.is_processing else ""
        queue_line  = ("Очередь: " + " → ".join(s.name for s in self._queue) + "\n") if self._queue else ""

        keyboard = []

        # Кнопка "Запустить все" если несколько сессий не в очереди
        free = [s for s in sessions if s.name not in queued_names and s.name != self.current_session]
        if len(free) > 1:
            keyboard.append([InlineKeyboardButton(
                f"▶ Запустить все ({len(free)})", callback_data="all_sessions"
            )])

        for s in sessions:
            if s.name == self.current_session:
                label = f"⏳ {s.name} (обрабатывается)"
            elif s.name in queued_names:
                pos = next(i + 1 for i, q in enumerate(self._queue) if q.name == s.name)
                label = f"#{pos} в очереди: {s.name}"
            else:
                cnt = s.file_count
                marker = " 📄" if (self._mode == "standard" and getattr(s, "has_scenario", False)) else ""
                label = f"{s.name}  ({cnt} файл{'а' if cnt in (2,3,4) else 'ов'}){marker}"
            keyboard.append([InlineKeyboardButton(label, callback_data=f"process:{s.name}")])

        await update.message.reply_text(
            f"{status_line}{queue_line}Выбери сессию — начнётся сразу или встанет в очередь:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        if self.is_processing:
            if self._queue:
                queue_line = "\n\n⏳ Очередь: " + " → ".join(s.name for s in self._queue)
            else:
                queue_line = ""
            await update.message.reply_text(
                f"⏳ Обрабатывается: <b>{self.current_session}</b>{queue_line}",
                parse_mode="HTML"
            )
        else:
            sessions = self.session_manager.scan_sessions()
            if self._queue:
                queue_line = "\n⏳ В очереди: " + " → ".join(s.name for s in self._queue)
            else:
                queue_line = ""
            await update.message.reply_text(
                f"✅ Свободно. Ожидает обработки: {len(sessions)} сессий.{queue_line}"
            )

    # ── Callback: выбор сессии ────────────────────────────────────────────────

    async def _on_all_sessions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Добавляет все свободные сессии в очередь."""
        query = update.callback_query
        await query.answer()
        if not self._is_allowed(update):
            return

        if self._mode == "standard":
            all_sessions = self.session_manager.scan_standard_sessions()
        else:
            all_sessions = self.session_manager.scan_sessions()

        queued_names = {s.name for s in self._queue}
        free = [s for s in all_sessions
                if s.name not in queued_names and s.name != self.current_session]

        if not free:
            await query.edit_message_text("Нет свободных сессий для добавления.")
            return

        added = []
        for s in free:
            if not self.is_processing and not added:
                status_msg = await self._app.bot.send_message(
                    chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                    text=f"▶ Начинаю обработку: <b>{s.name}</b>",
                    parse_mode="HTML",
                )
                asyncio.create_task(self._run_pipeline(s, status_msg))
            else:
                self._queue.append(s)
            added.append(s.name)

        names = "\n".join(f"  • {n}" for n in added)
        await query.edit_message_text(
            f"✅ Запущено/поставлено в очередь ({len(added)}):\n{names}",
            parse_mode="HTML",
        )

    async def _on_session_selected(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Вызывается когда пользователь нажимает кнопку с именем сессии."""
        query = update.callback_query
        await query.answer()

        if not self._is_allowed(update):
            return

        session_name = query.data.replace("process:", "")

        if any(s.name == session_name for s in self._queue):
            pos = next(i + 1 for i, s in enumerate(self._queue) if s.name == session_name)
            await query.edit_message_text(
                f"⚠️ <b>{session_name}</b> уже в очереди (позиция {pos}).",
                parse_mode="HTML"
            )
            return

        if self._mode == "standard":
            all_sessions = self.session_manager.scan_standard_sessions()
        else:
            all_sessions = self.session_manager.scan_sessions()

        session = next((s for s in all_sessions if s.name == session_name), None)

        if session is None:
            await query.edit_message_text(f"❌ Сессия '{session_name}' не найдена или уже обработана.")
            return

        if self.is_processing:
            self._queue.append(session)
            pos = len(self._queue)
            queue_names = " → ".join(s.name for s in self._queue)
            await query.edit_message_text(
                f"✅ <b>{session_name}</b> добавлена в очередь (позиция {pos}).\n\n"
                f"Текущая очередь: {queue_names}\n\n"
                f"Начнётся автоматически после завершения <b>{self.current_session}</b>.",
                parse_mode="HTML"
            )
            return

        mode_label = "[Стандартный]" if self._mode == "standard" else ""
        status_msg = await query.edit_message_text(
            f"▶ Начинаю обработку: <b>{session_name}</b> {mode_label}",
            parse_mode="HTML"
        )
        asyncio.create_task(self._run_pipeline(session, status_msg))

    # ── Основной пайплайн (запускается как asyncio task) ──────────────────────

    async def _run_pipeline(self, session, status_msg) -> None:
        """
        Запускает пайплайн обработки для одной сессии (Session или StandardSession).
        После завершения автоматически запускает следующую сессию из очереди.
        """
        session_name = session.name
        self.is_processing   = True
        self.current_session = session_name
        is_standard = isinstance(session, StandardSession)

        loop = asyncio.get_running_loop()

        async def progress(text: str) -> None:
            try:
                await status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass

        def sync_progress(text: str) -> None:
            future = asyncio.run_coroutine_threadsafe(progress(text), loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass

        pipeline_fn = self.standard_pipeline_fn if is_standard else self.auto_pipeline_fn
        try:
            await asyncio.to_thread(pipeline_fn, session, sync_progress)

            # Отправляем отдельное сообщение о загрузке — саммари в status_msg не трогаем
            upload_msg = await self._app.bot.send_message(
                chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                text="📤 Загружаю результаты на Google Drive...",
                parse_mode="HTML",
            )
            await asyncio.to_thread(self._upload_output, session_name)

            if config.DELETE_INPUT_AFTER_PROCESSING:
                await asyncio.to_thread(self._delete_input, session_name)

            queue_info = f"\n\n⏳ Следующая в очереди: <b>{self._queue[0].name}</b>" if self._queue else ""
            await upload_msg.edit_text(
                f"✅ <b>Готово!</b> Видео загружены на Google Drive:\n"
                f"<code>PROJECTS/Автомонтаж/output/{session_name}/</code>"
                f"{queue_info}",
                parse_mode="HTML",
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

            # Запускаем следующую сессию из очереди
            if self._queue:
                next_session = self._queue.pop(0)
                log.info(f"Очередь: запускаю следующую сессию — {next_session.name}")
                next_msg = await self._app.bot.send_message(
                    chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                    text=f"▶ Начинаю обработку из очереди: <b>{next_session.name}</b>",
                    parse_mode="HTML"
                )
                asyncio.create_task(self._run_pipeline(next_session, next_msg))

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _is_allowed(self, update: Update) -> bool:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id != config.TELEGRAM_ALLOWED_CHAT_ID:
            log.warning(f"Отклонён запрос от неизвестного chat_id: {chat_id}")
            return False
        return True

    async def _cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        if not self.is_processing:
            await update.message.reply_text("Ничего не обрабатывается.")
            return

        session_name  = self.current_session
        queue_cleared = len(self._queue)
        self._queue.clear()

        queue_msg = f"\nОчередь очищена ({queue_cleared} сессий)." if queue_cleared else ""
        await update.message.reply_text(
            f"⛔ Останавливаю обработку <b>{session_name}</b>...\n"
            f"Бот перезапустится через несколько секунд.{queue_msg}",
            parse_mode="HTML"
        )

        if session_name:
            import shutil
            output_path = config.OUTPUT_DIR / session_name
            if output_path.exists():
                shutil.rmtree(output_path)
                log.info(f"/cancel: удалён частичный output: {output_path}")

        log.warning("Получена команда /cancel — завершаю процесс.")
        os.kill(os.getpid(), signal.SIGTERM)

    async def _cmd_reset(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_allowed(update):
            return

        if self.is_processing:
            await update.message.reply_text(
                f"⏳ Сейчас идёт обработка: <b>{self.current_session}</b>\n"
                "Используй /cancel чтобы остановить её.",
                parse_mode="HTML"
            )
            return

        if not config.OUTPUT_DIR.exists():
            await update.message.reply_text("Нет незавершённых сессий.")
            return

        sessions_with_output = sorted(d for d in config.OUTPUT_DIR.iterdir() if d.is_dir())

        if not sessions_with_output:
            await update.message.reply_text("Нет незавершённых сессий.")
            return

        keyboard = [
            [InlineKeyboardButton(f"🗑 {d.name}", callback_data=f"reset:{d.name}")]
            for d in sessions_with_output
        ]
        await update.message.reply_text(
            "Выбери сессию для сброса.\n"
            "Удалит output-файлы — входные файлы останутся, можно запустить заново:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def _on_reset_selected(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if not self._is_allowed(update):
            return

        import shutil
        session_name = query.data.replace("reset:", "")
        output_path  = config.OUTPUT_DIR / session_name

        if output_path.exists():
            shutil.rmtree(output_path)
            log.info(f"/reset: удалён output: {output_path}")
            await query.edit_message_text(
                f"✅ Сессия <b>{session_name}</b> сброшена.\n"
                "Запусти обработку заново через /sessions.",
                parse_mode="HTML"
            )
        else:
            await query.edit_message_text(f"Папка <b>{session_name}</b> не найдена.", parse_mode="HTML")

    def _run_rclone_sync(self) -> None:
        remote_path = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_INPUT_PATH}"
        local_path  = str(config.INPUT_DIR)
        cmd = [
            "rclone", "sync",
            remote_path, local_path,
            "--min-age", "30s",
            "--progress",
        ]
        log.info(f"rclone sync: {remote_path} → {local_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rclone завершился с ошибкой:\n{result.stderr[-1000:]}")

    def _delete_input(self, session_name: str) -> None:
        import shutil
        input_path = config.INPUT_DIR / session_name
        if input_path.exists():
            shutil.rmtree(input_path)
            log.info(f"Удалена входная папка: {input_path}")

    def _upload_output(self, session_name: str) -> None:
        local_path  = str(config.OUTPUT_DIR / session_name)
        remote_path = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_OUTPUT_PATH}/{session_name}"
        cmd = ["rclone", "copy", local_path, remote_path, "--progress"]
        log.info(f"rclone upload: {local_path} → {remote_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rclone upload завершился с ошибкой:\n{result.stderr[-1000:]}")
