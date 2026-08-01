"""
bot.py — Telegram-бот для управления системой Автомонтаж.

Команды:
  /start    — приветствие
  /sync     — скачать файлы с Google Drive
  /sessions — список сессий для обработки
  /status   — текущий статус
  /cancel   — остановить текущую обработку
  /reset    — сбросить незавершённую сессию

Бот отвечает только пользователю с TELEGRAM_ALLOWED_CHAT_ID.
"""

import asyncio
import logging
import os
import shutil
import signal
import subprocess
from typing import Callable, Optional

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
from modules.utils import format_duration

log = logging.getLogger(__name__)

_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("📥 Sync"), KeyboardButton("📋 Сессии"), KeyboardButton("📊 Статус")],
        [KeyboardButton("⛔ Отмена"), KeyboardButton("🔄 Сброс"), KeyboardButton("🗑 Output")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)

_BUTTON_COMMANDS = {
    "📥 sync":    "_cmd_sync",
    "📋 сессии":  "_cmd_sessions",
    "📊 статус":  "_cmd_status",
    "⛔ отмена":  "_cmd_cancel",
    "🔄 сброс":   "_cmd_reset",
    "🗑 output":  "_cmd_clear_output",
}

class AutomontazhBot:

    def __init__(self, pipeline_fn: Callable):
        self.pipeline_fn     = pipeline_fn
        self.session_manager = SessionManager()
        self.is_processing   = False
        self.current_session: Optional[str] = None
        self._queue: list[Session] = []
        self._app            = None

    # ── Запуск ────────────────────────────────────────────────────────────────

    async def _post_init(self, app: Application) -> None:
        await app.bot.set_my_commands([
            BotCommand("sync",     "скачать файлы с Google Drive"),
            BotCommand("sessions", "список сессий для обработки"),
            BotCommand("status",   "статус обработки и очередь"),
            BotCommand("cancel",   "остановить обработку"),
            BotCommand("reset",    "сбросить незавершённую сессию"),
        ])

    def run(self) -> None:
        app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .post_init(self._post_init)
            .build()
        )
        self._app = app

        app.add_handler(CommandHandler("start",        self._cmd_start))
        app.add_handler(CommandHandler("sync",         self._cmd_sync))
        app.add_handler(CommandHandler("sessions",     self._cmd_sessions))
        app.add_handler(CommandHandler("status",       self._cmd_status))
        app.add_handler(CommandHandler("cancel",       self._cmd_cancel))
        app.add_handler(CommandHandler("reset",        self._cmd_reset))
        app.add_handler(CommandHandler("clear_output", self._cmd_clear_output))

        app.add_handler(CallbackQueryHandler(self._on_session_selected, pattern="^process:"))
        app.add_handler(CallbackQueryHandler(self._on_all_sessions,     pattern="^all_sessions$"))
        app.add_handler(CallbackQueryHandler(self._on_reset_selected,   pattern="^reset:"))
        app.add_handler(CallbackQueryHandler(self._on_clear_confirmed,  pattern="^clear_output_confirm$"))

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_keyboard_button))

        log.info("Telegram-бот запущен")
        app.run_polling(allowed_updates=Update.ALL_TYPES)

    # ── Команды ───────────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        await update.message.reply_text(
            "🎬 <b>Автомонтаж</b>\n\n"
            "/sync — скачать файлы с Google Drive\n"
            "/sessions — список сессий\n"
            "/status — статус обработки\n"
            "/cancel — остановить\n"
            "/reset — сбросить незавершённую сессию",
            reply_markup=_KEYBOARD,
            parse_mode="HTML",
        )

    async def _on_keyboard_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        text   = (update.message.text or "").lower()
        method = _BUTTON_COMMANDS.get(text)
        if method:
            await getattr(self, method)(update, ctx)

    async def _cmd_sync(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        msg = await update.message.reply_text("⏳ Синхронизирую с Google Drive...")
        try:
            before = {s.name for s in self.session_manager.scan_sessions()}
            await asyncio.to_thread(self._run_rclone_sync)
            after_sessions = self.session_manager.scan_sessions()
            new = {s.name for s in after_sessions} - before
            if new:
                names = "\n".join(f"  • {n}" for n in sorted(new))
                reply = (
                    f"✅ Готово. Новые сессии ({len(new)}):\n{names}\n\n"
                    "Используй /sessions для запуска."
                )
            else:
                reply = f"✅ Готово. Новых файлов нет. Доступно: {len(after_sessions)} сессий."
        except Exception as e:
            log.error(f"rclone sync: {e}")
            reply = f"❌ Ошибка синхронизации:\n<code>{e}</code>"
        try:
            await msg.edit_text(reply, parse_mode="HTML")
        except Exception as e:
            log.warning(f"sync reply edit failed: {e}")
            try:
                await update.message.reply_text(reply, parse_mode="HTML")
            except Exception:
                log.exception("sync fallback reply failed")

    async def _cmd_sessions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        sessions = self.session_manager.scan_sessions()
        if not sessions:
            await update.message.reply_text(
                "Нет сессий для обработки.\n"
                "Загрузи файлы на Google Drive и выполни /sync"
            )
            return

        queued_names = {s.name for s in self._queue}
        status_line  = f"⏳ Сейчас: <b>{self.current_session}</b>\n" if self.is_processing else ""
        queue_line   = ("Очередь: " + " → ".join(s.name for s in self._queue) + "\n") if self._queue else ""

        keyboard = []
        free = [s for s in sessions if s.name not in queued_names and s.name != self.current_session]
        if len(free) > 1:
            keyboard.append([InlineKeyboardButton(
                f"▶ Запустить все ({len(free)})", callback_data="all_sessions"
            )])

        for s in sessions:
            # Иконки наличия файлов
            icons = ("🖥" if s.has_screen else "") + ("🎥" if s.has_webcam else "")
            if s.name == self.current_session:
                label = f"⏳ {s.name} [{icons}] (обрабатывается)"
            elif s.name in queued_names:
                pos   = next(i + 1 for i, (q, _) in enumerate(self._queue) if q.name == s.name)
                label = f"#{pos} в очереди: {s.name} [{icons}]"
            else:
                label = f"{s.name}  [{icons}] ({s.file_count} файл(ов))"
            keyboard.append([InlineKeyboardButton(label, callback_data=f"process:{s.name}")])


        await update.message.reply_text(
            f"{status_line}{queue_line}Выбери сессию:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        if self.is_processing:
            q = ("\n\n⏳ Очередь: " + " → ".join(s.name for s in self._queue)) if self._queue else ""
            await update.message.reply_text(
                f"⏳ Обрабатывается: <b>{self.current_session}</b>{q}",
                parse_mode="HTML",
            )
        else:
            sessions = self.session_manager.scan_sessions()
            q = ("\n⏳ В очереди: " + " → ".join(s.name for s in self._queue)) if self._queue else ""
            await update.message.reply_text(
                f"✅ Свободно. Ожидает обработки: {len(sessions)} сессий.{q}"
            )

    async def _cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        if not self.is_processing:
            await update.message.reply_text("Ничего не обрабатывается.")
            return
        session_name  = self.current_session
        queue_cleared = len(self._queue)
        self._queue.clear()
        q_msg = f"\nОчередь очищена ({queue_cleared} сессий)." if queue_cleared else ""
        await update.message.reply_text(
            f"⛔ Останавливаю <b>{session_name}</b>...\n"
            f"Бот перезапустится через несколько секунд.{q_msg}",
            parse_mode="HTML",
        )
        if session_name:
            out = config.OUTPUT_DIR / session_name
            if out.exists():
                shutil.rmtree(out)
                log.info(f"/cancel: удалён частичный output: {out}")
        log.warning("/cancel — завершаю процесс")
        os.kill(os.getpid(), signal.SIGTERM)

    async def _cmd_reset(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        if self.is_processing:
            await update.message.reply_text(
                f"⏳ Сейчас идёт обработка: <b>{self.current_session}</b>\n"
                "Используй /cancel чтобы остановить.",
                parse_mode="HTML",
            )
            return
        if not config.OUTPUT_DIR.exists():
            await update.message.reply_text("Нет незавершённых сессий.")
            return
        folders = sorted(d for d in config.OUTPUT_DIR.iterdir() if d.is_dir())
        if not folders:
            await update.message.reply_text("Нет незавершённых сессий.")
            return
        keyboard = [
            [InlineKeyboardButton(f"🗑 {d.name}", callback_data=f"reset:{d.name}")]
            for d in folders
        ]
        await update.message.reply_text(
            "Выбери сессию для сброса (удалит output, input останется):",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _cmd_clear_output(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        if self.is_processing:
            await update.message.reply_text(
                f"⏳ Идёт обработка: <b>{self.current_session}</b>\n"
                "Дождись завершения или останови /cancel.",
                parse_mode="HTML",
            )
            return

        msg = await update.message.reply_text("⏳ Проверяю output...")
        local_folders, gdrive_folders = await asyncio.to_thread(self._list_output_folders)
        all_names = sorted(set(local_folders) | set(gdrive_folders))
        if not all_names:
            await msg.edit_text("Папка output пуста (сервер и Google Drive).")
            return

        lines = []
        for name in all_names:
            tags = []
            if name in local_folders:
                tags.append("сервер")
            if name in gdrive_folders:
                tags.append("GDrive")
            lines.append(f"  • {name} ({', '.join(tags)})")

        await msg.edit_text(
            f"Будет удалено ({len(all_names)} сессий):\n" + "\n".join(lines) + "\n\nПодтверждаешь?",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🗑 Да, удалить всё", callback_data="clear_output_confirm"),
                InlineKeyboardButton("Отмена",             callback_data="clear_output_cancel"),
            ]]),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def _on_all_sessions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if not self._allowed(update):
            return
        all_sessions = self.session_manager.scan_sessions()
        queued_names = {s.name for s in self._queue}
        free = [s for s in all_sessions if s.name not in queued_names and s.name != self.current_session]
        if not free:
            await query.edit_message_text("Нет свободных сессий.")
            return

        added = []
        for s in free:
            if not self.is_processing and not added:
                status_msg = await self._app.bot.send_message(
                    chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                    text=f"▶ <b>{s.name}</b>",
                    parse_mode="HTML",
                )
                asyncio.create_task(self._run_pipeline(s, status_msg))
            else:
                self._queue.append(s)
            added.append(s.name)

        names = "\n".join(f"  • {n}" for n in added)
        await query.edit_message_text(
            f"✅ Запущено/в очереди ({len(added)}):\n{names}",
            parse_mode="HTML",
        )

    async def _on_session_selected(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if not self._allowed(update):
            return

        session_name = query.data.replace("process:", "")

        if any(s.name == session_name for s in self._queue):
            pos = next(i + 1 for i, s in enumerate(self._queue) if s.name == session_name)
            await query.edit_message_text(
                f"⚠️ <b>{session_name}</b> уже в очереди (позиция {pos}).",
                parse_mode="HTML",
            )
            return

        session = next(
            (s for s in self.session_manager.scan_sessions() if s.name == session_name),
            None,
        )
        if session is None:
            await query.edit_message_text(f"❌ Сессия '{session_name}' не найдена.")
            return

        if self.is_processing:
            self._queue.append(session)
            pos         = len(self._queue)
            queue_names = " → ".join(s.name for s in self._queue)
            await query.edit_message_text(
                f"✅ <b>{session_name}</b> в очереди (позиция {pos}).\n\n"
                f"Очередь: {queue_names}\n\n"
                f"Начнётся после <b>{self.current_session}</b>.",
                parse_mode="HTML",
            )
            return

        status_msg = await query.edit_message_text(
            f"▶ <b>{session_name}</b>",
            parse_mode="HTML",
        )
        asyncio.create_task(self._run_pipeline(session, status_msg))

    async def _on_reset_selected(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if not self._allowed(update):
            return
        session_name = query.data.replace("reset:", "")
        out = config.OUTPUT_DIR / session_name
        if out.exists():
            shutil.rmtree(out)
            log.info(f"/reset: удалён output: {out}")
            await query.edit_message_text(
                f"✅ <b>{session_name}</b> сброшена. Запусти заново через /sessions.",
                parse_mode="HTML",
            )
        else:
            await query.edit_message_text(f"Папка <b>{session_name}</b> не найдена.", parse_mode="HTML")

    async def _on_clear_confirmed(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        if not self._allowed(update):
            return
        await query.edit_message_text("⏳ Удаляю...")
        count, gdrive_fail = await asyncio.to_thread(self._delete_all_output)
        if gdrive_fail:
            fail_list = "\n".join(f"  • {n}" for n in gdrive_fail)
            await query.edit_message_text(
                f"✅ Удалено локально: {count} сессий.\n\n"
                f"❌ Не удалось удалить с Google Drive:\n{fail_list}\n\n"
                "Проверь логи сервера.",
                parse_mode="HTML",
            )
        else:
            await query.edit_message_text(f"✅ Удалено {count} сессий (сервер + Google Drive).")

    # ── Пайплайн ──────────────────────────────────────────────────────────────

    async def _run_pipeline(self, session: Session, status_msg) -> None:
        session_name         = session.name
        self.is_processing   = True
        self.current_session = session_name
        loop = asyncio.get_running_loop()

        async def progress(text: str) -> None:
            try:
                await status_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass

        def sync_progress(text: str) -> None:
            fut = asyncio.run_coroutine_threadsafe(progress(text), loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass

        try:
            await asyncio.to_thread(self.pipeline_fn, session, sync_progress)

            upload_msg = await self._app.bot.send_message(
                chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                text="📤 Загружаю на Google Drive...",
                parse_mode="HTML",
            )
            await asyncio.to_thread(self._upload_output, session_name)

            if config.DELETE_INPUT_AFTER_PROCESSING:
                await asyncio.to_thread(self._delete_input, session_name)

            q_info = ""
            q_info = ""
            if self._queue:
                next_name = self._queue[0].name
                q_info = f"\n\n⏳ Следующая: <b>{next_name}</b>"
            success_text = (
                f"✅ <b>Готово!</b> Видео на Google Drive:\n"
                f"<code>PROJECTS/Автомонтаж/output/{session_name}/</code>"
                f"{q_info}"
            )
            try:
                await upload_msg.edit_text(success_text, parse_mode="HTML")
            except Exception as tg_err:
                log.warning(f"edit_text не удался для {session_name}: {tg_err}")
                try:
                    await self._app.bot.send_message(
                        chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                        text=success_text,
                        parse_mode="HTML",
                    )
                except Exception:
                    pass

        except Exception as e:
            log.error(f"Ошибка обработки {session_name}: {e}", exc_info=True)
            await progress(
                f"❌ <b>Ошибка: {session_name}</b>\n\n"
                f"<code>{str(e)[:500]}</code>\n\n"
                "Подробности в логах."
            )

        finally:
            self.is_processing   = False
            self.current_session = None
            if self._queue:
                next_s = self._queue.pop(0)
                log.info(f"Очередь: запускаю {next_s.name}")
                next_msg = await self._app.bot.send_message(
                    chat_id=config.TELEGRAM_ALLOWED_CHAT_ID,
                    text=f"▶ Из очереди: <b>{next_s.name}</b>",
                    parse_mode="HTML",
                )
                asyncio.create_task(self._run_pipeline(next_s, next_msg))

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _allowed(self, update: Update) -> bool:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id != config.TELEGRAM_ALLOWED_CHAT_ID:
            log.warning(f"Запрос от неизвестного chat_id: {chat_id}")
            return False
        return True

    def _run_rclone_sync(self) -> None:
        remote = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_INPUT_PATH}"
        local  = str(config.INPUT_DIR)
        cmd    = ["rclone", "sync", remote, local, "--min-age", "30s", "--stats", "30s"]
        log.info(f"rclone sync: {remote} → {local}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=config.RCLONE_SYNC_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"rclone sync timed out after {config.RCLONE_SYNC_TIMEOUT_SEC}s") from e
        if result.returncode != 0:
            raise RuntimeError(f"rclone sync:\n{result.stderr[-1000:]}")

    def _upload_output(self, session_name: str) -> None:
        local  = str(config.OUTPUT_DIR / session_name)
        remote = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_OUTPUT_PATH}/{session_name}"
        cmd    = ["rclone", "copy", local, remote, "--stats", "30s"]
        log.info(f"rclone upload: {local} → {remote}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=config.RCLONE_UPLOAD_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"rclone upload timed out after {config.RCLONE_UPLOAD_TIMEOUT_SEC}s") from e
        if result.returncode != 0:
            raise RuntimeError(f"rclone upload:\n{result.stderr[-1000:]}")

    def _delete_input(self, session_name: str) -> None:
        inp = config.INPUT_DIR / session_name
        if inp.exists():
            shutil.rmtree(inp)
            log.info(f"Удалена входная папка: {inp}")

    def _list_output_folders(self) -> tuple[list[str], list[str]]:
        local_folders = []
        if config.OUTPUT_DIR.exists():
            local_folders = sorted(d.name for d in config.OUTPUT_DIR.iterdir() if d.is_dir())
        remote = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_OUTPUT_PATH}"
        res = subprocess.run(["rclone", "lsf", "--dirs-only", remote], capture_output=True, text=True)
        gdrive_folders = sorted(
            name.rstrip("/") for name in res.stdout.splitlines() if name.strip()
        ) if res.returncode == 0 else []
        return local_folders, gdrive_folders

    def _delete_all_output(self) -> tuple[int, list[str]]:
        local_folders, gdrive_folders = self._list_output_folders()
        local_dirs = []
        if config.OUTPUT_DIR.exists():
            local_dirs = [d for d in config.OUTPUT_DIR.iterdir() if d.is_dir()]

        all_names = sorted(set(d.name for d in local_dirs) | set(gdrive_folders))
        count     = len(all_names)

        for d in local_dirs:
            shutil.rmtree(d)
            log.info(f"clear_output: удалена локальная папка {d.name}")

        remote_base = f"{config.RCLONE_REMOTE_NAME}:{config.RCLONE_YD_OUTPUT_PATH}"
        gdrive_fail = []
        for name in all_names:
            remote = f"{remote_base}/{name}"
            result = subprocess.run(["rclone", "purge", remote], capture_output=True, text=True)
            if result.returncode == 0:
                log.info(f"clear_output: удалено с GDrive {name}")
            else:
                log.warning(f"clear_output: GDrive purge не удался для {name}: {result.stderr[:500]}")
                gdrive_fail.append(name)

        return count, gdrive_fail
