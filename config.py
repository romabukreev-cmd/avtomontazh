"""
config.py — все настройки системы Автомонтаж в одном месте.
Меняй только этот файл, чтобы перенастроить поведение системы.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # загружаем переменные из .env файла

# ─── Пути ─────────────────────────────────────────────────────────────────────

# Корень проекта (папка, где лежит этот файл)
BASE_DIR = Path(__file__).parent

# Сессии хранятся как папки: input/2024-01-15_logo/, input/2024-01-22_website/ и т.д.
INPUT_DIR  = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR   = BASE_DIR / "temp"
LOGS_DIR   = BASE_DIR / "logs"

# ─── Именование файлов внутри сессии ─────────────────────────────────────────

# Glob-паттерны для поиска файлов внутри папки сессии
# Ожидаемый формат: screen_001.mp4/.mkv, webcam_001.mp4/.mkv и т.д.
SCREEN_FILE_PATTERN = "screen_*"
WEBCAM_FILE_PATTERN = "webcam_*"
VIDEO_EXTENSIONS    = {".mp4", ".mkv", ".avi", ".mov", ".ts"}

# ─── Whisper (транскрипция) ───────────────────────────────────────────────────

# Модель: tiny/base/small/medium/large-v3
WHISPER_MODEL    = "medium"
WHISPER_LANGUAGE = "ru"

# ─── Нарезка пауз (применяется после LLM-анализа) ────────────────────────────

# Пауза длиннее этого значения вырезается из финального видео
PAUSE_CUT_SEC = 0.6  # секунды

# Буфер с каждой стороны при нарезке (не рубить прямо по слову)
PAUSE_BUFFER_SEC = 0.2  # секунды

# ─── Разрешение входных файлов ────────────────────────────────────────────────

# Разрешение экранной записи (1920×1200 — нестандарт, нужен кроп до 1920×1080)
SCREEN_SOURCE_WIDTH  = 1920
SCREEN_SOURCE_HEIGHT = 1200  # если экран пишет 1920×1080 — поставь 1080
SCREEN_CROP_Y = (SCREEN_SOURCE_HEIGHT - 1080) // 2  # = 60 для 1200, = 0 для 1080

# ─── LLM (анализ контента через OpenRouter) ───────────────────────────────────

# Ключ берётся из .env файла — не вводи его прямо здесь!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Модель для анализа транскрипции
# Хорошие варианты: "anthropic/claude-3-5-haiku", "google/gemini-2.0-flash-001", "openai/gpt-4o-mini"
LLM_MODEL = "anthropic/claude-sonnet-4-6"

# Контекст видео — описание того, ЧТО снимается. LLM использует это для понимания структуры.
VIDEO_CONTEXT = (
    "Автор записывает процесс дизайна: сначала создаёт карточку товара (Figma/Photoshop), "
    "затем делает анимацию. Иногда встречаются рекламные интеграции. "
    "Видео должно охватывать весь процесс от начала до конца."
)

# ─── Выходные форматы ─────────────────────────────────────────────────────────

# Формат 1 — вертикальный 9:16, не более 9 минут
FORMAT_1 = {
    "name": "vertical_10min",
    "width": 1080,
    "height": 1920,
    "max_duration_sec": 540,   # 9 минут (запас под интро/аутро при монтаже)
    "layout": "split_vertical", # экран сверху, вебка снизу
}

# Формат 2 — вертикальный 9:16, не более 2:10 (лучшие моменты)
FORMAT_2 = {
    "name": "vertical_3min",
    "width": 1080,
    "height": 1920,
    "max_duration_sec": 130,   # 2 минуты 10 секунд (запас под интро/аутро)
    "layout": "split_vertical",
    "highlight_only": True,    # LLM выбирает самые сильные моменты
}

# Формат 3 — горизонтальный 16:9, не более 9 минут
FORMAT_3 = {
    "name": "horizontal_10min",
    "width": 1920,
    "height": 1080,
    "max_duration_sec": 540,   # 9 минут (запас под интро/аутро при монтаже)
    "layout": "pip",           # picture-in-picture: экран + кружок вебки
}

# ─── Настройки PiP (вебка, Формат 3) ────────────────────────────────────────
# Вебка — квадрат из центра исходника (1920×1080 → кроп 1080×1080 → масштаб)
# Позиция: правый нижний угол с отступами

PIP_WIDTH         = 350     # ширина квадрата (~⅓ высоты экрана)
PIP_HEIGHT        = 350     # высота квадрата
PIP_CORNER_RADIUS = 20      # скруглённость углов
PIP_MARGIN_RIGHT  = 60      # отступ от правого края
PIP_MARGIN_BOTTOM = 60      # отступ от нижнего края

# ─── FFmpeg ───────────────────────────────────────────────────────────────────

# Кодек видео: libx264 (CPU, совместимость) или h264_nvenc (GPU NVIDIA, быстро)
VIDEO_CODEC = "libx264"
VIDEO_CRF   = 23      # качество: 18 = лучше/тяжелее, 28 = хуже/легче
VIDEO_PRESET = "ultrafast" # ultrafast/fast/medium/slow

AUDIO_CODEC   = "aac"
AUDIO_BITRATE = "192k"

FFMPEG_THREADS = 0  # 0 = FFmpeg auto (использует все доступные ядра)

# ─── Логирование и поведение ──────────────────────────────────────────────────

LOG_LEVEL = "INFO"   # DEBUG / INFO / WARNING / ERROR

# Удалять временные файлы после успешной обработки?
CLEANUP_TEMP_ON_SUCCESS = True

# Удалять входную папку сессии после успешной обработки?
DELETE_INPUT_AFTER_PROCESSING = True

# Повторная попытка если LLM вернул ошибку
LLM_MAX_RETRIES = 3

# ─── Telegram-бот ────────────────────────────────────────────────────────────

# Токен бота — получить у @BotFather в Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Твой Telegram ID — узнать через @userinfobot
# Бот будет отвечать ТОЛЬКО этому пользователю (защита)
TELEGRAM_ALLOWED_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

# ─── Google Drive (rclone) ───────────────────────────────────────────────────

# Имя remote в rclone config (задаётся при настройке rclone)
RCLONE_REMOTE_NAME = "gdrive"

# Путь на Google Drive откуда скачиваются исходные файлы
RCLONE_YD_INPUT_PATH = "PROJECTS/Автомонтаж/input"

# Путь на Google Drive куда загружаются готовые видео
RCLONE_YD_OUTPUT_PATH = "PROJECTS/Автомонтаж/output"
