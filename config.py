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
# Ожидаемый формат: screen_001.mp4, screen_002.mp4 / webcam_001.mp4, webcam_002.mp4
SCREEN_FILE_PATTERN = "screen_*.mp4"
WEBCAM_FILE_PATTERN = "webcam_*.mp4"

# ─── Whisper (транскрипция) ───────────────────────────────────────────────────

# Модель: tiny/base/small/medium/large
# tiny  — очень быстро, хуже качество
# small — хороший баланс скорости и точности для русского
# large — лучшее качество, медленно (нужен мощный CPU или GPU)
WHISPER_MODEL    = "small"
WHISPER_LANGUAGE = "ru"   # явно указываем русский для точности

# Пауза: отрезок тишины длиннее порога будет вырезан
PAUSE_THRESHOLD_SEC = 0.8  # секунды (аналог Premiere Pro)

# Минимальная длина сохраняемого сегмента (избегаем микро-кусков)
MIN_SEGMENT_DURATION_SEC = 3.0  # секунды

# ─── LLM (анализ контента через OpenRouter) ───────────────────────────────────

# Ключ берётся из .env файла — не вводи его прямо здесь!
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Модель для анализа транскрипции
# Хорошие варианты: "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini", "google/gemini-flash-1.5"
LLM_MODEL = "anthropic/claude-3.5-sonnet"

# Сколько токенов транскрипции отправлять за раз (у длинных видео транскрипция большая)
LLM_CHUNK_SIZE_TOKENS = 8000

# ─── Критерии отбора сегментов (инструкция для LLM) ─────────────────────────

# Что оставлять: типы активности, которые интересны зрителю
KEEP_ACTIVITY_TYPES = [
    "генерация идей",
    "создание и редактирование дизайна",
    "обсуждение решений",
    "объяснение процесса",
    "интересные находки и открытия",
]

# Что вырезать
REMOVE_ACTIVITY_TYPES = [
    "долгое молчание",
    "технические задержки (загрузка, ожидание)",
    "нерелевантные разговоры",
    "отвлечения от работы",
    "повторяющиеся однотипные действия без комментариев",
]

# ─── Выходные форматы ─────────────────────────────────────────────────────────

# Формат 1 — вертикальный 9:16, до 10 минут
FORMAT_1 = {
    "name": "vertical_10min",
    "width": 1080,
    "height": 1920,
    "max_duration_sec": 600,   # 10 минут
    "layout": "split_vertical", # экран сверху, вебка снизу
}

# Формат 2 — вертикальный 9:16, до 3 минут (лучшие моменты)
FORMAT_2 = {
    "name": "vertical_3min",
    "width": 1080,
    "height": 1920,
    "max_duration_sec": 180,   # 3 минуты
    "layout": "split_vertical",
    "highlight_only": True,    # LLM выбирает самые сильные моменты
}

# Формат 3 — горизонтальный 16:9, до 10 минут
FORMAT_3 = {
    "name": "horizontal_10min",
    "width": 1920,
    "height": 1080,
    "max_duration_sec": 600,   # 10 минут
    "layout": "pip",           # picture-in-picture: экран + кружок вебки
}

# ─── Настройки PiP (вебка, Формат 3) ────────────────────────────────────────
# Вебка — квадрат из центра исходника (1920×1080 → кроп 1080×1080 → масштаб)
# Размер: ¼ высоты экрана = 270×270px
# Позиция: правый нижний угол с отступами

PIP_SIZE          = 350     # сторона квадрата (~⅓ высоты экрана)
PIP_WIDTH         = 350     # для совместимости с renderer.py
PIP_HEIGHT        = 350
PIP_CORNER_RADIUS = 20      # скруглённость углов
PIP_MARGIN_RIGHT  = 60      # отступ от правого края
PIP_MARGIN_BOTTOM = 60      # отступ от нижнего края

# ─── FFmpeg ───────────────────────────────────────────────────────────────────

# Кодек видео: libx264 (CPU, совместимость) или h264_nvenc (GPU NVIDIA, быстро)
VIDEO_CODEC = "libx264"
VIDEO_CRF   = 23      # качество: 18 = лучше/тяжелее, 28 = хуже/легче
VIDEO_PRESET = "fast" # ultrafast/fast/medium/slow

AUDIO_CODEC   = "aac"
AUDIO_BITRATE = "192k"

FFMPEG_THREADS = 4  # сколько потоков CPU отдать FFmpeg

# ─── Логирование и поведение ──────────────────────────────────────────────────

LOG_LEVEL = "INFO"   # DEBUG / INFO / WARNING / ERROR

# Удалять временные файлы после успешной обработки?
CLEANUP_TEMP_ON_SUCCESS = True

# Перемещать входные файлы в archive/ после обработки?
ARCHIVE_INPUT_AFTER_PROCESSING = True
ARCHIVE_DIR = BASE_DIR / "archive"

# Повторная попытка если LLM вернул ошибку
LLM_MAX_RETRIES = 3

# ─── Telegram-бот ────────────────────────────────────────────────────────────

# Токен бота — получить у @BotFather в Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Твой Telegram ID — узнать через @userinfobot
# Бот будет отвечать ТОЛЬКО этому пользователю (защита)
TELEGRAM_ALLOWED_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

# ─── Яндекс Диск (rclone) ────────────────────────────────────────────────────

# Имя remote в rclone config (задаётся при настройке rclone)
RCLONE_REMOTE_NAME = "yadisk"

# Путь на Яндекс Диске откуда скачиваются исходные файлы
RCLONE_YD_INPUT_PATH = "PROJECTS/Автомонтаж/input"

# Путь на Яндекс Диске куда загружаются готовые видео
RCLONE_YD_OUTPUT_PATH = "PROJECTS/Автомонтаж/output"
