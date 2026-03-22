"""
config.py — все настройки системы Автомонтаж.
Секреты (ключи, токены) — в .env файле.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Пути ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR   = BASE_DIR / "temp"
LOGS_DIR   = BASE_DIR / "logs"

# ── Файлы сессии ───────────────────────────────────────────────────────────────

# Поддерживаем оба варианта именования:
# screen_001.mp4 / webcam_001.mp4 и screen.mp4 / webcam.mp4
SCREEN_FILE_PATTERN = "screen*"
WEBCAM_FILE_PATTERN = "webcam*"
VIDEO_EXTENSIONS    = {".mp4", ".mkv", ".avi", ".mov", ".ts"}

# ── Whisper ────────────────────────────────────────────────────────────────────

WHISPER_MODEL    = "large-v3"
WHISPER_LANGUAGE = "ru"
VAD_MIN_SILENCE_MS = 400   # мс — минимальная тишина между сегментами
VAD_SPEECH_PAD_MS  = 200   # мс — буфер вокруг речевого региона

# ── Пороги ────────────────────────────────────────────────────────────────────

PAUSE_CUT_SEC = 0.5    # пауза >= этого → разрыв между блоками слов
TIMELINE_START_PAD_SEC = 0.05
TIMELINE_END_PAD_SEC   = 0.20

# ── LLM ───────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL          = "anthropic/claude-sonnet-4-6"
LLM_MAX_RETRIES    = 3

VIDEO_CONTEXT = (
    "Автор записывает процесс дизайна: сначала создаёт карточку товара (Figma/Photoshop), "
    "затем делает анимацию. Иногда встречаются рекламные интеграции. "
    "Видео должно охватывать весь процесс от начала до конца."
)

# ── Входное видео ──────────────────────────────────────────────────────────────

SCREEN_SOURCE_WIDTH  = 1920
SCREEN_SOURCE_HEIGHT = 1200   # если 1920×1080 — поставить 1080
SCREEN_CROP_Y        = (SCREEN_SOURCE_HEIGHT - 1080) // 2   # = 60 для 1200

# ── FFmpeg ─────────────────────────────────────────────────────────────────────

VIDEO_CODEC    = "libx264"
VIDEO_CRF      = 23
VIDEO_PRESET   = "ultrafast"
AUDIO_CODEC    = "aac"
AUDIO_BITRATE  = "192k"
FFMPEG_THREADS = 0   # 0 = auto

# ── PiP (горизонтальный формат) ────────────────────────────────────────────────

PIP_WIDTH         = 350
PIP_HEIGHT        = 350
PIP_MARGIN_RIGHT  = 60
PIP_MARGIN_BOTTOM = 60

# ── Поведение ──────────────────────────────────────────────────────────────────

CLEANUP_TEMP_ON_SUCCESS       = True
DELETE_INPUT_AFTER_PROCESSING = True

# ── Telegram ───────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

# ── Google Drive (rclone) ──────────────────────────────────────────────────────

RCLONE_REMOTE_NAME    = "gdrive"
RCLONE_YD_INPUT_PATH  = "PROJECTS/Автомонтаж/input"
RCLONE_YD_OUTPUT_PATH = "PROJECTS/Автомонтаж/output"
