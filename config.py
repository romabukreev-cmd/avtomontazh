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

# Входные файлы — уже готовые вертикальные видео (экран+вебка склеены заранее,
# например в OBS), пронумерованные части одной записи: 001.mp4, 002.mp4, ...
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".ts"}

# ── Whisper (Groq API через Cloudflare Worker-прокси) ─────────────────────────

# ── Deepgram ──────────────────────────────────────────────────────────────────

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL   = "nova-2"
DEEPGRAM_LANGUAGE = "ru"

# ── Пороги ────────────────────────────────────────────────────────────────────

PAUSE_CUT_SEC = 0.5    # пауза >= этого → разрыв между блоками слов
# Буфер по краям каждого отрезка. Нужен запас больше, чем возможная неточность
# word-level таймкодов Whisper (turbo-модель) — иначе срез попадает на живое
# слово вместо паузы.
TIMELINE_START_PAD_SEC = 0.35
TIMELINE_END_PAD_SEC   = 0.35

# ── LLM ───────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY", "")
# OpenRouter тоже режет запросы с RU IP (403 "Access denied by security
# policy") — прокси через Cloudflare Worker по тому же принципу, что и Groq.
OPENROUTER_PROXY_URL    = os.getenv("OPENROUTER_PROXY_URL", "https://openrouter.ai")
OPENROUTER_PROXY_SECRET = os.getenv("OPENROUTER_PROXY_SECRET", "")
LLM_MODEL          = "anthropic/claude-sonnet-4-6"
LLM_MAX_RETRIES    = 3

VIDEO_CONTEXT = (
    "Автор записывает процесс дизайна: сначала создаёт карточку товара (Figma/Photoshop), "
    "затем делает анимацию. Иногда встречаются рекламные интеграции. "
    "Видео должно охватывать весь процесс от начала до конца."
)

# ── FFmpeg ─────────────────────────────────────────────────────────────────────

SCREEN_CROP_Y  = int(os.getenv("SCREEN_CROP_Y", "60"))   # пикселей срезать сверху с экранной записи

VIDEO_CODEC    = "libx264"
VIDEO_CRF      = 23
VIDEO_PRESET   = "ultrafast"
AUDIO_CODEC    = "aac"
AUDIO_BITRATE  = "192k"
FFMPEG_THREADS = 0   # 0 = auto

# ── Конкатенация входных частей сессии ─────────────────────────────────────────
# Перекодируем (не -c copy), чтобы части с разными настройками энкодера
# (например смена кодировщика в OBS между записями) не давали рассинхрон
# видео/аудио на выходе — copy требует идентичных потоков во всех частях.
CONCAT_FPS        = 30
CONCAT_AUDIO_RATE = 48000
CONCAT_PIX_FMT    = "yuv420p"
CONCAT_CRF        = 18
CONCAT_PRESET     = "veryfast"

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
RCLONE_SYNC_TIMEOUT_SEC   = int(os.getenv("RCLONE_SYNC_TIMEOUT_SEC", "7200"))
RCLONE_UPLOAD_TIMEOUT_SEC = int(os.getenv("RCLONE_UPLOAD_TIMEOUT_SEC", "7200"))
