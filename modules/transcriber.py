"""
transcriber.py — транскрипция видео через Groq Whisper API.

Groq блокирует российские IP, поэтому запросы идут через Cloudflare
Worker-прокси (адрес и секрет в .env). При пустом GROQ_PROXY_URL идёт
прямо в api.groq.com (полезно для отладки с не-РФ IP).

Аудио извлекается в FLAC mono 16kHz — сжатие 6-8× против WAV, помещается
в лимит 25 MB Free tier'а даже для часовых видео.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)


class Transcriber:

    def transcribe(self, video_path: Path) -> List[Dict]:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = config.TEMP_DIR / f"{video_path.stem}_audio.flac"

        log.info(f"Извлекаю аудио из {video_path.name}...")
        self._extract_audio(video_path, audio_path)

        size_mb = audio_path.stat().st_size / 1024 / 1024
        log.info(f"Аудио готово: {audio_path.name} ({size_mb:.1f} MB)")
        if size_mb > config.GROQ_MAX_FILE_MB:
            audio_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Аудио {size_mb:.1f} MB > лимит Groq {config.GROQ_MAX_FILE_MB} MB. "
                f"Разбей видео на части или понизь битрейт."
            )

        log.info(f"Транскрипция через Groq ({config.WHISPER_MODEL})...")
        try:
            data = self._call_groq(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        segments = _build_segments(data.get("segments", []), data.get("words", []))
        log.info(f"Получено {len(segments)} сегментов")
        return segments

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "flac",
            "-compression_level", "8",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка извлечения аудио:\n{result.stderr[-1000:]}")

    def _call_groq(self, audio_path: Path) -> dict:
        url = f"{config.GROQ_PROXY_URL.rstrip('/')}/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        if config.GROQ_PROXY_SECRET:
            headers["X-Proxy-Secret"] = config.GROQ_PROXY_SECRET

        with open(audio_path, "rb") as f:
            content = f.read()
        files = {"file": (audio_path.name, content, "audio/flac")}
        data = {
            "model": config.WHISPER_MODEL,
            "language": config.WHISPER_LANGUAGE,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
            "temperature": "0",
        }
        with httpx.Client(timeout=600.0) as client:
            r = client.post(url, headers=headers, files=files, data=data)
        if r.status_code != 200:
            raise RuntimeError(f"Groq API {r.status_code}: {r.text[:500]}")
        return r.json()


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _dedup_consecutive(words: List[Dict]) -> List[Dict]:
    """Убирает подряд идущие одинаковые слова с паузой < 0.5с между ними."""
    if not words:
        return words
    result = [words[0]]
    for w in words[1:]:
        prev = result[-1]
        same = w["word"].strip().lower() == prev["word"].strip().lower()
        gap  = w["start"] - prev["end"]
        if same and gap < 0.5:
            continue
        result.append(w)
    return result


def _build_segments(segments_raw: List[Dict], words_raw: List[Dict]) -> List[Dict]:
    """
    Groq/OpenAI Whisper API отдаёт сегменты и слова плоскими списками.
    Распределяем слова по сегментам по попаданию во временной диапазон,
    затем приравниваем границы сегмента к границам его первого/последнего слова.
    """
    result = []
    wi = 0
    n_words = len(words_raw)

    for idx, seg in enumerate(segments_raw):
        seg_end = seg["end"]
        seg_words: List[Dict] = []
        while wi < n_words and words_raw[wi]["start"] < seg_end:
            w = words_raw[wi]
            if w["start"] >= seg["start"]:
                seg_words.append({
                    "word":  w["word"],
                    "start": w["start"],
                    "end":   w["end"],
                })
            wi += 1

        seg_words = _dedup_consecutive(seg_words)

        if seg_words:
            s_start = seg_words[0]["start"]
            s_end   = seg_words[-1]["end"]
        else:
            s_start = seg["start"]
            s_end   = seg["end"]

        result.append({
            "index": idx,
            "start": round(s_start, 3),
            "end":   round(s_end,   3),
            "text":  (seg.get("text") or "").strip(),
            "words": seg_words,
        })
    return result
