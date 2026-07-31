"""
transcriber.py — транскрипция видео через Deepgram API.

Аудио извлекается в FLAC mono 16kHz перед отправкой.
Deepgram не имеет лимита на размер файла — chunking не нужен.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)

_DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


class Transcriber:

    def transcribe(self, video_path: Path) -> List[Dict]:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = config.TEMP_DIR / f"{video_path.stem}_audio.flac"

        log.info(f"Извлекаю аудио из {video_path.name}...")
        self._extract_audio(video_path, audio_path)

        size_mb = audio_path.stat().st_size / 1024 / 1024
        log.info(f"Аудио готово: {audio_path.name} ({size_mb:.1f} MB)")

        try:
            log.info(f"Транскрипция через Deepgram ({config.DEEPGRAM_MODEL})...")
            data = self._call_deepgram(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        segments = _parse_response(data)
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

    def _call_deepgram(self, audio_path: Path) -> dict:
        params = {
            "model":       config.DEEPGRAM_MODEL,
            "language":    config.DEEPGRAM_LANGUAGE,
            "utterances":  "true",
            "words":       "true",
            "punctuate":   "true",
        }
        headers = {
            "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
            "Content-Type":  "audio/flac",
        }
        with open(audio_path, "rb") as f:
            content = f.read()

        with httpx.Client(timeout=600.0) as client:
            r = client.post(_DEEPGRAM_URL, params=params, headers=headers, content=content)

        if r.status_code != 200:
            raise RuntimeError(f"Deepgram API {r.status_code}: {r.text[:500]}")
        return r.json()


# ── Парсинг ответа ────────────────────────────────────────────────────────────

def _parse_response(data: dict) -> List[Dict]:
    utterances = data.get("results", {}).get("utterances") or []

    # Fallback: если utterances пусты — строим один сегмент из всех слов
    if not utterances:
        alt = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
        )
        words = _extract_words(alt.get("words", []))
        if not words:
            return []
        return [{
            "index": 0,
            "start": words[0]["start"],
            "end":   words[-1]["end"],
            "text":  alt.get("transcript", ""),
            "words": words,
        }]

    result = []
    for idx, utt in enumerate(utterances):
        words = _extract_words(utt.get("words", []))
        result.append({
            "index": idx,
            "start": round(utt["start"], 3),
            "end":   round(utt["end"],   3),
            "text":  utt.get("transcript", ""),
            "words": words,
        })
    return result


def _extract_words(raw: list) -> List[Dict]:
    return [
        {
            "word":  w.get("punctuated_word") or w.get("word", ""),
            "start": round(w.get("start", 0.0), 3),
            "end":   round(w.get("end",   0.0), 3),
        }
        for w in raw
        if w.get("start") is not None
    ]
