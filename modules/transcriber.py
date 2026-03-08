"""
transcriber.py — транскрипция аудио через Whisper.

Логика:
  1. FFmpeg извлекает аудио из видеофайла в WAV (16kHz, моно)
  2. Whisper транскрибирует с word_timestamps=True и vad_filter=True
  3. Возвращает сегменты Whisper с пословными таймстемпами

Модель загружается лениво: только когда приходит задача, и выгружается
сразу после завершения транскрипции, чтобы не занимать RAM постоянно.
"""

import ctypes
import gc
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from faster_whisper import WhisperModel

import config

log = logging.getLogger(__name__)


class Transcriber:

    def __init__(self):
        self._model: Optional[WhisperModel] = None
        log.info(f"Transcriber создан (модель '{config.WHISPER_MODEL}' загружается по требованию)")

    def transcribe(self, video_file: Path) -> List[Dict]:
        """
        Транскрибирует видеофайл, возвращает сегменты Whisper с пословными таймстемпами.

        Returns:
            [
                {
                    "index": 0,
                    "start": 1.52, "end": 14.3,
                    "text": "Итак, давайте нарисуем карточку...",
                    "words": [{"word": "Итак", "start": 1.52, "end": 1.8}, ...]
                },
                ...
            ]
        """
        audio_file = self._extract_audio(video_file)
        try:
            self._load_model()
            segments = self._transcribe_segments(audio_file)
        finally:
            audio_file.unlink(missing_ok=True)
            self._unload_model()

        log.info(f"Итого сегментов: {len(segments)}")
        return segments

    # ── Шаг 1: Извлечение аудио ───────────────────────────────────────────────

    def _extract_audio(self, video_file: Path) -> Path:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = config.TEMP_DIR / f"{video_file.stem}_audio.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-ar", "16000",   # 16 кГц — стандарт для Whisper
            "-ac", "1",       # моно
            "-vn",            # без видео
            str(audio_path),
        ]

        log.info(f"Извлечение аудио: {video_file.name} → {audio_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg не смог извлечь аудио из {video_file.name}:\n{result.stderr[-1000:]}"
            )

        return audio_path

    # ── Шаг 2: Загрузка / выгрузка модели ────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        log.info(f"Загрузка Whisper '{config.WHISPER_MODEL}'...")
        self._model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8",  # int8 квантизация: +30% скорость, качество не страдает
        )
        log.info("Whisper загружен")

    def _unload_model(self) -> None:
        if self._model is None:
            return
        log.info("Выгрузка Whisper из памяти...")
        del self._model
        self._model = None
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # возвращаем RAM операционной системе
        except Exception:
            pass
        log.info("Whisper выгружен")

    # ── Шаг 3: Транскрипция ───────────────────────────────────────────────────

    def _transcribe_segments(self, audio_file: Path) -> List[Dict]:
        """
        Whisper: WAV → сегменты с пословными таймстемпами.
        Сегменты Whisper соответствуют естественным речевым блокам (предложениям/фразам).
        """
        log.info(f"Транскрипция ({config.WHISPER_MODEL}, язык: {config.WHISPER_LANGUAGE})...")

        segments_gen, _info = self._model.transcribe(
            str(audio_file),
            language=config.WHISPER_LANGUAGE,
            word_timestamps=True,
            vad_filter=True,                      # Silero VAD: только реальная речь
            condition_on_previous_text=False,     # отключаем bias — меньше галлюцинаций
            hallucination_silence_threshold=2.0,  # не повторять текст в паузах > 2с
        )

        result = []
        for i, seg in enumerate(segments_gen):
            words = []
            for w in (seg.words or []):
                if w.start is not None and w.end is not None:
                    words.append({
                        "word":  w.word.strip(),
                        "start": float(w.start),
                        "end":   float(w.end),
                    })

            text = seg.text.strip()
            if not words and not text:
                continue

            result.append({
                "index": i,
                "start": round(float(seg.start), 3),
                "end":   round(float(seg.end), 3),
                "text":  text,
                "words": words,
            })

        log.info(f"Распознано сегментов: {len(result)}")
        return result
