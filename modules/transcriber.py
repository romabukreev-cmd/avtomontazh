"""
transcriber.py — транскрипция аудио и обнаружение пауз.

Логика:
  1. FFmpeg извлекает аудио из видеофайла в WAV (16kHz, моно — оптимум для Whisper)
  2. Whisper транскрибирует с word_timestamps=True — каждое слово имеет свой таймстемп
  3. По таймстемпам слов находим паузы: промежуток >= PAUSE_THRESHOLD_SEC без слов
  4. Из пауз собираем «сегменты речи» — непрерывные блоки говорения
  5. Слишком короткие сегменты (< MIN_SEGMENT_DURATION_SEC) отбрасываются

Важно:
  - Паузы определяются по словам Whisper, а не по уровню громкости аудио
  - Whisper работает только с screen_file (там лучше аудио — системный звук + микрофон)
  - Получившиеся временны́е отрезки применяются к обоим файлам (экран + вебка синхронны)
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List

from faster_whisper import WhisperModel

import config

log = logging.getLogger(__name__)


class Transcriber:

    def __init__(self):
        # Модель загружается один раз при создании объекта
        log.info(f"Загрузка Whisper модели '{config.WHISPER_MODEL}'...")
        self._model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8",  # int8 квантизация: +30% скорость, качество не страдает
        )
        log.info("Whisper готов")

    def transcribe_and_cut_pauses(self, video_file: Path) -> List[Dict]:
        """
        Главный метод: принимает видеофайл, возвращает список сегментов без пауз.

        Паузы определяются по промежуткам между словами Whisper (не по громкости аудио).

        Returns:
            [
                {"start": 0.0,  "end": 12.4, "text": "Хорошо, попробуем вот так..."},
                {"start": 14.1, "end": 28.7, "text": "Этот цвет мне нравится..."},
                ...
            ]
        """
        audio_file = self._extract_audio(video_file)
        try:
            words    = self._transcribe(audio_file)
            segments = self._split_by_pauses(words)
        finally:
            audio_file.unlink(missing_ok=True)

        log.info(f"Итого сегментов: {len(segments)}")
        return segments

    # ── Шаг 1: Извлечение аудио ───────────────────────────────────────────────

    def _extract_audio(self, video_file: Path) -> Path:
        """
        FFmpeg: video → WAV (16000Hz, моно).
        16kHz моно — стандарт для Whisper, меньше места и быстрее обработка.
        """
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = config.TEMP_DIR / f"{video_file.stem}_audio.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-ar", "16000",   # частота дискретизации 16 кГц
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

    # ── Шаг 2: Транскрипция ───────────────────────────────────────────────────

    def _transcribe(self, audio_file: Path) -> List[Dict]:
        """
        Whisper: WAV → список слов с таймстемпами.

        Returns:
            [{"word": "Хорошо", "start": 0.12, "end": 0.54}, ...]
        """
        log.info(f"Транскрипция ({config.WHISPER_MODEL}, язык: {config.WHISPER_LANGUAGE})...")

        segments_gen, _info = self._model.transcribe(
            str(audio_file),
            language=config.WHISPER_LANGUAGE,
            word_timestamps=True,
            vad_filter=True,   # Silero VAD: только реальная речь → точные таймстемпы слов
        )

        # Извлекаем плоский список слов из всех сегментов Whisper
        # faster-whisper возвращает генератор объектов (не dict), атрибуты через точку
        words = []
        for seg in segments_gen:
            seg_words = seg.words or []

            if seg_words:
                # Есть пословные таймстемпы — используем их
                for w in seg_words:
                    if w.start is not None and w.end is not None:
                        words.append({
                            "word":  w.word.strip(),
                            "start": float(w.start),
                            "end":   float(w.end),
                        })
            else:
                # Нет пословных таймстемпов — используем весь сегмент как одно слово
                words.append({
                    "word":  seg.text.strip(),
                    "start": float(seg.start),
                    "end":   float(seg.end),
                })

        log.info(f"Распознано слов: {len(words)}")
        return words

    # ── Шаг 3: Нарезка по паузам (по словам Whisper) ─────────────────────────

    def _split_by_pauses(self, words: List[Dict]) -> List[Dict]:
        """
        Определяет паузы по промежуткам между словами Whisper.
        Пауза = промежуток >= PAUSE_THRESHOLD_SEC без слов.
        Не зависит от уровня громкости аудио.
        """
        if not words:
            return []

        segments: List[Dict] = []
        current_words = [words[0]]

        for word in words[1:]:
            gap = word["start"] - current_words[-1]["end"]
            if gap >= config.PAUSE_THRESHOLD_SEC:
                self._flush_segment(current_words, segments)
                current_words = [word]
            else:
                current_words.append(word)

        self._flush_segment(current_words, segments)

        log.info(
            f"Сегментов после нарезки: {len(segments)} "
            f"(порог паузы: {config.PAUSE_THRESHOLD_SEC}с)"
        )
        return segments

    def _flush_segment(self, words: List[Dict], out: List[Dict]) -> None:
        """Собирает список слов в сегмент и добавляет в out (если достаточно длинный)."""
        if not words:
            return
        duration = words[-1]["end"] - words[0]["start"]
        if duration < config.MIN_SEGMENT_DURATION_SEC:
            return
        start = max(0.0, words[0]["start"] - config.SEGMENT_START_PADDING_SEC)
        end   = words[-1]["end"]
        text  = " ".join(w["word"] for w in words if w["word"]).strip()
        out.append({"start": round(start, 3), "end": round(end, 3), "text": text})
