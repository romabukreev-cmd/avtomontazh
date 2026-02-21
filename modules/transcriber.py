"""
transcriber.py — транскрипция аудио и обнаружение пауз.

Логика:
  1. FFmpeg извлекает аудио из видеофайла в WAV (16kHz, моно — оптимум для Whisper)
  2. Whisper транскрибирует с word_timestamps=True — каждое слово имеет свой таймстемп
  3. По таймстемпам слов находим промежутки тишины (пауза = нет слов N секунд)
  4. Из промежутков тишины собираем «сегменты речи» — непрерывные блоки говорения
  5. Слишком короткие сегменты (< MIN_SEGMENT_DURATION_SEC) отбрасываются

Важно:
  - Whisper работает только с screen_file (там лучше аудио — системный звук + микрофон)
  - Получившиеся временны́е отрезки применяются к обоим файлам (экран + вебка синхронны)
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import whisper

import config

log = logging.getLogger(__name__)


class Transcriber:

    def __init__(self):
        # Модель загружается один раз при создании объекта
        log.info(f"Загрузка Whisper модели '{config.WHISPER_MODEL}'...")
        self._model = whisper.load_model(config.WHISPER_MODEL)
        log.info("Whisper готов")

    def transcribe_and_cut_pauses(self, video_file: Path) -> List[Dict]:
        """
        Главный метод: принимает видеофайл, возвращает список сегментов без пауз.

        Returns:
            [
                {"start": 0.0,  "end": 12.4, "text": "Хорошо, попробуем вот так..."},
                {"start": 14.1, "end": 28.7, "text": "Этот цвет мне нравится..."},
                ...
            ]
        """
        audio_file = self._extract_audio(video_file)
        try:
            words = self._transcribe(audio_file)
            segments = self._split_by_pauses(words)
        finally:
            # Удаляем WAV после работы — он тяжёлый
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

        result = self._model.transcribe(
            str(audio_file),
            language=config.WHISPER_LANGUAGE,
            word_timestamps=True,
            verbose=False,
        )

        # Извлекаем плоский список слов из всех сегментов Whisper
        words = []
        for segment in result.get("segments", []):
            segment_words = segment.get("words", [])

            if segment_words:
                # Есть пословные таймстемпы — используем их
                for w in segment_words:
                    if w.get("start") is not None and w.get("end") is not None:
                        words.append({
                            "word":  w["word"].strip(),
                            "start": float(w["start"]),
                            "end":   float(w["end"]),
                        })
            else:
                # Нет пословных таймстемпов — используем весь сегмент как одно слово
                words.append({
                    "word":  segment.get("text", "").strip(),
                    "start": float(segment["start"]),
                    "end":   float(segment["end"]),
                })

        log.info(f"Распознано слов: {len(words)}")
        return words

    # ── Шаг 3: Нарезка по паузам ─────────────────────────────────────────────

    def _split_by_pauses(self, words: List[Dict]) -> List[Dict]:
        """
        По списку слов находит паузы и нарезает сегменты.

        Алгоритм:
          - Идём по словам
          - Если разрыв между концом предыдущего слова и началом следующего
            больше PAUSE_THRESHOLD_SEC — закрываем текущий сегмент
          - Собираем текст сегмента, фильтруем слишком короткие
        """
        if not words:
            log.warning("Слов не найдено — аудио пустое или тихое")
            return []

        segments = []
        seg_start   = words[0]["start"]
        seg_words   = [words[0]]

        for prev, curr in zip(words, words[1:]):
            gap = curr["start"] - prev["end"]

            if gap > config.PAUSE_THRESHOLD_SEC:
                # Пауза — закрываем сегмент
                self._close_segment(segments, seg_start, prev["end"], seg_words)
                seg_start = curr["start"]
                seg_words = [curr]
            else:
                seg_words.append(curr)

        # Закрываем последний сегмент
        self._close_segment(segments, seg_start, words[-1]["end"], seg_words)

        log.info(
            f"Сегментов после нарезки: {len(segments)} "
            f"(порог паузы: {config.PAUSE_THRESHOLD_SEC}с, "
            f"мин. длина: {config.MIN_SEGMENT_DURATION_SEC}с)"
        )
        return segments

    def _close_segment(
        self,
        segments: List[Dict],
        start: float,
        end: float,
        words: List[Dict],
    ) -> None:
        """Финализирует сегмент и добавляет в список если он достаточно длинный."""
        duration = end - start
        if duration < config.MIN_SEGMENT_DURATION_SEC:
            return

        text = " ".join(w["word"] for w in words if w["word"]).strip()
        segments.append({
            "start": round(start, 3),
            "end":   round(end, 3),
            "text":  text,
        })
