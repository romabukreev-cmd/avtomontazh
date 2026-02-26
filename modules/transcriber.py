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
import re
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

        Использует FFmpeg silencedetect для точного определения пауз (как в Premiere Pro),
        Whisper — только для транскрипции текста.

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
            segments = self._split_by_pauses(words, audio_file)
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

    # ── Шаг 3: Нарезка по паузам (через FFmpeg silencedetect) ───────────────────

    def _split_by_pauses(self, words: List[Dict], audio_file: Path) -> List[Dict]:
        """
        Определяет паузы через FFmpeg silencedetect (точно, как в Premiere Pro),
        строит речевые сегменты из не-тихих областей,
        заполняет текстом из Whisper-слов.
        """
        silence_regions = self._detect_silence_regions(audio_file)
        audio_duration  = self._get_audio_duration(audio_file)

        if not silence_regions:
            # Тишины не найдено — весь аудио одним сегментом
            text = " ".join(w["word"] for w in words if w["word"]).strip()
            if audio_duration >= config.MIN_SEGMENT_DURATION_SEC:
                return [{"start": 0.0, "end": round(audio_duration, 3), "text": text}]
            return []

        # Строим речевые интервалы — всё что НЕ тишина
        speech_regions = []
        cursor = 0.0
        for silence in sorted(silence_regions, key=lambda x: x["start"]):
            if silence["start"] - cursor >= config.MIN_SEGMENT_DURATION_SEC:
                speech_regions.append({"start": cursor, "end": silence["start"]})
            cursor = silence["end"]
        if audio_duration - cursor >= config.MIN_SEGMENT_DURATION_SEC:
            speech_regions.append({"start": cursor, "end": audio_duration})

        # Назначаем Whisper-текст каждому речевому интервалу
        segments = []
        for region in speech_regions:
            region_words = [
                w for w in words
                if w["end"] > region["start"] - 0.1 and w["start"] < region["end"] + 0.1
            ]
            text = " ".join(w["word"] for w in region_words if w["word"]).strip()

            # Начинаем сегмент с первого слова Whisper (минус небольшой зазор),
            # а не с момента когда FFmpeg засёк первый звук.
            # Это убирает вдохи/дыхание в начале каждого сегмента.
            if region_words:
                first_word_start = region_words[0]["start"]
                seg_start = max(region["start"], first_word_start - config.SEGMENT_START_PADDING_SEC)
            else:
                seg_start = region["start"]

            segments.append({
                "start": round(seg_start,       3),
                "end":   round(region["end"],   3),
                "text":  text,
            })

        log.info(
            f"Сегментов после нарезки: {len(segments)} "
            f"(порог паузы: {config.PAUSE_THRESHOLD_SEC}с, "
            f"уровень шума: {config.SILENCE_NOISE_DB}dB)"
        )
        return segments

    def _detect_silence_regions(self, audio_file: Path) -> List[Dict]:
        """
        FFmpeg silencedetect: находит все тихие участки в аудио.
        Возвращает список {start, end} тихих периодов.
        """
        cmd = [
            "ffmpeg", "-i", str(audio_file),
            "-af", f"silencedetect=noise={config.SILENCE_NOISE_DB}dB:duration={config.PAUSE_THRESHOLD_SEC}",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        silence_regions = []
        current_start   = None
        for line in result.stderr.split("\n"):
            s = re.search(r"silence_start: (\d+\.?\d*)", line)
            e = re.search(r"silence_end: (\d+\.?\d*)",   line)
            if s:
                current_start = float(s.group(1))
            if e and current_start is not None:
                silence_regions.append({"start": current_start, "end": float(e.group(1))})
                current_start = None

        # Аудио закончилось в тишине
        if current_start is not None:
            silence_regions.append({"start": current_start, "end": float("inf")})

        log.info(f"Найдено пауз: {len(silence_regions)}")
        return silence_regions

    def _get_audio_duration(self, audio_file: Path) -> float:
        """ffprobe: длительность аудиофайла в секундах."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0
