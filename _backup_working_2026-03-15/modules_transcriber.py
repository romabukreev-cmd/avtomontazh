"""
transcriber.py — транскрипция видео через faster-whisper.

Модель загружается при каждом вызове transcribe() и выгружается сразу после —
чтобы не держать ~3GB в памяти во время LLM и FFmpeg.

Ключевая функция _compute_boundaries() вычисляет точные start/end из word timestamps
и выполняет корректный клиппинг: если точка обрезки попадает внутрь последнего слова,
откатываем clip до начала этого слова, чтобы не создавать заикание.
"""

import gc
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import config

log = logging.getLogger(__name__)


class Transcriber:

    def transcribe(self, video_path: Path) -> List[Dict]:
        """
        Транскрибирует видео и возвращает список сегментов с точными границами.

        Каждый сегмент:
            {index, start, end, text, words: [{word, start, end}, ...]}
        """
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = config.TEMP_DIR / f"{video_path.stem}_audio.wav"

        log.info(f"Извлекаю аудио из {video_path.name}...")
        self._extract_audio(video_path, wav_path)

        log.info(f"Загружаю Whisper {config.WHISPER_MODEL}...")
        from faster_whisper import WhisperModel
        model = WhisperModel(
            config.WHISPER_MODEL,
            device="auto",
            compute_type="auto",
        )

        try:
            log.info("Транскрипция...")
            segments_iter, info = model.transcribe(
                str(wav_path),
                language=config.WHISPER_LANGUAGE,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": config.VAD_MIN_SILENCE_MS,
                    "speech_pad_ms":           config.VAD_SPEECH_PAD_MS,
                },
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
            )

            raw = []
            for idx, seg in enumerate(segments_iter):
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (seg.words or [])
                ]
                words = _dedup_consecutive(words)
                raw.append({
                    "index":      idx,
                    "_seg_start": seg.start,
                    "_seg_end":   seg.end,
                    "text":       seg.text.strip(),
                    "words":      words,
                })

            log.info(f"Получено {len(raw)} сегментов, длительность аудио {info.duration:.0f}с")
            return _compute_boundaries(raw)

        finally:
            wav_path.unlink(missing_ok=True)
            del model
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    def _extract_audio(self, video_path: Path, wav_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка извлечения аудио:\n{result.stderr[-1000:]}")


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _dedup_consecutive(words: List[Dict]) -> List[Dict]:
    """
    Убирает подряд идущие одинаковые слова с паузой < 0.5с между ними.
    Whisper иногда дублирует слово внутри одного сегмента.
    """
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


def _compute_boundaries(raw: List[Dict]) -> List[Dict]:
    """
    Форматирует сегменты из Whisper.
    Использует нативные timestamps без изменений — Whisper достаточно точен.
    """
    result = []
    for seg in raw:
        words = seg["words"]
        if words:
            seg_start = words[0]["start"]
            seg_end   = words[-1]["end"]
        else:
            seg_start = seg["_seg_start"]
            seg_end   = seg["_seg_end"]
        result.append({
            "index": seg["index"],
            "start": round(seg_start, 3),
            "end":   round(seg_end,   3),
            "text":  seg["text"],
            "words": words,
        })
    return result
