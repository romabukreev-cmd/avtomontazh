"""
timeline.py — конвертирует Whisper-сегменты в речевые блоки.

Каждый сегмент от Whisper = один блок. Границы сегментов определяются Silero VAD
(min_silence_duration_ms=VAD_MIN_SILENCE_MS), поэтому start/end надёжны и отражают
реальные паузы в аудио — независимо от точности Whisper word-timestamps.

Word-gap сплиттинг больше не нужен.
"""

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def build_blocks(whisper_segments: List[Dict]) -> List[Dict]:
    """
    Из Whisper-сегментов строит речевые блоки.

    Каждый сегмент → блок с теми же start/end (уже надёжные — из Silero VAD).
    Пустые сегменты пропускаются.

    Returns:
        [{"index": 0, "start": 0.3, "end": 5.0, "text": "...", "words": [...]}, ...]
    """
    blocks = []
    for seg in whisper_segments:
        text  = seg.get("text", "").strip()
        words = seg.get("words", [])
        if not text and not words:
            continue
        blocks.append({
            "index": len(blocks),
            "start": seg["start"],
            "end":   seg["end"],
            "text":  text,
            "words": words,
        })

    log.info(f"Речевых блоков: {len(blocks)}")
    return blocks


def total_duration(blocks: List[Dict]) -> float:
    """Суммарная длительность блоков в секундах."""
    return sum(b["end"] - b["start"] for b in blocks)
