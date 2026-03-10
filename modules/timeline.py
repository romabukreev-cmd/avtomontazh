"""
timeline.py — разбивает Whisper-слова на речевые блоки по паузам.

Блок = непрерывная речь без пауз >= VAD_MIN_SILENCE_MS.
Паузы между блоками вырезаются автоматически — в финальный таймлайн
попадают только сами блоки (LLM решает какие оставить, какие удалить).

Мелкие блоки (одна фраза) дают LLM нужную гранулярность для поиска перезаписей.
"""

import logging
from typing import Dict, List

import config

log = logging.getLogger(__name__)

_MAX_WORD_DURATION = 1.5  # сек: cap на длину слова — защита от плохих Whisper-таймстемпов


def build_blocks(whisper_segments: List[Dict]) -> List[Dict]:
    """
    Из Whisper-сегментов строит речевые блоки.

    Алгоритм:
      1. Собираем все слова из всех сегментов в единый поток
      2. Глобальная дедупликация — убираем кросс-сегментные галлюцинации Whisper
      3. Разбиваем по паузам >= VAD_MIN_SILENCE_MS → речевые блоки
      4. Каждый блок получает start/end с буфером VAD_SPEECH_PAD_MS
      5. Клиппинг буферов: блок не может заходить в территорию соседнего блока

    Мелкая нарезка (по словесным паузам) даёт LLM видимость перезаписей.

    Returns:
        [{"index": 0, "start": 0.3, "end": 5.0, "text": "...", "words": [...]}, ...]
    """
    all_words = []
    for seg in whisper_segments:
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                all_words.append(w)

    if not all_words:
        return []

    all_words.sort(key=lambda w: w["start"])

    start_buf = config.BLOCK_START_BUFFER_SEC  # маленький: не тянуть вдохи/хвосты
    end_buf   = config.BLOCK_END_BUFFER_SEC    # нормальный: естественное затухание слова
    thr       = config.VAD_MIN_SILENCE_MS / 1000.0  # мс → секунды

    blocks = []
    current = [all_words[0]]

    for word in all_words[1:]:
        # Используем кэпнутый end — иначе раздутый Whisper-таймстемп скрывает реальную паузу
        last_end = min(current[-1]["end"], current[-1]["start"] + _MAX_WORD_DURATION)
        gap = word["start"] - last_end
        if gap >= thr:
            blocks.append(_words_to_block(current, start_buf, end_buf))
            current = [word]
        else:
            current.append(word)
    blocks.append(_words_to_block(current, start_buf, end_buf))

    for i, b in enumerate(blocks):
        b["index"] = i

    log.info(f"Речевых блоков: {len(blocks)} (порог: {thr}с, start_buf: {start_buf}с, end_buf: {end_buf}с)")
    return blocks


def _words_to_block(words: List[Dict], start_buf: float, end_buf: float) -> Dict:
    last = words[-1]
    capped_end = min(last["end"], last["start"] + _MAX_WORD_DURATION)
    return {
        "start": max(0.0, words[0]["start"] - start_buf),
        "end":   capped_end + end_buf,
        "text":  " ".join(w["word"] for w in words).strip(),
        "words": words,
    }


def total_duration(blocks: List[Dict]) -> float:
    """Суммарная длительность блоков в секундах."""
    return sum(b["end"] - b["start"] for b in blocks)
