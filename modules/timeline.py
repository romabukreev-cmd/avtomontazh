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

    # Глобальная дедупликация: убираем одинаковые слова на стыках Whisper-сегментов.
    # Whisper иногда повторяет последнее слово сегмента в начале следующего.
    # per-segment dedup в transcriber.py не ловит эти кросс-сегментные дубли.
    all_words = _dedup_words(all_words)

    if not all_words:
        return []

    buf = config.VAD_SPEECH_PAD_MS / 1000.0   # мс → секунды
    thr = config.VAD_MIN_SILENCE_MS / 1000.0  # мс → секунды

    blocks = []
    current = [all_words[0]]

    for word in all_words[1:]:
        # Используем кэпнутый end — иначе раздутый Whisper-таймстемп скрывает реальную паузу
        last_end = min(current[-1]["end"], current[-1]["start"] + _MAX_WORD_DURATION)
        gap = word["start"] - last_end
        if gap >= thr:
            blocks.append(_words_to_block(current, buf))
            current = [word]
        else:
            current.append(word)
    blocks.append(_words_to_block(current, buf))

    # Клиппинг буферов: блок не может заходить в область слов соседнего блока.
    # Защита от неточных Whisper-таймстемпов на границах блоков.
    for i, block in enumerate(blocks):
        if i + 1 < len(blocks):
            next_word_start = blocks[i + 1]["words"][0]["start"]
            if block["end"] > next_word_start:
                block["end"] = next_word_start
        if i > 0:
            prev = blocks[i - 1]["words"][-1]
            prev_capped_end = min(prev["end"], prev["start"] + _MAX_WORD_DURATION)
            if block["start"] < prev_capped_end:
                block["start"] = prev_capped_end

    for i, b in enumerate(blocks):
        b["index"] = i

    log.info(f"Речевых блоков: {len(blocks)} (порог: {thr}с, буфер: {buf}с)")
    return blocks


def _words_to_block(words: List[Dict], buf: float) -> Dict:
    last = words[-1]
    capped_end = min(last["end"], last["start"] + _MAX_WORD_DURATION)
    return {
        "start": max(0.0, words[0]["start"] - buf),
        "end":   capped_end + buf,
        "text":  " ".join(w["word"] for w in words).strip(),
        "words": words,
    }


def total_duration(blocks: List[Dict]) -> float:
    """Суммарная длительность блоков в секундах."""
    return sum(b["end"] - b["start"] for b in blocks)


def _dedup_words(words: List[Dict]) -> List[Dict]:
    """
    Глобальная дедупликация слов после объединения всех Whisper-сегментов.

    Whisper иногда повторяет одно и то же слово на стыке сегментов:
    конец сегмента N и начало сегмента N+1 содержат одно слово.
    Per-segment dedup в transcriber.py это не ловит.

    Порог 0.3с — достаточно для граничных галлюцинаций, не удаляет реальные повторы.
    """
    if not words:
        return words
    result = [words[0]]
    for w in words[1:]:
        if (w["word"].lower() == result[-1]["word"].lower()
                and w["start"] - result[-1]["end"] < 0.3):
            log.debug(f"Global dedup: убран '{w['word']}' @ {w['start']:.2f}s")
            continue
        result.append(w)
    return result
