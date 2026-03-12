"""
timeline.py — разбивает Whisper-слова на речевые блоки по паузам.

Блок = непрерывная речь без пауз >= PAUSE_THRESHOLD_SEC.
Паузы между блоками вырезаются автоматически — в финальный таймлайн
попадают только сами блоки (LLM решает какие оставить, какие удалить).
"""

import logging
from typing import Dict, List

import config

log = logging.getLogger(__name__)

# Cap на длину слова: защита от раздутых Whisper-таймстемпов при детекции пауз.
# Например Whisper иногда отмечает "да" как длиной 4с — это скрывает реальную паузу.
_MAX_WORD_DURATION = 1.5  # секунды


def build_blocks(whisper_segments: List[Dict]) -> List[Dict]:
    """
    Из Whisper-сегментов строит речевые блоки.

    Алгоритм:
      1. Собираем все слова из всех сегментов в единый поток
      2. Разбиваем по паузам >= PAUSE_THRESHOLD_SEC → речевые блоки
      3. Каждый блок получает start/end с буферами из config

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
    all_words = _dedup_boundary_words(all_words)

    thr       = config.PAUSE_THRESHOLD_SEC
    start_buf = config.BLOCK_START_BUFFER_SEC
    end_buf   = config.BLOCK_END_BUFFER_SEC

    blocks  = []
    current = [all_words[0]]

    for word in all_words[1:]:
        # Используем кэпнутый end — иначе раздутый Whisper-таймстемп скрывает реальную паузу
        last_end = min(current[-1]["end"], current[-1]["start"] + _MAX_WORD_DURATION)
        gap = word["start"] - last_end
        if gap >= thr:
            blocks.append(_make_block(current, start_buf, end_buf))
            current = [word]
        else:
            current.append(word)
    blocks.append(_make_block(current, start_buf, end_buf))

    for i, b in enumerate(blocks):
        b["index"] = i

    log.info(f"Речевых блоков: {len(blocks)} (порог паузы: {thr}с)")
    return blocks


def total_duration(blocks: List[Dict]) -> float:
    """Суммарная длительность kept-блоков в секундах."""
    return sum(b["end"] - b["start"] for b in blocks)


def _dedup_boundary_words(words: List[Dict]) -> List[Dict]:
    """
    Удаляет слова-дубли на границах Whisper-сегментов.
    Whisper иногда записывает последнее слово сегмента 1 как первое слово сегмента 2.
    Критерий: то же слово (без пунктуации, без учёта регистра) в пределах 1.5с.
    """
    if len(words) < 2:
        return words
    result = [words[0]]
    for w in words[1:]:
        prev = result[-1]
        prev_clean = prev["word"].strip().lower().strip(".,!?;:-—«»")
        curr_clean = w["word"].strip().lower().strip(".,!?;:-—«»")
        if prev_clean and curr_clean == prev_clean and w["start"] - prev["end"] < 1.5:
            log.debug(f"Dedup boundary: '{w['word']}' @ {w['start']:.2f}s (дубль с {prev['start']:.2f}s)")
            continue
        result.append(w)
    return result


def _make_block(words: List[Dict], start_buf: float, end_buf: float) -> Dict:
    last       = words[-1]
    capped_end = min(last["end"], last["start"] + _MAX_WORD_DURATION)
    return {
        "start": max(0.0, words[0]["start"] - start_buf),
        "end":   capped_end + end_buf,
        "text":  " ".join(w["word"] for w in words).strip(),
        "words": words,
    }
