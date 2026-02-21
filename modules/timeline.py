"""
timeline.py — построение финального таймлайна из отобранных сегментов.

Логика для длинного таймлайна (Форматы 1 и 3, до 10 минут):
  1. Берём все keep=True сегменты в хронологическом порядке
  2. Объединяем соседние сегменты если между ними < 0.5 секунды (зазор)
  3. Суммарная длительность: если > 10 минут — обрезаем хвост

Логика для хайлайтов (Формат 2, до 3 минут):
  1. Сортируем keep=True сегменты по score (от высокого к низкому)
  2. Берём лучшие пока сумма не превысит 3 минуты
  3. Сортируем выбранные обратно в хронологический порядок (важно для восприятия!)

Результат — список временны́х отрезков [{start, end}], готовый для renderer.py
"""

import logging
from typing import Dict, List

log = logging.getLogger(__name__)

# Два соседних сегмента с зазором меньше этого значения объединяются в один
_MERGE_GAP_SEC = 0.5

# Минимальный балл для попадания в хайлайты (если LLM вернул keep=True но балл низкий)
_HIGHLIGHT_MIN_SCORE = 0.0  # берём все keep=True, сортируем по score


class TimelineBuilder:

    def build_long(self, segments: List[Dict], max_sec: float) -> List[Dict]:
        """
        Строит таймлайн для длинного видео (до max_sec секунд).

        Принимает scored_segments (с полями keep, score, start, end).
        Возвращает: [{"start": 0.0, "end": 12.4}, ...]
        """
        kept = [s for s in segments if s.get("keep", False)]
        if not kept:
            log.warning("build_long: нет ни одного keep=True сегмента")
            return []

        kept.sort(key=lambda s: s["start"])
        merged  = self._merge_close(kept)
        trimmed = self._trim_to_duration(merged, max_sec)

        actual = self.total_duration(trimmed)
        log.info(f"Длинный таймлайн: {len(trimmed)} отрезков, {actual:.1f}с")
        return trimmed

    def build_highlights(self, segments: List[Dict], max_sec: float) -> List[Dict]:
        """
        Строит таймлайн из лучших моментов (до max_sec секунд).

        Алгоритм:
          1. Сортируем по score убыванию
          2. Жадно берём сегменты пока не наберём max_sec
          3. Возвращаем в хронологическом порядке
        """
        kept = [s for s in segments if s.get("keep", False)]
        if not kept:
            log.warning("build_highlights: нет ни одного keep=True сегмента")
            return []

        # Сортируем по убыванию score
        by_score = sorted(kept, key=lambda s: s.get("score", 0.0), reverse=True)

        selected = []
        total = 0.0
        for seg in by_score:
            dur = seg["end"] - seg["start"]
            if total + dur > max_sec:
                # Можем взять только кусок чтобы не превысить лимит
                remaining = max_sec - total
                if remaining > 1.0:  # стоит только если > 1 секунды
                    selected.append({
                        "start": seg["start"],
                        "end":   round(seg["start"] + remaining, 3),
                    })
                break
            selected.append({"start": seg["start"], "end": seg["end"]})
            total += dur

        # Восстанавливаем хронологический порядок
        selected.sort(key=lambda s: s["start"])

        actual = self.total_duration(selected)
        log.info(f"Хайлайты: {len(selected)} отрезков, {actual:.1f}с")
        return selected

    def total_duration(self, timeline: List[Dict]) -> float:
        """Возвращает суммарную длительность таймлайна в секундах."""
        return sum(seg["end"] - seg["start"] for seg in timeline)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _merge_close(self, segments: List[Dict]) -> List[Dict]:
        """
        Объединяет соседние сегменты с зазором меньше _MERGE_GAP_SEC.
        Входные сегменты должны быть отсортированы по start.
        """
        if not segments:
            return []

        merged = [{"start": segments[0]["start"], "end": segments[0]["end"]}]

        for seg in segments[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= _MERGE_GAP_SEC:
                # Расширяем последний отрезок
                merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            else:
                merged.append({"start": seg["start"], "end": seg["end"]})

        return merged

    def _trim_to_duration(self, timeline: List[Dict], max_sec: float) -> List[Dict]:
        """
        Обрезает таймлайн до max_sec секунд.
        Последний отрезок может быть обрезан частично.
        """
        result = []
        total  = 0.0

        for seg in timeline:
            dur = seg["end"] - seg["start"]
            if total + dur >= max_sec:
                # Берём только нужную часть последнего отрезка
                cut_end = seg["start"] + (max_sec - total)
                result.append({"start": seg["start"], "end": round(cut_end, 3)})
                break
            result.append({"start": seg["start"], "end": seg["end"]})
            total += dur

        return result
