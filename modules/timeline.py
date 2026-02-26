"""
timeline.py — построение финального таймлайна из отобранных сегментов.

Логика для длинного таймлайна (Форматы 1 и 3, до 10 минут):
  1. Берём все keep=True сегменты в хронологическом порядке
  2. Объединяем соседние сегменты если между ними < 0.5 секунды
  3. Если суммарная длительность > max_sec — удаляем наименее важные сегменты
     из СЕРЕДИНЫ, защищая первые и последние 15% (сохраняем начало и конец)

Логика для хайлайтов (Формат 2, до 3 минут):
  Используется результат второго LLM-прохода (analyze_highlights).
  Строим из keep=True сегментов в хронологическом порядке.

Padding:
  Каждый отрезок расширяется на 0.5с с обеих сторон — берём кусок тишины
  вокруг речи. Это обеспечивает "воздух" между склейками.

Результат — список временны́х отрезков [{start, end}], готовый для renderer.py
"""

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Два соседних сегмента с зазором меньше этого значения объединяются в один
_MERGE_GAP_SEC = 0.5

# Padding вокруг каждого отрезка
# PRE_ROLL = 0: сегменты уже начинаются с первого слова - SEGMENT_START_PADDING_SEC
# (начало рассчитывается в transcriber.py как first_word.start - 0.1с).
# Добавлять ещё отступ сюда — значит вернуть вдохи/дыхание назад.
_PRE_ROLL_SEC  = 0.0
_POST_ROLL_SEC = 0.2   # после речи — короткий хвост, убирает длинные паузы

# Доля сегментов защищённых от удаления (начало и конец видео)
_PROTECT_FRACTION = 0.15


class TimelineBuilder:

    def build_long(self, segments: List[Dict], max_sec: float,
                   video_duration: Optional[float] = None) -> List[Dict]:
        """
        Строит таймлайн для длинного видео (до max_sec секунд).

        Если отобранного контента больше max_sec — удаляет наименее важные
        сегменты из середины, сохраняя начало и конец.

        video_duration — длина исходного видео для ограничения padding.
        """
        kept = [s for s in segments if s.get("keep", False)]
        if not kept:
            log.warning("build_long: нет ни одного keep=True сегмента")
            return []

        kept.sort(key=lambda s: s["start"])
        merged = self._merge_close(kept)

        # Если превышаем лимит — убираем наименее важные из середины
        total = self.total_duration(merged)
        if total > max_sec:
            merged = self._prune_by_priority(merged, max_sec)

        padded = self._add_padding(merged, video_duration)

        # Padding добавляет до 1с на каждый сегмент и может превысить лимит.
        # Второй проход: убираем наименее важные из середины уже padded-сегментов.
        actual = self.total_duration(padded)
        if actual > max_sec:
            padded = self._prune_by_priority(padded, max_sec)
            actual = self.total_duration(padded)

        log.info(f"Длинный таймлайн: {len(padded)} отрезков, {actual:.1f}с")
        return padded

    def build_highlights(self, segments: List[Dict],
                         video_duration: Optional[float] = None,
                         max_sec: Optional[float] = None) -> List[Dict]:
        """
        Строит таймлайн из сегментов прошедших второй LLM-проход (analyze_highlights).
        Если max_sec задан — обрезает до лимита (как build_long).
        """
        kept = [s for s in segments if s.get("keep", False)]
        if not kept:
            log.warning("build_highlights: нет ни одного keep=True сегмента")
            return []

        kept.sort(key=lambda s: s["start"])
        merged = self._merge_close(kept)

        if max_sec and self.total_duration(merged) > max_sec:
            merged = self._prune_by_priority(merged, max_sec)

        padded = self._add_padding(merged, video_duration)

        if max_sec and self.total_duration(padded) > max_sec:
            padded = self._prune_by_priority(padded, max_sec)

        actual = self.total_duration(padded)
        log.info(f"Хайлайты: {len(padded)} отрезков, {actual:.1f}с")
        return padded

    def total_duration(self, timeline: List[Dict]) -> float:
        """Возвращает суммарную длительность таймлайна в секундах."""
        return sum(seg["end"] - seg["start"] for seg in timeline)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _prune_by_priority(self, segments: List[Dict], max_sec: float) -> List[Dict]:
        """
        Удаляет наименее важные сегменты из середины пока total ≤ max_sec.
        Первые и последние _PROTECT_FRACTION сегментов защищены от удаления.
        """
        n = len(segments)
        protect_n = max(1, int(n * _PROTECT_FRACTION))

        protected_idx = set(range(protect_n)) | set(range(n - protect_n, n))
        # Интеграции защищены от удаления независимо от позиции в таймлайне
        protected_idx |= {i for i, s in enumerate(segments) if s.get("integration")}
        middle = [(i, s) for i, s in enumerate(segments) if i not in protected_idx]

        # Сортируем кандидатов на удаление по score возрастанию
        middle.sort(key=lambda x: x[1].get("score", 0.0))

        to_remove: set = set()
        total = self.total_duration(segments)

        for i, seg in middle:
            if total <= max_sec:
                break
            dur = seg["end"] - seg["start"]
            to_remove.add(i)
            total -= dur

        result = [s for i, s in enumerate(segments) if i not in to_remove]
        log.info(f"_prune_by_priority: убрано {len(to_remove)} сегментов из середины, осталось {len(result)}")
        return result

    def _merge_close(self, segments: List[Dict]) -> List[Dict]:
        """
        Объединяет соседние сегменты с зазором меньше _MERGE_GAP_SEC.
        Сохраняет score как максимум из объединяемых сегментов.
        Входные сегменты должны быть отсортированы по start.
        """
        if not segments:
            return []

        merged = [{
            "start": segments[0]["start"],
            "end":   segments[0]["end"],
            "score": segments[0].get("score", 0.5),
        }]

        for seg in segments[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= _MERGE_GAP_SEC:
                merged[-1]["end"]   = max(merged[-1]["end"], seg["end"])
                merged[-1]["score"] = max(merged[-1]["score"], seg.get("score", 0.5))
            else:
                merged.append({
                    "start": seg["start"],
                    "end":   seg["end"],
                    "score": seg.get("score", 0.5),
                })

        return merged

    def _add_padding(self, timeline: List[Dict],
                     video_duration: Optional[float]) -> List[Dict]:
        """
        Расширяет каждый отрезок на PRE_ROLL_SEC назад и POST_ROLL_SEC вперёд.
        Обеспечивает "воздух" вокруг речи — звук начинается не с первого сэмпла.
        Соседние расширенные отрезки объединяются если пересекаются.
        """
        if not timeline:
            return []

        max_end = video_duration if video_duration else float("inf")

        padded = []
        for seg in timeline:
            padded.append({
                "start": max(0.0, seg["start"] - _PRE_ROLL_SEC),
                "end":   min(max_end, seg["end"] + _POST_ROLL_SEC),
                "score": seg.get("score", 0.5),
            })

        # Объединяем пересекающиеся отрезки после расширения
        merged = [padded[0]]
        for seg in padded[1:]:
            if seg["start"] <= merged[-1]["end"]:
                merged[-1]["end"]   = max(merged[-1]["end"], seg["end"])
                merged[-1]["score"] = max(merged[-1]["score"], seg["score"])
            else:
                merged.append(seg)

        return merged
