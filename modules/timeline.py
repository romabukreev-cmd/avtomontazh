"""
timeline.py — построение таймлайна из отобранных LLM сегментов.

Алгоритм:
  Для каждого kept-сегмента с пословными таймстемпами:
    1. Сканируем промежутки между словами
    2. Промежуток > PAUSE_CUT_SEC → разрезаем здесь
       Оставляем PAUSE_BUFFER_SEC с каждой стороны (не рубим прямо по слову)
    3. Результат: список отрезков {start, end} для renderer.py

Пример:
  Слова: [1.0–1.5] "Итак" [1.6–2.0] "нарисуем" ... пауза 1.2с ... [3.4–3.9] "карточку"
  PAUSE_CUT_SEC=0.6, PAUSE_BUFFER_SEC=0.2:
    → отрезок 1: start=0.9, end=2.2   (буфер вокруг "нарисуем")
    → отрезок 2: start=3.2, end=4.1   (буфер вокруг "карточку")
"""

import logging
from typing import Dict, List

import config

log = logging.getLogger(__name__)


class TimelineBuilder:

    def build(self, segments: List[Dict]) -> List[Dict]:
        """
        Строит таймлайн из kept-сегментов.
        Внутренние паузы > config.PAUSE_CUT_SEC удаляются с буфером config.PAUSE_BUFFER_SEC.

        Returns: [{start, end}, ...] для renderer.py
        """
        kept = [s for s in segments if s.get("keep", False)]
        if not kept:
            log.warning("build: нет ни одного keep=True сегмента")
            return []

        kept.sort(key=lambda s: s["start"])

        cuts = []
        for seg in kept:
            cuts.extend(self._cut_segment(seg))

        # Первый отрезок начинается с 0.0 — убираем тишину в начале видео
        if cuts:
            cuts[0]["start"] = 0.0

        total = self.total_duration(cuts)
        log.info(f"Таймлайн: {len(cuts)} отрезков, {total:.1f}с")
        return cuts

    def total_duration(self, timeline: List[Dict]) -> float:
        return sum(seg["end"] - seg["start"] for seg in timeline)

    # ── Нарезка одного сегмента ───────────────────────────────────────────────

    def _cut_segment(self, seg: Dict) -> List[Dict]:
        """
        Разрезает один сегмент по паузам > PAUSE_CUT_SEC.
        Каждая пауза заменяется склейкой с буфером PAUSE_BUFFER_SEC с каждой стороны.
        Возвращает 1+ отрезков {start, end}.
        """
        words = seg.get("words", [])
        buf   = config.PAUSE_BUFFER_SEC
        thr   = config.PAUSE_CUT_SEC

        if not words:
            # Нет пословных таймстемпов — берём сегмент целиком
            return [{"start": seg["start"], "end": seg["end"]}]

        cuts      = []
        cut_start = max(0.0, words[0]["start"] - buf)
        prev_end  = words[0]["end"]

        for word in words[1:]:
            gap = word["start"] - prev_end
            if gap > thr:
                # Закрываем текущий отрезок с буфером
                cuts.append({"start": cut_start, "end": prev_end + buf})
                # Начинаем новый отрезок с буфером
                cut_start = max(0.0, word["start"] - buf)
            prev_end = word["end"]

        # Закрываем последний отрезок
        cuts.append({"start": cut_start, "end": prev_end + buf})
        return cuts
