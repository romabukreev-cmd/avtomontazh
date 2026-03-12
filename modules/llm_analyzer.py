"""
llm_analyzer.py — анализ транскрипции через LLM.

Получает Whisper-сегменты (sentence-level), отправляет чанками по 30 (~4 минуты)
с перекрытием 5 сегментов. LLM возвращает {"delete": [индексы]}.
Возвращает только те сегменты которые нужно ОСТАВИТЬ в видео.
"""

import json
import logging
import re
import time
from typing import Dict, List, Set

import httpx

import config

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Ты редактор видео-транскрипций. Твоя задача — найти ВСЕ места где автор повторяет одну и ту же мысль несколько раз подряд, и оставить только финальную попытку.

ОБЯЗАТЕЛЬНО: просмотри КАЖДЫЙ сегмент от первого до последнего. Повторы встречаются по всей длине видео, не только в начале.

Что удалять — все три случая:
1. Автор начинает фразу, останавливается, начинает заново → удалить все попытки кроме последней
2. Два соседних сегмента говорят одно и то же разными словами → удалить первый
3. Сегмент почти дословно повторяет предыдущий → удалить первый

Примеры:
- [1] "Итак, давайте сделаем карточку" → [2] "Итак, давайте нарисуем карточку" → удалить [1]
- [3] "Отправляюсь в синтекс" → [4] "Отправляюсь синтакс дизайн" → [5] "Открываю раздел Design" → удалить [3][4]
- [6] "перед этим перемещу" → [7] "перед этим перемещу" → [8] "перед этим перемещу" → удалить [6][7]
- [9] "набросить несколько тезисов" → [10] "набросить несколько тезисов" → удалить [9]
- [13] "И нажимаю ОК" → [14] "И нажимаю OK" → удалить [13]

Что НЕ удалять:
- Разные шаги одного процесса (это разные действия, не повторы)
- Тема возвращается через несколько минут с НОВОЙ информацией

Формат ответа — строго JSON, без пояснений до и после:
{"delete": [1, 3, 4]}

Где числа — номера в квадратных скобках перед временем.
Если повторов нет — {"delete": []}"""

# 30 сегментов ≈ 4 минуты. Перекрытие 5 страхует от пропуска повторов на стыке чанков.
_CHUNK_SIZE = 30
_OVERLAP    = 5
_STEP       = _CHUNK_SIZE - _OVERLAP  # 25


class LLMAnalyzer:

    def analyze(self, segments: List[Dict]) -> List[Dict]:
        """
        Анализирует Whisper-сегменты и возвращает те что нужно ОСТАВИТЬ.

        Args:
            segments: [{index, start, end, text, words}, ...]

        Returns:
            kept: те же dict-ы без повторов
        """
        if not segments:
            return []

        delete_indices = self._collect_deletions(segments)
        kept = [s for s in segments if s["index"] not in delete_indices]

        log.info(f"LLM итого: оставлено {len(kept)}/{len(segments)} сегментов")
        return kept

    # ── Чанковая обработка ────────────────────────────────────────────────────

    def _collect_deletions(self, segments: List[Dict]) -> Set[int]:
        all_delete: Set[int] = set()
        chunk_starts = list(range(0, len(segments), _STEP))
        total = len(chunk_starts)

        for num, chunk_start in enumerate(chunk_starts):
            chunk = segments[chunk_start:chunk_start + _CHUNK_SIZE]
            if not chunk:
                continue

            log.info(
                f"LLM чанк {num+1}/{total}: "
                f"сегменты [{chunk[0]['index']}]–[{chunk[-1]['index']}]"
            )

            raw = self._call_with_retry(self._format(chunk))
            if raw is None:
                log.error(f"  Чанк {num+1} пропущен — LLM не ответил")
                continue

            indices = self._parse(raw)
            log.info(f"  → удаляем: {sorted(indices)}")
            all_delete.update(indices)

        return all_delete

    # ── Форматирование ────────────────────────────────────────────────────────

    def _format(self, segments: List[Dict]) -> str:
        lines = [
            f"Контекст видео: {config.VIDEO_CONTEXT}",
            "",
            "Проанализируй транскрипцию и найди ВСЕ повторы:",
            "",
        ]
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            m, s = divmod(int(seg["start"]), 60)
            lines.append(f"[{seg['index']}] {m:02d}:{s:02d} --- {text}")
        return "\n".join(lines)

    # ── Парсинг ответа ────────────────────────────────────────────────────────

    def _parse(self, text: str) -> Set[int]:
        """{"delete": [1, 3, 5]} → {1, 3, 5}. При ошибке → пустое множество."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^}]*"delete"[^}]*\}', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            log.warning(f"Не удалось распарсить ответ LLM: {text[:200]!r}")
            return set()

        delete_list = data.get("delete", [])
        if not isinstance(delete_list, list):
            return set()

        return {int(x) for x in delete_list if isinstance(x, (int, float))}

    # ── API ───────────────────────────────────────────────────────────────────

    def _call_with_retry(self, user_prompt: str) -> str | None:
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                return self._call_api(user_prompt)
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"  Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)
        return None

    def _call_api(self, user_prompt: str) -> str:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/automontazh",
            },
            json={
                "model":       config.LLM_MODEL,
                "messages":    [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
