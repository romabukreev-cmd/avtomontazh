"""
llm_analyzer.py — анализ транскрипции через LLM.

Получает Whisper-сегменты (sentence-level), отправляет чанками по 30
с перекрытием 5 сегментов. LLM возвращает {"delete": [индексы]}.
Возвращает только сегменты, которые нужно ОСТАВИТЬ в видео.
"""

import json
import logging
import re
import time
from typing import Callable, Dict, List, Optional, Set

import httpx

import config

log = logging.getLogger(__name__)

_CHUNK_SIZE = 30
_OVERLAP    = 5
_STEP       = _CHUNK_SIZE - _OVERLAP  # 25

_SYSTEM_PROMPT = """\
Ты редактор видео-транскрипций. Задача видео — показать процесс дизайна динамично, \
с пояснениями что и как делается.
Твоя задача — убрать повторы и сегменты, не несущие ценности для зрителя.

Важно: анализируй весь текст целиком. Повторы встречаются по всей длине видео, не только в начале.

Что считается повтором:
Автор говорит одну и ту же мысль разными словами несколько раз подряд. \
Это не пересказ — это попытки сформулировать одно и то же. \
Между попытками нет новой информации. Оставляй только финальную попытку.

Примеры повторов:
- "Итак, давайте сделаем карточку" → "Итак, давайте нарисуем карточку" → оставляем последнее
- "Отправляюсь в синтекс" → "Отправляюсь синтакс дизайн" → "Открываю раздел Design" → оставляем последнее
- "перед этим перемещу" × 3 подряд → оставляем одно

Что НЕ является повтором:
- Автор возвращается к теме через несколько минут с новой информацией
- Автор подводит итог того что уже сделал
- Автор объясняет зачем он что-то делает

Также удаляй:
- Автор оправдывается или объясняет своё решение вместо того чтобы показывать процесс
- Комментарии о съёмке, технике, условиях записи — не о дизайне
- Лирические отступления, не связанные с тем что происходит на экране

Перед каждым удалением: "Если убрать этот сегмент — зритель потеряет понимание процесса?"
Если нет — удалять.

Формат ответа — строго JSON, без пояснений до и после:
{"delete": [1, 3, 4]}

Где числа — номера в квадратных скобках перед временем.
Если нечего удалять — {"delete": []}"""


class LLMAnalyzer:

    def analyze(
        self,
        segments: List[Dict],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[Dict]:
        """
        Анализирует сегменты чанками, возвращает те, что нужно ОСТАВИТЬ.

        on_progress(msg) вызывается при начале каждого чанка.
        """
        if not segments:
            return segments

        delete_indices = self._collect_deletions(segments, on_progress)
        kept = [s for s in segments if s["index"] not in delete_indices]

        log.info(
            f"LLM: удалено {len(delete_indices)} сегментов, "
            f"оставлено {len(kept)}/{len(segments)}"
        )
        return kept

    def _collect_deletions(
        self,
        segments: List[Dict],
        on_progress: Optional[Callable[[str], None]],
    ) -> Set[int]:
        n           = len(segments)
        all_delete  = set()
        chunk_starts = list(range(0, n, _STEP))
        total_chunks = len(chunk_starts)

        for num, chunk_start in enumerate(chunk_starts, 1):
            chunk = segments[chunk_start:chunk_start + _CHUNK_SIZE]

            if on_progress:
                on_progress(f"чанк {num}/{total_chunks}")

            log.info(f"LLM чанк {num}/{total_chunks}: сегменты {chunk[0]['index']}–{chunk[-1]['index']}")

            user_prompt = self._format_chunk(chunk)
            try:
                raw = self._call_with_retry(user_prompt)
                indices = self._parse_response(raw)
                all_delete.update(indices)
                log.info(f"  → удалить {sorted(indices)}")
            except Exception as e:
                log.error(f"  Ошибка в чанке {num}: {e}")

        return all_delete

    def _format_chunk(self, chunk: List[Dict]) -> str:
        lines = [f"Контекст видео: {config.VIDEO_CONTEXT}", "", "Проанализируй транскрипцию и найди ВСЕ повторы:", ""]
        for seg in chunk:
            mm = int(seg["start"]) // 60
            ss = int(seg["start"]) % 60
            lines.append(f"[{seg['index']}] {mm:02d}:{ss:02d} --- {seg['text']}")
        return "\n".join(lines)

    def _parse_response(self, text: str) -> Set[int]:
        # Ищем JSON в ответе (LLM иногда добавляет пояснения вокруг)
        m = re.search(r'\{[^{}]*"delete"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
        if not m:
            log.warning(f"Не нашли JSON в ответе LLM: {text[:200]}")
            return set()
        try:
            data = json.loads(m.group())
            return set(int(i) for i in data.get("delete", []))
        except Exception as e:
            log.warning(f"Ошибка парсинга JSON: {e} | {text[:200]}")
            return set()

    def _call_with_retry(self, user_prompt: str) -> str:
        last_error = None
        for attempt in range(1, config.LLM_MAX_RETRIES + 1):
            try:
                return self._call_api(user_prompt)
            except Exception as e:
                last_error = e
                log.warning(f"LLM попытка {attempt}/{config.LLM_MAX_RETRIES} не удалась: {e}")
                if attempt < config.LLM_MAX_RETRIES:
                    time.sleep(5 * attempt)
        raise RuntimeError(f"LLM не ответил после {config.LLM_MAX_RETRIES} попыток: {last_error}")

    def _call_api(self, user_prompt: str) -> str:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": config.LLM_MODEL,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                },
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
