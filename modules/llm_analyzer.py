"""
llm_analyzer.py — анализ транскрипции через LLM (OpenRouter API).

Сегментный режим (основной):
  LLM получает пронумерованные Whisper-сегменты в формате [N] MM:SS --- текст.
  LLM возвращает {"delete": [1, 3, 5]} — номера сегментов для удаления.
  Маппим сегменты → блоки по времени (mid блока попадает в диапазон сегмента).

Текстовый режим (fallback, если segments не переданы):
  LLM видит блоки как текст, возвращает {"repeats": [{"delete":"...","keep":"..."}]}.
  Текстовое сопоставление через SequenceMatcher.
"""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Set

import httpx

import config

log = logging.getLogger(__name__)


# ── Промпты: сегментный режим ──────────────────────────────────────────────────

_SEGMENT_SYSTEM_PROMPT = """\
Ты редактор видео-транскрипций. Твоя задача — найти ВСЕ места где автор повторяет одну и ту же мысль, и оставить только финальную версию.

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
- [11] "можно бросить плашку" → [12] "можно бросить плашку" → удалить [11]
- [13] "И нажимаю ОК" → [14] "И нажимаю OK" → удалить [13]

Что НЕ удалять:
- Разные шаги одного процесса (это разные действия, не повторы)
- Тема возвращается через несколько минут с НОВОЙ информацией

Формат ответа — строго JSON, без пояснений до и после:
{"delete": [1, 3, 4]}

Где числа — номера в квадратных скобках перед временем.
Если повторов нет — {"delete": []}\
"""

_SEGMENT_STANDARD_SYSTEM_PROMPT = """\
Ты монтажёр Reels. Автор записал ролик по сценарию, делая несколько дублей каждой части.

Твоя задача — для каждой части сценария найти все её дубли в транскрипции и оставить только ПОСЛЕДНИЙ. Все предыдущие неудачные дубли — удалять.

Важно: Анализируй весь текст целиком, не блоками.

Как определить дубли:
- Автор начинает говорить ту же часть сценария заново (перефразированно или почти дословно)
- Дубли всегда идут подряд или через 1-2 коротких сегмента (запинки, паузы)
- Последний дубль — тот, после которого автор переходит к следующей части сценария

Что НЕ является дублями (не удалять):
- Разные варианты CTA (призыв к действию) для разных платформ — даже если похожи по структуре. Признак: упоминают разные платформы (Telegram, Instagram, ВКонтакте), разные ссылки или разные действия. Все варианты CTA оставляй.
- Переходы и связки между частями сценария
- Блоки с новой информацией, которой нет в предыдущих дублях

Формат ответа — строго JSON, без пояснений:
{"delete": [1, 3, 4]}

Где числа — номера сегментов для удаления (цифры в квадратных скобках перед временем).
Если удалять нечего — {"delete": []}\
"""


# ── Промпты: текстовый режим (fallback) ───────────────────────────────────────

_SYSTEM_PROMPT = """\
Ты редактор видео-транскрипций. Твоя задача — найти все места где автор повторяет одну и ту же мысль несколько раз подряд, и оставить только финальную попытку.

Важно: Анализируй весь текст целиком, не блоками.

Что считается повтором:
Автор говорит одну и ту же мысль разными словами несколько раз подряд. Это не пересказ — это попытки сформулировать одно и то же. Между попытками нет новой информации, только переформулировка.

Примеры повторов:
- "Итак, давайте сделаем карточку" → "Итак, давайте нарисуем карточку" → оставляем последнее
- "Отправляюсь в синтекс" → "Отправляюсь синтакс дизайн" → "Отправляю синтакс раздел Design" → оставляем последнее
- "перед этим перемещу" × 3 подряд → оставляем одно

Что НЕ является повтором:
- Автор возвращается к теме через несколько минут с новой информацией
- Автор подводит итог того что уже сделал
- Автор объясняет зачем он что-то делает

Формат ответа — строго JSON, без пояснений до и после:
{
  "repeats": [
    {
      "delete": "точный текст который удаляем",
      "keep": "точный текст который оставляем",
      "reason": "почему это повтор"
    }
  ]
}

Если повторов нет — {"repeats": []}\
"""


_STANDARD_SYSTEM_PROMPT = """\
Ты монтажёр Reels. Автор записал ролик по сценарию, делая несколько дублей каждой части.

Твоя задача — для каждой части сценария найти все её дубли в транскрипции и оставить только ПОСЛЕДНИЙ. Все предыдущие неудачные дубли — удалять.

Важно: Анализируй весь текст целиком, не блоками.

Как определить дубли:
- Автор начинает говорить ту же часть сценария заново (перефразированно или почти дословно)
- Дубли всегда идут подряд или через 1-2 коротких блока (запинки, паузы)
- Последний дубль — тот, после которого автор переходит к следующей части сценария

Что НЕ является дублями (не удалять):
- Разные варианты CTA (призыв к действию) для разных платформ — даже если похожи по структуре. Признак: упоминают разные платформы (Telegram, Instagram, ВКонтакте), разные ссылки или разные действия. Все варианты CTA оставляй.
- Переходы и связки между частями сценария
- Блоки с новой информацией, которой нет в предыдущих дублях

Формат ответа — строго JSON, без пояснений:
{
  "repeats": [
    {
      "delete": "точный текст который удаляем",
      "keep": "точный текст который оставляем",
      "reason": "почему это повтор"
    }
  ]
}

Если удалять нечего — {"repeats": []}\
"""


class LLMAnalyzer:

    def analyze(self, blocks: List[Dict], segments: List[Dict] = None) -> List[Dict]:
        """
        Анализирует речевые блоки, маркирует повторы как keep=False.

        Args:
            blocks:   список блоков из timeline.build_blocks()
            segments: Whisper-сегменты (sentence-level) для формирования промпта.
                      Если переданы — сегментный режим (надёжнее, без эвристик).
                      Если None — текстовый режим (fallback).
        Returns:
            те же блоки с полем keep=True/False
        """
        if not blocks:
            return []

        if segments:
            return self._analyze_by_segments(blocks, segments, _SEGMENT_SYSTEM_PROMPT)
        else:
            return self._analyze_by_text(blocks, _SYSTEM_PROMPT)

    def analyze_with_scenario(
        self, blocks: List[Dict], scenario_text: str, segments: List[Dict] = None
    ) -> List[Dict]:
        """Анализирует блоки зная текст сценария."""
        if not blocks:
            return []

        if segments:
            return self._analyze_by_segments(
                blocks, segments, _SEGMENT_STANDARD_SYSTEM_PROMPT, scenario_text
            )
        else:
            return self._analyze_by_text_with_scenario(blocks, scenario_text)

    # ── Сегментный режим ───────────────────────────────────────────────────────

    def _analyze_by_segments(
        self,
        blocks: List[Dict],
        segments: List[Dict],
        system_prompt: str,
        scenario_text: str = None,
    ) -> List[Dict]:
        """Основной режим: LLM видит сегменты, возвращает {"delete": [индексы]}."""
        user_prompt = self._format_transcript_from_segments(segments)
        if scenario_text:
            user_prompt = (
                f"СЦЕНАРИЙ:\n---\n{scenario_text.strip()}\n---\n\n" + user_prompt
            )

        log.info(f"LLM анализ (сегментный): {len(segments)} сегментов → {len(blocks)} блоков")

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw = self._call_api(system_prompt, user_prompt)
                log.debug(f"LLM ответ: {raw[:500]}")
                seg_delete = self._parse_segment_indices(raw)
                log.info(f"LLM: удаляем сегменты {sorted(seg_delete)}")
                delete_set = self._segments_to_block_indices(seg_delete, segments, blocks)
                result = self._apply_decisions(blocks, delete_set)
                kept = sum(1 for b in result if b["keep"])
                log.info(f"LLM: оставлено {kept}/{len(result)} блоков")
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"LLM не ответил: {last_error}. Оставляю все блоки.")
        return self._apply_decisions(blocks, set())

    def _format_transcript_from_segments(self, segments: List[Dict]) -> str:
        """
        Форматирует Whisper-сегменты для LLM.
        Формат: [N] MM:SS --- текст сегмента
        Совпадает с тем как пользователь отправляет транскрипцию вручную.
        """
        lines = [
            f"Контекст видео: {config.VIDEO_CONTEXT}",
            "",
            "Проанализируй эту транскрипцию:",
            "",
        ]
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            start = seg["start"]
            m, s = divmod(int(start), 60)
            ts = f"{m:02d}:{s:02d}"
            lines.append(f"[{seg['index']}] {ts} --- {text}")
        return "\n".join(lines)

    def _parse_segment_indices(self, text: str) -> Set[int]:
        """Парсит {"delete": [1, 3, 5]} → {1, 3, 5}."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*"delete".*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            log.warning(f"Не удалось распарсить ответ LLM:\n{text[:500]}")
            return set()

        delete_list = data.get("delete", [])
        if not isinstance(delete_list, list):
            return set()

        result = set()
        for item in delete_list:
            if isinstance(item, (int, float)):
                result.add(int(item))
        return result

    def _segments_to_block_indices(
        self, seg_delete: Set[int], segments: List[Dict], blocks: List[Dict]
    ) -> Set[int]:
        """
        Маппит индексы удаляемых сегментов в индексы блоков.
        Блок попадает в сегмент если его середина (mid) лежит в [seg.start, seg.end].
        """
        if not seg_delete:
            return set()

        deleted_segs = [s for s in segments if s["index"] in seg_delete]

        delete_blocks: Set[int] = set()
        for b in blocks:
            mid = (b["start"] + b["end"]) / 2
            for seg in deleted_segs:
                if seg["start"] <= mid <= seg["end"]:
                    delete_blocks.add(b["index"])
                    break

        return delete_blocks

    # ── Текстовый режим (fallback) ─────────────────────────────────────────────

    def _analyze_by_text(self, blocks: List[Dict], system_prompt: str) -> List[Dict]:
        """Fallback: LLM видит блоки как текст, текстовое сопоставление."""
        user_prompt = self._format_transcript(blocks)
        log.info(f"LLM анализ (текстовый): {len(blocks)} блоков")

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw = self._call_api(system_prompt, user_prompt)
                log.debug(f"LLM ответ: {raw[:500]}")
                delete_set = self._parse_and_match(raw, blocks)
                result = self._apply_decisions(blocks, delete_set)
                kept = sum(1 for b in result if b["keep"])
                log.info(f"LLM: оставлено {kept}/{len(result)}, удалено {len(delete_set)}")
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"LLM не ответил: {last_error}. Оставляю все блоки.")
        return self._apply_decisions(blocks, set())

    def _analyze_by_text_with_scenario(
        self, blocks: List[Dict], scenario_text: str
    ) -> List[Dict]:
        user_prompt = (
            f"СЦЕНАРИЙ:\n---\n{scenario_text.strip()}\n---\n\n"
            + self._format_transcript(blocks)
        )
        log.info(f"LLM (сценарий, текстовый): {len(blocks)} блоков")

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw = self._call_api_with_model(
                    _STANDARD_SYSTEM_PROMPT, user_prompt, config.STANDARD_LLM_MODEL
                )
                log.debug(f"LLM (сценарий) ответ: {raw[:500]}")
                delete_set = self._parse_and_match(raw, blocks)
                result = self._apply_decisions(blocks, delete_set)
                kept = sum(1 for b in result if b["keep"])
                log.info(f"LLM (сценарий): оставлено {kept}/{len(result)}, удалено {len(delete_set)}")
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"LLM не ответил: {last_error}. Оставляю все блоки.")
        return self._apply_decisions(blocks, set())

    # ── Форматирование транскрипции (fallback) ─────────────────────────────────

    def _format_transcript(self, blocks: List[Dict]) -> str:
        lines = [
            f"Контекст видео: {config.VIDEO_CONTEXT}",
            "",
            "Проанализируй эту транскрипцию:",
            "",
        ]
        for b in blocks:
            text = b.get("text", "").strip()
            if text:
                lines.append(text)
        return "\n".join(lines)

    # ── API ────────────────────────────────────────────────────────────────────

    def _call_api(self, system: str, user: str, temperature: float = 0.4) -> str:
        return self._call_api_with_model(system, user, config.LLM_MODEL, temperature)

    def _call_api_with_model(self, system: str, user: str, model: str, temperature: float = 0.2) -> str:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/automontazh",
            },
            json={
                "model":       model,
                "messages":    [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": temperature,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ── Парсинг и сопоставление (fallback) ────────────────────────────────────

    def _parse_and_match(self, text: str, blocks: List[Dict]) -> Set[int]:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*"repeats".*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            log.warning(f"Не удалось распарсить ответ LLM:\n{text[:500]}")
            return set()

        repeats = data.get("repeats", [])
        if not isinstance(repeats, list):
            return set()

        delete_set: Set[int] = set()
        for item in repeats:
            if not isinstance(item, dict):
                continue
            delete_text = item.get("delete", "")
            if not delete_text:
                continue

            matched = self._match_text_to_blocks(delete_text, blocks)
            if matched:
                log.info(f"  Удаляем блоки {matched}: {delete_text[:80]!r}")
                delete_set.update(matched)
            else:
                log.warning(f"  Не удалось сопоставить с блоком: {delete_text[:80]!r}")

        return delete_set

    def _match_text_to_blocks(self, delete_text: str, blocks: List[Dict]) -> List[int]:
        delete_norm = _normalize(delete_text)
        delete_words = set(delete_norm.split())
        if not delete_words:
            return []

        results = []
        for b in blocks:
            block_norm  = _normalize(b.get("text", ""))
            block_words = set(block_norm.split())
            if not block_words:
                continue

            overlap = len(block_words & delete_words) / len(block_words)
            seq_ratio = SequenceMatcher(None, block_norm, delete_norm).ratio()

            if overlap >= 0.55 or seq_ratio >= 0.70:
                results.append(b["index"])

        return results

    # ── Применение решений ─────────────────────────────────────────────────────

    def _apply_decisions(self, blocks: List[Dict], delete_set: Set[int]) -> List[Dict]:
        delete_set = self._cleanup_orphan_fragments(blocks, delete_set)

        result = []
        for b in blocks:
            b_copy = dict(b)
            b_copy["keep"] = b["index"] not in delete_set
            if not b_copy["keep"]:
                log.info(f"  [{b['index']}] удалён: {b.get('text', '')[:70]!r}")
            result.append(b_copy)
        return result

    def _cleanup_orphan_fragments(self, blocks: List[Dict], delete_set: Set[int]) -> Set[int]:
        """Авто-удаляет осколки: блок ≤3 слов у которого оба соседа удалены."""
        result = set(delete_set)
        index_set = {b["index"] for b in blocks}

        for b in blocks:
            idx = b["index"]
            if idx in result:
                continue
            if idx - 1 not in index_set or idx + 1 not in index_set:
                continue
            if idx - 1 not in result or idx + 1 not in result:
                continue
            if len(b.get("text", "").split()) <= 3:
                result.add(idx)
                log.info(f"  [{idx}] авто-удалён (осколок): {b.get('text', '')[:60]!r}")

        return result


# ── Утилиты ────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Нижний регистр, убираем пунктуацию — для сопоставления текстов."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
