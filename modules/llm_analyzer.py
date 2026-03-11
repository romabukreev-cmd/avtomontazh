"""
llm_analyzer.py — анализ транскрипции через LLM (OpenRouter API).

Использует промпт в том же формате что работает у пользователя вручную:
LLM возвращает {"repeats": [{"delete": "текст", "keep": "текст"}]}.
Код сопоставляет текстовые фрагменты обратно с блоками по словесному перекрытию.

Это точнее чем просить LLM считать индексы блоков.
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

    def analyze(self, blocks: List[Dict]) -> List[Dict]:
        """
        Анализирует речевые блоки, маркирует повторы как keep=False.

        Args:
            blocks: список блоков из timeline.build_blocks()
        Returns:
            те же блоки с полем keep=True/False
        """
        if not blocks:
            return []

        user_prompt = self._format_transcript(blocks)
        log.info(f"LLM анализ: {len(blocks)} блоков")

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw = self._call_api(_SYSTEM_PROMPT, user_prompt)
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

    def analyze_with_scenario(self, blocks: List[Dict], scenario_text: str) -> List[Dict]:
        """Анализирует блоки зная текст сценария."""
        if not blocks:
            return []

        user_prompt = (
            f"СЦЕНАРИЙ:\n---\n{scenario_text.strip()}\n---\n\n"
            + self._format_transcript(blocks)
        )
        log.info(f"LLM (сценарий): {len(blocks)} блоков")

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

    # ── Форматирование транскрипции ────────────────────────────────────────────

    def _format_transcript(self, blocks: List[Dict]) -> str:
        """
        Формирует транскрипцию для LLM.
        Блоки идут сплошным текстом — LLM читает как связный документ.
        """
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

    def _call_api(self, system: str, user: str) -> str:
        return self._call_api_with_model(system, user, config.LLM_MODEL)

    def _call_api_with_model(self, system: str, user: str, model: str) -> str:
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
                "temperature": 0.2,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ── Парсинг и сопоставление с блоками ─────────────────────────────────────

    def _parse_and_match(self, text: str, blocks: List[Dict]) -> Set[int]:
        """
        Парсит ответ LLM в формате {"repeats": [{"delete": "...", "keep": "..."}]},
        сопоставляет текстовые фрагменты с блоками по словесному перекрытию.
        """
        # Чистим markdown-обёртки если есть
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Пробуем вырезать JSON из текста
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
        """
        Находит блоки, текст которых совпадает с фрагментом для удаления.

        Алгоритм: нормализуем текст (нижний регистр, без пунктуации),
        считаем долю совпадающих слов. Порог 0.55 — достаточно строгий
        чтобы не удалять лишнее, достаточно мягкий для незначительных расхождений.
        """
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

            # Доля слов блока, присутствующих в delete-тексте
            overlap = len(block_words & delete_words) / len(block_words)

            # Также проверяем через SequenceMatcher — ловит перефразировки
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
