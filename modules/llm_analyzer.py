"""
llm_analyzer.py — анализ транскрипции через LLM (OpenRouter API).

Находит места где автор повторяет одну и ту же мысль несколько раз подряд,
оставляет только финальную попытку. Возвращает блоки с полем keep=True/False.

Ключевой принцип промпта: анализировать весь текст целиком, а не блоки изолированно.
Блоки показываются как [N] + текст — без временных меток, чтобы LLM читал
транскрипцию как связный текст, а не набор независимых фрагментов.
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
Ты редактор видео-транскрипций. Твоя задача — найти все места где автор повторяет одну и ту же мысль несколько раз подряд, и оставить только финальную попытку.

Важно: Анализируй весь текст целиком, не блоками.

Что считается повтором:
Автор говорит одну и ту же мысль разными словами несколько раз подряд. Это не пересказ — это попытки сформулировать одно и то же. Между попытками нет новой информации, только переформулировка.

Примеры повторов:
- "Итак, давайте сделаем карточку" → "Итак, давайте нарисуем карточку" → удаляем все кроме последнего
- "Отправляюсь в синтекс" → "Отправляюсь синтакс дизайн" → "Отправляю синтакс раздел Design" → удаляем все кроме последнего
- "перед этим перемещу" × 3 подряд → удаляем все кроме одного

Что НЕ является повтором:
- Автор возвращается к теме через несколько минут с новой информацией
- Автор подводит итог того что уже сделал
- Автор объясняет зачем он что-то делает

Формат ответа — строго JSON, без пояснений до и после:
{"delete": [0, 3, 5]}

Числа — индексы блоков для удаления. Если удалять нечего — {"delete": []}\
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
{"delete": [0, 1, 3]}

Если удалять нечего — {"delete": []}\
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
                delete_set = self._parse_response(raw, len(blocks))
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
        """
        Анализирует блоки зная текст сценария.
        Оставляет последний дубль каждой части, удаляет предыдущие.
        """
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
                delete_set = self._parse_response(raw, len(blocks))
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

        Формат: [N] на отдельной строке + текст блока.
        Временные метки НЕ показываются — LLM должен читать как связный текст,
        а не анализировать блоки изолированно.
        """
        lines = [
            f"Контекст видео: {config.VIDEO_CONTEXT}",
            "",
            "Проанализируй эту транскрипцию:",
            "",
        ]
        for b in blocks:
            text = b.get("text", "").strip() or "[без текста]"
            lines.append(f"[{b['index']}]")
            lines.append(text)
            lines.append("")
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

    # ── Парсинг ────────────────────────────────────────────────────────────────

    def _parse_response(self, text: str, total_blocks: int) -> Set[int]:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{"delete"\s*:\s*\[[^\]]*\]\s*\}', text, re.DOTALL)
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

        valid = set()
        for idx in delete_list:
            if isinstance(idx, int) and 0 <= idx < total_blocks:
                valid.add(idx)
            else:
                log.warning(f"Некорректный индекс от LLM: {idx!r}")
        return valid

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
        """
        Авто-удаляет осколки: блок ≤3 слов у которого оба соседа удалены LLM.
        """
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

            word_count = len(b.get("text", "").split())
            if word_count <= 3:
                result.add(idx)
                log.info(f"  [{idx}] авто-удалён (осколок): {b.get('text', '')[:60]!r}")

        return result


# ── Утилита ────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"
