"""
llm_analyzer.py — анализ речевых блоков через LLM (OpenRouter API).

Получает блоки из timeline.build_blocks(), находит смысловые повторы,
возвращает блоки с полем keep=True/False.
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
Ты редактор видео-транскрипций. Твоя задача — найти все повторы и запинки и пометить их на удаление, оставив только финальную версию каждой мысли.

Что удалять:

1. ПОВТОРЫ — автор говорит одну и ту же мысль два и более раза (подряд или через 1-3 блока):
   - "Итак, давайте сделаем карточку" → "Итак, давайте нарисуем карточку" → удаляем все кроме последнего
   - "Отправляюсь в синтекс" → "Отправляюсь синтакс дизайн" → "Отправляю синтакс раздел Design" → удаляем все кроме последнего
   - "перед этим перемещу" × 3 → удаляем все кроме одного
   - Неважно перефразирован ли повтор или сказан дословно — если смысл тот же, удаляем ранние попытки

2. ЗАПИНКИ — оговорки, незаконченные фразы, бессмысленные звуки:
   - "э-э", "ну это", "вот как бы", незаконченные предложения перед перезаписью
   - Блок из одного-двух слов без смысла ("Итак,", "Ну,", "Значит,")

3. ТЕХНИЧЕСКИЕ ПОВТОРЫ — дословное или почти дословное повторение через несколько блоков:
   - Автор случайно начинает ту же фразу что уже сказал

Что НЕ удалять:
- Осознанное подведение итогов ("Итак, мы сделали X, теперь делаем Y")
- Объяснение своих действий ("Нажимаю сюда, потому что...")
- Возврат к теме с НОВОЙ информацией (новые детали, новый шаг)
- Связки и переходы между темами

Критерий удаления: если блок — это неудачная или промежуточная попытка сказать то, что автор потом сказал лучше — удаляй.

Формат ответа — строго JSON, без пояснений до и после:
{"delete": [0, 3, 5]}

Числа — индексы блоков для удаления.
Если удалять нечего — верни {"delete": []}\
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

    # ── Форматирование транскрипции ────────────────────────────────────────────

    def _format_transcript(self, blocks: List[Dict]) -> str:
        lines = [
            f"Контекст видео: {config.VIDEO_CONTEXT}",
            "",
            "Проанализируй эту транскрипцию:",
            "",
        ]
        for b in blocks:
            start = _fmt_time(b["start"])
            end   = _fmt_time(b["end"])
            text  = b.get("text", "").strip() or "[без текста]"
            lines.append(f"[{b['index']}] {start} → {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    # ── API ────────────────────────────────────────────────────────────────────

    def _call_api(self, system: str, user: str) -> str:
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
        # Убираем markdown-обёртку если модель добавила
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
        result = []
        for b in blocks:
            b_copy = dict(b)
            b_copy["keep"] = b["index"] not in delete_set
            if not b_copy["keep"]:
                log.info(f"  [{b['index']}] удалён: {b.get('text', '')[:70]!r}")
            result.append(b_copy)
        return result


# ── Утилита ────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"
