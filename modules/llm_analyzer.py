"""
llm_analyzer.py — анализ транскрипции через LLM (OpenRouter API).

Один проход: отправляем всю транскрипцию в Claude, он маркирует каждый сегмент
keep=true/false. Убирает повторы и незаконченные дубли.
"""

import json
import logging
import re
import time
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)


# ── Промпты ───────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """Ты редактор видео-транскрипций. Твоя задача — найти все места где автор повторяет одну и ту же мысль несколько раз подряд, и оставить только финальную попытку.

Важно: Повторы могут быть как внутри одного временного блока, так и на стыке нескольких блоков подряд. Анализируй весь текст целиком, не блоками.

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
{"repeats": [{"delete": [0, 1], "keep": 2, "reason": "почему это повтор"}]}

delete — список индексов сегментов которые удалить.
keep — индекс сегмента который оставить.
reason — одна фраза почему это повтор.
Если повторов нет — верни {"repeats": []}"""


def _build_user_prompt(segments: List[Dict]) -> str:
    lines = [f"Проанализируй эту транскрипцию:\n"]
    for i, seg in enumerate(segments):
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        text  = seg.get("text", "").strip() or "[тишина]"
        lines.append(f"[{i}] {start}–{end}  {text}")
        if i < len(segments) - 1:
            pause = segments[i + 1]["start"] - seg["end"]
            if pause >= 3:
                lines.append(f"    ═══ пауза {pause:.0f}с ═══")

    return "\n".join(lines)


# ── Основной класс ────────────────────────────────────────────────────────────

class LLMAnalyzer:

    def analyze(self, segments: List[Dict]) -> List[Dict]:
        """
        Отправляет всю транскрипцию в Claude одним запросом.
        Возвращает сегменты с полем keep=true/false и reason.
        """
        if not segments:
            return []

        total_duration = sum(s["end"] - s["start"] for s in segments)
        log.info(f"LLM анализ: {len(segments)} сег., {total_duration/60:.1f} мин")

        system = _build_system_prompt()
        user   = _build_user_prompt(segments)

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw       = self._call_api(system, user)
                repeats   = self._parse_response(raw)
                result    = self._apply_decisions(segments, repeats)
                kept = sum(1 for s in result if s.get("keep"))
                log.info(f"LLM итого: оставлено {kept}/{len(result)} сегментов")
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"⚠️ LLM не ответил после {config.LLM_MAX_RETRIES} попыток: {last_error}. Оставляю все сегменты.")
        return self._apply_decisions(segments, [])  # все keep=True

    # ── API ───────────────────────────────────────────────────────────────────

    def _call_api(self, system: str, user: str) -> str:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/automontazh",
            },
            json={
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": 0.2,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ── Парсинг ───────────────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> List[Dict]:
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text)

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "repeats" in data:
                return data["repeats"]
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*"repeats"\s*:\s*\[.*\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("repeats", [])
            except json.JSONDecodeError:
                pass

        log.warning(f"Не удалось распарсить ответ LLM:\n{text[:500]}")
        return []

    def _apply_decisions(self, segments: List[Dict], repeats: List[Dict]) -> List[Dict]:
        # По умолчанию все сегменты остаются
        delete_indices: set = set()

        for repeat in repeats:
            if not isinstance(repeat, dict):
                continue
            delete = repeat.get("delete", [])
            reason = str(repeat.get("reason", ""))
            if isinstance(delete, int):
                delete = [delete]
            for idx in delete:
                if isinstance(idx, int) and 0 <= idx < len(segments):
                    delete_indices.add(idx)
                    log.info(f"  [{idx}] удалён: {reason}")

        result = []
        for i, seg in enumerate(segments):
            seg_copy = dict(seg)
            seg_copy["keep"]   = i not in delete_indices
            seg_copy["reason"] = ""
            result.append(seg_copy)

        return result


# ── Утилита ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
