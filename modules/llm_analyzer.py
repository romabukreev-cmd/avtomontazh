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
    return f"""Ты — видеомонтажёр. Тебе дана транскрипция видео с временными метками.
Твоя задача — убрать повторы и незаконченные дубли.

КОНТЕКСТ ВИДЕО:
{config.VIDEO_CONTEXT}

═══════════════════════════════════════════════
ЧТО УБИРАТЬ
═══════════════════════════════════════════════

ПОВТОРЫ:
Автор говорит одну и ту же мысль несколько раз подряд — между попытками нет новой информации.
Из группы повторов оставь одну версию — самую полную и чёткую (обычно последнюю).
keep=false для всех остальных версий в группе.

Примеры:
- «Итак, давайте сделаем карточку» → «Итак, давайте нарисуем карточку» → оставляем последнее
- «Отправляюсь в синтекс» → «Отправляюсь синтакс дизайн» → «Отправляю синтакс, раздел Design» → последнее
- «перед этим перемещу» × 3 подряд → оставляем одно

НЕЗАКОНЧЕННЫЕ ДУБЛИ:
Сегмент обрывается на полуслове или не несёт самостоятельного смысла → keep=false.

═══════════════════════════════════════════════
ЧТО НЕ УБИРАТЬ
═══════════════════════════════════════════════
- Автор возвращается к теме с новой информацией
- Автор подводит итог или объясняет зачем что-то делает
- Любые призывы к действию, упоминания ссылок, кодовых слов — оставляем как есть

═══════════════════════════════════════════════
ФОРМАТ ОТВЕТА
═══════════════════════════════════════════════
Верни строго JSON без лишнего текста:
{{"analysis": "...", "segments": [{{"index": 0, "keep": true, "reason": "кратко"}}, ...]}}

analysis — 2-3 предложения: что нашёл, что убрал.
reason — одна фраза: зачем оставляем или почему убираем."""


def _build_user_prompt(segments: List[Dict]) -> str:
    lines = ["Полная транскрипция видео:\n"]
    for i, seg in enumerate(segments):
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        dur   = seg["end"] - seg["start"]
        text  = seg.get("text", "").strip() or "[тишина]"
        lines.append(f"[{i}] {start}–{end} ({dur:.0f}с)  {text}")
        if i < len(segments) - 1:
            pause = segments[i + 1]["start"] - seg["end"]
            if pause >= 3:
                lines.append(f"    ═══ пауза {pause:.0f}с ═══")

    lines.append(
        '\nВерни JSON: {"analysis": "...", "segments": [{"index": 0, "keep": true, "reason": "..."}, ...]}'
    )
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
                decisions = self._parse_response(raw, len(segments))
                result    = self._apply_decisions(segments, decisions)
                kept = sum(1 for s in result if s.get("keep"))
                log.info(f"LLM итого: оставлено {kept}/{len(result)} сегментов")
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"⚠️ LLM не ответил после {config.LLM_MAX_RETRIES} попыток: {last_error}. Оставляю все сегменты.")
        return self._apply_decisions(segments, [])

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

    def _parse_response(self, text: str, expected_count: int) -> List[Dict]:
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text)

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "segments" in data:
                if "analysis" in data:
                    log.info(f"LLM анализ: {data['analysis'][:300]}")
                return data["segments"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*"segments"\s*:\s*\[.*\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "analysis" in data:
                    log.info(f"LLM анализ: {data['analysis'][:300]}")
                return data.get("segments", [])
            except json.JSONDecodeError:
                pass

        log.warning(f"Не удалось распарсить ответ LLM:\n{text[:500]}")
        return []

    def _apply_decisions(self, segments: List[Dict], decisions: List[Dict]) -> List[Dict]:
        decision_map = {
            item["index"]: item
            for item in decisions
            if isinstance(item, dict) and "index" in item
        }

        result = []
        for i, seg in enumerate(segments):
            seg_copy = dict(seg)
            llm_data = decision_map.get(i, {})
            raw_keep = llm_data.get("keep", None)
            seg_copy["keep"]   = bool(raw_keep) if raw_keep is not None else True
            seg_copy["reason"] = str(llm_data.get("reason", ""))
            result.append(seg_copy)

        return result


# ── Утилита ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
