"""
llm_analyzer.py — анализ транскрипции через LLM (OpenRouter API).

Логика:
  1. Берём список сегментов от Transcriber
  2. Разбиваем на чанки если транскрипция очень длинная (>8000 токенов)
  3. Отправляем каждый чанк в LLM с инструкцией:
     - Оцени каждый сегмент по 0..1 (насколько это интересный дизайн-процесс)
     - Пометь keep=True если score >= 0.5
  4. Склеиваем результаты чанков обратно
  5. При ошибке API — повторяем до LLM_MAX_RETRIES раз

Формат ответа LLM (JSON):
  {"segments": [{"index": 0, "score": 0.9, "keep": true, "reason": "..."}, ...]}
"""

import json
import logging
import re
import time
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)

# Минимальный балл для keep=True (если LLM вернул keep=True, но score ниже — перезаписываем)
_KEEP_THRESHOLD = 0.5

SYSTEM_PROMPT = """Ты — опытный видеомонтажёр. Тебе дана транскрипция экранной записи дизайнера.

Твоя задача — оценить каждый сегмент и решить, стоит ли он включения в финальное видео для аудитории.

ОСТАВЛЯЙ (высокий score, keep=true):
- Генерация идей, поиск решений, обсуждение концепций
- Активное создание и редактирование дизайна с комментариями
- Объяснение своего процесса, рассуждения вслух
- Интересные находки, неожиданные решения, "эврика"-моменты
- Критика и итерации над работой

ВЫРЕЗАЙ (низкий score, keep=false):
- Долгое молчание или невнятное бормотание
- Технические задержки: загрузка файлов, ожидание рендера, зависания
- Нерелевантные разговоры, не связанные с работой
- Монотонные повторяющиеся действия без комментариев (клик-клик-клик в тишине)
- Административная рутина: сохранение файлов, открытие программ

Верни строго JSON в формате:
{"segments": [{"index": 0, "score": 0.85, "keep": true, "reason": "краткое объяснение"}, ...]}

Оценивай честно. score=1.0 — только для исключительно интересных моментов."""


class LLMAnalyzer:

    def analyze(self, segments: List[Dict]) -> List[Dict]:
        """
        Принимает сегменты от Transcriber, возвращает их же с добавленными полями:
          - "score":  float 0..1 (важность)
          - "keep":   bool (оставить или вырезать)
          - "reason": str (объяснение LLM — для отладки)
        """
        if not segments:
            return []

        chunks = self._split_into_chunks(segments)
        log.info(f"LLM анализ: {len(segments)} сегментов, {len(chunks)} чанков")

        result = []
        for i, chunk in enumerate(chunks):
            log.info(f"Анализирую чанк {i+1}/{len(chunks)} ({len(chunk)} сегментов)...")
            analyzed = self._analyze_chunk(chunk)
            result.extend(analyzed)

        kept = sum(1 for s in result if s.get("keep"))
        log.info(f"LLM: оставлено {kept}/{len(result)} сегментов")
        return result

    # ── Разбивка на чанки ────────────────────────────────────────────────────

    def _split_into_chunks(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Разбивает сегменты на части по ~LLM_CHUNK_SIZE_TOKENS токенов.
        Грубая оценка: 1 токен ≈ 4 символа.
        """
        max_chars = config.LLM_CHUNK_SIZE_TOKENS * 4

        chunks: List[List[Dict]] = []
        current: List[Dict] = []
        current_chars = 0

        for seg in segments:
            seg_chars = len(seg.get("text", ""))
            if current and current_chars + seg_chars > max_chars:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(seg)
            current_chars += seg_chars

        if current:
            chunks.append(current)

        return chunks

    # ── Анализ одного чанка ───────────────────────────────────────────────────

    def _analyze_chunk(self, chunk: List[Dict]) -> List[Dict]:
        """
        Отправляет чанк в OpenRouter, получает оценки, применяет к сегментам.
        При ошибке — retry с экспоненциальной паузой.
        """
        user_prompt = self._build_prompt(chunk)

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw_response = self._call_api(user_prompt)
                scored = self._parse_response(raw_response, len(chunk))
                return self._apply_scores(chunk, scored)

            except Exception as e:
                last_error = e
                wait = 2 ** attempt  # 1с, 2с, 4с
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES} не удалась: {e}. Жду {wait}с...")
                time.sleep(wait)

        # Все попытки исчерпаны — возвращаем сегменты с нейтральными оценками
        log.error(f"LLM не ответил после {config.LLM_MAX_RETRIES} попыток: {last_error}")
        return self._apply_scores(chunk, [])

    def _call_api(self, user_prompt: str) -> str:
        """Делает HTTP запрос к OpenRouter API, возвращает текст ответа."""
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": 0.2,  # низкая температура = стабильный JSON
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ── Промпт ────────────────────────────────────────────────────────────────

    def _build_prompt(self, segments: List[Dict]) -> str:
        """
        Формирует текст запроса к LLM с нумерованными сегментами.
        """
        lines = ["Вот сегменты транскрипции. Оцени каждый:\n"]
        for i, seg in enumerate(segments):
            start = _fmt_time(seg["start"])
            end   = _fmt_time(seg["end"])
            text  = seg.get("text", "").strip() or "[тишина]"
            lines.append(f"[{i}] {start}–{end}  {text}")

        lines.append(
            "\nВерни JSON: "
            '{"segments": [{"index": 0, "score": 0.0, "keep": false, "reason": "..."}, ...]}'
        )
        return "\n".join(lines)

    # ── Парсинг ответа ────────────────────────────────────────────────────────

    def _parse_response(self, text: str, expected_count: int) -> List[Dict]:
        """
        Парсит JSON из ответа LLM.
        Пробует прямой json.loads, затем ищет JSON в тексте через regex.
        """
        # Попытка 1: прямой парсинг
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "segments" in data:
                return data["segments"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Попытка 2: извлечь JSON-объект из текста (LLM иногда добавляет преамбулу)
        match = re.search(r'\{.*"segments"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("segments", [])
            except json.JSONDecodeError:
                pass

        # Попытка 3: найти JSON-массив
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        log.warning(f"Не удалось распарсить ответ LLM:\n{text[:500]}")
        return []

    def _apply_scores(self, chunk: List[Dict], scored: List[Dict]) -> List[Dict]:
        """
        Применяет оценки LLM к сегментам чанка.
        Если LLM не вернул оценку для сегмента — keep=False, score=0.5.
        """
        # Индексируем оценки по полю "index"
        score_map: Dict[int, Dict] = {item["index"]: item for item in scored if "index" in item}

        result = []
        for i, seg in enumerate(chunk):
            seg_copy = dict(seg)
            llm_data = score_map.get(i, {})

            raw_score = float(llm_data.get("score", 0.5))
            raw_keep  = llm_data.get("keep", None)

            # Если LLM не прислал keep — определяем по порогу
            if raw_keep is None:
                keep = raw_score >= _KEEP_THRESHOLD
            else:
                keep = bool(raw_keep) and raw_score >= _KEEP_THRESHOLD

            seg_copy["score"]  = round(raw_score, 3)
            seg_copy["keep"]   = keep
            seg_copy["reason"] = str(llm_data.get("reason", ""))
            result.append(seg_copy)

        return result


# ── Утилита ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Форматирует секунды в MM:SS для промпта."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
