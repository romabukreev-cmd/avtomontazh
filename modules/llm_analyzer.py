"""
llm_analyzer.py — анализ транскрипции через LLM (OpenRouter API).

Два прохода:
  1. analyze(segments) — полный анализ всей транскрипции:
       - оценивает важность каждого сегмента (score 0..1)
       - удаляет незаконченные мысли и фальстарты
       - результат используется для 10-минутных версий
  2. analyze_highlights(segments, max_sec) — второй проход для 3-минутной версии:
       - принимает уже отфильтрованные сегменты из 10-мин таймлайна
       - просит LLM выбрать лучшие max_sec секунд как самостоятельный ролик
"""

import json
import logging
import re
import time
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)

_KEEP_THRESHOLD = 0.5


def _build_system_prompt() -> str:
    return f"""Ты — опытный видеомонтажёр для YouTube. Тебе дана ПОЛНАЯ транскрипция видео.

КОНТЕКСТ ВИДЕО:
{config.VIDEO_CONTEXT}

═══════════════════════════════════════════════
ГЛАВНЫЙ ПРИНЦИП — СОХРАНИ ЛОГИЧЕСКУЮ ДУГУ
═══════════════════════════════════════════════
Финальное видео должно иметь начало, середину и конец.
Цель: финальное видео = 40–60% от оригинала.

НАЧАЛО: оставляй с первой фразы где автор говорит связно и по делу.
  Удаляй: паузы перед стартом, запинки, незаконченные вступления.
  Первый keep=true сегмент — это начало ролика.

КОНЕЦ: последние 3–5 сегментов — почти всегда keep=true.
  Зритель должен увидеть финальный результат и завершающую мысль.
  Не обрезай конец — это критически важно.

СЕРЕДИНА: сжимай, оставляя ключевые моменты из каждого этапа процесса.

═══════════════════════════════════════════════
ПРАВИЛА УДАЛЕНИЯ
═══════════════════════════════════════════════

1. НЕЗАКОНЧЕННАЯ МЫСЛЬ
   Если сегмент не несёт самостоятельного смысла — обрывается на полуслове,
   начинает мысль но не завершает её — ставь keep=false.
   Тест: "Если убрать этот сегмент, зритель ничего не потеряет?"
   Если да — убирай.

2. ФАЛЬСТАРТ / ПОВТОР
   Если автор запнулся, не договорил, и в следующем сегменте начинает
   ту же мысль заново (дословно или другими словами) — предыдущий сегмент
   это неудавшаяся попытка. Удаляй его, оставляй только завершённую версию.

   Признаки фальстарта:
   - Текущий сегмент обрывается или незакончен
   - Следующий сегмент начинает похожую/ту же мысль
   - Повтор не обязательно дословный: одна мысль разными словами тоже повтор

3. ТЕХНИЧЕСКИЕ ПАУЗЫ
   Сегменты где автор молчит или ждёт (загрузка, рендер, зависание).

4. НЕРЕЛЕВАНТНЫЕ ОТСТУПЛЕНИЯ
   Разговоры не по теме работы.

═══════════════════════════════════════════════
ПРАВИЛА СОХРАНЕНИЯ
═══════════════════════════════════════════════
- Объяснения что делает автор
- Ключевые решения и действия
- Интересные находки и "эврика"-моменты
- Критика, итерации, изменение решений
- Финальный результат и выводы

═══════════════════════════════════════════════
ИНТЕГРАЦИИ (рекламные вставки / CTA)
═══════════════════════════════════════════════
Интеграция — блок где автор напрямую обращается к зрителю с призывом к действию:
перейти по ссылке, написать кодовое слово, посмотреть другое видео и т.д.
Интеграции НЕЛЬЗЯ удалять — они обязательны, score=1.0, keep=true (кроме неудачных дублей).

КАК ОПРЕДЕЛИТЬ ТИП интеграции (поле "integration" в ответе):

"youtube" — интеграция для YouTube (горизонтальный формат).
  Признаки: автор говорит о ссылке которая находится ПОД видео, в описании к видео,
  упоминает другое видео в своём канале, говорит «закреплю/прикреплю ссылку» в комментарии под видео.
  Ключевой контекст: зритель видит видеоплеер и может кликнуть описание или закреплённый комментарий.
  Примеры: «ссылка в описании под видео», «смотрите видео у меня в канале», «прикреплю ссылку здесь».

"social" — интеграция для TikTok / Instagram (вертикальный формат).
  Признаки: автор говорит о ссылке в шапке профиля или описании профиля (аккаунта, не видео),
  либо просит написать кодовое слово в комментарии или директ.
  Ключевой контекст: в TikTok/Инстаграм нет кликабельных ссылок под видео — ссылку кладут
  в шапку профиля, зритель должен сам перейти в профиль.
  Примеры: «ссылка в шапке профиля», «ссылка в описании профиля», «напишите мне слово [X]».

null — обычный контент, не интеграция.

Если одна и та же интеграция записана несколько раз (дубли/пересъёмки):
- ВСЕ попытки кроме последней по времени → keep=false, "integration": "<тип>" (фальстарты)
- Последняя запись → keep=true, score=1.0, "integration": "<тип>"

═══════════════════════════════════════════════
ФОРМАТ ОТВЕТА
═══════════════════════════════════════════════
Верни строго JSON без лишнего текста:
{{"segments": [{{"index": 0, "score": 0.85, "keep": true, "reason": "кратко", "integration": null}}]}}

score=1.0 — ключевой момент, score=0.0 — однозначно вырезать.
reason — одна фраза, зачем оставляем или почему убираем.
integration — "youtube", "social", или null."""


def _build_highlights_prompt(segments: List[Dict], max_sec: float) -> str:
    max_min = max_sec / 60
    lines = [
        f"Это уже отобранные фрагменты из длинной версии видео. "
        f"Выбери из них лучшие суммарно до {max_min:.0f} минут.\n",
        "ТРЕБОВАНИЯ к 3-минутной версии:",
        "- Должна быть самостоятельным роликом с началом, серединой и концом",
        "- Начало: первый или один из первых сегментов — анонс темы",
        "- Конец: один из последних сегментов — финальный результат или вывод",
        "- Середина: самые яркие и интересные моменты процесса",
        "- НЕ бери 3 минуты из одного места — покрывай весь процесс\n",
        "Фрагменты:\n",
    ]
    for i, seg in enumerate(segments):
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        text  = seg.get("text", "").strip() or "[тишина]"
        dur   = seg["end"] - seg["start"]
        lines.append(f"[{i}] {start}–{end} ({dur:.0f}с)  {text}")

    lines.append(
        f"\nВерни JSON: "
        f'{{"segments": [{{"index": 0, "score": 0.0, "keep": false, "reason": "..."}}]}}'
    )
    return "\n".join(lines)


class LLMAnalyzer:

    def analyze(self, segments: List[Dict]) -> List[Dict]:
        """
        Полный анализ транскрипции. Возвращает сегменты с полями score, keep, reason.
        Отправляет всё одним запросом для понимания полного контекста.
        При слишком большом объёме — разбивает на чанки (fallback).
        """
        if not segments:
            return []

        total_chars = sum(len(s.get("text", "")) for s in segments)
        max_chars = config.LLM_CHUNK_SIZE_TOKENS * 4

        if total_chars <= max_chars:
            log.info(f"LLM анализ: {len(segments)} сегментов одним запросом")
            result = self._analyze_chunk(segments, _build_system_prompt())
        else:
            chunks = self._split_into_chunks(segments)
            log.info(f"LLM анализ: {len(segments)} сегментов, {len(chunks)} чанков")
            result = []
            for i, chunk in enumerate(chunks):
                log.info(f"Анализирую чанк {i+1}/{len(chunks)} ({len(chunk)} сегментов)...")
                result.extend(self._analyze_chunk(chunk, _build_system_prompt()))

        kept = sum(1 for s in result if s.get("keep"))
        log.info(f"LLM: оставлено {kept}/{len(result)} сегментов")
        return result

    def analyze_highlights(self, segments: List[Dict], max_sec: float) -> List[Dict]:
        """
        Второй проход: из уже отобранных сегментов (10-мин таймлайн) выбирает
        лучшие max_sec секунд для 3-минутной версии.

        Возвращает те же сегменты с обновлёнными keep/score.
        """
        if not segments:
            return []

        log.info(f"LLM хайлайты: выбираю лучшие {max_sec/60:.0f} мин из {len(segments)} сегментов")

        system = (
            "Ты — опытный видеомонтажёр для YouTube. "
            "Твоя задача — выбрать лучшие фрагменты для короткой версии ролика. "
            "Отвечай строго JSON без лишнего текста."
        )
        user = _build_highlights_prompt(segments, max_sec)

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw = self._call_api_with_system(system, user)
                scored = self._parse_response(raw, len(segments))
                result = self._apply_scores(segments, scored)
                kept = sum(1 for s in result if s.get("keep"))
                kept_dur = sum(s["end"] - s["start"] for s in result if s.get("keep"))
                log.info(f"LLM хайлайты: {kept} сегментов, {kept_dur:.0f}с")
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES}: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"LLM хайлайты не ответил: {last_error}")
        return self._apply_scores(segments, [])

    # ── Разбивка на чанки (fallback для очень длинных видео) ─────────────────

    def _split_into_chunks(self, segments: List[Dict]) -> List[List[Dict]]:
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

    def _analyze_chunk(self, chunk: List[Dict], system_prompt: str) -> List[Dict]:
        user_prompt = self._build_prompt(chunk)

        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                raw_response = self._call_api_with_system(system_prompt, user_prompt)
                scored = self._parse_response(raw_response, len(chunk))
                return self._apply_scores(chunk, scored)

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                log.warning(f"Попытка {attempt+1}/{config.LLM_MAX_RETRIES} не удалась: {e}. Жду {wait}с...")
                time.sleep(wait)

        log.error(f"LLM не ответил после {config.LLM_MAX_RETRIES} попыток: {last_error}")
        return self._apply_scores(chunk, [])

    def _call_api_with_system(self, system: str, user: str) -> str:
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
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": 0.2,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ── Промпт ────────────────────────────────────────────────────────────────

    def _build_prompt(self, segments: List[Dict]) -> str:
        lines = ["Вот полная транскрипция видео. Оцени каждый сегмент:\n"]
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
        # Попытка 1: прямой парсинг
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "segments" in data:
                return data["segments"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Попытка 2: извлечь JSON-объект из текста
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
        score_map: Dict[int, Dict] = {item["index"]: item for item in scored if "index" in item}

        result = []
        for i, seg in enumerate(chunk):
            seg_copy = dict(seg)
            llm_data = score_map.get(i, {})

            raw_score = float(llm_data.get("score", 0.5))
            raw_keep  = llm_data.get("keep", None)

            if raw_keep is None:
                keep = raw_score >= _KEEP_THRESHOLD
            else:
                keep = bool(raw_keep) and raw_score >= _KEEP_THRESHOLD

            seg_copy["score"]       = round(raw_score, 3)
            seg_copy["keep"]        = keep
            seg_copy["reason"]      = str(llm_data.get("reason", ""))
            seg_copy["integration"] = llm_data.get("integration", None) or None
            result.append(seg_copy)

        return result


# ── Утилита ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Форматирует секунды в MM:SS для промпта."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
