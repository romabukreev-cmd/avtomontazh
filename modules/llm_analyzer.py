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


def _build_system_prompt(total_duration: float, max_sec: float) -> str:
    total_min = total_duration / 60
    max_min   = max_sec / 60
    return f"""Ты — опытный видеомонтажёр для YouTube. Тебе дана ПОЛНАЯ транскрипция видео.

КОНТЕКСТ ВИДЕО:
{config.VIDEO_CONTEXT}

Исходный хронометраж: ~{total_min:.0f} мин
Лимит итогового видео: {max_min:.0f} мин ({max_sec:.0f}с) — но только если нужно (см. ниже)

Важно: паузы (тишина) уже вырезаны до тебя. Ты работаешь только с речевыми сегментами.

═══════════════════════════════════════════════
ПАУЗЫ МЕЖДУ СЕГМЕНТАМИ
═══════════════════════════════════════════════
Между сегментами указана длина молчания (═══ пауза Xс ═══).
- Длинная пауза (≥ 10с) — граница между частями процесса.
- Короткая пауза (< 5с) — нормальный ритм речи. Повторы и переформулировки
  происходят именно здесь: та же мысль в двух соседних сегментах с короткой паузой.

═══════════════════════════════════════════════
ПРИОРИТЕТЫ — СТРОГО В ЭТОМ ПОРЯДКЕ
═══════════════════════════════════════════════

▶▶ ПРИОРИТЕТ 1 — ОБЯЗАТЕЛЬНЫЙ (выполни всегда, независимо от хронометража):
   УБЕРИ ВСЕ ПОВТОРЫ, ФАЛЬСТАРТЫ И НЕЗАКОНЧЕННЫЕ МЫСЛИ

   Это главная задача. Даже если после этого останется 7 мин вместо {max_min:.0f} —
   это правильный результат. Чистое короткое видео лучше длинного с повторами.

   1а. ЦЕПОЧКИ ПОПЫТОК / СМЫСЛОВЫЕ ПОВТОРЫ
       Автор часто проговаривает одну мысль несколько раз, каждый раз иначе:
         первая попытка (обрывается) → вторая → третья → финальная формулировка
       Это ОДНА мысль. Оставь только самую полную/чёткую (обычно последнюю).

       Алгоритм для каждого сегмента:
       1. Сформулируй его суть в 5–7 словах.
       2. Просмотри предыдущие 10–15 сегментов: есть ли там сегменты с той же сутью?
          (даже если слова абсолютно разные — сравнивай смысл)
       3. Найди ВСЮ группу (2, 5, 10+ сегментов — не ограничено).
       4. Из группы оставь одну лучшую версию, всем остальным — keep=false.

       Исключение: автор углубляет или добавляет новое — это развитие, не повтор.

   1б. НЕЗАКОНЧЕННЫЕ МЫСЛИ
       Сегмент обрывается на полуслове или не несёт самостоятельного смысла → keep=false.
       Тест: «Если убрать этот сегмент, зритель ничего не потеряет?» Если да — убирай.

▶ ПРИОРИТЕТ 2 — УСЛОВНЫЙ:
  Выполняй ТОЛЬКО если после Приоритета 1 хронометраж всё ещё превышает {max_sec:.0f}с.
  Тогда дополнительно удаляй менее важное, пока не уложишься в лимит:
  - Технические паузы: автор ждёт загрузки, рендера, зависания инструмента
  - Нерелевантные отступления: разговоры не по теме работы
  - Менее важные пояснения (если суть уже раскрыта в других сегментах)

  Что НЕ трогать при Приоритете 2:
  - Начало видео (первые связные сегменты где автор обозначает тему)
  - Конец видео (финальный результат, вывод — последние 3–5 сегментов)
  - Ключевые моменты: решения, находки, итерации, смена подхода
  - Интеграции (рекламные вставки — обязательны)

═══════════════════════════════════════════════
ПРАВИЛА СОХРАНЕНИЯ
═══════════════════════════════════════════════
- Объяснения что делает автор
- Ключевые решения и действия
- Интересные находки и «эврика»-моменты
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
  Примеры: «ссылка в описании под видео», «смотрите видео у меня в канале».

"social" — интеграция для TikTok / Instagram (вертикальный формат).
  Признаки: автор говорит о ссылке в шапке профиля или описании профиля (аккаунта, не видео),
  либо просит написать кодовое слово в комментарии или директ.
  Примеры: «ссылка в шапке профиля», «напишите мне слово [X]».

null — обычный контент, не интеграция.

Если одна и та же интеграция записана несколько раз (дубли/пересъёмки):
- ВСЕ попытки кроме последней по времени → keep=false, "integration": "<тип>"
- Последняя запись → keep=true, score=1.0, "integration": "<тип>"

═══════════════════════════════════════════════
ОБЯЗАТЕЛЬНАЯ ФАЗА АНАЛИЗА (выполни ДО оценки сегментов)
═══════════════════════════════════════════════
Прочитай ВСЕ сегменты целиком. Запиши в поле "analysis":
1. На какие смысловые части делится видео
2. Какие темы/мысли повторяются и в каких конкретно сегментах (индексы)
3. Что точно стоит убрать и почему

Только после этого анализа — расставляй keep/score.

═══════════════════════════════════════════════
ФОРМАТ ОТВЕТА
═══════════════════════════════════════════════
Верни строго JSON без лишнего текста:
{{"analysis": "...", "segments": [{{"index": 0, "score": 0.85, "keep": true, "reason": "кратко", "integration": null}}]}}

score=1.0 — ключевой момент, score=0.0 — однозначно вырезать.
reason — одна фраза, зачем оставляем или почему убираем.
integration — "youtube", "social", или null.

ФИНАЛЬНАЯ ПРОВЕРКА перед ответом:
1. Пройдись по всем сегментам с keep=true.
   Для каждого — проверь, нет ли в предыдущих 10–15 сегментах других с той же сутью.
   Из каждой группы оставь только самую полную/чёткую, остальным — keep=false.
   Это Приоритет 1 — не пропускай ни одной группы повторов.

2. Оцени суммарный хронометраж keep=true сегментов.
   Если он превышает {max_sec:.0f}с — применяй Приоритет 2 (условный),
   убирай менее важные сегменты пока не уложишься в лимит."""


def _build_highlights_prompt(segments: List[Dict], max_sec: float) -> str:
    max_min = max_sec / 60
    lines = [
        f"Это уже отобранные фрагменты из длинной версии видео. "
        f"Выбери из них лучшие суммарно до {max_min:.1f} минут ({max_sec:.0f} секунд).\n",
        "ТРЕБОВАНИЯ к короткой версии:",
        "",
        "ПРОПОРЦИИ (строго соблюдай):",
        f"- Середина (сам процесс): не менее 60% итогового хронометража",
        f"- Начало + конец вместе: не более 40%",
        "",
        "НАЧАЛО (≤ 20 секунд): 1–2 коротких сегмента где автор обозначает тему.",
        "  Не нужно долгое вступление.",
        "",
        "СЕРЕДИНА (≥ 60%): самые интересные и показательные моменты САМОГО ПРОЦЕССА.",
        "  Конкретные решения, приёмы, результаты промежуточных шагов.",
        "  Избегай: объяснений что будет дальше, технических пауз, чтения инструкций.",
        "",
        "КОНЕЦ (≤ 20 секунд): момент когда виден ГОТОВЫЙ результат или финальный вывод.",
        "  НЕ ПОДХОДИТ: чтение промптов, подготовка к следующему шагу, рассуждения о том что будет.",
        "  ПОДХОДИТ: 'вот что получилось', демонстрация финала, ключевой вывод.",
        "",
        "НЕ бери контент только из начала или только из конца — покрывай весь процесс.",
        "НЕ включай: чтение технических инструкций, настройку инструментов, рекламные интеграции.\n",
        "Фрагменты:\n",
    ]
    for i, seg in enumerate(segments):
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        text  = seg.get("text", "").strip() or "[тишина]"
        dur   = seg["end"] - seg["start"]
        lines.append(f"[{i}] {start}–{end} ({dur:.0f}с)  {text}")

    lines.append(
        f"\nСначала в поле \"analysis\" опиши: какие фрагменты покрывают начало/середину/конец "
        f"процесса, какие наиболее показательны, какие можно пропустить. Затем выбери сегменты.\n"
        f"Верни JSON: "
        f'{{"analysis": "...", "segments": [{{"index": 0, "score": 0.0, "keep": false, "reason": "..."}}]}}'
    )
    return "\n".join(lines)


class LLMAnalyzer:

    def analyze(self, segments: List[Dict], max_sec: float = None) -> List[Dict]:
        """
        Полный анализ транскрипции. Возвращает сегменты с полями score, keep, reason.

        Приоритеты:
          1. Удалить ВСЕ повторы и фальстарты (всегда, даже если результат < max_sec)
          2. Если после п.1 хронометраж > max_sec — дополнительно убрать менее важное

        max_sec: лимит итогового видео в секундах (по умолчанию FORMAT_1)
        """
        if not segments:
            return []

        if max_sec is None:
            max_sec = config.FORMAT_1["max_duration_sec"]

        total_duration = sum(s["end"] - s["start"] for s in segments)
        total_chars    = sum(len(s.get("text", "")) for s in segments)
        max_chars      = config.LLM_CHUNK_SIZE_TOKENS * 4

        system = _build_system_prompt(total_duration, max_sec)

        if total_chars <= max_chars:
            log.info(
                f"LLM анализ: {len(segments)} сегментов, "
                f"{total_duration/60:.1f} мин → лимит {max_sec/60:.1f} мин"
            )
            result = self._analyze_chunk(segments, system)
        else:
            chunks = self._split_into_chunks(segments)
            log.info(
                f"LLM анализ: {len(segments)} сегментов, {len(chunks)} чанков, "
                f"{total_duration/60:.1f} мин → лимит {max_sec/60:.1f} мин"
            )
            result = []
            for i, chunk in enumerate(chunks):
                log.info(f"Анализирую чанк {i+1}/{len(chunks)} ({len(chunk)} сегментов)...")
                result.extend(self._analyze_chunk(chunk, system))

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
            "Твоя задача — выбрать лучшие фрагменты для короткой версии ролика.\n\n"
            f"КОНТЕКСТ ВИДЕО:\n{config.VIDEO_CONTEXT}\n\n"
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

        log.error(f"⚠️ LLM хайлайты не ответил: {last_error}. Проверь OPENROUTER_API_KEY и баланс.")
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

        log.error(f"⚠️ LLM НЕ ОТВЕТИЛ после {config.LLM_MAX_RETRIES} попыток: {last_error}. "
                  f"Все сегменты будут сохранены (fallback). Проверь OPENROUTER_API_KEY и баланс.")
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
            if i < len(segments) - 1:
                pause = segments[i + 1]["start"] - seg["end"]
                if pause >= 3:
                    lines.append(f"    ═══ пауза {pause:.0f}с ═══")

        lines.append(
            "\nВерни JSON: "
            '{"analysis": "...", "segments": [{"index": 0, "score": 0.0, "keep": false, "reason": "..."}, ...]}'
        )
        return "\n".join(lines)

    # ── Парсинг ответа ────────────────────────────────────────────────────────

    def _parse_response(self, text: str, expected_count: int) -> List[Dict]:
        # Снимаем markdown-обёртку (```json ... ```)
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text)

        # Попытка 1: прямой парсинг
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "segments" in data:
                if "analysis" in data:
                    log.info(f"LLM анализ видео: {data['analysis']}")
                return data["segments"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Попытка 2: извлечь JSON-объект из текста (жадный поиск — берём наибольший объект)
        match = re.search(r'\{.*"segments"\s*:\s*\[.*\]\s*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "analysis" in data:
                    log.info(f"LLM анализ видео: {data['analysis']}")
                return data.get("segments", [])
            except json.JSONDecodeError:
                pass

        # Попытка 3: найти JSON-массив объектов (жадный; проверяем что элементы — dict, не int)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list) and all(isinstance(item, dict) for item in result):
                    return result
            except json.JSONDecodeError:
                pass

        log.warning(f"Не удалось распарсить ответ LLM:\n{text[:500]}")
        return []

    def _apply_scores(self, chunk: List[Dict], scored: List[Dict]) -> List[Dict]:
        score_map: Dict[int, Dict] = {
            item["index"]: item
            for item in scored
            if isinstance(item, dict) and "index" in item
        }

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
