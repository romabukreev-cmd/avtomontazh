# Автомонтаж — паспорт проекта

Система автоматического монтажа видео. Принимает сырые записи с экрана и вебкой,
возвращает три готовых ролика (вертикальный 9 мин, вертикальный 2 мин, горизонтальный 9 мин).

---

## Архитектура пайплайна

```
Telegram-бот
    └── process_session(session, progress, transcriber)
            ├── 1. SessionManager.concat_files()      → screen.mp4 + webcam.mp4
            ├── 2. Transcriber.transcribe_and_cut_pauses()  → speech_segments
            ├── 3. LLMAnalyzer.analyze()              → scored_segments (keep/score)
            ├── 4. TimelineBuilder.build_long() ×2    → timeline_vertical, timeline_horizontal
            ├── 5. LLMAnalyzer.analyze_highlights()   → highlights_scored
            ├── 6. TimelineBuilder.build_highlights() → timeline_highlight
            └── 7. VideoRenderer.render_vertical() ×2 + render_horizontal()
```

---

## Карта файлов

| Файл | Что делает |
|------|-----------|
| `main.py` | Точка входа. Создаёт Transcriber ОДИН РАЗ, передаёт в pipeline через closure |
| `config.py` | Все настройки. Менять поведение системы только здесь |
| `modules/bot.py` | Telegram-бот. Управление очередью, отправка прогресса, rclone sync/upload |
| `modules/transcriber.py` | faster-whisper large-v3. Аудио → сегменты речи с таймстемпами |
| `modules/llm_analyzer.py` | OpenRouter API. Два прохода: analyze() + analyze_highlights() |
| `modules/timeline.py` | Сборка таймлайна из сегментов. Merge, padding, prune by priority |
| `modules/renderer.py` | FFmpeg. Две функции: render_vertical() + render_horizontal() |
| `modules/session_manager.py` | Сканирование input/, concat нескольких частей в один файл |
| `modules/utils.py` | setup_logging, ensure_dirs, format_duration |

---

## Форматы выходных видео

| Формат | Файл | Размер | Откуда | Лимит |
|--------|------|--------|--------|-------|
| Формат 1 | `vertical_9min.mp4` | 1080×1920 | `timeline_vertical` | 540с (9 мин) |
| Формат 2 | `vertical_2min.mp4` | 1080×1920 | `timeline_highlight` | 130с (2:10) |
| Формат 3 | `horizontal_9min.mp4` | 1920×1080 | `timeline_horizontal` | 540с (9 мин) |

- Форматы 1 и 2 — split-screen: экран сверху, вебка снизу
- Формат 3 — полный экран + PiP-кружок вебки в правом нижнем углу

---

## Ключевые константы config.py

```python
WHISPER_MODEL = "large-v3"          # лучший для русского языка
WHISPER_LANGUAGE = "ru"
PAUSE_THRESHOLD_SEC = 0.8           # пауза длиннее → вырезается
SILENCE_NOISE_DB = -35              # порог тишины для silencedetect
MIN_SEGMENT_DURATION_SEC = 1.5      # короче → выбрасывается
SEGMENT_START_PADDING_SEC = 0.1     # зазор перед первым словом (убирает вдохи)

LLM_MODEL = "anthropic/claude-sonnet-4-6"  # через OpenRouter
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "ultrafast"          # скорость vs качество
FFMPEG_THREADS = 0                  # 0 = FFmpeg сам определяет

SCREEN_SOURCE_HEIGHT = 1200         # если 1920×1200 → кроп 60px сверху/снизу
SCREEN_CROP_Y = 60                  # (SCREEN_SOURCE_HEIGHT - 1080) // 2

PIP_WIDTH = PIP_HEIGHT = 350        # размер PiP-кружка
PIP_CORNER_RADIUS = 20
PIP_MARGIN_RIGHT = PIP_MARGIN_BOTTOM = 60

FORMAT_1/2/3["max_duration_sec"]    # лимиты для каждого формата
```

---

## Архитектурные правила

### Transcriber — синглтон, создаётся один раз
Модель large-v3 весит ~3GB и грузится несколько минут.
`Transcriber()` создаётся в `main()` и передаётся в каждую сессию через closure.
**Никогда не создавать внутри process_session или в цикле.**

### Segment timing: два места, один принцип
- `transcriber.py`: `seg_start = first_word.start - SEGMENT_START_PADDING_SEC` (0.1с)
- `timeline.py`: `_PRE_ROLL_SEC = 0.0` — намеренно ноль!

Нельзя увеличивать `_PRE_ROLL_SEC` — это вернёт вдохи/дыхание в ролик.
Весь pre-roll управляется только через `SEGMENT_START_PADDING_SEC` в config.

### LLM-анализ: два прохода
1. `analyze(segments, max_sec)` — полная транскрипция → keep/score для всех сегментов
2. `analyze_highlights(segments, max_sec)` — второй проход только по лучшим сегментам

Приоритеты в промпте:
- **P1 (обязательный)**: убрать все повторы и незаконченные мысли — всегда
- **P2 (условный)**: убрать менее важное — только если после P1 хронометраж > max_sec

### Интеграции
Сегменты с `integration: "youtube"` → только в `timeline_horizontal` (Формат 3)
Сегменты с `integration: "social"` → только в `timeline_vertical` (Форматы 1 и 2)
Это фильтруется в `main.py`:
```python
kept_vertical   = [s for s in kept if s.get("integration") != "youtube"]
kept_horizontal = [s for s in kept if s.get("integration") != "social"]
kept_content    = [s for s in kept if not s.get("integration")]
```

### Сообщения в Telegram
Весь прогресс обновляет одно `status_msg` (edit_text).
После завершения рендеров — финальный саммари (остаётся как предпоследнее).
Загрузка на Drive — отдельное новое сообщение, редактируется в "✅ Готово!".

---

## Типичные ошибки и решения

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `argument of type 'int' is not iterable` | LLM вернул массив чисел в поле analysis | `isinstance(item, dict)` guard в `_apply_scores` |
| LLM оставляет 90%+ сегментов | Слишком мягкий промпт | Проверь формулировку P1 в `_build_system_prompt` |
| Вдохи в начале сегментов | `_PRE_ROLL_SEC > 0` | Вернуть в `0.0` |
| Повторы не удаляются | LLM обрабатывает блоки, не целиком | В промпте: "Анализируй весь текст ЦЕЛИКОМ" |

---

## Деплой

```bash
cd ~/Автомонтаж && git pull && sudo systemctl restart automontazh
```

Первый запуск после добавления зависимости:
```bash
pip install <пакет> && sudo systemctl restart automontazh
```
