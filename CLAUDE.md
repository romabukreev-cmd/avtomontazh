# Автомонтаж — паспорт проекта

Система автоматического монтажа видео. Принимает сырые записи с экрана и вебкой,
возвращает два готовых ролика: вертикальный 9:16 и горизонтальный 16:9.

---

## Деплой — строгий порядок

1. Изменения вносить **только локально**
2. `git add` + `git commit` + `git push`
3. Только после пуша — деплой на сервер:
   ```bash
   ssh -i "C:/Users/Роман/.ssh/id_ed25519" root@85.239.33.163 "cd ~/Автомонтаж && git pull && sudo systemctl restart automontazh && echo OK"
   ```

**Никогда не редактировать файлы напрямую на сервере.**

---

## Архитектура пайплайна (v3, с 2026-03-13)

```
Telegram-бот
    └── _pipeline(session, progress)
            ├── 1. SessionManager.concat_files()     → screen.mp4 + webcam.mp4
            ├── 2. Transcriber.transcribe()          → segments (sentence-level, word timestamps)
            ├── 3. LLMAnalyzer.analyze(segments)     → kept_segments
            └── 4. renderer.render_vertical()
                 renderer.render_horizontal()
```

**Ключевая идея:** LLM получает Whisper-сегменты (sentence-level) чанками по 30 штук
с перекрытием 5, возвращает `{"delete": [...]}`. Kept-сегменты напрямую идут в renderer.

---

## Карта файлов

| Файл | Что делает |
|------|-----------|
| `main.py` | Точка входа. Transcriber создаётся ОДИН РАЗ, pipeline через closure |
| `config.py` | Все настройки. Менять поведение только здесь |
| `modules/bot.py` | Telegram-бот. Очередь сессий, rclone sync/upload |
| `modules/transcriber.py` | faster-whisper large-v3. Аудио → сегменты с word timestamps |
| `modules/llm_analyzer.py` | OpenRouter API. Чанки по 30 сегментов, `analyze()` → kept |
| `modules/renderer.py` | FFmpeg. `render_vertical()` + `render_horizontal()` |
| `modules/session_manager.py` | Сканирование input/, concat нескольких частей |
| `modules/utils.py` | setup_logging, ensure_dirs, format_duration |

---

## Форматы выходных видео

| Формат | Файл | Размер | Лимит |
|--------|------|--------|-------|
| Вертикальный | `vertical_9min.mp4` | 1080×1920 | 9 мин |
| Горизонтальный | `horizontal_9min.mp4` | 1920×1080 | 9 мин |

- Вертикальный — split-screen: экран сверху, вебка снизу
- Горизонтальный — полный экран + PiP-кружок вебки в правом нижнем углу

---

## Ключевые константы config.py

```python
WHISPER_MODEL    = "large-v3"
WHISPER_LANGUAGE = "ru"
LLM_MODEL        = "anthropic/claude-sonnet-4-6"   # через OpenRouter

VAD_MIN_SILENCE_MS = 400    # граница между речевыми регионами
VAD_SPEECH_PAD_MS  = 200    # буфер вокруг речи

VIDEO_CODEC   = "libx264"
VIDEO_PRESET  = "ultrafast"
FFMPEG_THREADS = 0           # 0 = FFmpeg сам определяет

SCREEN_SOURCE_HEIGHT = 1200  # если 1920×1200 → кроп до 1080
SCREEN_CROP_Y = 60

DELETE_INPUT_AFTER_PROCESSING = True
CLEANUP_TEMP_ON_SUCCESS       = True
```

---

## Архитектурные правила

### Transcriber — синглтон, создаётся один раз
Модель large-v3 весит ~3GB. `Transcriber()` создаётся в `main()` и передаётся
в pipeline через closure. **Никогда не создавать внутри pipeline или в цикле.**

### LLM-анализ: чанки, не один запрос
- `CHUNK_SIZE = 30` сегментов на один вызов (~4 мин видео)
- `OVERLAP = 5`, `STEP = 25`
- Абсолютные индексы сегментов не меняются между чанками
- LLM возвращает `{"delete": [indices]}`, `_collect_deletions` объединяет через `set`

### Transcriber: границы из word timestamps
- `seg_start = first_word.start - 0.05`
- `seg_end = min(last_word.end, last_word.start + 1.5) + 0.15`
- Clipping: сегмент N не может перекрывать начало N+1

### Renderer принимает список сегментов
`render_vertical(kept_segments, ...)` и `render_horizontal(kept_segments, ...)` —
каждый сегмент: `{start, end, ...}`. FFmpeg concat demuxer с inpoint/outpoint.

---

## Типичные ошибки

| Симптом | Причина | Решение |
|---------|---------|---------|
| LLM оставляет 90%+ сегментов | Промпт слишком мягкий | Смотреть `_build_prompt` в llm_analyzer |
| Технические повторы слов | Двойная обработка границ сегментов | Clipping в transcriber |
| `git pull` падает с конфликтом | Прямая правка на сервере | `git reset --hard HEAD` → `git pull` |
