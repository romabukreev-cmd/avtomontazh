# Инструкция по настройке проекта

Делаем всё по шагам. Шаги 0–1 — один раз на твоём компьютере. Остальные — на VPS через консоль Timeweb.

---

## Шаг 0. Подключить проект к GitHub

Код пишем локально в VS Code, храним на GitHub. На сервер заливаем через `git pull` — быстро и надёжно.

### 0.1 Создай репозиторий на GitHub

1. Зайди на [github.com](https://github.com) → **New repository**
2. Название: `avtomontazh` (или любое)
3. Выбери **Private**
4. **Не** добавляй README, .gitignore — они уже есть
5. Нажми **Create repository**
6. Скопируй URL вида `https://github.com/твой-юзер/avtomontazh.git`

### 0.2 Инициализируй git в папке проекта

Открой терминал в VS Code (`Ctrl + \``) и выполни:

```bash
cd "d:\DOC\Documents\PROJECTS\Автомонтаж"

git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/твой-юзер/avtomontazh.git
git push -u origin main
```

> После `git push` GitHub попросит войти — войди через браузер (откроется автоматически).

### 0.3 Рабочий процесс — как обновлять код

Каждый раз когда ты что-то изменил в коде:

```bash
git add .
git commit -m "описание что изменил"
git push
```

На сервере чтобы получить обновления:

```bash
cd ~/Автомонтаж
git pull
```

---

## Шаг 1. Установка зависимостей на VPS

Заходи в консоль Timeweb (кнопка «Консоль» на странице сервера).

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg git
```

Проверить:
```bash
python3 --version   # должно быть 3.9+
ffmpeg -version
git --version
```

---

## Шаг 2. Установка rclone

```bash
curl https://rclone.org/install.sh | sudo bash
rclone version   # проверить
```

---

## Шаг 3. Подключить Яндекс Диск к rclone

```bash
rclone config
```

В диалоге отвечай:

```
n              ← новый remote
Имя: yadisk    ← должно совпадать с RCLONE_REMOTE_NAME в config.py
Тип: webdav    ← Яндекс Диск работает по протоколу WebDAV
```

Дальше:
```
URL: https://webdav.yandex.ru
Vendor: other
User: твой_логин@yandex.ru
Password: y  → вводи пароль приложения (не основной!)
```

**Как получить пароль приложения для Яндекс Диска:**
1. Зайди на id.yandex.ru → Безопасность → Пароли приложений
2. Создай новый пароль для «WebDAV»
3. Скопируй его — он показывается только один раз

Проверить подключение:
```bash
rclone ls yadisk:
```
Должен показать файлы твоего Яндекс Диска.

---

## Шаг 4. Создать папки на Яндекс Диске

На Яндекс Диске (через браузер или приложение) создай:

```
📁 Автомонтаж/
    📁 input/    ← сюда будешь класть папки сессий
    📁 output/   ← сюда система будет загружать результаты
```

---

## Шаг 5. Как загружать файлы на Яндекс Диск

Система не знает ничего об OBS или о том чем ты снимаешь.
Она просто берёт файлы из папки на Яндекс Диске — и всё.

**Твой процесс:**
1. Снял видео (чем угодно и как угодно)
2. Переименовал файлы по правилу ниже
3. Создал папку сессии на Яндекс Диске, положил файлы
4. Открыл Telegram → `/sync` → нажал кнопку → ждёшь результата

**Структура папки на Яндекс Диске:**

```
Яндекс Диск / Автомонтаж / input /
  2024-01-15_logo-design/       ← одна съёмка = одна папка (имя придумываешь сам)
    screen_001.mp4              ← запись экрана, часть 1
    screen_002.mp4              ← часть 2 (если была пауза во время записи)
    webcam_001.mp4              ← запись вебки, часть 1
    webcam_002.mp4              ← часть 2
```

**Правила именования:**
- Файлы экрана: `screen_001.mp4`, `screen_002.mp4`, ...
- Файлы вебки: `webcam_001.mp4`, `webcam_002.mp4`, ...
- Количество файлов экрана и вебки **должно совпадать**
- Нумерация строго с тремя цифрами: `001`, `002`, `003` (не `1`, `2`, `3`)
- Имя папки: латиница, без пробелов, пробелы заменяй на `-`

---

## Шаг 6. Скачать проект на VPS с GitHub

```bash
cd ~
git clone https://github.com/твой-юзер/avtomontazh.git Автомонтаж
cd Автомонтаж
```

---

## Шаг 7. Создать .env файл с ключами

```bash
cd ~/Автомонтаж
cp .env.example .env
nano .env
```

Заполни:
- `OPENROUTER_API_KEY` — ключ с openrouter.ai
- `TELEGRAM_BOT_TOKEN` — токен от @BotFather (шаг 8)
- `TELEGRAM_CHAT_ID` — твой числовой ID (шаг 9)

Сохрани: `Ctrl+O`, Enter, `Ctrl+X`

> **.env никогда не попадёт на GitHub** — он уже добавлен в .gitignore. Ключи хранятся только на сервере.

---

## Шаг 8. Создать Telegram-бота

1. Открой Telegram, напиши **@BotFather**
2. Команда `/newbot`
3. Придумай имя бота (например: `Автомонтаж`)
4. Придумай username (например: `automontazh_bot`) — должен заканчиваться на `bot`
5. BotFather пришлёт токен вида `1234567890:AAHdqTcvCH...`
6. Скопируй его в `.env` → `TELEGRAM_BOT_TOKEN`

---

## Шаг 9. Узнать свой Telegram ID

1. Напиши **@userinfobot** в Telegram
2. Он ответит сообщением с `Id: 123456789`
3. Скопируй это число в `.env` → `TELEGRAM_CHAT_ID`

> Это важно — бот будет принимать команды ТОЛЬКО от этого ID.

---

## Шаг 10. Установить Python-зависимости

```bash
cd ~/Автомонтаж

# Создать изолированное окружение
python3 -m venv venv

# Активировать
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

> Установка занимает несколько минут — Whisper скачивает pytorch (~2 ГБ).

---

## Шаг 11. Запустить как постоянный сервис

```bash
sudo nano /etc/systemd/system/automontazh.service
```

Вставить:

```ini
[Unit]
Description=Автомонтаж — автоматический монтаж видео
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/Автомонтаж
ExecStart=/root/Автомонтаж/venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Сохрани: `Ctrl+O`, Enter, `Ctrl+X`

```bash
sudo systemctl daemon-reload
sudo systemctl enable automontazh
sudo systemctl start automontazh

# Проверить что запустилось
sudo systemctl status automontazh
```

---

## Как обновить код на сервере

Когда поменял что-то локально и запушил на GitHub:

```bash
cd ~/Автомонтаж
git pull
sudo systemctl restart automontazh
```

---

## Готово! Как пользоваться

**После каждой съёмки:**

1. Переименуй файлы по правилу: `screen_001.mp4`, `webcam_001.mp4`, и т.д.
2. Создай папку на Яндекс Диске: `Автомонтаж/input/2024-01-15_название/`
3. Положи файлы в эту папку

**В Telegram (в чате с твоим ботом):**

```
/sync      ← скачать новые файлы с Яндекс Диска
/sessions  ← посмотреть доступные сессии
```

Нажми кнопку с нужной сессией → бот начнёт обработку и будет присылать прогресс.

Когда закончит — три видео появятся в `Яндекс Диск / Автомонтаж / output / название_сессии/`.

---

## Команды бота

| Команда    | Что делает |
|------------|-----------|
| `/start`   | Приветствие и список команд |
| `/sync`    | Скачать новые файлы с Яндекс Диска |
| `/sessions`| Показать сессии, готовые к обработке |
| `/status`  | Статус текущей обработки |

---

## Частые проблемы

**Бот не отвечает:**
```bash
sudo journalctl -u automontazh -n 50
```

**rclone не находит файлы:**
```bash
rclone ls yadisk:Автомонтаж/input/
```

**Whisper медленно работает:**
В `config.py` поменяй `WHISPER_MODEL = "tiny"` — быстрее, чуть хуже качество.

**Не хватает места на VPS:**
Видеофайлы большие. После обработки исходники в `input/` можно удалять вручную.

**Смотреть лог в реальном времени:**
```bash
tail -f ~/Автомонтаж/logs/automontazh.log
```
