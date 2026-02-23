"""
renderer.py — сборка финальных видео через FFmpeg.

Как работает нарезка без промежуточных файлов:
  FFmpeg умеет принимать список отрезков через «concat demuxer»:
    1. Создаём временный текстовый файл (concat list):
         file '/path/to/source.mp4'
         inpoint 0.0
         outpoint 12.4
         file '/path/to/source.mp4'
         inpoint 14.1
         outpoint 28.7
         ...
    2. FFmpeg читает этот список и склеивает отрезки в один поток
    3. Поверх применяем фильтры (кроп, масштаб, наложение вебки)

Для каждого формата своя цепочка FFmpeg-фильтров (filtergraph):

─── Формат 1 и 2 (вертикальный split-screen 1080×1920) ──────────────────────

  [screen]  → crop центр до 9:8 (1215×1080) → scale 1080×960  → [top]
  [webcam]  → crop центр до 9:8 (1215×1080) → scale 1080×960  → [bottom]
  [top][bottom] → vstack → [out]  (1080×1920)

  FFmpeg filtergraph:
    [0:v]crop=1215:1080:352:0,scale=1080:960[top];
    [1:v]crop=1215:1080:352:0,scale=1080:960[bot];
    [top][bot]vstack[out]

─── Формат 3 (горизонтальный PiP 1920×1080) ─────────────────────────────────

  [screen]  → scale 1920×1080 (уже нужного размера) → [bg]
  [webcam]  → scale до 340×255 → округлые углы (маска через geq) → [pip]
  [bg][pip] → overlay x=1920-340-30 y=1080-260-30 → [out]

  FFmpeg filtergraph:
    [0:v]scale=1920:1080[bg];
    [1:v]scale=340:255,
         geq='if(gt(hypot(X-W/2,Y-H/2),min(W,H)/2),0,p(X,Y))':
              'if(gt(hypot(X-W/2,Y-H/2),min(W,H)/2),0,p(X,Y)):...'[pip];
    [bg][pip]overlay=x=1550:y=790[out]

  Примечание: скруглённые углы реализуются через PNG-маску или корнер-радиус.
  Точный метод будет выбран при реализации (geq или alphaextract).
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config

log = logging.getLogger(__name__)


class VideoRenderer:

    def __init__(self, screen_file: Path, webcam_file: Path, session_name: str):
        self.screen_file  = screen_file
        self.webcam_file  = webcam_file
        self.session_name = session_name
        self.temp_files: List[Path] = []  # для уборки после обработки

    def render_vertical(
        self,
        timeline: List[Dict],
        output_dir: Path,
        output_filename: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Рендерит вертикальное видео 1080×1920 (split-screen: экран сверху, вебка снизу).
        Используется для Формата 1 (10 мин) и Формата 2 (3 мин / хайлайты).
        """
        output_file = output_dir / output_filename

        screen_list = self._write_concat_list(self.screen_file, timeline, "screen")
        webcam_list = self._write_concat_list(self.webcam_file, timeline, "webcam")

        # Filtergraph: два потока → кроп центра → vstack
        # Экран может быть 1920×1200 — кропаем по Y чтобы получить 1920×1080
        # Вебка нормализуется до 1920×1080 на случай нестандартного разрешения/SAR
        cy = config.SCREEN_CROP_Y  # 60 для 1920×1200, 0 для 1920×1080
        filtergraph = (
            f"[0:v]crop=1215:1080:352:{cy},scale=1080:960[top];"
            "[1:v]scale=1920:1080:flags=lanczos,setsar=1,crop=1215:1080:352:0,scale=1080:960[bot];"
            "[top][bot]vstack[out]"
        )

        audio_fade = self._build_audio_fade_filter(timeline)
        filtergraph += f";[0:a]{audio_fade}[aout]"

        total_duration = sum(s["end"] - s["start"] for s in timeline)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(screen_list),
            "-f", "concat", "-safe", "0", "-i", str(webcam_list),
            "-filter_complex", filtergraph,
            "-map", "[out]",
            "-map", "[aout]",
            "-c:v", config.VIDEO_CODEC,
            "-crf", str(config.VIDEO_CRF),
            "-preset", config.VIDEO_PRESET,
            "-c:a", config.AUDIO_CODEC,
            "-b:a", config.AUDIO_BITRATE,
            "-threads", str(config.FFMPEG_THREADS),
            str(output_file),
        ]

        log.info(f"Рендер vertical → {output_filename}")
        self._run_ffmpeg(cmd, total_duration, progress_callback)
        return output_file

    def render_horizontal(
        self,
        timeline: List[Dict],
        output_dir: Path,
        output_filename: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """
        Рендерит горизонтальное видео 1920×1080 (полный экран + вебка-кружок PiP).
        Формат 3.
        """
        output_file = output_dir / output_filename

        screen_list = self._write_concat_list(self.screen_file, timeline, "screen_h")
        webcam_list = self._write_concat_list(self.webcam_file, timeline, "webcam_h")

        # Позиция PiP-окошка (правый нижний угол с отступами)
        pip_x = config.FORMAT_3["width"]  - config.PIP_WIDTH  - config.PIP_MARGIN_RIGHT
        pip_y = config.FORMAT_3["height"] - config.PIP_HEIGHT - config.PIP_MARGIN_BOTTOM

        # Filtergraph:
        #   [0:v] экран → scale 1920×1080 → [bg]
        #   [1:v] вебка → crop квадрат из центра (1080×1080 из 1920×1080)
        #               → scale до PIP_SIZE×PIP_SIZE
        #               → скруглённые углы через geq
        #               → [pip]
        #   [bg][pip] → overlay в правый нижний угол
        r  = config.PIP_CORNER_RADIUS
        s  = config.PIP_SIZE
        cy = config.SCREEN_CROP_Y        # кроп экрана по вертикали
        cx = (1920 - 1080) // 2          # = 420, кроп вебки по горизонтали

        # Правильная формула скруглённых углов (CSS border-radius):
        # Делаем ПРОЗРАЧНЫМИ только угловые зоны за пределами окружности,
        # всё остальное остаётся видимым.
        geq_alpha = (
            f"a='if("
            f"(lt(X,{r})*lt(Y,{r})*gt(hypot(X-{r},Y-{r}),{r}))+"
            f"(gt(X,{s}-{r}-1)*lt(Y,{r})*gt(hypot(X-({s}-{r}),Y-{r}),{r}))+"
            f"(lt(X,{r})*gt(Y,{s}-{r}-1)*gt(hypot(X-{r},Y-({s}-{r})),{r}))+"
            f"(gt(X,{s}-{r}-1)*gt(Y,{s}-{r}-1)*gt(hypot(X-({s}-{r}),Y-({s}-{r})),{r})),"
            f"0,255)'"
        )

        filtergraph = (
            # Экран: сначала кропаем 1920×1200 → 1920×1080, потом масштабируем
            f"[0:v]crop={config.FORMAT_3['width']}:1080:0:{cy},"
            f"scale={config.FORMAT_3['width']}:{config.FORMAT_3['height']}[bg];"
            # Вебка: нормализуем до 1920×1080, кропаем центральный квадрат 1080×1080
            f"[1:v]scale=1920:1080:flags=lanczos,setsar=1,"
            f"crop=1080:1080:{cx}:0,scale={s}:{s},format=yuva420p,"
            f"geq=lum='p(X,Y)':{geq_alpha}[pip];"
            f"[bg][pip]overlay={pip_x}:{pip_y}[out]"
        )

        audio_fade = self._build_audio_fade_filter(timeline)
        filtergraph += f";[0:a]{audio_fade}[aout]"

        total_duration = sum(s["end"] - s["start"] for s in timeline)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(screen_list),
            "-f", "concat", "-safe", "0", "-i", str(webcam_list),
            "-filter_complex", filtergraph,
            "-map", "[out]",
            "-map", "[aout]",
            "-c:v", config.VIDEO_CODEC,
            "-crf", str(config.VIDEO_CRF),
            "-preset", config.VIDEO_PRESET,
            "-c:a", config.AUDIO_CODEC,
            "-b:a", config.AUDIO_BITRATE,
            "-threads", str(config.FFMPEG_THREADS),
            str(output_file),
        ]

        log.info(f"Рендер horizontal → {output_filename}")
        self._run_ffmpeg(cmd, total_duration, progress_callback)
        return output_file

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _write_concat_list(self, source_file: Path, timeline: List[Dict], tag: str) -> Path:
        """
        Создаёт временный .txt файл со списком отрезков для FFmpeg concat demuxer.
        Формат:
            file '/abs/path/to/video.mp4'
            inpoint 0.000
            outpoint 12.400
        """
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        list_path = config.TEMP_DIR / f"{self.session_name}_{tag}_list.txt"

        with open(list_path, "w", encoding="utf-8") as f:
            for seg in timeline:
                escaped = str(source_file.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
                f.write(f"inpoint {seg['start']:.3f}\n")
                f.write(f"outpoint {seg['end']:.3f}\n")

        self.temp_files.append(list_path)
        return list_path

    def _run_ffmpeg(
        self,
        cmd: List[str],
        total_duration: float,
        progress_callback: Optional[Callable[[float], None]],
    ) -> None:
        """
        Запускает FFmpeg, парсит прогресс из stderr и вызывает progress_callback(%).
        Бросает исключение если FFmpeg завершился с ошибкой.
        """
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stderr_lines = []
        last_pct = -1

        for line in process.stderr:
            stderr_lines.append(line)

            # Парсим текущее время из строк вида: time=00:01:23.45
            match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if match and total_duration > 0 and progress_callback:
                h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
                current_sec = h * 3600 + m * 60 + s
                pct = min(current_sec / total_duration * 100, 99)
                # Вызываем колбэк только при изменении на 5%+ (не спамим)
                if pct - last_pct >= 5:
                    last_pct = pct
                    try:
                        progress_callback(pct)
                    except Exception:
                        pass

        process.wait()

        if process.returncode != 0:
            error_tail = "".join(stderr_lines[-30:])
            raise RuntimeError(f"FFmpeg завершился с ошибкой:\n{error_tail}")

        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass

        log.debug(f"FFmpeg завершён успешно")

    def _build_audio_fade_filter(self, timeline: List[Dict]) -> str:
        """
        Строит цепочку afade-фильтров для плавного звука на всех стыках склейки.

        Для каждой точки склейки добавляет:
          - fade-out 0.5с перед стыком
          - fade-in  0.5с после стыка
        Плюс fade-in в начале и fade-out в конце всего ролика.

        Результат: 'afade=t=in:st=0:d=0.5,afade=t=out:st=9.5:d=0.5,...'
        """
        d = 0.5  # длительность fade в секундах
        fades = []

        # Fade-in в начале
        fades.append(f"afade=t=in:st=0:d={d}")

        # Вычисляем позиции стыков в выходном потоке
        t = 0.0
        for i, seg in enumerate(timeline):
            dur = seg["end"] - seg["start"]
            t += dur
            if i < len(timeline) - 1:
                # Fade-out перед стыком
                fade_out_st = max(0.0, t - d)
                fades.append(f"afade=t=out:st={fade_out_st:.3f}:d={d}")
                # Fade-in после стыка
                fades.append(f"afade=t=in:st={t:.3f}:d={d}")

        # Fade-out в конце
        fade_out_st = max(0.0, t - d)
        fades.append(f"afade=t=out:st={fade_out_st:.3f}:d={d}")

        return ",".join(fades)

    def cleanup_temp(self) -> None:
        """Удаляет все временные файлы, созданные при рендере."""
        for f in self.temp_files:
            f.unlink(missing_ok=True)
        self.temp_files.clear()
