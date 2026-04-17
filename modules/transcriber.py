"""
transcriber.py — транскрипция видео через Groq Whisper API.

Groq блокирует российские IP, поэтому запросы идут через Cloudflare
Worker-прокси (адрес и секрет в .env). При пустом GROQ_PROXY_URL идёт
прямо в api.groq.com (полезно для отладки с не-РФ IP).

Принимает список частей сессии (screen_001.mp4, screen_002.mp4, ...):
извлекает аудио из каждой части, транскрибирует её отдельно и сливает
timestamps со сдвигом на сумму длительностей предыдущих частей. Резка
на чанки включается только если конкретная часть сама по себе больше
лимита Groq — например если кто-то запишет одним файлом на 40+ минут.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)


class Transcriber:

    def transcribe(self, video_paths: List[Path]) -> List[Dict]:
        """
        Транскрибирует последовательность частей сессии. Возвращает плоский
        список сегментов с timestamps, привязанными к совокупной timeline.
        """
        if not video_paths:
            return []

        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        merged_segments: List[Dict] = []
        merged_words: List[Dict] = []
        offset = 0.0
        n = len(video_paths)

        for i, vpath in enumerate(video_paths, 1):
            log.info(f"Часть {i}/{n}: {vpath.name}")
            audio_path = config.TEMP_DIR / f"{vpath.stem}_audio.flac"
            try:
                self._extract_audio(vpath, audio_path)
                size_mb = audio_path.stat().st_size / 1024 / 1024
                part_dur = self._probe_duration(vpath)
                log.info(f"  Аудио {size_mb:.1f} MB, длительность {part_dur:.1f}с")

                if size_mb > config.GROQ_MAX_FILE_MB:
                    log.info(
                        f"  > лимит Groq {config.GROQ_MAX_FILE_MB} MB — "
                        f"режу часть на чанки."
                    )
                    data = self._transcribe_chunked(audio_path, size_mb)
                else:
                    log.info(f"  Транскрипция через Groq ({config.WHISPER_MODEL})...")
                    data = self._call_groq(audio_path)

                for seg in data.get("segments", []):
                    s = dict(seg)
                    s["start"] = s.get("start", 0.0) + offset
                    s["end"]   = s.get("end",   0.0) + offset
                    merged_segments.append(s)
                for w in data.get("words", []):
                    x = dict(w)
                    x["start"] = x.get("start", 0.0) + offset
                    x["end"]   = x.get("end",   0.0) + offset
                    merged_words.append(x)

                offset += part_dur
            finally:
                audio_path.unlink(missing_ok=True)

        segments = _build_segments(merged_segments, merged_words)
        log.info(f"Всего сегментов: {len(segments)} (итог ~{offset:.1f}с)")
        return segments

    def _transcribe_chunked(self, audio_path: Path, size_mb: float) -> dict:
        """
        Для аудио, превышающего лимит Groq: режет на N равных по времени частей,
        транскрибирует каждую, сливает timestamps с offset внутри этого файла.
        """
        duration = self._probe_duration(audio_path)
        target_mb = config.GROQ_MAX_FILE_MB * 0.8
        n_chunks  = max(2, int(size_mb / target_mb) + 1)
        chunk_sec = duration / n_chunks
        log.info(
            f"  Делю на {n_chunks} частей по ~{chunk_sec:.1f}с "
            f"(~{size_mb / n_chunks:.1f} MB/часть)."
        )

        merged_segments: List[Dict] = []
        merged_words: List[Dict] = []
        chunk_paths: List[Path] = []

        try:
            for i in range(n_chunks):
                start = i * chunk_sec
                chunk_path = config.TEMP_DIR / f"{audio_path.stem}_part{i + 1}.flac"
                self._extract_chunk(audio_path, chunk_path, start, chunk_sec)
                chunk_paths.append(chunk_path)

                chunk_mb = chunk_path.stat().st_size / 1024 / 1024
                if chunk_mb > config.GROQ_MAX_FILE_MB:
                    raise RuntimeError(
                        f"Часть {i + 1}/{n_chunks} всё ещё {chunk_mb:.1f} MB "
                        f"> лимит {config.GROQ_MAX_FILE_MB} MB."
                    )

                log.info(
                    f"  Транскрипция чанка {i + 1}/{n_chunks} "
                    f"({chunk_mb:.1f} MB, offset {start:.1f}с)..."
                )
                data = self._call_groq(chunk_path)

                for seg in data.get("segments", []):
                    s = dict(seg)
                    s["start"] = s.get("start", 0.0) + start
                    s["end"]   = s.get("end",   0.0) + start
                    merged_segments.append(s)
                for w in data.get("words", []):
                    x = dict(w)
                    x["start"] = x.get("start", 0.0) + start
                    x["end"]   = x.get("end",   0.0) + start
                    merged_words.append(x)
        finally:
            for p in chunk_paths:
                p.unlink(missing_ok=True)

        return {"segments": merged_segments, "words": merged_words}

    def _probe_duration(self, path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка ffprobe:\n{result.stderr[-1000:]}")
        return float(result.stdout.strip())

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "flac",
            "-compression_level", "8",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка извлечения аудио:\n{result.stderr[-1000:]}")

    def _extract_chunk(
        self,
        audio_path: Path,
        chunk_path: Path,
        start: float,
        duration: float,
    ) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "flac",
            "-compression_level", "8",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка разбиения аудио:\n{result.stderr[-1000:]}")

    def _call_groq(self, audio_path: Path) -> dict:
        url = f"{config.GROQ_PROXY_URL.rstrip('/')}/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        if config.GROQ_PROXY_SECRET:
            headers["X-Proxy-Secret"] = config.GROQ_PROXY_SECRET

        with open(audio_path, "rb") as f:
            content = f.read()
        files = {"file": (audio_path.name, content, "audio/flac")}
        data = {
            "model": config.WHISPER_MODEL,
            "language": config.WHISPER_LANGUAGE,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
            "temperature": "0",
        }
        with httpx.Client(timeout=600.0) as client:
            r = client.post(url, headers=headers, files=files, data=data)
        if r.status_code != 200:
            raise RuntimeError(f"Groq API {r.status_code}: {r.text[:500]}")
        return r.json()


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _dedup_consecutive(words: List[Dict]) -> List[Dict]:
    """Убирает подряд идущие одинаковые слова с паузой < 0.5с между ними."""
    if not words:
        return words
    result = [words[0]]
    for w in words[1:]:
        prev = result[-1]
        same = w["word"].strip().lower() == prev["word"].strip().lower()
        gap  = w["start"] - prev["end"]
        if same and gap < 0.5:
            continue
        result.append(w)
    return result


def _build_segments(segments_raw: List[Dict], words_raw: List[Dict]) -> List[Dict]:
    """
    Groq/OpenAI Whisper API отдаёт сегменты и слова плоскими списками.
    Распределяем слова по сегментам по попаданию во временной диапазон,
    затем приравниваем границы сегмента к границам его первого/последнего слова.
    """
    result = []
    wi = 0
    n_words = len(words_raw)

    for idx, seg in enumerate(segments_raw):
        seg_end = seg["end"]
        seg_words: List[Dict] = []
        while wi < n_words and words_raw[wi]["start"] < seg_end:
            w = words_raw[wi]
            if w["start"] >= seg["start"]:
                seg_words.append({
                    "word":  w["word"],
                    "start": w["start"],
                    "end":   w["end"],
                })
            wi += 1

        seg_words = _dedup_consecutive(seg_words)

        if seg_words:
            s_start = seg_words[0]["start"]
            s_end   = seg_words[-1]["end"]
        else:
            s_start = seg["start"]
            s_end   = seg["end"]

        result.append({
            "index": idx,
            "start": round(s_start, 3),
            "end":   round(s_end,   3),
            "text":  (seg.get("text") or "").strip(),
            "words": seg_words,
        })
    return result
