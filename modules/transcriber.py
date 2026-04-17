"""
transcriber.py — транскрипция видео через Groq Whisper API.

Groq блокирует российские IP, поэтому запросы идут через Cloudflare
Worker-прокси (адрес и секрет в .env). При пустом GROQ_PROXY_URL идёт
прямо в api.groq.com (полезно для отладки с не-РФ IP).

Аудио извлекается в FLAC mono 16kHz — сжатие 6-8× против WAV, помещается
в лимит 25 MB Free tier'а даже для часовых видео.
"""

import bisect
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import httpx

import config

log = logging.getLogger(__name__)


class Transcriber:

    def transcribe(self, video_path: Path) -> List[Dict]:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = config.TEMP_DIR / f"{video_path.stem}_audio.flac"

        log.info(f"Извлекаю аудио из {video_path.name}...")
        self._extract_audio(video_path, audio_path)

        size_mb = audio_path.stat().st_size / 1024 / 1024
        log.info(f"Аудио готово: {audio_path.name} ({size_mb:.1f} MB)")

        try:
            if size_mb > config.GROQ_MAX_FILE_MB:
                log.info(
                    f"Аудио {size_mb:.1f} MB > лимит Groq "
                    f"{config.GROQ_MAX_FILE_MB} MB — режу на части."
                )
                data = self._transcribe_chunked(audio_path, size_mb)
            else:
                log.info(f"Транскрипция через Groq ({config.WHISPER_MODEL})...")
                data = self._call_groq(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        segments = _build_segments(data.get("segments", []), data.get("words", []))
        log.info(f"Получено {len(segments)} сегментов")
        return segments

    def _transcribe_chunked(self, audio_path: Path, size_mb: float) -> dict:
        """
        Режет аудио на N равных частей по времени и транскрибирует каждую.
        Timestamps сегментов и слов сдвигаются на offset чанка и сливаются
        в единые списки — для рендера файл остаётся неделимым целым.
        """
        duration = self._probe_duration(audio_path)
        # Целимся в чанки ~80% от лимита, с запасом на вариативность размера.
        target_mb = config.GROQ_MAX_FILE_MB * 0.8
        n_chunks = max(2, int(size_mb / target_mb) + 1)
        chunk_sec = duration / n_chunks
        log.info(
            f"Делю на {n_chunks} частей по ~{chunk_sec:.1f}с "
            f"(~{size_mb / n_chunks:.1f} MB/часть)."
        )

        merged_segments: List[Dict] = []
        merged_words: List[Dict] = []
        chunk_paths: List[Path] = []

        try:
            for i in range(n_chunks):
                start = i * chunk_sec
                chunk_path = (
                    config.TEMP_DIR / f"{audio_path.stem}_part{i + 1}.flac"
                )
                self._extract_chunk(audio_path, chunk_path, start, chunk_sec)
                chunk_paths.append(chunk_path)

                chunk_mb = chunk_path.stat().st_size / 1024 / 1024
                if chunk_mb > config.GROQ_MAX_FILE_MB:
                    raise RuntimeError(
                        f"Часть {i + 1}/{n_chunks} всё ещё "
                        f"{chunk_mb:.1f} MB > лимит {config.GROQ_MAX_FILE_MB} MB."
                    )

                log.info(
                    f"Транскрипция части {i + 1}/{n_chunks} "
                    f"({chunk_mb:.1f} MB, offset {start:.1f}с)..."
                )
                data = self._call_groq(chunk_path)

                for seg in data.get("segments", []):
                    seg = dict(seg)
                    seg["start"] = seg.get("start", 0.0) + start
                    seg["end"] = seg.get("end", 0.0) + start
                    merged_segments.append(seg)
                for w in data.get("words", []):
                    w = dict(w)
                    w["start"] = w.get("start", 0.0) + start
                    w["end"] = w.get("end", 0.0) + start
                    merged_words.append(w)
        finally:
            for p in chunk_paths:
                p.unlink(missing_ok=True)

        return {"segments": merged_segments, "words": merged_words}

    def _probe_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка ffprobe:\n{result.stderr[-1000:]}")
        return float(result.stdout.strip())

    def _extract_chunk(
        self,
        audio_path: Path,
        chunk_path: Path,
        start: float,
        duration: float,
    ) -> None:
        # -ss до -i — быстрый input-level seek. Перекодируем (не -c copy),
        # чтобы границы чанка были точными и размер оставался управляемым.
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
    Распределяем слова по сегментам через bisect-поиск по диапазону
    [seg.start, seg.end), а не pointer-ом — pointer ломается когда
    Whisper-сегмент из чанка N имеет end > начала следующего чанка и
    «съедает» слова, оставляя последующие сегменты без слов.
    """
    words_sorted = sorted(words_raw, key=lambda w: w.get("start", 0.0))
    word_starts  = [w.get("start", 0.0) for w in words_sorted]

    empty_segs = 0
    result = []
    for idx, seg in enumerate(segments_raw):
        lo = bisect.bisect_left(word_starts,  seg["start"])
        hi = bisect.bisect_left(word_starts,  seg["end"])
        seg_words = [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            for w in words_sorted[lo:hi]
        ]
        seg_words = _dedup_consecutive(seg_words)
        if not seg_words:
            empty_segs += 1

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

    if empty_segs:
        log.warning(f"_build_segments: {empty_segs}/{len(segments_raw)} сегментов без слов → используем границы Whisper")
    log.info(f"_build_segments: {len(segments_raw)} сег, {len(words_raw)} слов")
    return result
