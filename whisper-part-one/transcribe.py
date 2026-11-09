#!/usr/bin/env python3
"""
Kaggle GPU kernel: transcribe "The GT50 Tapes — Part One: A Hundred Pounds
of Promise" (AMP HAUS — We Measure Everything) with faster-whisper large-v3
on CUDA float16.

Outputs (written to the kernel working directory):
  transcript.txt  - plain text, [mm:ss] at each segment start
  transcript.srt  - subtitles
  transcript.json - segments (+ word timestamps)
  run-log.txt     - GPU info, model load time, audio duration, wall time,
                    real-time factor, segment count, detected language
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

AUDIO_URL = (
    "https://anchor.fm/s/119555c5c/podcast/play/128326708/"
    "https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2026-9-29%2F"
    "38091000-6edf-c300-89cf-7bd5465b07a9.mp3"
)
EPISODE_TITLE = "The GT50 Tapes — Part One: A Hundred Pounds of Promise"
AUDIO_FILE = "part-one.mp3"
WORKDIR = os.getcwd()

LOG_LINES = []


def log(msg):
    line = str(msg)
    print(line, flush=True)
    LOG_LINES.append(line)


def ensure_faster_whisper():
    try:
        import faster_whisper  # noqa: F401
        return
    except ImportError:
        log("faster-whisper not present; installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "faster-whisper"]
        )
        import faster_whisper  # noqa: F401


def gpu_info():
    """Best-effort GPU inventory: torch first, then nvidia-smi, then ctranslate2."""
    info = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info.append(f"torch cuda[{i}]: {torch.cuda.get_device_name(i)}")
            info.append(f"torch version: {torch.__version__}, "
                        f"cuda version: {torch.version.cuda}")
        else:
            info.append("torch present but CUDA not available")
    except Exception as e:  # torch may not be installed
        info.append(f"torch unavailable: {e}")
    if not any("cuda[" in line for line in info):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                info.append("nvidia-smi:\n" + out.stdout.strip())
        except Exception as e:
            info.append(f"nvidia-smi unavailable: {e}")
    try:
        import ctranslate2

        info.append(
            f"ctranslate2 cuda device count: {ctranslate2.get_cuda_device_count()}"
        )
    except Exception as e:
        info.append(f"ctranslate2 probe failed: {e}")
    return info


def fmt_ts(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def fmt_srt_ts(seconds):
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def download_audio():
    dest = os.path.join(WORKDIR, AUDIO_FILE)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        log(f"Audio already present: {dest} ({os.path.getsize(dest)} bytes)")
        return dest
    log(f"Downloading audio: {AUDIO_URL}")
    req = urllib.request.Request(
        AUDIO_URL, headers={"User-Agent": "Mozilla/5.0 (Kaggle kernel)"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    log(f"Downloaded {os.path.getsize(dest)} bytes -> {dest}")
    return dest


def main():
    log("=" * 70)
    log(f"Episode: {EPISODE_TITLE}")
    log("Model: faster-whisper large-v3 (cuda, float16, device_index 0)")
    log("=" * 70)

    for line in gpu_info():
        log(line)

    ensure_faster_whisper()
    from faster_whisper import WhisperModel

    audio_path = download_audio()

    t0 = time.time()
    model = WhisperModel(
        "large-v3", device="cuda", device_index=0, compute_type="float16"
    )
    load_s = time.time() - t0
    log(f"Model load time: {load_s:.2f}s")

    t1 = time.time()
    segments_gen, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
    )
    segments = []
    for seg in segments_gen:
        segments.append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            }
        )
    wall_s = time.time() - t1

    audio_duration = float(info.duration)
    rtf = audio_duration / wall_s if wall_s > 0 else 0.0

    log(f"Detected language: {info.language} "
        f"(probability {info.language_probability:.4f})")
    log(f"Audio duration: {audio_duration:.2f}s ({fmt_ts(audio_duration)})")
    log(f"Transcription wall time: {wall_s:.2f}s")
    log(f"Real-time factor: {rtf:.2f}x")
    log(f"Segment count: {len(segments)}")

    # transcript.txt — [mm:ss] at each segment start
    txt_path = os.path.join(WORKDIR, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{fmt_ts(seg['start'])}] {seg['text']}\n")
    log(f"Wrote {txt_path}")

    # transcript.srt
    srt_path = os.path.join(WORKDIR, "transcript.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(
                f"{i}\n{fmt_srt_ts(seg['start'])} --> "
                f"{fmt_srt_ts(seg['end'])}\n{seg['text']}\n\n"
            )
    log(f"Wrote {srt_path}")

    # transcript.json
    json_path = os.path.join(WORKDIR, "transcript.json")
    payload = {
        "episode": EPISODE_TITLE,
        "audio_url": AUDIO_URL,
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": audio_duration,
        "segments": segments,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"Wrote {json_path}")

    # run-log.txt (everything logged so far)
    log_path = os.path.join(WORKDIR, "run-log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    log(f"Wrote {log_path}")
    log("DONE")


if __name__ == "__main__":
    main()
