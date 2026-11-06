"""Step 0: a song is just a long list of numbers.

We load any audio file, turn it into one channel (mono), and resample it to
11,025 numbers per second. That is plenty for finding the loud notes, and it
keeps everything small and fast.
"""
import shutil
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 11025  # numbers per second that we keep


def _load_with_ffmpeg(path):
    """Decode any format (mp3, wav, m4a, ...) with ffmpeg into mono floats."""
    cmd = [
        "ffmpeg", "-v", "quiet", "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def _load_wav(path):
    """Fallback for plain 16-bit WAV files when ffmpeg is not installed."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("Without ffmpeg, only 16-bit WAV files are supported.")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = data.reshape(-1, channels).mean(axis=1) / 32768.0
    if rate != SAMPLE_RATE:  # simple resampling by straight-line interpolation
        t_old = np.arange(len(x)) / rate
        t_new = np.arange(int(len(x) * SAMPLE_RATE / rate)) / SAMPLE_RATE
        x = np.interp(t_new, t_old, x)
    return x


def load_audio(path):
    """Return the song as one long list of numbers between -1 and 1."""
    if shutil.which("ffmpeg"):
        return _load_with_ffmpeg(path)
    return _load_wav(path)


def save_wav(path, x, rate=SAMPLE_RATE):
    """Save numbers back to a 16-bit WAV, so you can listen to a noisy clip."""
    x = np.clip(x / max(1e-9, np.abs(x).max()), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
