"""Make simple made-up songs so you can test the code without any music files.

    python make_demo_songs.py 50 demo_songs/

Each song is a random melody plus a bass line and a drum beat. These are only
for testing; use real songs for the numbers you put in the video.
"""
import sys
from pathlib import Path

import numpy as np

from tinyshazam import SAMPLE_RATE, save_wav


def note(freq, seconds, rng, timbre):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = sum(w * np.sin(2 * np.pi * freq * k * t) for k, w in enumerate(timbre, 1))  # this song's sound
    return tone * np.exp(-3 * t / seconds) * rng.uniform(0.6, 1.0)


def song(seed, seconds=60):
    rng = np.random.default_rng(seed)
    bpm = rng.uniform(70, 150)
    beat = 60 / bpm
    root = 220 * 2 ** rng.uniform(-1, 1)            # every song in its own key
    scales = [[0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10], [0, 3, 5, 7, 10]]
    steps = np.array(scales[rng.integers(len(scales))])
    steps = np.concatenate([steps, steps + 12])
    scale = root * 2 ** (steps / 12)
    timbre = rng.uniform(0, 1, 5) / np.arange(1, 6)  # every song has its own instrument sound
    timbre[0] = 1.0
    out = np.zeros(int(seconds * SAMPLE_RATE))
    t = 0.0
    while t < seconds - 1:
        dur = beat * rng.choice([0.5, 1, 1, 2])
        a = int(t * SAMPLE_RATE)
        n = note(rng.choice(scale), dur, rng, timbre)
        out[a:a + len(n)] += n[: len(out) - a]
        if rng.random() < 0.5:  # bass
            b = note(rng.choice(scale) / 4, dur, rng, timbre) * 0.7
            out[a:a + len(b)] += b[: len(out) - a]
        t += dur
    drum_colour = rng.standard_normal(rng.integers(3, 30))
    drum_colour /= np.abs(drum_colour).sum()
    for k in np.arange(0, seconds - 0.2, beat):  # drums: short noise bursts
        a = int(k * SAMPLE_RATE)
        d = rng.standard_normal(int(0.05 * SAMPLE_RATE)) * np.exp(-np.linspace(0, 6, int(0.05 * SAMPLE_RATE)))
        d = np.convolve(d, drum_colour, mode="same")  # every song has its own drum sound
        out[a:a + len(d)] += 0.4 * d
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    folder = Path(sys.argv[2] if len(sys.argv) > 2 else "demo_songs")
    folder.mkdir(exist_ok=True)
    for i in range(n):
        save_wav(folder / f"demo_song_{i + 1:02d}.wav", song(i))
    print(f"Made {n} demo songs in {folder}/")
