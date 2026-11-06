"""Trick 4: the vote.

For every fingerprint in the clip that we find in the index, we work out

    offset = time in the song - time in the clip

Matches from the right song all give the same offset, because the whole clip
starts at one place in the song. Chance matches give random offsets. So for each
song we count the biggest pile of votes on a single offset, and the right song
shows up as one tall spike.
"""
from collections import Counter, defaultdict

import numpy as np

from .audio import SAMPLE_RATE
from .fingerprint import fingerprints
from .peaks import find_stars
from .spectrogram import slice_seconds, spectrogram


def stars_of(x):
    return find_stars(spectrogram(x))


def vote(index, clip):
    """Return a list of results, best first.

    Each result is a dict with the song name, its winning vote count, the
    offset (in seconds) where the clip starts in that song, and the full
    offset histogram for plotting.
    """
    offsets = defaultdict(Counter)  # song_id -> Counter(offset -> votes)
    for code, t_clip in fingerprints(stars_of(clip)):
        for song_id, t_song in index.table.get(code, ()):
            offsets[song_id][t_song - t_clip] += 1

    results = []
    for song_id, counter in offsets.items():
        best_offset, votes = counter.most_common(1)[0]
        results.append({
            "song": index.songs[song_id],
            "song_id": song_id,
            "votes": votes,
            "start_seconds": best_offset * slice_seconds(),
            "histogram": counter,
        })
    results.sort(key=lambda r: r["votes"], reverse=True)
    return results


def add_noise(clip, noise_ratio, noise=None, rng=None):
    """Mix noise into a clip.

    noise_ratio is how loud the noise is compared with the music
    (1.0 = equally loud, 2.0 = twice as loud), measured by average power (RMS).
    Pass a recording of a noisy room as `noise`, or leave it out for hiss.
    """
    rng = rng or np.random.default_rng()
    if noise is None:
        noise = rng.standard_normal(len(clip))
    else:
        start = rng.integers(0, max(1, len(noise) - len(clip)))
        noise = np.resize(noise[start:start + len(clip)], len(clip))
    rms = lambda a: np.sqrt(np.mean(a ** 2)) + 1e-12
    return clip + noise * (noise_ratio * rms(clip) / rms(noise))


def cut_clip(song, start_seconds, length_seconds):
    a = int(start_seconds * SAMPLE_RATE)
    return song[a:a + int(length_seconds * SAMPLE_RATE)]
