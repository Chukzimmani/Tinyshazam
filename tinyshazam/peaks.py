"""Trick 2: keep only the loudest points, the "stars".

A point is a star only if it is louder than every point around it, and loud
enough overall. We keep its time and its note, and we throw away how loud it
was, so turning the volume up or down cannot move the stars.
"""
import numpy as np


def _sliding_max(a, size, axis):
    """Biggest value in a window of `size` centred on each point, along one axis."""
    pad = size // 2
    widths = [(0, 0)] * a.ndim
    widths[axis] = (pad, pad)
    padded = np.pad(a, widths, mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
    return windows.max(axis=-1)


STARS_PER_BLOCK = 8   # keep at most this many stars...
BLOCK_SLICES = 22     # ...in every block of 22 slices (about one second)


def find_stars(spec, neighbourhood=(21, 21), min_above_median_db=10.0,
               per_block=STARS_PER_BLOCK, block=BLOCK_SLICES):
    """Return the stars as an array of (time_slice, note_bin) pairs, sorted by time.

    neighbourhood: how many notes x how many slices a star must beat.
    min_above_median_db: a star must also be this much louder than the
        song's typical loudness, so silence and hiss do not make stars.
    per_block / block: in every short block of time we keep only the loudest
        few stars. Noise adds lots of weak stars, but the song's strongest
        notes stay on top, so this is what lets a noisy clip still match.
    """
    f_size, t_size = neighbourhood
    # The loudest value around every point (done in two easy passes).
    local_max = _sliding_max(_sliding_max(spec, f_size, axis=0), t_size, axis=1)
    is_star = (spec == local_max) & (spec > np.median(spec) + min_above_median_db)
    notes, times = np.nonzero(is_star)
    level = spec[notes, times]
    order = np.lexsort((notes, times))
    times, notes, level = times[order], notes[order], level[order]

    # A star survives only if fewer than `per_block` other stars within half a
    # block on either side of it are louder. The window moves with each star,
    # so a clip picks the same stars as the full song, wherever it starts.
    half = block // 2
    lo = np.searchsorted(times, times - half, side="left")
    hi = np.searchsorted(times, times + half, side="right")
    keep = np.array([np.sum(level[lo[i]:hi[i]] > level[i]) < per_block
                     for i in range(len(times))], dtype=bool)
    return np.stack([times[keep], notes[keep]], axis=1).astype(int).reshape(-1, 2)
