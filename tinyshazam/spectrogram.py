"""Trick 1: every sound is a mix of pure notes.

We chop the song into tiny slices (about one tenth of a second each), and for
every slice the Fourier transform tells us how loud each note is. Stacking the
slices side by side gives the spectrogram: time goes across, pitch goes up.
"""
import numpy as np

from .audio import SAMPLE_RATE

WINDOW = 1024  # samples per slice -> 1024 / 11025 = about 0.093 seconds
HOP = 512      # how far we move between slices -> about 0.046 seconds


def spectrogram(x, window=WINDOW, hop=HOP):
    """Return a 2-D array: rows are notes (low to high), columns are time slices.

    Values are loudness in decibels, so quiet and loud parts are easy to compare.
    """
    if len(x) < window:
        x = np.pad(x, (0, window - len(x)))
    n_slices = 1 + (len(x) - window) // hop
    # Every row of `frames` is one slice of the song.
    idx = np.arange(window)[None, :] + hop * np.arange(n_slices)[:, None]
    frames = x[idx] * np.hanning(window)  # soften the edges of each slice
    # The Fourier transform: how much of each pure note is in each slice.
    loudness = np.abs(np.fft.rfft(frames, axis=1))
    return 20 * np.log10(loudness.T + 1e-10)  # shape: (notes, time)


def slice_seconds(hop=HOP):
    """How many seconds one column of the spectrogram covers."""
    return hop / SAMPLE_RATE


def bin_hz(window=WINDOW):
    """How many hertz one row of the spectrogram covers."""
    return SAMPLE_RATE / window
