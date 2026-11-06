"""Tiny Shazam: song recognition from scratch with NumPy.

Pipeline: audio -> spectrogram -> stars -> fingerprints -> index -> vote.
"""
from .audio import SAMPLE_RATE, load_audio, save_wav
from .fingerprint import Index, fingerprints, glue, pairs, unglue
from .match import add_noise, cut_clip, stars_of, vote
from .peaks import find_stars
from .spectrogram import spectrogram
