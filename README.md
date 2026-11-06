# Tiny Shazam

A tiny version of Shazam, built from scratch with NumPy, to understand the math behind it.
This is the companion code for the Math With Chukz video **"I Built a Tiny Shazam to Understand the Math"**.

It follows the same four tricks as the real Shazam (Avery Wang, *An Industrial-Strength Audio Search Algorithm*, 2003):

| Step | File | What it does |
|---|---|---|
| Sound is numbers | `tinyshazam/audio.py` | Loads any audio file as 11,025 numbers per second |
| Trick 1: notes over time | `tinyshazam/spectrogram.py` | Chops the song into ~0.1 s slices and uses the Fourier transform to get a spectrogram |
| Trick 2: keep the stars | `tinyshazam/peaks.py` | Keeps only the loudest points (louder than everything around them), giving the constellation map |
| Trick 3: fingerprints | `tinyshazam/fingerprint.py` | Joins each star to a few stars just after it and glues *note 1, note 2, time gap* into one code, stored in an index |
| Trick 4: the vote | `tinyshazam/match.py` | For every matching fingerprint, works out *time in song − time in clip*; the right song piles its votes on one answer |

## Run it

You need Python 3, NumPy and Matplotlib. ffmpeg is optional and lets you load mp3/m4a files; without it, use 16-bit WAV files.

```bash
pip install numpy matplotlib

# 1. Put your songs in a folder, then build the index
python build_index.py songs/

# 2. Name the song in a real recording (for example, your phone held up to a speaker)
python identify.py my_recording.m4a

# 3. Run every experiment from the video and save the plots in outputs/
python video_numbers.py songs/
python video_numbers.py songs/ --noise noisy_room.wav   # use a real noisy-room recording instead of hiss
```

No music files? Make 50 test songs first with `python make_demo_songs.py 50 demo_songs/`.

## What `video_numbers.py` measures

- **Stars:** how many numbers one song's spectrogram has, and how few stars survive
- **Fingerprints:** how many fingerprints the whole index holds
- **The vote:** for a 5-second noisy clip, the winning song's votes against the runner-up's
- **Turning the noise up:** how loud the noise can get (compared with the music) before the right song stops winning at least 90% of the time
- **Making the clip shorter:** the shortest clip that still wins at least 90% of the time

## Settings worth playing with

- `peaks.py`: `neighbourhood` (how big an area a star must beat) and `STARS_PER_BLOCK` (how many stars to keep per second)
- `fingerprint.py`: `FAN_OUT` (partners per anchor) and the target zone (`MIN_GAP`, `MAX_GAP`, `MAX_NOTE_JUMP`)

More stars and more partners make more fingerprints. That helps with short clips, but it also lets noise in.

## The real Shazam goes further

The real Shazam searches tens of millions of songs, so it uses a much faster database, carefully tuned settings, and many years of engineering. The core idea is the same: **stars, pairs, and a vote.**
