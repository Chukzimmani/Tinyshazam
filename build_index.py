"""Build the fingerprint index for every song in a folder.

    python build_index.py songs/

Prints, for each song, how many numbers its spectrogram has, how many stars
survive, and how many fingerprints it adds, then saves index.pkl.
"""
import pickle
import sys
import time
from pathlib import Path

from tinyshazam import Index, find_stars, load_audio, spectrogram

AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def song_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO)


def build(folder, verbose=True):
    index, stats = Index(), []
    for path in song_files(folder):
        x = load_audio(path)
        spec = spectrogram(x)
        stars = find_stars(spec)
        n_fp = index.add_song(path.name, stars)
        stats.append({"song": path.name, "seconds": len(x) / 11025,
                      "grid_points": spec.size, "stars": len(stars), "fingerprints": n_fp})
        if verbose:
            print(f"{path.name:40s} grid={spec.size:>10,}  stars={len(stars):>6,}  fingerprints={n_fp:>7,}")
    return index, stats


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "songs"
    t0 = time.time()
    index, stats = build(folder)
    with open("index.pkl", "wb") as f:
        pickle.dump(index, f)
    print(f"\n{len(index.songs)} songs, {len(index):,} fingerprints in the index "
          f"({len(index.table):,} different codes), built in {time.time() - t0:.1f}s -> index.pkl")
