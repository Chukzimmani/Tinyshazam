"""Name the song in a real recording (for example, your phone held up to a speaker).

    python identify.py my_recording.m4a

Needs index.pkl from build_index.py.
"""
import pickle
import sys
import time

from tinyshazam import load_audio, vote

with open("index.pkl", "rb") as f:
    index = pickle.load(f)

clip = load_audio(sys.argv[1])
t0 = time.time()
results = vote(index, clip)
took = time.time() - t0

if not results:
    print("No fingerprints matched at all.")
else:
    print(f"Searched {len(index.songs)} songs in {took:.2f}s\n")
    for rank, r in enumerate(results[:3], 1):
        print(f"{rank}. {r['song']:40s} votes={r['votes']:>4}  clip starts at {r['start_seconds']:.1f}s")
