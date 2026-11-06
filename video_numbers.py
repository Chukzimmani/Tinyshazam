"""Run every experiment from the video and print the numbers for the script.

    python video_numbers.py songs/                     # hiss as the noise
    python video_numbers.py songs/ --noise room.wav    # a real noisy-room recording

Prints the seven placeholders ({GRID_POINTS}, {STARS}, {FINGERPRINTS},
{WIN_VOTES}, {RUNNER_UP_VOTES}, {NOISE_LIMIT}, {MIN_SECONDS}) and saves the
plots used on screen into outputs/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_index import build, song_files
from tinyshazam import (SAMPLE_RATE, add_noise, cut_clip, find_stars, fingerprints, pairs,
                        load_audio, save_wav, spectrogram, vote)
from tinyshazam.spectrogram import bin_hz, slice_seconds

AMBER, TEAL, GREY = "#F5A623", "#2EC4B6", "#5A5F6B"
plt.style.use("dark_background")


def trial(index, songs, rng, clip_seconds, noise_ratio, noise, song_id=None):
    """Cut a random clip from a random song, add noise, and see if the vote is right."""
    sid = rng.integers(len(songs)) if song_id is None else song_id
    x = songs[sid]
    start = rng.uniform(0, max(0.0, len(x) / SAMPLE_RATE - clip_seconds))
    clip = add_noise(cut_clip(x, start, clip_seconds), noise_ratio, noise, rng)
    res = vote(index, clip)
    ok = bool(res) and res[0]["song_id"] == sid
    return ok, res, clip, sid, start


def accuracy(index, songs, rng, clip_seconds, noise_ratio, noise, trials):
    return np.mean([trial(index, songs, rng, clip_seconds, noise_ratio, noise)[0] for _ in range(trials)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--clip", type=float, default=5.0, help="clip length in seconds for the hero demo")
    ap.add_argument("--noise-ratio", type=float, default=1.0, help="noise loudness vs music for the hero demo")
    ap.add_argument("--noise", help="optional recording of a noisy room to use as the noise")
    ap.add_argument("--trials", type=int, default=20, help="random clips per level in the sweeps")
    ap.add_argument("--pass-rate", type=float, default=0.9, help="share of trials that must be right to 'keep winning'")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = Path("outputs"); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    noise = load_audio(a.noise) if a.noise else None

    print("Building the index...\n")
    index, stats = build(a.folder, verbose=False)
    songs = [load_audio(p) for p in song_files(a.folder)]
    hero = stats[0]

    # ---------- Act 2/3: one song's picture and its stars ----------
    spec = spectrogram(songs[0])
    stars = find_stars(spec)
    secs = min(20, spec.shape[1] * slice_seconds())
    cols = int(secs / slice_seconds())
    ext = [0, secs, 0, spec.shape[0] * bin_hz()]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(spec[:, :cols], origin="lower", aspect="auto", extent=ext, cmap="magma")
    ax.set(xlabel="time (s)", ylabel="pitch (Hz)", title="Spectrogram")
    fig.savefig(out / "1_spectrogram.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    s = stars[stars[:, 0] < cols]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(s[:, 0] * slice_seconds(), s[:, 1] * bin_hz(), s=10, c=AMBER)
    ax.set(xlim=ext[:2], ylim=ext[2:], xlabel="time (s)", ylabel="pitch (Hz)", title="Constellation map (stars only)")
    fig.savefig(out / "2_star_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    s3 = stars[stars[:, 0] < int(3 / slice_seconds())]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(s3[:, 0] * slice_seconds(), s3[:, 1] * bin_hz(), s=18, c=AMBER, zorder=3)
    times, notes = s3[:, 0], s3[:, 1]
    anchors = set(range(0, len(s3), max(1, len(s3) // 4)))  # show a few anchors, not all
    for i, j in pairs(s3):
        if i in anchors:
            ax.plot([times[i] * slice_seconds(), times[j] * slice_seconds()],
                    [notes[i] * bin_hz(), notes[j] * bin_hz()], c=TEAL, lw=1.2)
    ax.scatter(times[list(anchors)] * slice_seconds(), notes[list(anchors)] * bin_hz(), s=60,
               facecolors="none", edgecolors=AMBER, zorder=4)
    ax.set(xlabel="time (s)", ylabel="pitch (Hz)", title="Pairs of stars -> fingerprints")
    fig.savefig(out / "3_pairs.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------- Act 5: the hero noisy clip ----------
    ok, res, clip, sid, start = trial(index, songs, rng, a.clip, a.noise_ratio, noise, song_id=0)
    save_wav(out / "noisy_clip.wav", clip)
    win = res[0]
    runner = res[1]["votes"] if len(res) > 1 else 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axx, r, col in zip(axes, res[:2], (AMBER, GREY)):
        h = r["histogram"]
        axx.bar([k * slice_seconds() for k in h], list(h.values()), width=0.2, color=col)
        axx.set(title=f"{r['song']}  ({r['votes']} votes)", xlabel="time in song - time in clip (s)")
    axes[0].set_ylabel("votes")
    fig.savefig(out / "4_vote_histograms.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------- Act 6: try to break it ----------
    print(f"Noise sweep ({a.clip:g}s clips, {a.trials} random clips per level):")
    ratios = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    noise_acc = []
    for r in ratios:
        acc = accuracy(index, songs, rng, a.clip, r, noise, a.trials)
        noise_acc.append(acc); print(f"  noise {r:>4g}x as loud as the music -> {acc:5.0%} right")
    passing = [r for r, acc in zip(ratios, noise_acc) if acc >= a.pass_rate]
    noise_limit = max(passing) if passing else None

    print(f"\nClip-length sweep (noise {a.noise_ratio:g}x as loud):")
    lengths = [5, 4, 3, 2, 1.5, 1, 0.75, 0.5, 0.25]
    len_acc = []
    for L in lengths:
        acc = accuracy(index, songs, rng, L, a.noise_ratio, noise, a.trials)
        len_acc.append(acc); print(f"  {L:>5g} s clip -> {acc:5.0%} right")
    passing = [L for L, acc in zip(lengths, len_acc) if acc >= a.pass_rate]
    min_seconds = min(passing) if passing else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ratios, noise_acc, "o-", c=AMBER); axes[0].set(xscale="log", xlabel="noise loudness / music loudness", ylabel="right song", title="Turning the noise up")
    axes[1].plot(lengths, len_acc, "o-", c=TEAL); axes[1].set(xlabel="clip length (s)", title="Making the clip shorter"); axes[1].invert_xaxis()
    for axx in axes: axx.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axx.set_ylim(-0.05, 1.05)
    fig.savefig(out / "5_stress_tests.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---------- Summary for the script ----------
    print("\n================ NUMBERS FOR THE SCRIPT ================")
    print(f"Songs in the index        : {len(index.songs)}")
    print(f"Hero song                 : {hero['song']} ({hero['seconds']:.0f}s)")
    print(f"{{GRID_POINTS}}  (beat 3.6) : {hero['grid_points']:,}")
    print(f"{{STARS}}        (beat 3.6) : {hero['stars']:,}")
    print(f"{{FINGERPRINTS}} (beat 4.6) : {len(index):,}")
    print(f"Hero clip                 : {a.clip:g}s from {start:.1f}s, noise {a.noise_ratio:g}x, matched={'YES' if ok else 'NO'}")
    print(f"{{WIN_VOTES}}    (beat 5.7) : {win['votes']}   ({win['song']})")
    print(f"{{RUNNER_UP_VOTES}} (5.7)   : {runner}")
    print(f"{{NOISE_LIMIT}}  (beat 6.2) : {noise_limit}x as loud as the music (>= {a.pass_rate:.0%} right)")
    print(f"{{MIN_SECONDS}}  (beat 6.3) : {min_seconds} seconds (>= {a.pass_rate:.0%} right)")
    print("Plots saved in outputs/; the noisy hero clip is outputs/noisy_clip.wav")


if __name__ == "__main__":
    main()
