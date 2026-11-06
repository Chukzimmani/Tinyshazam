"""Trick 3: join stars into pairs and turn each pair into a fingerprint.

Each star takes a turn as the anchor, and we pair it with a few stars that come
just after it (its "target zone"). A pair gives three numbers:

    note 1, note 2, and the time gap between them

and we glue them together into one short code. The time gap measures the
distance between the two stars, not the clock time, so a fingerprint stays the
same wherever in the song a clip begins.
"""
from collections import defaultdict

import numpy as np

FAN_OUT = 5          # how many partners each anchor gets
MIN_GAP = 1          # target zone starts 1 slice after the anchor...
MAX_GAP = 63         # ...and ends 63 slices later (about 2.9 seconds)
MAX_NOTE_JUMP = 200  # partners must be within this many note bins of the anchor


def glue(note1, note2, gap):
    """Pack three small numbers into one integer code (10 + 10 + 6 bits)."""
    return (int(note1) << 16) | (int(note2) << 6) | int(gap)


def unglue(code):
    """Undo `glue`, handy for showing a fingerprint on screen."""
    return code >> 16, (code >> 6) & 0x3FF, code & 0x3F


def pairs(stars):
    """Yield (anchor index, partner index) for every pair, using the target zone."""
    times, notes = stars[:, 0], stars[:, 1]
    for i in range(len(stars)):
        made = 0
        for j in range(i + 1, len(stars)):
            gap = times[j] - times[i]
            if gap < MIN_GAP:
                continue
            if gap > MAX_GAP or made == FAN_OUT:
                break
            if abs(int(notes[j]) - int(notes[i])) > MAX_NOTE_JUMP:
                continue
            yield i, j
            made += 1


def fingerprints(stars):
    """Yield (code, anchor_time) for every pair of stars."""
    times, notes = stars[:, 0], stars[:, 1]
    for i, j in pairs(stars):
        yield glue(notes[i], notes[j], times[j] - times[i]), int(times[i])


class Index:
    """The big table, like the index at the back of a book.

    code -> list of (song_id, time in the song where that pair starts)
    """

    def __init__(self):
        self.table = defaultdict(list)
        self.songs = []  # song_id -> song name

    def add_song(self, name, stars):
        song_id = len(self.songs)
        self.songs.append(name)
        count = 0
        for code, t in fingerprints(stars):
            self.table[code].append((song_id, t))
            count += 1
        return count

    def __len__(self):
        return sum(len(v) for v in self.table.values())
