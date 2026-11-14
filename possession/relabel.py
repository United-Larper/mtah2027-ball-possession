"""Coarsening of per-frame possession labels: erase brief touches inside the opponent's possession.

merge_short_spells(label, period, min_len_frames, rule="sandwich") -> new int8 label array.

A possession spell is a maximal run of H or of A inside one period (the scorer's definition, a D or
X frame ends it). Rules (all iterate, always handling the SHORTEST qualifying spell first, until
no spell qualifies; each period is processed separately):

  "sandwich" (default)  a spell shorter than min_len_frames whose left AND right neighbouring
                        runs are alive (H/A) and both of the OTHER team is relabelled to that team,
                        so the three runs fuse into one. Spells adjacent to a D/X run, or at a
                        period edge, are left alone. Rationale: "a touch without control, a
                        deflection or a won duel without control is not a change of possession".
  "prev"                a spell shorter than min_len_frames whose LEFT neighbour is an alive run
                        of the other team is relabelled to that team regardless of what follows
                        (right neighbour may be D, X or the same team as the left one).
  "sandwich_d"          like "sandwich", and also when the right neighbour is a D run that is
                        itself followed by an alive run of the other team (touch, stoppage, then
                        the opponent restarts): the touch is erased.

Frames labelled D or X are never changed.
"""
from __future__ import annotations

import numpy as np

H, A, D, X = 0, 1, 2, 3


def _runs(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    b = np.concatenate(([0], ch, [len(lab)]))
    return [[int(b[i]), int(b[i + 1]), int(lab[b[i]])] for i in range(len(b) - 1)]


def _merge_period(lab, min_len, rule):
    runs = _runs(lab)
    n_changed = 0
    while True:
        best = None
        for i, (s, e, c) in enumerate(runs):
            if c not in (H, A) or e - s >= min_len or i == 0:
                continue
            o = 1 - c
            left = runs[i - 1]
            if left[2] != o:
                continue
            ok = False
            if rule == "prev":
                ok = True
            elif i + 1 < len(runs):
                right = runs[i + 1]
                if right[2] == o:
                    ok = True
                elif rule == "sandwich_d" and right[2] == D and i + 2 < len(runs) and runs[i + 2][2] == o:
                    ok = True
            if ok and (best is None or e - s < runs[best][1] - runs[best][0]):
                best = i
        if best is None:
            break
        i = best
        s, e, c = runs[i]
        runs[i][2] = 1 - c
        # fuse with equal-label neighbours
        lo, hi = i, i
        if i > 0 and runs[i - 1][2] == runs[i][2]:
            lo = i - 1
        if i + 1 < len(runs) and runs[i + 1][2] == runs[i][2]:
            hi = i + 1
        merged = [runs[lo][0], runs[hi][1], runs[i][2]]
        runs[lo:hi + 1] = [merged]
        n_changed += 1
    out = np.empty_like(lab)
    for s, e, c in runs:
        out[s:e] = c
    return out


def merge_short_spells(label, period, min_len_frames, rule="sandwich"):
    """See module docstring. label: int8 [T] (0=H 1=A 2=D 3=X), period: [T]. Returns a new array."""
    label = np.asarray(label)
    period = np.asarray(period)
    out = label.astype(np.int8).copy()
    if min_len_frames <= 1:
        return out
    if rule not in ("sandwich", "prev", "sandwich_d"):
        raise ValueError(rule)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = _merge_period(out[idx], int(min_len_frames), rule)
    return out
