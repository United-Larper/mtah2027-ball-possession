"""Canonical orientation: the home team always attacks towards +x; y is left alone.

The tactical camera sits on the y<0 side in every match, so y must not be mirrored. Training
augmentation swaps the teams instead of flipping y: after `swap_teams` the old away side is "home"
and attacks +x.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .match import Match, H, A


def _sign_per_frame(m: Match) -> np.ndarray:
    s = np.ones(m.T, np.float32)
    for p, sl in m.period_slices().items():
        s[sl] = float(m.home_dir.get(p, 1))
    return s


def canonicalise(m: Match) -> Match:
    """Mirror x per period so that the home team attacks +x. Returns a new Match."""
    s = _sign_per_frame(m)
    return replace(m, hx=m.hx * s[:, None], ax=m.ax * s[:, None], bx=m.bx * s,
                   hy=m.hy.copy(), ay=m.ay.copy(), by=m.by.copy(), bz=m.bz.copy(),
                   home_dir={1: 1, 2: 1})


def decanonicalise_labels(m: Match, labels: np.ndarray) -> np.ndarray:
    """Labels are orientation-free (H/A/D), so this is the identity. Kept for readability."""
    return labels


def swap_label(label: np.ndarray) -> np.ndarray:
    out = label.copy()
    out[label == H] = A
    out[label == A] = H
    return out


def swap_teams(m: Match) -> Match:
    """Team-swap augmentation of a CANONICAL match: new home = old away, x mirrored, H<->A."""
    return replace(m, hx=-m.ax, hy=m.ay.copy(), ax=-m.hx, ay=m.hy.copy(),
                   bx=-m.bx, by=m.by.copy(), bz=m.bz.copy(),
                   label=swap_label(m.label), match_id=m.match_id + "_swap")
