"""Target match with the tactical ball filled from the live broadcast ball where the tactical ball is missing.

Fill candidates: tactical ball missing and ``bstate == 4`` (live broadcast, ball present). Plausibility guard,
computed per consecutive run of candidates: the ball must lie inside the pitch rectangle extended by 3 m and
must not move faster than 40 m/s from the previous kept frame of the same run. Failing frames are dropped.
Tactical positions are never altered.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .aux_streams import build_aux
from .match import Match
from .target import load_target_match

MARGIN = 3.0
MAX_SPEED = 40.0
_STATS: dict = {}


def _fill(m: Match, aux: dict):
    tact = np.isfinite(m.bx) & np.isfinite(m.by)
    bxb, byb = aux["bball_x"], aux["bball_y"]
    cand = ~tact & (aux["bstate"] == 4) & np.isfinite(bxb) & np.isfinite(byb)
    L, W = m.pitch
    inside = (np.abs(bxb) <= L / 2 + MARGIN) & (np.abs(byb) <= W / 2 + MARGIN)
    keep = np.zeros(m.T, bool)
    d = np.diff(np.concatenate([[0], cand.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    dropped_rect = dropped_jump = 0
    for s, e in zip(starts, ends):
        last = -1
        for t in range(s, e):
            if not inside[t]:
                dropped_rect += 1
                continue
            if last >= 0:
                v = np.hypot(bxb[t] - bxb[last], byb[t] - byb[last]) / ((t - last) / m.fps)
                if v > MAX_SPEED:
                    dropped_jump += 1
                    continue
            keep[t] = True
            last = t
    src = np.zeros(m.T, np.int8)
    src[tact] = 1
    src[keep] = 2
    stats = dict(candidates=int(cand.sum()), candidate_runs=int(starts.size), filled=int(keep.sum()),
                 dropped_rect=dropped_rect, dropped_jump=dropped_jump)
    return keep, src, stats


def fused_ball_source() -> np.ndarray:
    """int8 [T]: 0 no ball, 1 tactical, 2 broadcast fill."""
    return _fill(load_target_match(), build_aux())[1]


def fill_stats() -> dict:
    return _fill(load_target_match(), build_aux())[2]


def load_target_match_fused() -> Match:
    m = load_target_match()
    aux = build_aux()
    keep, _, _ = _fill(m, aux)
    bx, by = m.bx.copy(), m.by.copy()
    bx[keep] = aux["bball_x"][keep]
    by[keep] = aux["bball_y"][keep]
    return dataclasses.replace(m, bx=bx, by=by)
