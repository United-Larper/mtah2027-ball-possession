"""Locate and load the cached external matches (7 DFL + 3 Metrica sample games) as v1 training data.

* DFL labels are coarsened with `relabel.merge_short_spells` (rule "sandwich", 1.0 s; threshold calibrated
  on the Metrica sample games, see research/metrica-sample-report.md).
* Metrica sample games blank the ball during every stoppage; `prepare_metrica` makes the ball track usable:
  short gaps (< 2 s) in alive play are interpolated linearly, and every ball gap that overlaps a dead stretch
  is filled by linear interpolation between the last position before and the first position after it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .match import Match, D
from .paths import cache_dir
from .relabel import merge_short_spells

COARSEN_FRAMES = 25
COARSEN_RULE = "sandwich"
SHORT_GAP_FRAMES = 50          # 2 s


def external_ids() -> list[str]:
    root = cache_dir() / "external"
    dfl = sorted(p.stem.removeprefix("dfl_") for p in root.glob("dfl_*.npz"))
    met = sorted(p.stem for p in root.glob("metrica_g*.npz"))
    return dfl + met


def _path(i: str):
    root = cache_dir() / "external"
    return root / (i + ".npz") if i.startswith("metrica_") else root / f"dfl_{i}.npz"


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def prepare_metrica(m: Match):
    """Fill the ball track (see module docstring). Returns (match, stats dict)."""
    bx, by = m.bx.astype(np.float64).copy(), m.by.astype(np.float64).copy()
    stats = dict(short_alive=0, dead_gaps=0, dead_gap_frames=0, left_missing_frames=0)
    for p, sl in m.period_slices().items():
        x, y = bx[sl], by[sl]
        dead = m.label[sl] == D
        n = len(x)
        for s, e in _runs(~(np.isfinite(x) & np.isfinite(y))):
            if s == 0 or e == n:
                continue                               # no anchor on one side: leave missing
            overlaps_dead = dead[s:e].any()
            if overlaps_dead or (e - s) < SHORT_GAP_FRAMES:
                w = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
                x[s:e] = x[s - 1] + w * (x[e] - x[s - 1])
                y[s:e] = y[s - 1] + w * (y[e] - y[s - 1])
                stats["dead_gaps" if overlaps_dead else "short_alive"] += 1
                stats["dead_gap_frames"] += (e - s) if overlaps_dead else 0
            else:
                stats["left_missing_frames"] += e - s
    return replace(m, bx=bx.astype(np.float32), by=by.astype(np.float32)), stats


def load_external(ids=None) -> list[Match]:
    out = []
    for i in (ids or external_ids()):
        m = Match.load(_path(i))
        if m.source == "dfl":
            m = replace(m, label=merge_short_spells(m.label, m.period, COARSEN_FRAMES, COARSEN_RULE))
        else:
            m, _ = prepare_metrica(m)
        out.append(m)
    return out
