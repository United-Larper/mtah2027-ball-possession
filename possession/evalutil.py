"""Evaluation helpers around the official scorer (`evaluation/possession_metrics.py`)."""
from __future__ import annotations

import sys

import numpy as np

from .paths import REPO

sys.path.insert(0, str(REPO / "evaluation"))
import possession_metrics as pm  # noqa: E402

GAP = 100  # X frames inserted between halves so that the scorer's runs stay separate


def eval_arrays(m, pred_by_frame: np.ndarray):
    """Concatenate the graded halves of an external match with an X gap; returns (gt, pred)."""
    gt_parts, pr_parts = [], []
    for p, sl in m.period_slices().items():
        gt_parts += [m.label[sl], np.full(GAP, 3, np.int8)]
        pr_parts += [pred_by_frame[sl], np.full(GAP, 3, np.int8)]
    return np.concatenate(gt_parts[:-1]).astype(np.int8), np.concatenate(pr_parts[:-1]).astype(np.int8)


def full_eval(gt, pred):
    return pm.evaluate(gt, pred)


def _spells(codes, runs):
    s = pm.possession_spells(codes, runs)
    a = np.array(s, dtype=np.int64).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


def fast_spell_f1(gt, pred) -> float:
    """spell_f1_mean, same result as the scorer, with the pair enumeration vectorised."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs, ge, gc = _spells(gt, runs)
    ps, pe, pc = _spells(filled, runs)
    if len(gs) == 0 or len(ps) == 0:
        return 0.0
    inter = np.minimum(ge[:, None], pe[None, :]) - np.maximum(gs[:, None], ps[None, :])
    union = np.maximum(ge[:, None], pe[None, :]) - np.minimum(gs[:, None], ps[None, :])
    ok = (inter > 0) & (gc[:, None] == pc[None, :])
    i, j = np.nonzero(ok)
    iou = inter[i, j] / union[i, j]
    order = np.lexsort((j, i, -iou))
    i, j, iou = i[order], j[order], iou[order]
    f1s = []
    for thr in pm.SPELL_IOUS:
        used_g, used_p, tp = set(), set(), 0
        for a, b, v in zip(i.tolist(), j.tolist(), iou.tolist()):
            if v < thr:
                break
            if a in used_g or b in used_p:
                continue
            used_g.add(a); used_p.add(b); tp += 1
        f1s.append(2 * tp / (len(gs) + len(ps)))
    return float(np.mean(f1s))


# ---- organisers' baseline: nearest player's team, D when the ball is missing, +-w majority vote
def nearest_team_codes(m) -> np.ndarray:
    """-1 where the ball is missing, else the team (0/1) of the nearest tracked player."""
    bx, by = m.bx.astype(np.float64)[:, None], m.by.astype(np.float64)[:, None]
    dh = np.hypot(m.hx - bx, m.hy - by)
    da = np.hypot(m.ax - bx, m.ay - by)
    d1h = np.min(np.where(np.isfinite(dh), dh, np.inf), 1)
    d1a = np.min(np.where(np.isfinite(da), da, np.inf), 1)
    out = np.where(d1h <= d1a, 0, 1).astype(np.int8)
    out[~np.isfinite(m.bx) | ~np.isfinite(m.by) | ~np.isfinite(np.minimum(d1h, d1a))] = -1
    return out


def majority_vote(codes: np.ndarray, window: int = 50) -> np.ndarray:
    n = len(codes)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(n, 2, np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], 0, 1)
    return out


def baseline_labels(m, window: int = 50) -> np.ndarray:
    out = np.full(m.T, 3, np.int8)
    codes = nearest_team_codes(m)
    for p, sl in m.period_slices().items():
        out[sl] = majority_vote(codes[sl], window)
    return out
