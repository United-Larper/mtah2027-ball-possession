"""Auxiliary-evidence fit with a confidence mask, and the v1 stream definitions.

`fusion.fit_aux_tables` has no per-frame masking, so this wrapper re-uses its internals: the E-step (forward-backward
over ALL frames, so the chain stays intact) is unchanged, but the M-step counts only frames where the classifier is
confident (max state probability >= `conf`). Frames outside the mask therefore do not teach the tables, which keeps EM
from absorbing the classifier's own uncertain-region bias. Nothing here uses labels.
"""
from __future__ import annotations

import numpy as np

from . import fusion as fu

# fixed in advance (not tuned on anything)
E_SCALE = 0.3
GAMMA = 0.3
CLIP = 3.0
CONF = 0.9
STAY = 0.995
STREAM_NAMES = ("bstate", "sp_phase", "goal_win", "att_phase")


def make_streams(aux: dict) -> list[dict]:
    """bstate / sp_phase / goal_win tied alive-vs-dead; att_phase team-swap tied (home<->away codes 1<->2)."""
    return [
        dict(name="bstate", values=aux["bstate"].astype(np.int64), n_values=5, sym="alive_dead"),
        dict(name="sp_phase", values=aux["sp_phase"].astype(np.int64), n_values=9, sym="alive_dead"),
        dict(name="goal_win", values=aux["goal_win"].astype(np.int64), n_values=3, sym="alive_dead"),
        dict(name="att_phase", values=aux["att_phase"].astype(np.int64), n_values=4, sym="team", swap=np.array([0, 2, 1, 3])),
    ]


def fit_aux_masked(log_e, period, streams, mask, stay=STAY, n_iter=100, alpha=1.0, gamma=GAMMA, clip=CLIP,
                   tol=0.5, e_scale=E_SCALE) -> dict:
    """Same output as `fusion.fit_aux_tables`, with the M-step restricted to frames where `mask` is True."""
    mstreams = [dict(st, values=np.where(mask, st["values"], -1)) for st in streams]
    params = [fu._init_params(st) for st in streams]
    hist, prev = [], None
    for _ in range(n_iter):
        logt = fu._log_tables(streams, params)
        aux = gamma * sum((fu._stream_loglik(st, lt) for st, lt in zip(streams, logt)), np.zeros(log_e.shape))
        g, ll = fu.posterior(log_e, aux, period, stay, e_scale)
        hist.append(float(ll))
        params = [fu._mstep(st, g, alpha) for st in mstreams]
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    sel = period > 0
    out = dict(stay=stay, e_scale=e_scale, gamma=gamma, clip=clip, alpha=alpha, conf_mask_share=float(mask[sel].mean()),
               occupancy=g[sel].mean(0).tolist(), loglik=hist, streams={})
    for st, p in zip(streams, params):
        out["streams"][st["name"]] = dict(sym=st["sym"], n_values=int(st["n_values"]),
                                          swap=None if st["sym"] != "team" else np.asarray(st["swap"]).tolist(),
                                          table=fu._expand(st, p).tolist())
    return out


def fit_aux(ps: np.ndarray, period: np.ndarray, streams: list[dict], conf: float = CONF) -> dict:
    """Fit tables from classifier state probabilities ps [T,3] on graded frames, confident frames only."""
    log_e = np.log(np.clip(ps.astype(np.float64), 1e-6, 1.0))
    mask = (period > 0) & (ps.max(1) >= conf)
    return fit_aux_masked(log_e, period, streams, mask)


def log_aux(tables: dict, streams: list[dict], period: np.ndarray) -> np.ndarray:
    return fu.apply_aux_tables(tables, streams, period, gamma=GAMMA, clip=CLIP)
