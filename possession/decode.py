"""Viterbi decoding of H / A / D with learned, time-varying switching scores.

Path score = sum_t alpha * log p_state[t, s_t] + sum_t tau(s_{t-1}, s_t, t), where tau = 0 for staying and,
for a switch of type k (0 team change, 1 alive->D, 2 D->alive) at frame t,
    tau = beta_k * clip(logit(p_k[t]), -L, L) - c_k.
beta = 0 gives the plain constant-cost Viterbi. Each half is decoded separately.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numba import njit


@dataclass
class DecoderParams:
    alpha: float = 1.0
    beta: tuple = (0.0, 0.0, 0.0)     # team change, alive->D, D->alive
    c: tuple = (20.0, 12.0, 12.0)
    L: float = 8.0

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(alpha=d["alpha"], beta=tuple(d["beta"]), c=tuple(d["c"]), L=d["L"])


@njit(cache=True)
def _viterbi(e, sw):
    n = e.shape[0]
    back = np.zeros((n, 3), np.int8)
    d0, d1, d2 = e[0, 0], e[0, 1], e[0, 2]
    for t in range(1, n):
        tc, ad, da = sw[t, 0], sw[t, 1], sw[t, 2]
        # state H (0): from H stay, from A team change, from D restart
        a0, a1, a2 = d0, d1 + tc, d2 + da
        if a0 >= a1 and a0 >= a2:
            n0, b0 = a0, 0
        elif a1 >= a2:
            n0, b0 = a1, 1
        else:
            n0, b0 = a2, 2
        a0, a1, a2 = d0 + tc, d1, d2 + da
        if a1 >= a0 and a1 >= a2:
            n1, b1 = a1, 1
        elif a0 >= a2:
            n1, b1 = a0, 0
        else:
            n1, b1 = a2, 2
        a0, a1, a2 = d0 + ad, d1 + ad, d2
        if a2 >= a0 and a2 >= a1:
            n2, b2 = a2, 2
        elif a0 >= a1:
            n2, b2 = a0, 0
        else:
            n2, b2 = a1, 1
        d0, d1, d2 = n0 + e[t, 0], n1 + e[t, 1], n2 + e[t, 2]
        back[t, 0], back[t, 1], back[t, 2] = b0, b1, b2
    s = 0
    if d1 > d0 and d1 >= d2:
        s = 1
    elif d2 > d0 and d2 > d1:
        s = 2
    out = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = s
        s = back[t, s]
    return out


def switch_scores(p_trans: np.ndarray, prm: DecoderParams) -> np.ndarray:
    """[n,3] switching score per frame from transition-model probabilities [n,4] (none, team, alive->D, D->alive)."""
    p = np.clip(p_trans[:, 1:4].astype(np.float64), 1e-6, 1 - 1e-6)
    lg = np.clip(np.log(p / (1 - p)), -prm.L, prm.L)
    return lg * np.asarray(prm.beta)[None, :] - np.asarray(prm.c)[None, :]


def decode_half(p_state: np.ndarray, p_trans: np.ndarray | None, prm: DecoderParams) -> np.ndarray:
    e = prm.alpha * np.log(np.clip(p_state.astype(np.float64), 1e-6, 1.0))
    if p_trans is None:
        sw = np.tile(-np.asarray(prm.c, np.float64), (len(e), 1))
    else:
        sw = switch_scores(p_trans, prm)
    return _viterbi(np.ascontiguousarray(e), np.ascontiguousarray(sw))


def decode(p_state, p_trans, period, prm: DecoderParams, fill: int = 3) -> np.ndarray:
    """Decode each period (1, 2) of a full-length array; other frames get `fill` (X)."""
    out = np.full(len(period), fill, np.int8)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx):
            sl = slice(int(idx[0]), int(idx[-1]) + 1)
            out[sl] = decode_half(p_state[sl], None if p_trans is None else p_trans[sl], prm)
    return out
