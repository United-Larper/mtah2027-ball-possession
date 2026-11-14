"""Label-free calibration of auxiliary categorical streams against a fixed classifier (HMM + EM).

States are (H, A, D) = (0, 1, 2). The classifier supplies per-frame log-likelihood-like scores ``log_e`` that are
treated as FIXED evidence. Each auxiliary stream is a per-frame categorical value ``v_t`` (negative = missing)
with a table P(v | state) learnt by EM (forward-backward over the sticky chain, Dirichlet smoothing). No labels
are used anywhere.

Symmetry specs (what is tied):
  ``alive_dead``: P(v|H) = P(v|A); D has its own table.
  ``team``: a value permutation ``swap`` (an involution exchanging home and away values) is given;
            P(v|A) = P(swap(v)|H); the D table is symmetrised under the same swap (no home advantage when dead).

The fused evidence for downstream decoding is ``log_aux[t, s] = sum_k gamma_k * clip(log P_k(v|s) - log P_k(v),
-c, c)``, with ``log P_k(v)`` the model marginal under the average state occupancy (so it is a log-ratio).
``e_scale`` tempers the classifier scores inside the EM only (an overconfident classifier makes the posteriors, and
hence the tables, echo its mistakes); the downstream decoder is free to use its own scale. Tables are JSON-serialisable (``tables_to_json`` / ``tables_from_json``) so a fit can be reused deterministically.

Streams are dicts: ``dict(name=str, values=int array [T], n_values=int, sym="alive_dead"|"team", swap=int array)``.
"""
from __future__ import annotations

import json

import numpy as np
from numba import njit

H, A, D = 0, 1, 2


@njit(cache=True)
def _fb(logB, stay):
    """Scaled forward-backward on a sticky 3-state chain. logB (n,3). Returns gamma (n,3), loglik."""
    n = logB.shape[0]
    move = (1.0 - stay) / 2.0
    B = np.empty((n, 3))
    sc = np.empty(n)
    for t in range(n):
        m = max(logB[t, 0], max(logB[t, 1], logB[t, 2]))
        sc[t] = m
        for s in range(3):
            B[t, s] = np.exp(logB[t, s] - m)
    al = np.empty((n, 3))
    ll = 0.0
    for s in range(3):
        al[0, s] = B[0, s] / 3.0
    z = al[0, 0] + al[0, 1] + al[0, 2]
    ll += np.log(z) + sc[0]
    for s in range(3):
        al[0, s] /= z
    for t in range(1, n):
        tot = al[t - 1, 0] + al[t - 1, 1] + al[t - 1, 2]
        for s in range(3):
            al[t, s] = (stay * al[t - 1, s] + move * (tot - al[t - 1, s])) * B[t, s]
        z = al[t, 0] + al[t, 1] + al[t, 2]
        ll += np.log(z) + sc[t]
        for s in range(3):
            al[t, s] /= z
    be = np.empty((n, 3))
    for s in range(3):
        be[n - 1, s] = 1.0
    for t in range(n - 2, -1, -1):
        w0 = B[t + 1, 0] * be[t + 1, 0]
        w1 = B[t + 1, 1] * be[t + 1, 1]
        w2 = B[t + 1, 2] * be[t + 1, 2]
        tot = w0 + w1 + w2
        be[t, 0] = stay * w0 + move * (tot - w0)
        be[t, 1] = stay * w1 + move * (tot - w1)
        be[t, 2] = stay * w2 + move * (tot - w2)
        z = be[t, 0] + be[t, 1] + be[t, 2]
        for s in range(3):
            be[t, s] /= z
    g = al * be
    for t in range(n):
        z = g[t, 0] + g[t, 1] + g[t, 2]
        for s in range(3):
            g[t, s] /= z
    return g, ll


def _segments(period: np.ndarray):
    out = []
    for p in np.unique(period):
        if p <= 0:
            continue
        idx = np.flatnonzero(period == p)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def posterior(log_e: np.ndarray, log_aux: np.ndarray | None, period: np.ndarray, stay: float = 0.995,
              e_scale: float = 1.0):
    """Smoothed posteriors [T,3] of the chain given log_e (+ optional log_aux), per period. Period 0 -> 1/3."""
    T = log_e.shape[0]
    g = np.full((T, 3), 1.0 / 3.0)
    ll = 0.0
    tot = e_scale * log_e if log_aux is None else e_scale * log_e + log_aux
    for a, b in _segments(period):
        gg, l = _fb(np.ascontiguousarray(tot[a:b], dtype=np.float64), float(stay))
        g[a:b] = gg
        ll += l
    return g, ll


# --------------------------------------------------------------------------------------------- tables
def _expand(stream, params):
    """Full 3xV table from tied parameters."""
    V = stream["n_values"]
    if stream["sym"] == "alive_dead":
        return np.stack([params["alive"], params["alive"], params["dead"]])
    swap = np.asarray(stream["swap"])
    return np.stack([params["home"], params["home"][swap], params["dead"]])


def _init_params(stream):
    V = stream["n_values"]
    u = np.full(V, 1.0 / V)
    if stream["sym"] == "alive_dead":
        return dict(alive=u.copy(), dead=u.copy())
    return dict(home=u.copy(), dead=u.copy())


def _mstep(stream, gamma, alpha):
    V = stream["n_values"]
    v = stream["values"]
    ok = v >= 0
    cnt = np.zeros((3, V))
    for s in range(3):
        cnt[s] = np.bincount(v[ok], weights=gamma[ok, s], minlength=V)[:V]
    nrm = lambda c: (c + alpha) / (c + alpha).sum()  # noqa: E731
    if stream["sym"] == "alive_dead":
        return dict(alive=nrm(cnt[H] + cnt[A]), dead=nrm(cnt[D]))
    swap = np.asarray(stream["swap"])
    # P(v|A) = q[swap v]  =>  q[u] collects home counts at u and away counts at swap(u)
    home = cnt[H] + cnt[A][swap]
    dead = cnt[D] + cnt[D][swap]
    return dict(home=nrm(home), dead=nrm(dead))


def _log_tables(streams, params):
    return [np.log(np.maximum(_expand(st, p), 1e-12)) for st, p in zip(streams, params)]


def _stream_loglik(stream, logtab):
    """[T,3] log P(v_t | s); 0 where v is missing."""
    v = stream["values"]
    out = np.zeros((v.size, 3))
    ok = v >= 0
    out[ok] = logtab[:, v[ok]].T
    return out


def fit_aux_tables(log_e, period, streams, stay: float = 0.995, n_iter: int = 100, alpha: float = 1.0,
                   gamma: float = 1.0, clip: float = 3.0, tol: float = 0.5, e_scale: float = 1.0,
                   verbose: bool = False) -> dict:
    """EM for the tied tables (stops when the log-likelihood gain is below ``tol`` nats). The E-step tempers each
    stream's emission by ``gamma`` (a stream that repeats one value over many frames, like tags, would otherwise be
    counted as many independent observations; gamma ~ 1 / typical hold length). Returns a JSON-able dict with ``streams`` (tables), ``occupancy``, ``stay``, ``gamma``,
    ``clip`` and the log-likelihood history."""
    params = [_init_params(st) for st in streams]
    hist = []
    prev = None
    for it in range(n_iter):
        logt = _log_tables(streams, params)
        aux = gamma * sum((_stream_loglik(st, lt) for st, lt in zip(streams, logt)), np.zeros(log_e.shape))
        g, ll = posterior(log_e, aux, period, stay, e_scale)
        hist.append(float(ll))
        if verbose:
            print(f"  EM iter {it}: loglik {ll:.2f}")
        params = [_mstep(st, g, alpha) for st in streams]
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    sel = period > 0
    occ = g[sel].mean(0)
    out = dict(stay=stay, e_scale=e_scale, gamma=gamma, clip=clip, alpha=alpha, occupancy=occ.tolist(), loglik=hist, streams={})
    for st, p in zip(streams, params):
        tab = _expand(st, p)
        out["streams"][st["name"]] = dict(sym=st["sym"], n_values=int(st["n_values"]),
                                          swap=None if st["sym"] != "team" else np.asarray(st["swap"]).tolist(),
                                          table=tab.tolist())
    return out


def tables_to_json(tables: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(tables, f)


def tables_from_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_aux_tables(tables: dict, streams, period=None, gamma: float | None = None, clip: float | None = None):
    """log_aux [T,3] = sum_k gamma * clip(log P_k(v|s) - log P_k(v), -c, c); 0 where a value is missing.
    The marginal uses the stored state occupancy."""
    gam = tables["gamma"] if gamma is None else gamma
    c = tables["clip"] if clip is None else clip
    occ = np.asarray(tables["occupancy"])
    T = len(streams[0]["values"])
    out = np.zeros((T, 3))
    for st in streams:
        tab = np.asarray(tables["streams"][st["name"]]["table"])
        lt = np.log(np.maximum(tab, 1e-12))
        lm = np.log(np.maximum(occ @ tab, 1e-12))
        v = st["values"]
        ok = v >= 0
        r = np.clip(lt[:, v[ok]] - lm[v[ok]][None, :], -c, c)
        out[ok] += gam * r.T
    if period is not None:
        out[period <= 0] = 0.0
    return out


def fuse(log_e, period, streams, tables=None, **fit_kw):
    """Convenience: fit (if no tables), apply, return (tables, log_aux, posterior of log_e + log_aux)."""
    if tables is None:
        tables = fit_aux_tables(log_e, period, streams, **fit_kw)
    log_aux = apply_aux_tables(tables, streams, period)
    g, _ = posterior(log_e, log_aux, period, tables["stay"], tables.get("e_scale", 1.0))
    return tables, log_aux, g
