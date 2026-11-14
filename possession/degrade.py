"""Make clean external (DFL) tracking look like the competition match's tracking.

The target tracking is a camera-based product: the ball is a 2-D image detection projected on the
ground plane (airborne balls are displaced away from the camera), the ball is lost for long blocks
that are tied to stoppages, and players outside the panning camera's field of view are not
tracked. Each component below is switchable. Parameters given as `(lo, hi)` tuples are drawn
uniformly per degraded copy (domain randomisation); `mid=True` uses the mid-points instead.

Input and output are CANONICAL matches (home attacks +x, camera on the y<0 side).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .camera import project_to_ground
from .match import Match, D

FPS = 25.0
STATS_PATH = Path(__file__).resolve().parent / "assets" / "target_stats.json"

# Target ball-missing run lengths (frames), from the EDA: (lo, hi, count) per bin of >=126 frames.
MISSING_RUN_BINS = [(126, 250, 53), (251, 500, 35), (501, 1000, 31), (1000, 3000, 13)]


@dataclass
class DegradeParams:
    camera: tuple = (0.77, 81.69, 29.99)     # camera is on the +y side (target_stats.json)
    aerial: bool = False                     # the target ball track is NOT back-projected: off by default
    ball_smooth: float = 6.0                 # Gaussian smoothing (frames): target tracks are smooth
    player_smooth: float = 3.0
    ball_sigma: float = 0.0
    mask_blocks: bool = True
    p_dead_block: object = (0.5, 0.95)       # probability a dead stretch (>2 s) triggers a block
    reacq_median_s: object = (2.0, 8.0)      # median re-acquisition delay after the stretch ends
    reacq_sigma: float = 0.6                 # lognormal sigma of that delay
    missing_share: object = (0.3, 0.5)       # overall target share of ball-missing frames
    dead_min_s: float = 2.0
    min_block_s: float = 5.0
    min_present_s: float = 1.0
    fov: bool = True
    fov_half_extent: tuple = (30.5, 4.0)     # mean, std (m); varies slowly in time
    fov_margin: float = 3.0                  # added to the half extent (calibrated with scripts/calib_fov.py)
    fov_persist_q: object = (0.4, 0.55)             # share of time an out-of-view player is nevertheless still tracked
    fov_persist_tau_s: float = 20.0          # slow on/off switching of that state (dropouts are slow, not flicker)
    fov_lag_s: float = 1.5
    fov_tau_alive_s: float = 1.0
    fov_tau_dead_s: float = 6.0
    player_sigma: float = 0.0


def default_params() -> DegradeParams:
    """Defaults, overridden by `assets/target_stats.json` where it provides a camera centre."""
    p = DegradeParams()
    try:
        s = json.loads(STATS_PATH.read_text())
        c = s.get("camera_centre_m")
        if c is not None and len(c) == 3:
            p = replace(p, camera=tuple(float(v) for v in c))
    except Exception:  # noqa: BLE001 - optional file
        pass
    return p


def _draw(v, rng, mid):
    if isinstance(v, tuple) and len(v) == 2 and not isinstance(v[0], bool):
        return 0.5 * (v[0] + v[1]) if mid else float(rng.uniform(v[0], v[1]))
    return v


def nan_gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 that ignores NaNs (normalised convolution); NaNs stay NaN."""
    if sigma <= 0:
        return x
    from scipy.ndimage import gaussian_filter1d
    ok = np.isfinite(x)
    num = gaussian_filter1d(np.where(ok, x, 0.0).astype(np.float64), sigma, axis=0, mode="nearest", truncate=3.0)
    den = gaussian_filter1d(ok.astype(np.float64), sigma, axis=0, mode="nearest", truncate=3.0)
    with np.errstate(all="ignore"):
        out = num / den
    return np.where(ok, out, np.nan).astype(x.dtype)


def _runs(mask: np.ndarray):
    """[(start, stop)) of True runs."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def _live_block_length(rng) -> int:
    w = np.array([b[2] for b in MISSING_RUN_BINS], float)
    lo, hi, _ = MISSING_RUN_BINS[rng.choice(len(w), p=w / w.sum())]
    return int(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def missing_mask(dead: np.ndarray, rng, p_dead, reacq_med_s, sigma, share, prm: DegradeParams) -> np.ndarray:
    """Boolean [n] ball-missing mask for one half. `dead` is the true dead-ball indicator."""
    n = len(dead)
    miss = np.zeros(n, bool)
    for s, e in _runs(dead):
        if e - s <= prm.dead_min_s * FPS or rng.random() > p_dead:
            continue
        a = s + int(rng.uniform(0, 2.0) * FPS)
        delay = float(np.clip(rng.lognormal(np.log(reacq_med_s), sigma), 0.0, 25.0))
        miss[a:min(n, e + int(delay * FPS))] = True
    guard = 0
    while miss.mean() < share and guard < 3000:
        guard += 1
        L = _live_block_length(rng)
        a = int(rng.integers(0, max(1, n - L)))
        miss[a:a + L] = True
    # constraints: short present gaps are filled, short missing blocks are extended
    for s, e in _runs(~miss):
        if e - s < prm.min_present_s * FPS and s > 0 and e < n:
            miss[s:e] = True
    for s, e in _runs(miss):
        if e - s < prm.min_block_s * FPS:
            miss[s:min(n, s + int(prm.min_block_s * FPS))] = True
    return miss


def _slow_process(n: int, mean: float, std: float, rng, phi: float = 0.97) -> np.ndarray:
    """AR(1) at 1 Hz with stationary std `std`, linearly interpolated to frames."""
    k = int(n / FPS) + 3
    z = np.empty(k)
    z[0] = rng.normal()
    eps = rng.normal(size=k) * np.sqrt(1 - phi ** 2)
    for i in range(1, k):
        z[i] = phi * z[i - 1] + eps[i]
    return mean + std * np.interp(np.arange(n) / FPS, np.arange(k), z)


def _persist_state(n: int, ncol: int, q: float, tau_s: float, rng) -> np.ndarray:
    """[n, ncol] slow two-state Markov chains (stationary P(on)=q, mean switching time ~tau_s)."""
    k = int(n / FPS) + 2
    on = np.empty((k, ncol), bool)
    on[0] = rng.random(ncol) < q
    u = rng.random((k, ncol))
    p_off, p_on = (1 - q) / tau_s, q / tau_s
    for i in range(1, k):
        on[i] = np.where(on[i - 1], u[i] > p_off, u[i] < p_on)
    return np.repeat(on, int(FPS), axis=0)[:n]


def _camera_centre_x(bx: np.ndarray, dead: np.ndarray, prm: DegradeParams) -> np.ndarray:
    """Camera centre-x: the ball's x, lagged, low-pass filtered (slower while the ball is dead)."""
    n = len(bx)
    b = bx.astype(np.float64)
    idx = np.arange(n)
    ok = np.isfinite(b)
    b = np.interp(idx, idx[ok], b[ok]) if ok.any() else np.zeros(n)
    lag = int(prm.fov_lag_s * FPS)
    tgt = np.concatenate([np.full(lag, b[0]), b[:n - lag]]) if lag else b
    a_alive = 1 - np.exp(-1 / (prm.fov_tau_alive_s * FPS))
    a_dead = 1 - np.exp(-1 / (prm.fov_tau_dead_s * FPS))
    alpha = np.where(dead, a_dead, a_alive)
    out = np.empty(n)
    c = tgt[0]
    for i in range(n):
        c += alpha[i] * (tgt[i] - c)
        out[i] = c
    return np.clip(out, -40.0, 40.0)


def degrade(m: Match, rng: np.random.Generator, params: DegradeParams | None = None, mid: bool = False) -> Match:
    """A degraded copy of a canonical match (labels unchanged)."""
    prm = params or default_params()
    hx, hy, ax, ay = (a.astype(np.float32).copy() for a in (m.hx, m.hy, m.ax, m.ay))
    bx, by = m.bx.astype(np.float64).copy(), m.by.astype(np.float64).copy()
    clean_bx = m.bx.astype(np.float64)

    for p, sl in m.period_slices().items():          # smoothing never crosses a half
        for arr in (hx, hy, ax, ay):
            arr[sl] = nan_gauss(arr[sl], prm.player_smooth)
        bx[sl] = nan_gauss(bx[sl], prm.ball_smooth)
        by[sl] = nan_gauss(by[sl], prm.ball_smooth)
    if prm.aerial and np.isfinite(m.bz).any():
        z = np.nan_to_num(m.bz.astype(np.float64), nan=0.0)
        bx, by = project_to_ground(bx, by, z, prm.camera)
    if prm.ball_sigma > 0:
        bx = bx + rng.normal(0, prm.ball_sigma, bx.shape)
        by = by + rng.normal(0, prm.ball_sigma, by.shape)

    p_dead = float(_draw(prm.p_dead_block, rng, mid))
    reacq = float(_draw(prm.reacq_median_s, rng, mid))
    share = float(_draw(prm.missing_share, rng, mid))
    fov_off = float(rng.normal(0, 1.0)) if not mid else 0.0

    for p, sl in m.period_slices().items():
        dead = m.label[sl] == D
        if prm.mask_blocks:
            miss = missing_mask(dead, rng, p_dead, reacq, prm.reacq_sigma, share, prm)
            bx[sl][miss] = np.nan
            by[sl][miss] = np.nan
        if prm.fov:
            n = sl.stop - sl.start
            cx = _camera_centre_x(clean_bx[sl], dead, prm)
            if mid:
                half = np.full(n, prm.fov_half_extent[0])
            else:
                half = _slow_process(n, prm.fov_half_extent[0] + fov_off, prm.fov_half_extent[1], rng)
            half = np.clip(half, 20.0, 45.0) + prm.fov_margin
            q = float(_draw(prm.fov_persist_q, rng, mid))
            for X, Y in ((hx, hy), (ax, ay)):
                out = np.abs(X[sl] - cx[:, None]) > half[:, None]
                if q > 0:
                    out &= ~_persist_state(n, X.shape[1], q, prm.fov_persist_tau_s, rng)
                X[sl][out] = np.nan
                Y[sl][out] = np.nan
    if prm.player_sigma > 0:
        for arr in (hx, hy, ax, ay):
            arr += rng.normal(0, prm.player_sigma, arr.shape).astype(np.float32)
    return replace(m, hx=hx, hy=hy, ax=ax, ay=ay, bx=bx.astype(np.float32), by=by.astype(np.float32),
                   bz=np.full(m.T, np.nan, np.float32), label=m.label.copy(), match_id=m.match_id + "_deg")
