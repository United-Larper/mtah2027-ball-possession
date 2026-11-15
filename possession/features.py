"""Identity-free per-frame features from a CANONICAL match (home attacks +x; camera on the +y side).

`extract(match) -> (X float32 [T, F], names)`. Rows with period 0 (ungraded frames of the target) are
NaN. Everything is computed per half, so no window crosses a half. NaN is allowed (LightGBM handles
it). No absolute time, frame index, period or score is used.

Feature groups (by name prefix): avail_, ball_, geo_, touch_, carry_, shape_, ctx_.
Convention: "near" touchline = +y (camera side), "far" = -y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .match import Match

CAP_S = 30.0          # cap for "seconds since / until" features
PRESENT_CAP_S = 60.0
TOUCH_DV = 3.0        # m/s velocity change over 0.2 s needed for a touch candidate
TOUCH_DIST = 2.0      # m, a player of the touching team must be this close
WINDOWS = (10, 25, 50, 125)   # 0.4, 1, 2, 5 s


# ---------------------------------------------------------------- small helpers
def shift(x: np.ndarray, k: int) -> np.ndarray:
    """y[t] = x[t-k] (NaN padded); k<0 looks ahead."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def roll(x: np.ndarray, w: int, how: str, causal: bool) -> np.ndarray:
    """Rolling mean/max/min/sum over finite values; window ends at t (causal) or starts at t."""
    s = pd.Series(x if causal else x[::-1])
    r = s.rolling(w, min_periods=1)
    out = getattr(r, how)().to_numpy()
    return out if causal else out[::-1]


def ffill(x):
    return pd.Series(x).ffill().to_numpy()


def bfill(x):
    return pd.Series(x).bfill().to_numpy()


def since(flag: np.ndarray, fps: float, cap: float) -> np.ndarray:
    idx = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, idx, -10 ** 9))
    return np.minimum((idx - last) / fps, cap)


def until(flag: np.ndarray, fps: float, cap: float) -> np.ndarray:
    idx = np.arange(len(flag))
    nxt = np.minimum.accumulate(np.where(flag, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum((nxt - idx) / fps, cap)


def run_info(present: np.ndarray, fps: float):
    """Length (s) of the current present/missing run, position within it and frames left (s)."""
    n = len(present)
    change = np.concatenate(([True], present[1:] != present[:-1]))
    start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    ends = np.concatenate((np.flatnonzero(change)[1:], [n]))
    run_id = np.cumsum(change) - 1
    end = ends[run_id]
    return (end - start) / fps, (np.arange(n) - start) / fps, (end - 1 - np.arange(n)) / fps


def two_smallest(d: np.ndarray):
    """Smallest and second-smallest over axis 1 ignoring NaN; NaN where fewer than 1 / 2 values."""
    f = np.where(np.isfinite(d), d, np.inf)
    f = np.partition(f, 1, axis=1)[:, :2] if f.shape[1] > 1 else np.concatenate([f, np.full_like(f, np.inf)], 1)
    f = np.sort(f, axis=1)
    f[~np.isfinite(f)] = np.nan
    return f[:, 0], f[:, 1]


def masked_mean(x: np.ndarray, axis: int = 1):
    ok = np.isfinite(x)
    n = ok.sum(axis)
    with np.errstate(all="ignore"):
        return np.where(n > 0, np.where(ok, x, 0.0).sum(axis) / n, np.nan), n


def masked_std(x: np.ndarray):
    mu, n = masked_mean(x)
    ok = np.isfinite(x)
    with np.errstate(all="ignore"):
        v = np.where(ok, (x - mu[:, None]) ** 2, 0.0).sum(1) / n
    return np.where(n > 1, np.sqrt(v), np.nan)


def nth_extreme(x: np.ndarray, highest: bool):
    """(extreme, second extreme) of finite entries per row; NaN where missing."""
    f = np.where(np.isfinite(x), -x if highest else x, np.inf)
    f = np.sort(f, axis=1)[:, :2] if f.shape[1] > 1 else np.concatenate([f, np.full_like(f, np.inf)], 1)
    f = np.where(np.isfinite(f), -f if highest else f, np.nan)
    return f[:, 0], f[:, 1]


def player_velocity(X, Y, fps, k=2, vmax=12.0):
    """Central-difference velocity (m/s); implausible speeds (tracklet swaps) become NaN."""
    h = 2 * k / fps
    vx = (shift(X, -k) - shift(X, k)) / h
    vy = (shift(Y, -k) - shift(Y, k)) / h
    sp = np.hypot(vx, vy)
    bad = sp > vmax
    return np.where(bad, np.nan, vx), np.where(bad, np.nan, vy), np.where(bad, np.nan, sp)


class _F:
    """Ordered feature collector."""

    def __init__(self):
        self.d: dict[str, np.ndarray] = {}

    def add(self, name, arr):
        self.d[name] = np.asarray(arr, np.float32)


# ---------------------------------------------------------------- per-half computation
def _half(m: Match, sl: slice, pitch) -> dict:
    fps = m.fps
    L, W = pitch
    f = _F()
    hx, hy = m.hx[sl].astype(np.float64), m.hy[sl].astype(np.float64)
    ax, ay = m.ax[sl].astype(np.float64), m.ay[sl].astype(np.float64)
    bx, by = m.bx[sl].astype(np.float64), m.by[sl].astype(np.float64)
    n = len(bx)
    t = np.arange(n)
    present = np.isfinite(bx) & np.isfinite(by)

    # ---- availability
    since_p = since(present, fps, PRESENT_CAP_S)
    until_p = until(present, fps, PRESENT_CAP_S)
    rl, rp, rr = run_info(present, fps)
    f.add("avail_present", present)
    f.add("avail_since_present", since_p)
    f.add("avail_until_present", until_p)
    f.add("avail_run_len", np.minimum(rl, 120))
    f.add("avail_run_pos", np.minimum(rp, 120))
    f.add("avail_run_rem", np.minimum(rr, 120))

    # ---- ball kinematics
    sx = pd.Series(bx).rolling(5, min_periods=3, center=True).mean().to_numpy().copy()
    sy = pd.Series(by).rolling(5, min_periods=3, center=True).mean().to_numpy().copy()
    sx[~present] = np.nan
    sy[~present] = np.nan
    k = 3
    h6 = 2 * k / fps
    vbx = (shift(sx, -k) - shift(sx, k)) / h6
    vby = (shift(sy, -k) - shift(sy, k)) / h6
    speed = np.hypot(vbx, vby)
    abx = (shift(vbx, -k) - shift(vbx, k)) / h6
    aby = (shift(vby, -k) - shift(vby, k)) / h6
    acc = np.hypot(abx, aby)
    v0x, v0y, v1x, v1y = shift(vbx, k), shift(vby, k), shift(vbx, -k), shift(vby, -k)
    n0, n1 = np.hypot(v0x, v0y), np.hypot(v1x, v1y)
    with np.errstate(all="ignore"):
        cosang = (v0x * v1x + v0y * v1y) / (n0 * n1)
    head = np.where((n0 > 1.0) & (n1 > 1.0), np.arccos(np.clip(cosang, -1, 1)), np.nan)
    m_near = np.where(present, np.maximum(0, by - W / 2), np.nan)       # beyond the near (+y) touchline
    m_far = np.where(present, np.maximum(0, -by - W / 2), np.nan)       # beyond the far (-y) touchline
    m_goal = np.where(present, np.maximum(0, np.abs(bx) - L / 2), np.nan)
    d_tl = W / 2 - np.abs(by)
    d_gl = L / 2 - np.abs(bx)
    f.add("ball_x", bx); f.add("ball_y", by); f.add("ball_absy", np.abs(by))
    box_w = np.abs(by) < 20.16                                            # canonical: home attacks +x
    f.add("ball_box_opp", np.where(present, ((bx > L / 2 - 16.5) & box_w).astype(float), np.nan))
    f.add("ball_box_own", np.where(present, ((bx < -L / 2 + 16.5) & box_w).astype(float), np.nan))
    f.add("ball_speed", speed); f.add("ball_acc", acc); f.add("ball_head_change", head)
    f.add("ball_d_touchline", d_tl); f.add("ball_d_goalline", d_gl)
    f.add("ball_out_near", m_near); f.add("ball_out_far", m_far); f.add("ball_out_goal", m_goal)

    # ---- player velocities
    hvx, hvy, hsp = player_velocity(hx, hy, fps)
    avx, avy, asp = player_velocity(ax, ay, fps)

    # ---- ball-player geometry
    bxc, byc = bx[:, None], by[:, None]
    dh = np.hypot(hx - bxc, hy - byc)
    da = np.hypot(ax - bxc, ay - byc)
    d1h, d2h = two_smallest(dh)
    d1a, d2a = two_smallest(da)
    fill = 40.0
    d1h_f, d1a_f = np.where(np.isfinite(d1h), d1h, fill), np.where(np.isfinite(d1a), d1a, fill)
    dd = np.where(present, np.clip(d1a_f - d1h_f, -30, 30), np.nan)         # >0: home closer
    ns = np.where(present, np.sign(d1a_f - d1h_f), np.nan)
    d1min = np.where(present, np.minimum(d1h_f, d1a_f), np.nan)
    f.add("geo_d1_h", d1h); f.add("geo_d2_h", d2h); f.add("geo_d1_a", d1a); f.add("geo_d2_a", d2a)
    f.add("geo_diff_d1", dd); f.add("geo_nearest_sign", ns); f.add("geo_d1_min", d1min)
    for r in (1, 2, 4):
        ch = np.where(present, (dh < r).sum(1), np.nan)
        ca = np.where(present, (da < r).sum(1), np.nan)
        f.add(f"geo_cnt{r}_h", ch); f.add(f"geo_cnt{r}_a", ca)
    rel = {}
    for tag, dist, vx_, vy_, sp_ in (("h", dh, hvx, hvy, hsp), ("a", da, avx, avy, asp)):
        dfill = np.where(np.isfinite(dist), dist, np.inf)
        j = np.argmin(dfill, 1)
        r_ = np.arange(n)
        has = np.isfinite(dfill[r_, j])
        pvx, pvy, psp = vx_[r_, j], vy_[r_, j], sp_[r_, j]
        relsp = np.where(has & present, np.hypot(pvx - vbx, pvy - vby), np.nan)
        rel[tag] = relsp
        f.add(f"geo_nearest_speed_{tag}", np.where(has & present, psp, np.nan))
        f.add(f"geo_nearest_relspeed_{tag}", relsp)
        with_ = np.where(present & has & np.isfinite(relsp), ((dfill[r_, j] < 2.0) & (relsp < 1.5)).astype(float), np.nan)
        f.add(f"geo_moves_with_{tag}", with_)

    # ---- touch events
    s5 = 5
    pre_x, pre_y = (sx - shift(sx, s5)) * fps / s5, (sy - shift(sy, s5)) * fps / s5
    post_x, post_y = (shift(sx, -s5) - sx) * fps / s5, (shift(sy, -s5) - sy) * fps / s5
    dv = np.hypot(post_x - pre_x, post_y - pre_y)
    locmax = dv >= pd.Series(dv).rolling(13, min_periods=1, center=True).max().to_numpy() - 1e-9
    cand = np.isfinite(dv) & (dv >= TOUCH_DV) & locmax
    near_h = np.where(np.isfinite(d1h), d1h, np.inf) <= TOUCH_DIST
    near_a = np.where(np.isfinite(d1a), d1a, np.inf) <= TOUCH_DIST
    th = cand & near_h & (d1h_f <= d1a_f)
    ta = cand & near_a & (d1a_f < d1h_f)
    tany = th | ta
    f.add("touch_dv", dv)
    sh, sa = since(th, fps, CAP_S), since(ta, fps, CAP_S)
    uh, ua = until(th, fps, CAP_S), until(ta, fps, CAP_S)
    f.add("touch_since_h", sh); f.add("touch_since_a", sa)
    f.add("touch_until_h", uh); f.add("touch_until_a", ua)
    last_team = np.where((sh >= CAP_S) & (sa >= CAP_S), 0.0, np.where(sh < sa, 1.0, -1.0))
    next_team = np.where((uh >= CAP_S) & (ua >= CAP_S), 0.0, np.where(uh < ua, 1.0, -1.0))
    f.add("touch_last_team", last_team); f.add("touch_next_team", next_team)
    thf, taf = th.astype(float), ta.astype(float)
    w3 = int(3 * fps)
    for tag, fl in (("h", thf), ("a", taf)):
        f.add(f"touch_n_past3_{tag}", roll(fl, w3, "sum", True))
        f.add(f"touch_n_fut3_{tag}", roll(fl, w3, "sum", False))
    f.add("touch_gap_to_next_s", np.minimum(until(tany, fps, CAP_S), CAP_S))

    # ---- carry-over across missing blocks
    miss = ~present
    pack = {"x": bx, "y": by, "near": m_near, "far": m_far, "goal": m_goal, "ns": ns, "d1": d1min, "dd": dd}
    for name, arr in pack.items():
        f.add(f"carry_last_{name}", np.where(miss, ffill(arr), np.nan))
        f.add(f"carry_next_{name}", np.where(miss, bfill(arr), np.nan))
    f.add("carry_last_speed", np.where(miss, ffill(speed), np.nan))
    f.add("carry_last_acc", np.where(miss, ffill(acc), np.nan))
    f.add("carry_last_dtl", np.where(miss, ffill(d_tl), np.nan))
    f.add("carry_next_dtl", np.where(miss, bfill(d_tl), np.nan))
    f.add("carry_last_touch_team_at_loss", np.where(miss, ffill(np.where(present, last_team, np.nan)), np.nan))
    f.add("carry_ago_s", np.where(miss, since_p, np.nan))
    f.add("carry_ahead_s", np.where(miss, until_p, np.nan))

    # ---- team shape and motion
    allsp = np.concatenate([hsp, asp], 1)
    mean_sp_all, _ = masked_mean(allsp)
    std_sp_all = masked_std(allsp)
    team_vx = {}
    nearest_opp = np.full((n, 2), np.nan)
    for ti, (tag, X, Y, VX, SP, o) in enumerate((("h", hx, hy, hvx, hsp, 1.0), ("a", ax, ay, avx, asp, -1.0))):
        ok = np.isfinite(X) & np.isfinite(Y)
        cnt = ok.sum(1)
        s = o * X                                    # attack-oriented x: own goal at -L/2
        f.add(f"shape_cnt_{tag}", cnt)
        cx_, _ = masked_mean(X); cy_, _ = masked_mean(Y)
        f.add(f"shape_cx_{tag}", cx_); f.add(f"shape_cy_{tag}", cy_)
        f.add(f"shape_sdx_{tag}", masked_std(X)); f.add(f"shape_sdy_{tag}", masked_std(Y))
        d1_, d2_ = nth_extreme(s, False)
        h1_, h2_ = nth_extreme(s, True)
        f.add(f"shape_deep1_{tag}", d1_); f.add(f"shape_deep2_{tag}", d2_)
        f.add(f"shape_high1_{tag}", h1_); f.add(f"shape_high2_{tag}", h2_)
        msp, nsp = masked_mean(SP)
        f.add(f"shape_speed_{tag}", msp)
        vxo, _ = masked_mean(o * VX)
        team_vx[tag] = vxo
        f.add(f"shape_vx_{tag}", vxo)
        okv = np.isfinite(SP)
        nv = np.maximum(okv.sum(1), 1)
        for thr, name in ((0.5, "lt05"), (1.5, "lt15")):
            f.add(f"shape_share_{name}_{tag}", np.where(okv.sum(1) > 0, (SP < thr).sum(1) / nv, np.nan))
        for thr, name in ((4.0, "gt4"), (6.0, "gt6")):
            f.add(f"shape_share_{name}_{tag}", np.where(okv.sum(1) > 0, (SP > thr).sum(1) / nv, np.nan))
        c = np.maximum(cnt, 1)
        own = (s < -L / 6) & ok; fin = (s > L / 6) & ok
        f.add(f"shape_n_own3_{tag}", own.sum(1)); f.add(f"shape_n_mid3_{tag}", (ok & ~own & ~fin).sum(1))
        f.add(f"shape_n_fin3_{tag}", fin.sum(1))
        inbox_w = np.abs(Y) < 20.16
        f.add(f"shape_n_ownbox_{tag}", (ok & (s < -L / 2 + 16.5) & inbox_w).sum(1))
        f.add(f"shape_n_oppbox_{tag}", (ok & (s > L / 2 - 16.5) & inbox_w).sum(1))
        f.add(f"shape_n_outside_{tag}", (ok & ((np.abs(X) > L / 2) | (np.abs(Y) > W / 2))).sum(1))
        f.add(f"shape_own_half_{tag}", np.where(cnt > 0, (ok & (s < 0)).sum(1) / c, np.nan))
        f.add(f"shape_n_near_tl_{tag}", (ok & (Y > W / 2 - 3)).sum(1))      # within 3 m of the near touchline
        f.add(f"shape_n_far_tl_{tag}", (ok & (Y < -W / 2 + 3)).sum(1))
        with np.errstate(all="ignore"):
            f.add(f"shape_min_dtl_{tag}", np.where(cnt > 0, np.where(ok, W / 2 - np.abs(Y), np.inf).min(1), np.nan))
            for cxs, cys, cn in ((-1, -1, "mm"), (1, -1, "pm"), (-1, 1, "mp"), (1, 1, "pp")):
                dcor = np.hypot(X - cxs * L / 2, Y - cys * W / 2)
                f.add(f"shape_min_corner_{cn}_{tag}", np.where(cnt > 0, np.where(ok, dcor, np.inf).min(1), np.nan))
        # nearest opponent distance (chunked to bound memory)
        OX, OY = (ax, ay) if ti == 0 else (hx, hy)
        for a in range(0, n, 20000):
            b = min(n, a + 20000)
            dd_ = np.hypot(X[a:b, :, None] - OX[a:b, None, :], Y[a:b, :, None] - OY[a:b, None, :])
            dd_ = np.where(np.isfinite(dd_), dd_, np.inf).min(2)
            dd_[~np.isfinite(dd_)] = np.nan
            nearest_opp[a:b, ti], _ = masked_mean(dd_)
    f.add("shape_nearest_opp_mean", np.nanmean(nearest_opp, 1) if n else nearest_opp[:, 0])
    f.add("shape_speed_all", mean_sp_all); f.add("shape_speed_all_std", std_sp_all)
    allx = np.concatenate([hx, ax], 1); ally = np.concatenate([hy, ay], 1)
    f.add("shape_cx_all", masked_mean(allx)[0]); f.add("shape_cy_all", masked_mean(ally)[0])
    f.add("shape_vx_all", masked_mean(np.concatenate([hvx, avx], 1))[0])
    f.add("shape_cnt_all", np.isfinite(allx).sum(1))

    # ---- temporal context
    sig = {"ns": ns, "dd": dd, "d1min": d1min, "bspeed": speed, "bacc": acc, "present": present.astype(float),
           "pspeed": mean_sp_all, "vx_h": team_vx["h"], "vx_a": team_vx["a"], "touch_h": thf, "touch_a": taf,
           "touch_any": tany.astype(float)}
    for name, x in sig.items():
        wins = WINDOWS + ((250,) if name == "pspeed" else ())
        for w in wins:
            c = roll(x, w, "mean", True)
            a = roll(x, w, "mean", False)
            f.add(f"ctx_{name}_past{w}", c)
            f.add(f"ctx_{name}_fut{w}", a)
            f.add(f"ctx_{name}_diff{w}", a - c)
    for name, x, hows in (("d1min", d1min, ("min",)), ("bspeed", speed, ("max",)), ("bacc", acc, ("max",)),
                          ("dd", dd, ("min", "max")), ("pspeed", mean_sp_all, ("max",))):
        for w in WINDOWS:
            for how in hows:
                f.add(f"ctx_{name}_{how}_past{w}", roll(x, w, how, True))
                f.add(f"ctx_{name}_{how}_fut{w}", roll(x, w, how, False))
    return f.d


def extract(m: Match) -> tuple[np.ndarray, list[str]]:
    """Features for every frame of a canonical match; rows with period==0 are NaN."""
    sls = m.period_slices()
    cols: dict[str, np.ndarray] = {}
    for p, sl in sls.items():
        d = _half(m, sl, m.pitch)
        for k, v in d.items():
            if k not in cols:
                cols[k] = np.full(m.T, np.nan, np.float32)
            cols[k][sl] = v
    names = list(cols)
    X = np.empty((m.T, len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = cols[k]
    X[m.period == 0] = np.nan
    return X, names
