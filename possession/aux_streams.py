"""Target-side auxiliary evidence streams (per tactical video frame, int8 categorical codes).

All streams are label-free: they come from the broadcast recording (via the sync mapping) and from
Smart Tagging. Cached to ``cache/aux_streams.npz``.

Code tables
-----------
bstate   0 broadcast frame unmapped | 1 mapped, homography invalid | 2 mapped, homography valid but NOT LIVE
         (replay) | 3 live, no ball | 4 live, ball present
sp_phase 0 none | time relative to the START of the nearest SET PIECES window the frame is in or precedes:
         1 [-10,-5) s | 2 [-5,0) | 3 [0,2) | 4 [2,4) | 5 [4,6) | 6 [6,9) | 7 [9,14) | 8 [14 s, window end]
sp_type  0 none | 1 Corner Kick | 2 Free Kick | 3 Goal Kick | 4 Kick Off (After Goal) | 5 Kick Off (Start)
         | 6 Throw-in | 7 other
sp_team  0 none / N/A | 1 home (Barcelona) | 2 away (Athletic)
goal_win 0 none | 1 first 20 % of a GOALS window | 2 remaining 80 %
att_phase 0 none | 1 home attacking-phase code active | 2 away | 3 both
         (BUILD UP, PROGRESSION, ATTACKING TRANSITION, LONG BALL)
cc_win   0 none | 1/2/3 first / middle / last third of a CREATING CHANCES window
Float arrays ``bball_x``, ``bball_y``: broadcast ball, metres pitch-centred, NaN unless bstate == 4.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import sync_broadcast as sb
from .io_tags import load_tags
from .paths import cache_dir

FPS = 25
T_TAC = sb.T_TAC
VERSION = 1
LIVE_RADIUS = 2.0       # m, same-team tactical player counts as a match
MIN_PLAYERS = 4         # players needed on both recordings to judge
SMOOTH_HALF = 12        # frames, median over +-12

SP_TYPES = {"Corner Kick": 1, "Free Kick": 2, "Goal Kick": 3, "Kick Off (After Goal)": 4,
            "Kick Off (Start)": 5, "Throw-in": 6}
SP_EDGES_S = [-10, -5, 0, 2, 4, 6, 9, 14]      # phase k+1 starts at SP_EDGES_S[k]
ATT_CODES = ("BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "LONG BALL")


def _team_code(d: dict) -> int:
    t = (d.get("Team") or "").lower()
    return 1 if t.startswith("barcelona") else 2 if t.startswith("atletic") or t.startswith("athletic") else 0


def live_statistic(mapping: np.ndarray, S: "sb.Sources", flip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per tactical frame: share of broadcast players with a same-team tactical player within 2 m (NaN if
    unmapped / invalid homography / fewer than MIN_PLAYERS on either recording), and the two player counts."""
    bx, by, bpx, bpy = sb.apply_flip(S, *flip)
    T = mapping.size
    share = np.full(T, np.nan)
    nb = np.zeros(T, np.int16)
    nt = np.zeros(T, np.int16)
    ok = np.flatnonzero((mapping >= 0))
    ok = ok[S.bvalid[mapping[ok]]]
    bt = S.bteam
    tt = S.tteam
    for a in range(0, ok.size, 3000):
        t = ok[a:a + 3000]
        b = mapping[t]
        hit = np.zeros((t.size, bpx.shape[1]), bool)
        for team in (0, 1):
            bs = np.flatnonzero(bt == team)
            ts = np.flatnonzero(tt == team)
            dx = bpx[b][:, bs][:, :, None] - S.tpx[t][:, ts][:, None, :]
            dy = bpy[b][:, bs][:, :, None] - S.tpy[t][:, ts][:, None, :]
            d = np.hypot(dx, dy)
            hit[:, bs] = np.where(np.isfinite(d), d, np.inf).min(2) < LIVE_RADIUS
        pb = np.isfinite(bpx[b]) & np.isfinite(bpy[b])
        n_b = pb.sum(1)
        n_t = (np.isfinite(S.tpx[t]) & np.isfinite(S.tpy[t])).sum(1)
        nb[t], nt[t] = n_b, n_t
        s = (hit & pb).sum(1) / np.maximum(n_b, 1)
        good = (n_b >= MIN_PLAYERS) & (n_t >= MIN_PLAYERS)
        share[t[good]] = s[good]
    return share, nb, nt


def _rolling_nanmedian(x: np.ndarray, half: int) -> np.ndarray:
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(win, axis=1)


def otsu_threshold(v: np.ndarray, nbins: int = 100) -> float:
    h, e = np.histogram(v, bins=nbins, range=(0, 1))
    p = h / h.sum()
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * c)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb2 = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
    sb2[~np.isfinite(sb2)] = -1
    return float(e[int(np.argmax(sb2)) + 1])


def build_bstate(S=None, sync=None) -> dict:
    S = S or sb.load_sources()
    sync = sync or sb.load_sync()
    mapping, flip = sync["mapping"], sync["flip"]
    T = mapping.size
    mapped = mapping >= 0
    mb = np.where(mapped, mapping, 0)
    valid = mapped & S.bvalid[mb]
    share, nb, nt = live_statistic(mapping, S, flip)
    sm = _rolling_nanmedian(np.where(valid, share, np.nan), SMOOTH_HALF)
    sm[~valid] = np.nan
    thr = otsu_threshold(sm[np.isfinite(sm)])
    hist10 = np.histogram(sm[np.isfinite(sm)], bins=10, range=(0, 1))[0]
    live = sm >= thr
    decided = np.isfinite(sm)
    # homography-valid runs on the tactical timeline (also broken where the broadcast frame index jumps)
    brk = np.ones(T, bool)
    brk[1:] = ~(valid[1:] & valid[:-1] & (np.diff(mb) == 1))
    run_id = np.cumsum(brk) - 1
    run_id[~valid] = -1
    idx = np.arange(T)
    prev = np.maximum.accumulate(np.where(decided, idx, -1))
    nxt = np.minimum.accumulate(np.where(decided, idx, 10 ** 9)[::-1])[::-1]
    pv = prev >= 0
    nx = nxt < 10 ** 9
    same_p = pv & (run_id[np.where(pv, prev, 0)] == run_id)
    same_n = nx & (run_id[np.where(nx, nxt, 0)] == run_id)
    dp = np.where(same_p, idx - prev, 10 ** 9)
    dn = np.where(same_n, nxt - idx, 10 ** 9)
    src = np.where(dp <= dn, prev, nxt)
    have = valid & (np.minimum(dp, dn) < 10 ** 9)
    live_f = np.zeros(T, bool)
    live_f[have] = live[src[have]]
    n_undecided_run = int((valid & ~have).sum())     # valid frames in runs with no decided frame: treated as NOT live
    bP = S.bP[mb] & mapped
    bstate = np.zeros(T, np.int8)
    bstate[mapped & ~valid] = 1
    bstate[valid & ~live_f] = 2
    bstate[valid & live_f] = 3
    bstate[valid & live_f & bP] = 4
    bx, by, _, _ = sb.apply_flip(S, *flip)
    L, W = sb.PX, sb.PY
    bball_x = np.full(T, np.nan, np.float32)
    bball_y = np.full(T, np.nan, np.float32)
    m4 = bstate == 4
    bball_x[m4] = (bx[mb[m4]] / L - 0.5) * L
    bball_y[m4] = (by[mb[m4]] / W - 0.5) * W
    return dict(bstate=bstate, bball_x=bball_x, bball_y=bball_y, live_stat=sm.astype(np.float32),
                live_thr=np.float64(thr), live_hist10=hist10, n_undecided_run=np.int64(n_undecided_run),
                n_decided_by_carry=np.int64(int((have & ~decided).sum())))


def build_tag_streams(T: int = T_TAC) -> dict:
    tg = load_tags()
    tg["type"] = tg.tags.map(lambda d: d.get("Type"))
    tg["team"] = tg.tags.map(_team_code)
    code = tg.code.str.upper()
    out = {k: np.zeros(T, np.int8) for k in ("sp_phase", "sp_type", "sp_team", "goal_win", "att_phase", "cc_win")}
    best = np.full(T, 10 ** 9)
    edges = np.array(SP_EDGES_S) * FPS
    for _, r in tg[code == "SET PIECES"].iterrows():
        s, e = int(r.start_frame), int(r.end_frame)
        lo, hi = max(0, s - 10 * FPS), min(T - 1, e)
        t = np.arange(lo, hi + 1)
        d = np.abs(t - s)
        upd = d < best[t]
        tu = t[upd]
        best[tu] = d[upd]
        out["sp_phase"][tu] = np.searchsorted(edges, tu - s, side="right").astype(np.int8)
        out["sp_type"][tu] = SP_TYPES.get(r.type, 7)
        out["sp_team"][tu] = r.team
    for _, r in tg[code == "GOALS"].iterrows():
        s, e = max(0, int(r.start_frame)), min(T, int(r.end_frame) + 1)
        cut = s + int(round(0.2 * (e - s)))
        out["goal_win"][s:cut] = 1
        out["goal_win"][cut:e] = 2
    for _, r in tg[code == "CREATING CHANCES"].iterrows():
        s, e = max(0, int(r.start_frame)), min(T, int(r.end_frame) + 1)
        n = e - s
        t = np.arange(s, e)
        out["cc_win"][s:e] = (1 + np.minimum(2, ((t - s) * 3) // max(n, 1))).astype(np.int8)
    for _, r in tg[code.isin(ATT_CODES)].iterrows():
        if r.team in (1, 2):
            s, e = max(0, int(r.start_frame)), min(T, int(r.end_frame) + 1)
            out["att_phase"][s:e] |= np.int8(r.team)
    return out


def build_aux(use_cache: bool = True) -> dict:
    path = cache_dir() / "aux_streams.npz"
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) == VERSION:
                return {k: z[k] for k in z.files if k != "version"}
    d = build_bstate()
    d.update(build_tag_streams())
    np.savez_compressed(path, version=VERSION, **d)
    return d
