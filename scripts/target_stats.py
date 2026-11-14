"""Aggregate, label-free statistics of the target tracking, for fitting a noise simulator.

Writes possession/assets/target_stats.json (no per-frame values, no timing). Run:
``uv run python scripts/target_stats.py``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import games  # noqa: E402
from possession.camera import estimate_camera_centre, plausibility_check  # noqa: E402
from possession.io_homography import load_homography, pixel_to_pitch  # noqa: E402
from possession.target import load_player_speed, load_target_match  # noqa: E402

FPS = 25.0
OUT = Path(__file__).resolve().parent.parent / "possession" / "assets" / "target_stats.json"


def pcts(a, qs, nd=4):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {str(q): None for q in qs}
    return {str(q): round(float(v), nd) for q, v in zip(qs, np.percentile(a, qs))}


def run_lengths(mask: np.ndarray, blocks) -> list[int]:
    out = []
    for _, a, b in blocks:
        m = mask[a:b + 1].astype(np.int8)
        d = np.diff(np.concatenate([[0], m, [0]]))
        out.append(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return sorted(int(v) for v in np.concatenate(out))


def d2_windows(x: np.ndarray, y: np.ndarray, speed: np.ndarray, blocks, lo: float, hi: float):
    """Second differences at t (x[t+1]-2x[t]+x[t-1]) where t-2..t+2 are all present and the speed at
    t lies in [lo, hi). x, y, speed: (T,) or (T,S); windows never straddle a block edge."""
    dxs, dys = [], []
    for _, a, b in blocks:
        xs, ys, sp = x[a:b + 1], y[a:b + 1], speed[a:b + 1]
        pres = np.isfinite(xs) & np.isfinite(ys)
        n = pres.shape[0]
        ok = np.ones_like(pres)
        ok[:2] = False
        ok[-2:] = False
        for k in (-2, -1, 1, 2):
            ok[2:-2] &= pres[2 + k:n - 2 + k] if n > 4 else True
        ok &= pres
        ok &= (sp >= lo) & (sp < hi)
        d2x = np.full(xs.shape, np.nan)
        d2y = np.full(ys.shape, np.nan)
        d2x[1:-1] = xs[2:] - 2 * xs[1:-1] + xs[:-2]
        d2y[1:-1] = ys[2:] - 2 * ys[1:-1] + ys[:-2]
        dxs.append(d2x[ok])
        dys.append(d2y[ok])
    return np.concatenate(dxs), np.concatenate(dys)


def jitter(dx, dy):
    def rs(v):  # robust sigma
        return float(1.4826 * np.median(np.abs(v - np.median(v))))
    return dict(n=int(dx.size), std_x_m=float(dx.std()), std_y_m=float(dy.std()),
                robust_std_x_m=rs(dx), robust_std_y_m=rs(dy))


def autocorr_drop(series_by_block: list[np.ndarray], thresh=0.5) -> float:
    acs = []
    for s in series_by_block:
        s = s - s.mean()
        n = len(s)
        f = np.fft.rfft(s, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        acs.append(ac / ac[0])
    n = min(len(a) for a in acs)
    ac = np.mean([a[:n] for a in acs], axis=0)
    below = np.flatnonzero(ac < thresh)
    return float(below[0] / FPS) if below.size else float("nan")


def main() -> None:
    t0 = time.time()
    m = load_target_match()
    hs, as_ = load_player_speed()
    T = m.T
    L, W = m.pitch
    blocks = games.half_video_blocks("BAR-ATH")
    G = np.zeros(T, bool)
    period = np.zeros(T, np.int8)
    for p, a, b in blocks:
        G[a:b + 1] = True
        period[a:b + 1] = p
    px = np.concatenate([m.hx, m.ax], axis=1).astype(np.float64)
    py = np.concatenate([m.hy, m.ay], axis=1).astype(np.float64)
    sp = np.concatenate([hs, as_], axis=1).astype(np.float64)
    nh = m.hx.shape[1]
    isH = np.zeros(px.shape[1], bool)
    isH[:nh] = True
    pres = np.isfinite(px) & np.isfinite(py)
    bx, by = m.bx.astype(np.float64), m.by.astype(np.float64)
    P = np.isfinite(bx) & np.isfinite(by)
    S: dict = {}

    cam = estimate_camera_centre()
    S["camera_centre_m"] = cam["centre"]
    S["camera_centre_iqr"] = cam["iqr"]
    S["camera_centre_detail"] = {k: cam[k] for k in
                                 ("p05", "p95", "rejected_share", "focal_px_p10_50_90", "n_frames", "n_ok")}
    S["camera_plausibility"] = plausibility_check(cam)
    S["home_dir"] = {str(k): v for k, v in m.home_dir.items()}

    # ---- ball presence
    S["ball_present_runs"] = run_lengths(P, blocks)
    S["ball_missing_runs"] = run_lengths(~P, blocks)
    S["ball_missing_share"] = {f"half{p}": float(1 - P[a:b + 1].mean()) for p, a, b in blocks}
    S["ball_missing_share"]["overall"] = float(1 - P[G].mean())

    # ---- players tracked
    nH, nA = pres[:, isH].sum(1), pres[:, ~isH].sum(1)
    S["players_tracked_hist"] = {
        "home": np.bincount(np.minimum(nH[G], 16), minlength=17).tolist(),
        "away": np.bincount(np.minimum(nA[G], 16), minlength=17).tolist(),
        "note": "index = count 0..16, 16 means 16 or more",
    }
    tot = (nH + nA).astype(np.float64)
    S["players_tracked_autocorr_s"] = autocorr_drop([tot[a:b + 1] for _, a, b in blocks])

    # ---- field of view from the homography
    Hh, _, valid = load_homography("tactical")
    gi = np.flatnonzero(G)
    gi = gi[valid[gi]]
    cxn, _ = pixel_to_pitch(Hh[gi], 960.0, 540.0)
    lxn, _ = pixel_to_pitch(Hh[gi], 0.0, 540.0)
    rxn, _ = pixel_to_pitch(Hh[gi], 1920.0, 540.0)
    vcx = (cxn - 0.5) * L
    xl, xr = (lxn - 0.5) * L, (rxn - 0.5) * L
    ext = np.abs(xr - xl)
    lo, hi = np.minimum(xl, xr), np.maximum(xl, xr)
    qs = [1, 10, 50, 90, 99]
    fov = dict(view_centre_x_m=pcts(vcx, qs, 2), visible_extent_m=pcts(ext, qs, 2))
    # players outside the visible x-extent
    PXg = px[gi]
    inside = (PXg >= lo[:, None]) & (PXg <= hi[:, None])
    pg = pres[gi]
    fov["share_players_outside_visible_x"] = float((pg & ~inside).sum() / pg.sum())
    # deepest-player tracked, per team defending each end (from the inferred home_dir)
    per = period[gi]
    hdir = np.where(per == 1, m.home_dir[1], m.home_dir[2])           # home attacks +x if +1
    with np.errstate(all="ignore"):
        xH = np.where(pres[:, isH], px[:, isH], np.nan)[gi]
        xA = np.where(pres[:, ~isH], px[:, ~isH], np.nan)[gi]
        minH, maxH = np.nanmin(np.where(np.isfinite(xH), xH, np.inf), 1), np.nanmax(np.where(np.isfinite(xH), xH, -np.inf), 1)
        minA, maxA = np.nanmin(np.where(np.isfinite(xA), xA, np.inf), 1), np.nanmax(np.where(np.isfinite(xA), xA, -np.inf), 1)
    # team defending x<0 is the home team when it attacks +x
    neg_ok = np.where(hdir == 1, minH, minA) < -40
    pos_ok = np.where(hdir == 1, maxA, maxH) > 40
    fov["deepest_tracked_share"] = dict(team_defending_neg_x=float(neg_ok.mean()),
                                        team_defending_pos_x=float(pos_ok.mean()))
    bins = {"view_centre_x<-15": vcx < -15, "view_centre_x_-15..15": (vcx >= -15) & (vcx <= 15),
            "view_centre_x>15": vcx > 15}
    fov["deepest_tracked_share_by_view_centre"] = {
        k: dict(n=int(v.sum()), team_defending_neg_x=float(neg_ok[v].mean()),
                team_defending_pos_x=float(pos_ok[v].mean())) for k, v in bins.items()}
    fov["note"] = ("view centre = pitch-x (m, pitch-centred) at image centre, from the homography; "
                   "this is the conditioning variable (not the constant camera position)")
    S["fov"] = fov

    # ---- player jitter
    jit = {}
    for name, (slo, shi) in {"speed_lt_1": (0, 1), "speed_3_5": (3, 5)}.items():
        dx, dy = d2_windows(px, py, sp, blocks, slo, shi)
        jit[name] = jitter(dx, dy)
    S["player_jitter"] = jit

    # ---- ball speed and jitter
    d = np.full(T, np.nan)
    d[1:] = np.hypot(bx[1:] - bx[:-1], by[1:] - by[:-1]) * FPS
    ok = np.zeros(T, bool)
    ok[1:] = P[1:] & P[:-1] & G[1:] & G[:-1] & (period[1:] == period[:-1])
    bs = d[ok]
    S["ball_speed_pcts"] = pcts(bs, [5, 25, 50, 75, 90, 95, 99], 3)
    S["ball_step_gt36"] = float((bs > 36).mean())
    cs = np.full(T, np.nan)   # central-difference speed (m/s)
    cs[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * FPS / 2
    bj = {}
    for name, (slo, shi) in {"speed_lt_2": (0, 2), "speed_5_10": (5, 10)}.items():
        dx, dy = d2_windows(bx, by, cs, blocks, slo, shi)
        bj[name] = jitter(dx, dy)
    S["ball_jitter"] = bj

    gp = G & P
    S["ball_outside"] = dict(
        near_touchline_neg_y=float((by[gp] < -W / 2).mean()), far_touchline_pos_y=float((by[gp] > W / 2).mean()),
        goal_line_neg_x=float((bx[gp] < -L / 2).mean()), goal_line_pos_x=float((bx[gp] > L / 2).mean()),
        any=float(((by[gp] < -W / 2) | (by[gp] > W / 2) | (bx[gp] < -L / 2) | (bx[gp] > L / 2)).mean()),
        note="names follow +y = camera side (see camera_plausibility.camera_side)")

    # ---- ball to nearest player
    dist = np.where(pres & P[:, None], np.hypot(px - bx[:, None], py - by[:, None]), np.inf)
    dh = np.where(isH[None, :], dist, np.inf).min(1)
    da = np.where(~isH[None, :], dist, np.inf).min(1)
    dn = np.minimum(dh, da)
    fin = lambda a: np.where(np.isfinite(a), a, np.nan)  # noqa: E731
    q6 = [5, 25, 50, 75, 90, 95]
    d0 = dn[gp][np.isfinite(dn[gp])]
    nteam = np.where(np.isfinite(dn), np.where(dh <= da, 0, 1), -1)
    pair = np.zeros(T, bool)
    pair[1:] = ok[1:] & (nteam[1:] >= 0) & (nteam[:-1] >= 0)
    flips = pair & (nteam != np.roll(nteam, 1))
    minutes = (gp & (nteam >= 0)).sum() / FPS / 60
    S["ball_nearest"] = dict(
        nearest_player_m=pcts(fin(dn)[gp], q6, 3), nearest_home_m=pcts(fin(dh)[gp], q6, 3),
        nearest_away_m=pcts(fin(da)[gp], q6, 3),
        share_lt1=float((d0 < 1).mean()), share_lt2=float((d0 < 2).mean()), share_lt3=float((d0 < 3).mean()),
        share_gt5=float((d0 > 5).mean()), share_gt10=float((d0 > 10).mean()),
        nearest_team_flips_per_min=float(flips.sum() / minutes))

    # ---- track fragments (per slot, clipped at half boundaries)
    lens = []
    for s in range(px.shape[1]):
        lens.extend(run_lengths(pres[:, s], blocks))
    S["track_fragment_len_s"] = pcts(np.array(lens) / FPS, [10, 50, 90], 3)
    S["track_fragment_len_s"]["n_runs"] = len(lens)

    # ---- mean per-frame speed
    with np.errstate(all="ignore"):
        mps = np.nanmean(np.where(pres, sp, np.nan), axis=1)
    S["mean_player_speed_pcts"] = pcts(mps[G], [5, 25, 50, 75, 95], 3)

    S["runtime_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(S, indent=1), encoding="utf-8")
    # aggregates only on stdout
    summ = {k: v for k, v in S.items() if k not in ("ball_present_runs", "ball_missing_runs")}
    summ["ball_present_runs"] = f"n={len(S['ball_present_runs'])}"
    summ["ball_missing_runs"] = f"n={len(S['ball_missing_runs'])}"
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - never echo data through a traceback
        import traceback
        fr = [f"{Path(x.filename).name}:{x.lineno}" for x in traceback.extract_tb(exc.__traceback__)]
        print(f"FAILED: {type(exc).__name__} at {fr[-3:]}")
        sys.exit(1)
