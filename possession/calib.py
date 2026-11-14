"""Aggregate statistics of a (canonical) match's tracking, to compare degraded DFL with the target.

Only aggregates are returned (shares, percentiles, histograms), never per-frame values.
"""
from __future__ import annotations

import numpy as np

RUN_BINS = [(1, 2), (3, 5), (6, 12), (13, 25), (26, 50), (51, 125), (126, 250), (251, 500), (501, 1000), (1001, 10 ** 9)]
RUN_LABELS = ["1-2", "3-5", "6-12", "13-25", "26-50", "51-125", "126-250", "251-500", "501-1000", ">1000"]
CNT_BINS = [(0, 5), (6, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 99)]
CNT_LABELS = ["0-5", "6-8", "9", "10", "11", "12", "13+"]


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    s, t = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    return t - s


def _hist(lengths, bins):
    lengths = np.asarray(lengths)
    return [int(((lengths >= a) & (lengths <= b)).sum()) for a, b in bins]


def _jitter(m) -> dict:
    """Std of the second difference (m / frame^2) of players (by speed band m/s) and ball."""
    out = {}
    fps = m.fps
    pl_x, pl_y, bl_x = [], [], []
    res = {k: [] for k in ("p_lt1", "p_3_5", "b_lt2", "b_5_10")}
    for p, sl in m.period_slices().items():
        for X, Y in ((m.hx[sl], m.hy[sl]), (m.ax[sl], m.ay[sl])):
            X = X.astype(np.float64); Y = Y.astype(np.float64)
            d2x = X[2:] - 2 * X[1:-1] + X[:-2]
            d2y = Y[2:] - 2 * Y[1:-1] + Y[:-2]
            sp = np.hypot(X[2:] - X[:-2], Y[2:] - Y[:-2]) * fps / 2
            for k, mk in (("p_lt1", sp < 1), ("p_3_5", (sp >= 3) & (sp < 5))):
                v = np.concatenate([d2x[mk], d2y[mk]])
                res[k].append(v[np.isfinite(v)])
        X, Y = m.bx[sl].astype(np.float64), m.by[sl].astype(np.float64)
        d2x = X[2:] - 2 * X[1:-1] + X[:-2]
        d2y = Y[2:] - 2 * Y[1:-1] + Y[:-2]
        sp = np.hypot(X[2:] - X[:-2], Y[2:] - Y[:-2]) * fps / 2
        for k, mk in (("b_lt2", sp < 2), ("b_5_10", (sp >= 5) & (sp < 10))):
            v = np.concatenate([d2x[mk], d2y[mk]])
            res[k].append(v[np.isfinite(v)])
    return {k: float(np.std(np.concatenate(v))) for k, v in res.items()}


def summarise(m) -> dict:
    """Statistics over frames with period > 0, per half where run lengths are concerned."""
    L, W = m.pitch
    miss_runs, pres_runs = [], []
    sel_all = m.period > 0
    for p, sl in m.period_slices().items():
        pres = np.isfinite(m.bx[sl]) & np.isfinite(m.by[sl])
        pres_runs.extend(_runs(pres))
        miss_runs.extend(_runs(~pres))
    bx, by = m.bx[sel_all].astype(np.float64), m.by[sel_all].astype(np.float64)
    pres = np.isfinite(bx) & np.isfinite(by)
    out = dict(n_frames=int(sel_all.sum()), missing_share=float(1 - pres.mean()),
               missing_runs=_hist(miss_runs, RUN_BINS), present_runs=_hist(pres_runs, RUN_BINS))
    x, y = bx[pres], by[pres]
    # camera is on the +y side: near touchline = +y, far = -y
    out["outside"] = dict(far_touchline=float((y < -W / 2).mean()), near_touchline=float((y > W / 2).mean()),
                          goal_line=float((np.abs(x) > L / 2).mean()),
                          near_over_10m=float((y > W / 2 + 10).mean()), far_over_5m=float((y < -W / 2 - 5).mean()))
    out["jitter"] = _jitter(m)
    # nearest player
    hx, hy = m.hx[sel_all].astype(np.float64)[pres], m.hy[sel_all].astype(np.float64)[pres]
    ax, ay = m.ax[sel_all].astype(np.float64)[pres], m.ay[sel_all].astype(np.float64)[pres]
    d = np.concatenate([np.hypot(hx - x[:, None], hy - y[:, None]), np.hypot(ax - x[:, None], ay - y[:, None])], 1)
    d = np.where(np.isfinite(d), d, np.inf).min(1)
    d = d[np.isfinite(d)]
    out["nearest_dist_pct"] = dict(zip([5, 25, 50, 75, 90, 95], np.percentile(d, [5, 25, 50, 75, 90, 95]).round(2).tolist()))
    # tracked players
    for name, X in (("home", m.hx), ("away", m.ax)):
        c = np.isfinite(X[sel_all]).sum(1)
        out[f"tracked_{name}"] = [float(((c >= a) & (c <= b)).mean()) for a, b in CNT_BINS]
        out[f"tracked_{name}_mean"] = float(c.mean())
    # ball speed on consecutive present frames (within a half)
    sp = []
    for p, sl in m.period_slices().items():
        bxx, byy = m.bx[sl].astype(np.float64), m.by[sl].astype(np.float64)
        v = np.hypot(np.diff(bxx), np.diff(byy)) * m.fps
        sp.append(v[np.isfinite(v)])
    sp = np.concatenate(sp)
    out["ball_speed_pct"] = dict(zip([5, 25, 50, 75, 90, 95, 99], np.percentile(sp, [5, 25, 50, 75, 90, 95, 99]).round(2).tolist()))
    return out


def format_comparison(rows: dict) -> str:
    """rows: {name: summarise-dict}. Markdown table block."""
    names = list(rows)
    lines = []

    def tab(title, labels, getter, fmt):
        lines.append(f"\n**{title}**\n")
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for i, lab in enumerate(labels):
            lines.append(f"| {lab} | " + " | ".join(fmt(getter(rows[n], i)) for n in names) + " |")

    tab("Scalars", ["ball missing share", "beyond far (-y) touchline", "beyond near (+y) touchline", "beyond goal lines",
                    "far > 5 m beyond", "near > 10 m beyond", "mean tracked home", "mean tracked away",
                    "2nd-diff std player <1 m/s", "2nd-diff std player 3-5 m/s", "2nd-diff std ball <2 m/s", "2nd-diff std ball 5-10 m/s"],
        lambda r, i: [r["missing_share"], r["outside"]["far_touchline"], r["outside"]["near_touchline"], r["outside"]["goal_line"],
                      r["outside"]["far_over_5m"], r["outside"]["near_over_10m"], r["tracked_home_mean"], r["tracked_away_mean"],
                      r["jitter"]["p_lt1"], r["jitter"]["p_3_5"], r["jitter"]["b_lt2"], r["jitter"]["b_5_10"]][i],
        lambda v: f"{v:.4f}")
    tab("Ball-missing runs per half-match (count, bins in frames)", RUN_LABELS,
        lambda r, i: r["missing_runs"][i], lambda v: f"{v}")
    tab("Ball-present runs (count)", RUN_LABELS, lambda r, i: r["present_runs"][i], lambda v: f"{v}")
    tab("Nearest player distance (m) percentile", [5, 25, 50, 75, 90, 95],
        lambda r, i: list(r["nearest_dist_pct"].values())[i], lambda v: f"{v:.2f}")
    tab("Ball speed (m/s) percentile", [5, 25, 50, 75, 90, 95, 99],
        lambda r, i: list(r["ball_speed_pct"].values())[i], lambda v: f"{v:.2f}")
    tab("Tracked home players (share of frames)", CNT_LABELS, lambda r, i: r["tracked_home"][i], lambda v: f"{v:.3f}")
    tab("Tracked away players (share of frames)", CNT_LABELS, lambda r, i: r["tracked_away"][i], lambda v: f"{v:.3f}")
    return "\n".join(lines)
