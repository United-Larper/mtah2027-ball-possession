"""Second aggregate-only exploratory report: broadcast sync (B), whistle audio (A), restarts / ball-missing blocks (C).

Writes a markdown report with counts / percentiles / correlations / event-triggered averages only; it never
prints or stores per-frame values or individual tag instances. Run: ``uv run python scripts/eda2.py``.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from possession import audio as au  # noqa: E402
from possession import sync_broadcast as sb  # noqa: E402
from possession.io_atd import load_atd  # noqa: E402
from possession.io_homography import load_homography  # noqa: E402
from possession.io_tags import load_tags  # noqa: E402
from possession.paths import REPO  # noqa: E402
import games  # noqa: E402

warnings.filterwarnings("ignore")

FPS = 25
PXM, PYM = games.GAMES["BAR-ATH"]["pitch"]
HALVES = [(11185, 90651), (110808, 190197)]
WIN0, WIN1 = 11185, 18684          # public ground-truth window
SEC_BINS = [(0, 2, "<2 s"), (2, 5, "2-5 s"), (5, 10, "5-10 s"), (10, 30, "10-30 s"), (30, 60, "30-60 s"),
            (60, 1e12, ">60 s")]
OUT: list[str] = []


def w(s: str = "") -> None:
    OUT.append(s)


def f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def pct(a, qs, nd=3) -> str:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n=0"
    v = np.percentile(a, qs)
    return f"n={a.size}; " + ", ".join(f"p{q:g}={f(x, nd)}" for q, x in zip(qs, v))


def table(headers: list[str], rows: list[list]) -> None:
    w("| " + " | ".join(headers) + " |")
    w("|" + "---|" * len(headers))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def runs(mask: np.ndarray, lo: int = 0, hi: int | None = None):
    """(starts, ends_exclusive) of maximal True runs of mask[lo:hi], in absolute indices."""
    hi = mask.size if hi is None else hi
    m = mask[lo:hi].astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return np.flatnonzero(d == 1) + lo, np.flatnonzero(d == -1) + lo


def graded_runs(mask: np.ndarray):
    s_all, e_all = [], []
    for a, b in HALVES:
        s, e = runs(mask, a, b + 1)
        s_all.append(s)
        e_all.append(e)
    return np.concatenate(s_all), np.concatenate(e_all)


def run_hist_s(lengths_frames: np.ndarray) -> None:
    sec = lengths_frames / FPS
    rows = []
    for lo, hi, nme in SEC_BINS:
        sel = (sec >= lo) & (sec < hi)
        rows.append([nme, int(sel.sum()), int(lengths_frames[sel].sum())])
    table(["run length", "runs", "total frames"], rows)


def binned(series, centers, pre, post, bin_frames=FPS, mask=None):
    """(n_events, pre+post) mean of series over consecutive bins relative to each centre (NaN-aware)."""
    series = np.asarray(series, float)
    T = series.size
    ok = np.isfinite(series)
    if mask is not None:
        ok &= mask
    c1 = np.concatenate([[0.0], np.cumsum(np.where(ok, series, 0.0))])
    c2 = np.concatenate([[0], np.cumsum(ok)])
    j = np.arange(-pre, post)
    lo = np.asarray(centers)[:, None] + j[None, :] * bin_frames
    hi = lo + bin_frames
    valid = (lo >= 0) & (hi <= T)
    lo, hi = np.clip(lo, 0, T), np.clip(hi, 0, T)
    n = c2[hi] - c2[lo]
    return np.where(valid & (n > 0), (c1[hi] - c1[lo]) / np.maximum(n, 1), np.nan)


def eta(series, centers, pre, post, mask=None):
    m = binned(series, centers, pre, post, mask=mask)
    return np.nanmean(m, axis=0), (np.isfinite(m)).sum(0)


def within(events: np.ndarray, targets: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """True for each event e if some target t satisfies e+lo <= t <= e+hi (targets sorted)."""
    if targets.size == 0:
        return np.zeros(events.size, bool)
    a = np.searchsorted(targets, events + lo, side="left")
    b = np.searchsorted(targets, events + hi, side="right")
    return b > a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO.parent / "research" / "eda2-report.md"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # ------------------------------------------------------------------ data
    ta = load_atd("tactical")
    T = ta.T
    G = np.zeros(T, bool)
    half = np.zeros(T, np.int8)
    for i, (a, b) in enumerate(HALVES):
        G[a:b + 1] = True
        half[a:b + 1] = i + 1
    gidx = np.flatnonzero(G)
    P = np.isfinite(ta.bx) & np.isfinite(ta.by)
    bxm, bym = ta.bx * PXM, ta.by * PYM
    outside = P & ((ta.bx < 0) | (ta.bx > 1) | (ta.by < 0) | (ta.by > 1))
    pres = np.isfinite(ta.px) & np.isfinite(ta.py)
    ps = ta.ps.astype(np.float64)
    mps = np.where(pres.sum(1) > 0, np.nanmean(np.where(pres, ps, np.nan), axis=1), np.nan)   # m/s
    prevok = np.zeros(T, bool)
    prevok[1:] = G[1:] & G[:-1] & (half[1:] == half[:-1])
    bsp = np.full(T, np.nan)
    bsp[1:] = np.hypot(np.diff(bxm), np.diff(bym)) * FPS
    bsp[~(prevok & P & np.roll(P, 1))] = np.nan

    S = sb.load_sources()
    Tb = S.Tb
    _, _, bvalid = load_homography("broadcast")
    bP = S.bP
    bpres = np.isfinite(S.bpx) & np.isfinite(S.bpy)

    sync = sb.load_sync()
    mapping = sync["mapping"]
    plats = sync["plateaus"]
    mapped = mapping >= 0
    mb = np.where(mapped, mapping, 0)
    # tactical-frame broadcast state: 0 unmapped, 1 mapped & homography invalid, 2 mapped valid & ball missing,
    # 3 mapped valid & ball present (ball present on an invalid frame is counted as invalid; see B3 note)
    state = np.zeros(T, np.int8)
    state[mapped & ~bvalid[mb]] = 1
    state[mapped & bvalid[mb] & ~bP[mb]] = 2
    state[mapped & bvalid[mb] & bP[mb]] = 3
    STATE_NAMES = ["unmapped", "mapped, homography invalid", "mapped, valid, ball missing", "mapped, ball present"]
    inv_t = state == 1

    # long broadcast invalid runs (> 2 s), expressed on the tactical timeline
    s_inv, e_inv = runs(~bvalid)
    long_inv = (e_inv - s_inv) > 2 * FPS
    inv_long_t = np.zeros(T, bool)
    inv_run_id = np.full(Tb, -1, np.int32)
    for i, (s_, e_) in enumerate(zip(s_inv, e_inv)):
        inv_run_id[s_:e_] = i
    t_of_b = np.full(Tb, -1, np.int64)
    ok_t = np.flatnonzero(mapped)
    t_of_b[mapping[ok_t]] = ok_t
    for s_, e_ in zip(s_inv[long_inv], e_inv[long_inv]):
        tt = t_of_b[s_:e_]
        inv_long_t[tt[tt >= 0]] = True

    tags = load_tags()
    tags["type"] = tags.tags.map(lambda d: d.get("Type"))
    sp = tags[tags.code.str.upper() == "SET PIECES"].copy()
    goals = tags[tags.code.str.upper() == "GOALS"].copy()
    sp_start = np.sort(sp.start_frame.values.astype(int))
    goal_start = np.sort(goals.start_frame.values.astype(int))

    gt = pd.read_csv(REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv")
    lab = np.full(T, "X", dtype="<U1")
    lab[gt.frame.values] = gt.label.values
    win = np.zeros(T, bool)
    win[WIN0:WIN1 + 1] = True
    isD = win & (lab == "D")
    isAl = win & np.isin(lab, ["H", "A"])

    w("# EDA report 2: broadcast sync, whistle audio, restarts (aggregates only)")
    w()
    w(f"Generated by `scripts/eda2.py`. Tactical timeline T={T} frames at 25 fps; graded frames "
      f"{HALVES[0][0]}..{HALVES[0][1]} and {HALVES[1][0]}..{HALVES[1][1]} ({int(G.sum())} frames). Metres use "
      f"{PXM} x {PYM}. Player 'speed' field is in m/s (the metadata calls it kph). No per-frame values, "
      "no individual tag instances, no plots are included. The public 5-minute ground truth is used only in the "
      "B4, A4 and C6 semantics checks.")

    # ================================================================== B1
    w("\n## B1. Broadcast inventory")
    ba = load_atd("broadcast")
    w(f"- Broadcast ATD records {Tb}; slots {ba.px.shape[1]}: team 0 (Barcelona) {int((ba.team == 0).sum())}, "
      f"team 1 (Athletic) {int((ba.team == 1).sum())} (tactical: {int((ta.team == 0).sum())} / {int((ta.team == 1).sum())}).")
    w(f"- Ball present on {f(bP.mean())} of broadcast frames; homography valid on {f(bvalid.mean())}.")
    w(f"- Share of ball-present frames that have a valid homography: {f(bvalid[bP].mean())} "
      f"(ball present on {f(bP[~bvalid].mean())} of invalid frames and on {f(bP[bvalid].mean())} of valid frames).")
    npl = bpres.sum(1)
    w(f"- Tracked players per frame on valid-homography frames: p10/p50/p90 = "
      f"{'/'.join(str(int(x)) for x in np.percentile(npl[bvalid], [10, 50, 90]))}; on invalid frames "
      f"{'/'.join(str(int(x)) for x in np.percentile(npl[~bvalid], [10, 50, 90]))}.")
    for nm_, m_ in (("homography-VALID runs", bvalid), ("homography-INVALID runs", ~bvalid),
                    ("ball-PRESENT runs", bP), ("ball-MISSING runs", ~bP)):
        s_, e_ = runs(m_)
        w(f"\n**{nm_}** (whole broadcast recording; {s_.size} runs)\n")
        run_hist_s(e_ - s_)

    # ================================================================== B2
    w("\n## B2. Mapping tactical_frame = broadcast_frame + delta")
    w("Method: broadcast ball track cut into 1500-frame chunks (stride 750); each chunk with >= 100 ball-present "
      "frames is matched against the whole tactical ball track by masked FFT correlation (score(delta) = N - SSD/8^2 "
      "over frames where both balls are present), the best delta is refined to single-frame resolution "
      "(smallest median ball distance within +-8 frames; n2 = frames within 2 m). A chunk is *confident* if n2 >= 100 "
      "and its best coarse score is at least 2x the best score at any delta >= 50 frames away. Confident chunks are "
      "clustered by delta (gap > 12 frames starts a new cluster); clusters with >= 3 chunks are plateaus. A "
      "plateau's extent is then the span of broadcast frames supported by the independent player check (same-team "
      "tactical player within 2 m for >= 35 % of tracked players, 251-frame window), unsupported gaps up to 10,000 "
      "frames are bridged. Delta is not hard-coded anywhere.")
    fs = sync["flip_stats"]
    w("\n**Coordinate convention** (flip test on 40 seeded chunks, confident chunks / median n2):\n")
    rows = []
    for k_, v in fs.items():
        rows.append([k_, v["n"], v["confident"], f(v["median_n2"], 1)])
    table(["flip (x,y) of broadcast", "chunks tested", "confident", "median n2"], rows)
    flip = sync["flip"]
    w(f"Chosen flip: x flipped = {flip[0]}, y flipped = {flip[1]} (so the two ATDs share the coordinate convention "
      f"if both are False).")
    bx, by, bpx, bpy = sb.apply_flip(S, *flip)
    chunks = sync["chunks"]
    n_valid_chunks = sum(c is not None for c in chunks)
    conf = [c for c in chunks if sb.is_confident(c)]
    w(f"\n- Chunks: {len(chunks)} total, {n_valid_chunks} with enough ball frames, {len(conf)} confident "
      f"({f(len(conf) / max(n_valid_chunks, 1), 3)} of searchable).")
    w(f"- Confident-chunk delta clusters outside any plateau (< 3 chunks, interpreted as replays / outliers): "
      f"{len(sync['dropped'])} cluster(s) containing {sum(d['n_chunks'] for d in sync['dropped'])} chunk(s); "
      f"their deltas differ from the nearest plateau by "
      f"{', '.join(str(abs(d['delta'] - min((p['delta'] for p in plats), key=lambda q: abs(q - d['delta'])))) for d in sync['dropped']) or 'n/a'} frames.")
    w("\n**Plateaus** (one row per supported broadcast segment; frame ranges inclusive):\n")
    rows = []
    for p in plats:
        rows.append([p["delta"], f"{p['b0']}..{p['b1']}", f"{p['t0']}..{p['t1']}", f"{(p['b1'] - p['b0'] + 1) / FPS / 60:.1f}",
                     p["n_chunks"]])
    table(["delta", "broadcast frames", "tactical frames", "minutes", "supporting confident chunks"], rows)
    w(f"Number of distinct delta values: {len({p['delta'] for p in plats})}; segments: {len(plats)}. "
      f"The jump between the first and second plateau is {plats[-1]['delta'] - plats[0]['delta']} frames "
      f"({(plats[-1]['delta'] - plats[0]['delta']) / FPS / 60:.2f} min), which is the halftime cut "
      f"(tactical halftime = frames 90652..110807 = {(110807 - 90652 + 1) / FPS / 60:.2f} min).")

    # drift + fractional delta per plateau
    w("\n**Drift and sub-frame offset per plateau**\n")
    rows = []
    allb = np.arange(Tb)
    seg_b = np.zeros(Tb, bool)
    for p in plats:
        seg_b[p["b0"]:p["b1"] + 1] = True
    for p in plats:
        cs = [c for c in conf if c["b0"] >= p["b0"] and c["b1"] <= p["b1"] + 1 and abs(c["delta"] - p["delta"]) <= 12]
        xs, ys = [], []
        for c in cs:
            d = sb.chunk_delta_frac(S, bx, by, c["b0"], c["b1"], p["delta"], half_range=3.0, step=0.25)
            if np.isfinite(d):
                xs.append((c["b0"] + c["b1"]) / 2 / FPS / 60)
                ys.append(d)
        xs, ys = np.array(xs), np.array(ys)
        ts = sps.theilslopes(ys, xs)
        ols = sps.linregress(xs, ys)
        # pooled fractional optimum on all joint-present frames of the confident chunks
        best = (np.inf, p["delta"])
        for s in np.arange(-1.5, 1.51, 0.25):
            dist = np.concatenate([sb.ball_dist(S, bx, by, c["b0"], c["b1"], p["delta"] + s) for c in cs])
            v = float(np.median(dist))
            if v < best[0]:
                best = (v, p["delta"] + s)
        rows.append([p["delta"], len(xs), f"{ts[0]:+.4f} [{ts[2]:+.4f}, {ts[3]:+.4f}]", f"{ols.slope:+.4f} +- {ols.stderr:.4f}",
                     f(np.std(ys - np.polyval(np.polyfit(xs, ys, 1), xs)), 2), f(best[1], 2), f(best[0], 3)])
    table(["delta", "chunks", "Theil-Sen slope frames/min [95% CI]", "OLS slope +- SE", "per-chunk scatter sd (frames)",
           "pooled fractional optimum delta", "median dist at optimum (m)"], rows)
    w("The fractional optimum is found with the tactical ball linearly interpolated at b + delta + s, s in "
      "[-1.5, 1.5] step 0.25. The chunk-level scatter of about one frame is the noise floor of the two independent "
      "ball tracks, not a sign of cuts.")

    # coverage
    cov_all = mapped[G].mean()
    w("\n**Coverage of tactical frames by a mapped broadcast frame**\n")
    rows = [["graded (both halves)", f(cov_all, 4), int(mapped[G].sum())]]
    for hi_, (a, b) in enumerate(HALVES, 1):
        rows.append([f"graded half {hi_}", f(mapped[a:b + 1].mean(), 4), int(mapped[a:b + 1].sum())])
    rows.append(["ungraded", f(mapped[~G].mean(), 4), int(mapped[~G].sum())])
    table(["frames", "share mapped", "count"], rows)
    gm = mapped & G
    uncovered_runs_s, uncovered_runs_e = graded_runs(~mapped)
    ln = uncovered_runs_e - uncovered_runs_s
    big = np.argsort(-ln)[:6]
    w(f"- Unmapped graded frames: {int((~mapped[G]).sum())} in {ln.size} runs; the longest run lengths (s): "
      f"{', '.join(f(ln[i] / FPS, 1) for i in big)}.")
    w(f"- Mapped broadcast frames: {int(mapped.sum())} of {Tb} ({f(mapped.sum() / Tb, 4)}); "
      f"mapping is injective: {len(np.unique(mapping[mapped])) == int(mapped.sum())}.")

    # player-based confirmation
    w("\n**Independent player-based confirmation** (per mapped broadcast frame with valid homography and >= 1 tracked "
      "player: fraction of tracked broadcast players with a same-team tactical player within 2 m at the tactical "
      "frame b + delta; averaged over the plateau segments)\n")
    seg_valid = np.flatnonzero(seg_b & bvalid)
    delta_seg = np.array(sync["delta_of_b"])[seg_valid].astype(int)

    def player_rate(shift_fn):
        vals = []
        for d in sorted({int(x) for x in delta_seg}):
            sel = seg_valid[delta_seg == d]
            vals.append(sb.player_support(S, bpx, bpy, sel, shift_fn(d)))
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        return v.mean(), (v >= 0.5).mean(), v.size

    rows = []
    for nm_, fn in [("found delta", lambda d: d), ("delta - 25", lambda d: d - 25), ("delta + 25", lambda d: d + 25)]:
        m_, s50, n_ = player_rate(fn)
        rows.append([nm_, f(m_, 4), f(s50, 4), n_])
    for r_ in range(5):
        off = int(rng.integers(500, 20000)) * (1 if r_ % 2 == 0 else -1)
        m_, s50, n_ = player_rate(lambda d, off=off: d + off)
        rows.append([f"random delta (draw {r_ + 1}, delta{off:+d})", f(m_, 4), f(s50, 4), n_])
    table(["candidate", "mean fraction matched", "share of frames with >= 50 % matched", "frames"], rows)

    # residual ball distance at delta, +-1, +-2
    w("**Residual ball distance** (metres) between the broadcast ball at b and the tactical ball at b + delta + s, "
      "over all frames of the plateau segments where both balls are present; second block restricts to frames inside "
      "confident chunks (excludes replays with valid homography)\n")
    conf_b = np.zeros(Tb, bool)
    for c in conf:
        if seg_b[c["b0"]]:
            conf_b[c["b0"]:c["b1"]] = True

    def resid(shift, maskb):
        out = []
        for p in plats:
            b = np.flatnonzero(maskb[p["b0"]:p["b1"] + 1]) + p["b0"]
            t = b + p["delta"] + shift
            ok = (t >= 0) & (t < T)
            b, t = b[ok], t[ok]
            both = bP[b] & P[t]
            out.append(np.hypot(bx[b[both]] - bxm[t[both]], by[b[both]] - bym[t[both]]))
        return np.concatenate(out)

    for title, mk in (("all segment frames", seg_b), ("frames inside confident chunks", conf_b)):
        rows = []
        for sft in (-2, -1, 0, 1, 2):
            d = resid(sft, mk)
            q = np.percentile(d, [10, 50, 90, 99])
            rows.append([f"delta{sft:+d}" if sft else "delta", d.size, f(q[0], 3), f(q[1], 3), f(q[2], 3), f(q[3], 2),
                         f((d < 1).mean(), 3)])
        w(f"_{title}_\n")
        table(["shift", "pairs", "p10", "p50", "p90", "p99", "share < 1 m"], rows)
    w("Frame-exactness reading: the median residual has its minimum at the found delta and rises on both sides "
      "(about 0.1-0.3 m per frame), p10 likewise. The noise floor between the two independent ball tracks (median "
      "~0.8 m) is, however, larger than one frame of ball motion (median 0.2 m/frame), so the ball alone pins delta to "
      "about +-1 frame; the pooled fractional optimum in the drift table is within 0.3 frame of the integer delta in "
      "both plateaus, i.e. no half-frame phase offset is detectable.")

    # ================================================================== B3
    w("\n## B3. What the broadcast adds over tactical graded frames")
    Gm = G
    ct = np.zeros((2, 4), int)
    for i_, pm in enumerate((P, ~P)):
        for s_ in range(4):
            ct[i_, s_] = int((Gm & pm & (state == s_)).sum())
    n_g = int(G.sum())
    w(f"\nGraded frames n = {n_g}. Rows: tactical ball; columns: broadcast state. Counts (share of all graded frames).\n")
    rows = []
    for nm_, i_ in (("tactical ball present", 0), ("tactical ball missing", 1)):
        rows.append([nm_] + [f"{ct[i_, s_]} ({f(ct[i_, s_] / n_g, 4)})" for s_ in range(4)] +
                    [f"{ct[i_].sum()} ({f(ct[i_].sum() / n_g, 4)})"])
    rows.append(["total"] + [f"{ct[:, s_].sum()} ({f(ct[:, s_].sum() / n_g, 4)})" for s_ in range(4)] + [n_g])
    table(["tactical \\ broadcast"] + STATE_NAMES + ["total"], rows)
    w("Row shares (share of each tactical row):\n")
    rows = [[nm_] + [f(ct[i_, s_] / ct[i_].sum(), 4) for s_ in range(4)] for nm_, i_ in
            (("tactical ball present", 0), ("tactical ball missing", 1))]
    table(["tactical \\ broadcast"] + STATE_NAMES, rows)
    w(f"Note: the 'ball present' broadcast state requires a valid homography; broadcast ball present on an invalid-homography "
      f"mapped graded frame (counted under 'invalid'): {int((G & mapped & ~bvalid[mb] & bP[mb]).sum())} frames.")
    gap = G & ~P & (state == 3)
    s_, e_ = runs(gap, 0, T)
    gs, ge = graded_runs(gap)
    w(f"\n**Tactical-missing and broadcast-ball-present graded frames**: {int(gap.sum())} frames "
      f"({f(gap.sum() / (G & ~P).sum(), 4)} of tactical-missing graded frames), {gs.size} runs (runs cut at half boundaries).\n")
    run_hist_s(ge - gs)
    both = G & P & (state == 3)
    dd = np.hypot(bx[mb[both]] - bxm[both], by[mb[both]] - bym[both])
    w(f"**Both balls present (graded, delta-mapped)**: {int(both.sum())} frames; distance (m) {pct(dd, [10, 50, 90, 99], 3)}; "
      f"share within 1 m {f((dd < 1).mean(), 4)}, 2 m {f((dd < 2).mean(), 4)}, 5 m {f((dd < 5).mean(), 4)}.")
    ms_, me_ = graded_runs(~P)
    blk = (me_ - ms_) >= 5 * FPS
    bs, be = ms_[blk], me_[blk]
    n_blk = bs.size
    cov_ball = np.array([(state[a:b] == 3).mean() for a, b in zip(bs, be)])
    cov_inv = np.array([(state[a:b] == 1).mean() for a, b in zip(bs, be)])
    cov_un = np.array([(state[a:b] == 0).mean() for a, b in zip(bs, be)])
    w(f"\n**Tactical ball-missing blocks >= 5 s** (graded): {n_blk} blocks, {int((be - bs).sum())} frames "
      f"({f((be - bs).sum() / (G & ~P).sum(), 4)} of all graded missing frames). Percentiles 10/25/50/75/90 of the "
      f"per-block share covered by:\n")
    rows = []
    for nm_, a_ in (("broadcast ball present (valid)", cov_ball), ("homography invalid", cov_inv), ("unmapped", cov_un)):
        rows.append([nm_] + [f(x, 3) for x in np.percentile(a_, [10, 25, 50, 75, 90])] + [f(a_.mean(), 3)])
    table(["covered by", "p10", "p25", "p50", "p75", "p90", "mean"], rows)
    w(f"Blocks with >= 50 % broadcast-ball coverage: {int((cov_ball >= 0.5).sum())}; with >= 90 %: {int((cov_ball >= 0.9).sum())}; "
      f"with zero broadcast-ball frames: {int((cov_ball == 0).sum())}.")

    # ================================================================== B4
    w("\n## B4. Semantics check on the public window only (frames 11185..18684)")
    rows = []
    wl = win & np.isin(lab, ["H", "A", "D"])
    nD, nAl = int(isD.sum()), int(isAl.sum())
    for s_, nm_ in enumerate(STATE_NAMES):
        m_ = wl & (state == s_)
        n_ = int(m_.sum())
        d_ = int((isD & (state == s_)).sum())
        a_ = int((isAl & (state == s_)).sum())
        rows.append([nm_, n_, f(d_ / n_, 4) if n_ else "nan", f(1 - d_ / n_, 4) if n_ else "nan",
                     f(d_ / nD, 4), f(a_ / nAl, 4)])
    table(["broadcast state", "window frames", "share D", "share alive", "P(state | D)", "P(state | alive)"], rows)
    w(f"Window: {int(wl.sum())} labelled frames, D = {nD}, alive = {nAl}.")
    inv_ids = np.unique(inv_run_id[mb[win & mapped]][~bvalid[mb[win & mapped]]])
    rows = []
    for nm_, minlen in (("all invalid runs", 0), ("invalid runs > 2 s", 2 * FPS)):
        shares = []
        for rid in inv_ids:
            if rid < 0 or (e_inv[rid] - s_inv[rid]) <= minlen:
                continue
            tt = t_of_b[s_inv[rid]:e_inv[rid]]
            tt = tt[(tt >= 0) & win[np.clip(tt, 0, T - 1)] & np.isin(lab[np.clip(tt, 0, T - 1)], ["H", "A", "D"])]
            if tt.size:
                shares.append((lab[tt] == "D").mean())
        rows.append([nm_, len(shares), f(np.mean(shares), 4) if shares else "nan",
                     int(sum(s > 0 for s in shares))])
    table(["homography-invalid runs overlapping the window", "runs", "mean share of their window frames labelled D",
           "runs with any D"], rows)
    w(f"(Window base rate: D share {f(nD / (nD + nAl), 4)}.)")

    # ================================================================== B5
    w("\n## B5. Homography-invalid runs as a stoppage cue (whole match, unsupervised)")
    sel = []
    for s_, e_ in zip(s_inv[long_inv], e_inv[long_inv]):
        t_a, t_b = t_of_b[s_], t_of_b[e_ - 1]
        if t_a >= 0 and t_b >= 0 and (t_b - t_a) == (e_ - 1 - s_):   # entire run mapped, contiguous
            sel.append((t_a, t_b + 1))
    sel = np.array(sel)
    starts_t = sel[:, 0] if sel.size else np.array([], int)
    ends_t = sel[:, 1] if sel.size else np.array([], int)
    ok_s = G[starts_t] if starts_t.size else np.array([], bool)
    ok_e = G[np.minimum(ends_t, T - 1)] if ends_t.size else np.array([], bool)
    series = dict(speed=mps, present=P.astype(float), outside=outside.astype(float))
    w(f"- Broadcast invalid runs > 2 s: {int(long_inv.sum())}; fully mapped runs: {len(sel)}; with start in graded "
      f"frames: {int(ok_s.sum())}; with end in graded frames: {int(ok_e.sum())}. Bins are 1 s, offset = bin start "
      f"relative to the run start / end; frames outside graded excluded.")
    w(f"- Graded-frame reference levels: mean speed {f(np.nanmean(mps[G]), 3)} m/s, ball present {f(P[G].mean(), 3)}, "
      f"ball outside pitch {f(outside[G].mean(), 4)}.\n")
    res_s = {k: eta(v, starts_t[ok_s], 15, 15, mask=G) for k, v in series.items()}
    res_e = {k: eta(v, ends_t[ok_e], 15, 15, mask=G) for k, v in series.items()}
    rows = []
    for j, off in enumerate(range(-15, 15)):
        rows.append([off] + [f(res_s[k][0][j], 3) for k in series] + [res_s["speed"][1][j]] +
                    [f(res_e[k][0][j], 3) for k in series] + [res_e["speed"][1][j]])
    table(["offset (s)", "START: mean speed", "START: ball present", "START: ball outside", "n",
           "END: mean speed", "END: ball present", "END: ball outside", "n"], rows)

    # ================================================================== A1
    w("\n## A1. Audio levels")
    waves = {}
    for src in ("tactical", "broadcast"):
        au.decode(src)
        waves[src] = au.read_wav(src)
        st = au.level_stats(waves[src])
        w(f"\n**{src}**: decoded to `cache/audio_{src}_16k.wav` (mono 16 kHz); duration {f(st['duration_s'], 1)} s "
          f"({st['n_windows']} one-second windows); digital-silence windows {f(st['silent_share'], 4)}.")
        w(f"- 1 s RMS (dBFS, non-silent windows) p1/p10/p50/p90/p99 = "
          f"{' / '.join(f(v, 1) for v in st['rms_pcts'].values())}")
        w("- long-term average spectrum, octave band levels (dB re full scale, band centre Hz): " +
          ", ".join(f"{c}: {f(v, 1)}" for c, v in st["octave_db"].items()))

    # ================================================================== A2
    w("\n## A2. Whistle candidates (tactical audio)")
    x = waves["tactical"]
    ton, shr, pk = au.stft_features(x)
    w(f"STFT: Hann 64 ms window (1024 samples), 10 ms hop, {ton.size} hops. Per hop: tonality = 10 log10(peak bin power "
      f"in 2.5-4.5 kHz / median bin power in 1-7 kHz); band share = power 2.5-4.5 kHz / power 1-7 kHz. A hop is 'on' "
      f"when both features exceed (median + k * MAD) of the whole file (raw MAD, unscaled); runs >= 0.15 s are kept and "
      f"candidates closer than 0.4 s are merged. Feature medians / MADs: tonality {f(np.median(ton), 2)} / "
      f"{f(np.median(np.abs(ton - np.median(ton))), 2)} dB; band share {f(np.median(shr), 4)} / "
      f"{f(np.median(np.abs(shr - np.median(shr))), 4)}. The extra rows k=3 and k=4.5 are not requested but show "
      f"how the count scales.\n")
    KS = [3.0, 4.5, 6.0, 9.0, 12.0]
    cands = {}
    for k in KS:
        c, _ = au.detect(ton, shr, pk, k)
        cands[k] = np.array(c).reshape(-1, 3)
    region_edges = [(0, HALVES[0][0], "pre-match"), (HALVES[0][0], HALVES[0][1] + 1, "half 1 (graded)"),
                    (HALVES[0][1] + 1, HALVES[1][0], "halftime"), (HALVES[1][0], HALVES[1][1] + 1, "half 2 (graded)"),
                    (HALVES[1][1] + 1, 10 ** 9, "post-match")]
    rows = []
    for k in KS:
        c = cands[k]
        fr = np.round(c[:, 0] * FPS).astype(int) if c.size else np.array([], int)
        cnt = [int(((fr >= lo) & (fr < hi)).sum()) for lo, hi, _ in region_edges]
        dur = c[:, 1] - c[:, 0] if c.size else np.array([])
        gaps = c[1:, 0] - c[:-1, 1] if len(c) > 1 else np.array([])
        rows.append([k, len(c)] + cnt + [pct(dur, [10, 50, 90], 2), pct(c[:, 2], [10, 50, 90], 0) if c.size else "n=0",
                                         pct(gaps, [10, 50, 90], 1)])
    table(["k", "total"] + [r[2] for r in region_edges] + ["duration s", "peak freq Hz", "gap end->start s"], rows)

    # ================================================================== A3
    w("\n## A3. Unsupervised validation of the candidates")
    miss_start = np.zeros(T, bool)
    miss_start[1:] = ~P[1:] & P[:-1] & G[1:] & (half[1:] == half[:-1])
    ms_t = np.flatnonzero(miss_start)
    gms, gme = graded_runs(~P)
    blk5 = (gme - gms) >= 5 * FPS
    bstart, bend = gms[blk5], gme[blk5]            # bend exclusive: first present frame after the block (or half end)
    sp_t = sp_start[(sp_start >= 0) & (sp_start < T)]
    inv_start_t = np.flatnonzero(inv_t & ~np.roll(inv_t, 1) & np.roll(state != 0, 1))
    inv_idx = np.flatnonzero(inv_t)
    NRAND = 200

    def metrics(fr: np.ndarray) -> dict:
        return dict(
            miss5=within(fr, ms_t, 1, 5 * FPS).mean(),
            near_start=within(fr, bstart, -3 * FPS, 3 * FPS).mean(),
            near_end=within(fr, bend, -3 * FPS, 3 * FPS).mean(),
            near_sp=within(fr, sp_t, -3 * FPS, 3 * FPS).mean(),
            inv_any=within(fr, inv_idx, 0, 10 * FPS).mean(),
            inv_start=within(fr, inv_start_t, 0, 10 * FPS).mean())

    w(f"Reference: graded-time base rates are computed from {NRAND} seeded draws of N random graded frames (N = number "
      f"of candidates); mean +- sd over draws. 'ball goes missing' = a present->missing transition in the next 5 s. "
      f"Block start/end = start / first-present frame after the end of a >= 5 s missing block ({bstart.size} blocks). "
      f"Broadcast columns use the B2 mapping: any invalid-homography frame within the next 10 s ('any'), or the start "
      f"of an invalid run within the next 10 s ('start'); the candidate frame itself need not be mapped.\n")
    rows = []
    eta_rows = {}
    for k in KS:
        c = cands[k]
        fr_all = np.round(c[:, 0] * FPS).astype(int) if c.size else np.array([], int)
        fr = fr_all[(fr_all >= 0) & (fr_all < T) & G[np.clip(fr_all, 0, T - 1)]]
        n = fr.size
        if n < 3:
            rows.append([k, n] + ["n/a"] * 6)
            continue
        m_c = metrics(fr)
        draws = [metrics(rng.choice(gidx, size=n, replace=False)) for _ in range(NRAND)]
        def cell(key):
            mu, sd = np.mean([d[key] for d in draws]), np.std([d[key] for d in draws])
            return f"{f(m_c[key], 3)} vs {f(mu, 3)}+-{f(sd, 3)} (z={f((m_c[key] - mu) / sd, 1)})"
        rows.append([k, n] + [cell(key) for key in ("miss5", "near_start", "near_end", "near_sp", "inv_any", "inv_start")])
        e_c, n_c = eta(mps, fr, 10, 15, mask=G)
        e_r = np.mean([eta(mps, rng.choice(gidx, size=n, replace=False), 10, 15, mask=G)[0] for _ in range(NRAND)], axis=0)
        eta_rows[k] = (e_c, e_r)
    w("Candidate rate vs random-time rate (z = (candidate - random mean) / random sd):\n")
    table(["k", "graded candidates", "followed by ball going missing <= 5 s", "within 3 s of block START",
           "within 3 s of block END", "within 3 s of SET PIECES start", "invalid frame within 10 s (any)",
           "invalid run starts within 10 s"], rows)
    w("Event-triggered mean tracked-player speed (m/s), offsets in seconds relative to the candidate start; each cell is "
      "candidates / random-time average:\n")
    ks = list(eta_rows)
    rows = []
    for j, off in enumerate(range(-10, 15)):
        rows.append([off] + [f"{f(eta_rows[k][0][j], 3)} / {f(eta_rows[k][1][j], 3)}" for k in ks])
    table(["offset (s)"] + [f"k={k}" for k in ks], rows)
    w(f"(Mean speed over graded frames: {f(np.nanmean(mps[G]), 3)} m/s.)")

    # ================================================================== A4
    w("\n## A4. Whistle candidates vs the public window (counts only)")
    lab_ad = np.where(isD, 1, np.where(isAl, 0, -1))
    tr_ad = np.flatnonzero((lab_ad[1:] == 1) & (lab_ad[:-1] == 0)) + 1       # first D frame
    tr_da = np.flatnonzero((lab_ad[1:] == 0) & (lab_ad[:-1] == 1)) + 1       # first alive frame after D
    d_s, d_e = runs(isD, WIN0, WIN1 + 1)
    w(f"Window has {len(d_s)} D stretches, {tr_ad.size} alive->D and {tr_da.size} D->alive transitions. Categories are "
      f"exclusive in the order shown (candidate start frame, +-2 s = 50 frames).\n")
    rows = []
    for k in KS:
        c = cands[k]
        fr = np.round(c[:, 0] * FPS).astype(int) if c.size else np.array([], int)
        fr = fr[(fr >= WIN0) & (fr <= WIN1)]
        a1 = within(fr, tr_ad, -50, 50)
        a2 = within(fr, tr_da, -50, 50) & ~a1
        inside = np.zeros(fr.size, bool)
        for s_, e_ in zip(d_s, d_e):
            inside |= (fr > s_) & (fr < e_)
        inside &= ~a1 & ~a2
        other = ~(a1 | a2 | inside)
        rows.append([k, fr.size, int(a1.sum()), int(a2.sum()), int(inside.sum()), int(other.sum())])
    table(["k", "in window", "within 2 s of alive->D", "within 2 s of D->alive", "strictly inside D (else)", "elsewhere"], rows)

    # ================================================================== A5
    w("\n## A5. Audio as an independent sync check")
    et = au.frame_envelope(waves["tactical"], T)
    eb = au.frame_envelope(waves["broadcast"], Tb)
    w("Loudness envelope per video frame: log RMS, high-passed by subtracting a centred moving average. Pearson "
      "correlation between the broadcast envelope at b and the tactical envelope at b + delta + shift over all frames of "
      "the plateau segment. Two variants: (E1) 100 ms window, 10 s high-pass; (E2) 1 s window, 120 s high-pass. 'null' = "
      "mean +- sd of the correlation at 300 seeded random lags with 250 <= |lag| <= 7500 frames (10 s .. 5 min).\n")
    rows = []
    for var, (hw, hp) in (("E1", (800, 250)), ("E2", (8000, 3000))):
        et = au.frame_envelope(waves["tactical"], T, hw, hp)
        eb = au.frame_envelope(waves["broadcast"], Tb, hw, hp)
        for p in plats:
            b = np.arange(p["b0"], p["b1"] + 1)

            def cor(sft, b=b, p=p, et=et, eb=eb):
                t = b + p["delta"] + sft
                ok = (t >= 0) & (t < T)
                return np.corrcoef(eb[b[ok]], et[t[ok]])[0, 1]

            row = [var, f"{p['delta']} ({p['b0']}..{p['b1']})"] + [f(cor(s_), 4) for s_ in (0, -25, 25, -125, 125)]
            cs = np.array([cor(s_) for s_ in range(-250, 251)])
            lags = rng.integers(250, 7501, 300) * rng.choice([-1, 1], 300)
            nul = np.array([cor(int(l_)) for l_ in lags])
            row += [int(np.argmax(cs) - 250), f(cs.max(), 4), f"{f(nul.mean(), 4)} +- {f(nul.std(), 4)}",
                    f((cor(0) - nul.mean()) / nul.std(), 1)]
            rows.append(row)
    table(["variant", "plateau delta (broadcast frames)", "corr at delta", "delta-25 (-1 s)", "delta+25 (+1 s)",
           "delta-125 (-5 s)", "delta+125 (+5 s)", "best lag in +-250 frames", "corr at best lag", "null", "z at delta"], rows)

    # ================================================================== C1 / C2
    w("\n## C1. SET PIECES windows: event-triggered averages at window start")
    sp["sf"] = sp.start_frame.astype(int)
    sp_g = sp[(sp.sf >= 0) & (sp.sf < T) & G[np.clip(sp.sf.values, 0, T - 1)]].copy()
    types = ["(pooled)"] + sorted(sp_g.type.fillna("(none)").unique())
    counts = {t_: (len(sp_g) if t_ == "(pooled)" else int((sp_g.type.fillna("(none)") == t_).sum())) for t_ in types}
    w(f"Instances (starting inside graded frames) by type: {counts}; excluded because the start is ungraded: "
      f"{len(sp) - len(sp_g)}. Fixed window length by type is 35 s (kick-offs 5 s). Offsets are 1 s bins starting at the "
      f"offset shown, -10..+39 s; frames outside the graded halves and unmapped frames are excluded from the averages.\n")
    ser_c1 = [("(i) share of instances with tactical ball present", P.astype(float), G),
              ("(ii) mean of mean tracked-player speed (m/s)", mps, G),
              ("(iii) share of frames with ball outside the pitch rectangle", outside.astype(float), G),
              ("(iv) share of frames in a broadcast homography-invalid state (among mapped frames)", inv_t.astype(float), G & mapped)]
    first50 = {}
    for title, ser, mk in ser_c1:
        cols = []
        for t_ in types:
            ev = sp_g.sf.values if t_ == "(pooled)" else sp_g.sf.values[(sp_g.type.fillna("(none)") == t_).values]
            e_, n_ = eta(ser, ev, 10, 40, mask=mk)
            cols.append(e_)
            if title.startswith("(i)"):
                above = np.flatnonzero(e_[10:] >= 0.5)
                first50[t_] = (int(above[0]) if above.size else None, e_[10])
        w(f"**{title}**\n")
        rows = [[off] + [f(cols[i][j], 3) for i in range(len(types))] for j, off in enumerate(range(-10, 40))]
        table(["offset (s)"] + [f"{t_} (n={counts[t_]})" for t_ in types], rows)
    w("Offset at which ball-present share first reaches 50 % at or after the window start (share at start in brackets): " +
      ", ".join(f"{t_}: {'never' if v[0] is None else str(v[0]) + ' s'} ({f(v[1], 2)})" for t_, v in first50.items()) + ".")

    w("\n## C2. Delay from window start to the first ball-present frame (instances with ball missing at the start)")
    edges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 35)]
    names = ["0-1 s", "1-2 s", "2-3 s", "3-5 s", "5-8 s", "8-12 s", "12-20 s", "20-35 s", "never in window"]
    rows = []
    for t_ in types:
        sub = sp_g if t_ == "(pooled)" else sp_g[sp_g.type.fillna("(none)") == t_]
        cnts = [0] * 9
        n_miss = 0
        delays = []
        for sf, ef in zip(sub.sf, sub.end_frame):
            if P[sf]:
                continue
            n_miss += 1
            seg = P[sf:min(int(ef) + 1, T)]
            idx = np.flatnonzero(seg)
            if idx.size == 0:
                cnts[8] += 1
                continue
            dly = idx[0] / FPS
            delays.append(dly)
            for i_, (lo, hi) in enumerate(edges):
                if lo <= dly < hi:
                    cnts[i_] += 1
                    break
            else:
                cnts[8] += 1
        rows.append([t_, len(sub), n_miss] + cnts + [f(np.median(delays), 1) if delays else "nan"])
    table(["type", "instances", "ball missing at start"] + names + ["median delay (s)"], rows)

    # ================================================================== C3
    w("\n## C3. GOALS windows")
    rows = []
    gp = np.zeros(10)
    gs_ = np.zeros(10)
    delays = []
    for _, r in goals.iterrows():
        a, b = int(r.start_frame), int(r.end_frame)
        edges_ = np.linspace(a, b, 11).astype(int)
        for i_ in range(10):
            seg = slice(edges_[i_], edges_[i_ + 1])
            gp[i_] += P[seg].mean() / len(goals)
            gs_[i_] += np.nanmean(mps[seg]) / len(goals)
        ko = sp[(sp.type.fillna("").str.contains("after goal", case=False)) & (sp.sf >= a)]
        if len(ko):
            delays.append((ko.sf.min() - a) / FPS)
    w(f"{len(goals)} GOALS instances, window length {f(((goals.end_frame - goals.start_frame) / FPS).mean(), 1)} s "
      f"(range {f(((goals.end_frame - goals.start_frame) / FPS).min(), 1)}-{f(((goals.end_frame - goals.start_frame) / FPS).max(), 1)} s).\n")
    table(["tenth of window", "share ball present (mean over instances)", "mean player speed m/s"],
          [[i_ + 1, f(gp[i_], 3), f(gs_[i_], 3)] for i_ in range(10)])
    w(f"Delay from GOALS window start to the next 'Kick Off (After Goal)' window start (s): n={len(delays)}, mean "
      f"{f(np.mean(delays), 1)}, min {f(np.min(delays), 1)}, max {f(np.max(delays), 1)}.")

    # ================================================================== C4
    w("\n## C4. Anatomy of the >= 5 s ball-missing blocks (graded frames)")
    Pidx = np.flatnonzero(P)

    def last_seen(s):
        i = np.searchsorted(Pidx, s) - 1
        return Pidx[i] if i >= 0 else -1

    def first_seen(e):
        i = np.searchsorted(Pidx, e)
        return Pidx[i] if i < Pidx.size else -1

    def cat_a(x, y):
        out_ = (x < 0) | (x > PXM) | (y < 0) | (y > PYM)
        tl = (y <= 2) | (y >= PYM - 2)
        gl = (x <= 2) | (x >= PXM - 2)
        pen = ((x <= 16.5) | (x >= PXM - 16.5)) & (np.abs(y - PYM / 2) <= 20.16)
        return np.where(out_, 0, np.where(tl, 1, np.where(gl, 2, np.where(pen, 3, 4))))

    CAT_A = ["outside pitch rectangle", "within 2 m inside touchline", "within 2 m inside goal line",
             "inside a penalty area", "elsewhere interior"]
    ls = np.array([last_seen(s) for s in bstart])
    fs_ = np.array([first_seen(e) for e in bend])
    okl = ls >= 0
    okf = fs_ >= 0
    w(f"Blocks: {bstart.size}; with a ball sighting before: {int(okl.sum())}; after: {int(okf.sum())}.\n")
    ca = cat_a(bxm[ls[okl]], bym[ls[okl]])
    w("**(a) Ball LAST seen before the block** (exclusive, priority in the order listed):\n")
    table(["category", "blocks", "share"], [[CAT_A[i], int((ca == i).sum()), f((ca == i).mean(), 3)] for i in range(5)])
    spd = np.full(ls.size, np.nan)
    for i, l_ in enumerate(ls):
        if l_ >= 1 and P[l_ - 1]:
            spd[i] = np.hypot(bxm[l_] - bxm[l_ - 1], bym[l_] - bym[l_ - 1]) * FPS
    w(f"- last-seen ball speed (m/s, last two frames): {pct(spd, [10, 50, 90], 2)}")
    toward, n_tw = 0, 0
    for l_ in ls[okl]:
        if l_ >= 9 and P[l_ - 9:l_ + 1].all():
            n_tw += 1
            v = np.array([bxm[l_] - bxm[l_ - 9], bym[l_] - bym[l_ - 9]])
            x_, y_ = bxm[l_], bym[l_]
            sd = np.array([x_, PXM - x_, y_, PYM - y_])      # signed distance to left/right/top/bottom (negative outside)
            nrm = [(-1, 0), (1, 0), (0, -1), (0, 1)][int(np.argmin(sd))]
            toward += int(v @ np.array(nrm) > 0)
    w(f"- last 10 frames velocity points towards the nearest boundary (outward normal of the nearest side): "
      f"{toward} of {n_tw} blocks ({f(toward / max(n_tw, 1), 3)}) (blocks whose last 10 frames are all ball-present).")
    x_f, y_f = bxm[fs_[okf]], bym[fs_[okf]]
    near_t = (np.abs(y_f) <= 2) | (np.abs(y_f - PYM) <= 2)
    corners = np.array([(0, 0), (0, PYM), (PXM, 0), (PXM, PYM)])
    near_c = (np.min(np.hypot(x_f[:, None] - corners[:, 0], y_f[:, None] - corners[:, 1]), axis=1) <= 3)
    in_ga = (((x_f >= 0) & (x_f <= 5.5)) | ((x_f <= PXM) & (x_f >= PXM - 5.5))) & (np.abs(y_f - PYM / 2) <= 9.16)
    in_pa = (((x_f >= 0) & (x_f <= 16.5)) | ((x_f <= PXM) & (x_f >= PXM - 16.5))) & (np.abs(y_f - PYM / 2) <= 20.16)
    near_cs = np.hypot(x_f - PXM / 2, y_f - PYM / 2) <= 3
    pa_only = in_pa & ~in_ga
    excl = np.select([near_c, near_t, in_ga, near_cs, pa_only], [0, 1, 2, 3, 4], 5)
    names_b = ["within 3 m of a corner", "within 2 m of a touchline (inside or outside)", "inside a goal area (six-yard box)",
               "within 3 m of the centre spot", "inside a penalty area, outside the goal area", "elsewhere"]
    w("\n**(b) Ball FIRST seen after the block**: non-exclusive shares, then exclusive (priority corner > touchline > goal "
      "area > centre spot > penalty area > elsewhere):\n")
    flags = [near_c, near_t, in_ga, near_cs, pa_only]
    rows = [[names_b[i], int(flags[i].sum()), f(flags[i].mean(), 3), int((excl == i).sum()), f((excl == i).mean(), 3)]
            for i in range(5)]
    rows.append([names_b[5], int((~(near_c | near_t | in_ga | near_cs | pa_only)).sum()),
                 f((~(near_c | near_t | in_ga | near_cs | pa_only)).mean(), 3), int((excl == 5).sum()), f((excl == 5).mean(), 3)])
    table(["category", "blocks (non-excl.)", "share", "blocks (excl.)", "share"], rows)
    w("**(c) Last-seen category x block length**:\n")
    L = ((bend - bstart) / FPS)[okl]
    lb = [(5, 10, "5-10 s"), (10, 20, "10-20 s"), (20, 40, "20-40 s"), (40, 1e9, ">40 s")]
    rows = []
    for i in range(5):
        rows.append([CAT_A[i]] + [int(((ca == i) & (L >= lo) & (L < hi)).sum()) for lo, hi, _ in lb] + [int((ca == i).sum())])
    rows.append(["total"] + [int(((L >= lo) & (L < hi)).sum()) for lo, hi, _ in lb] + [L.size])
    table(["last-seen category"] + [n for _, _, n in lb] + ["all"], rows)
    be_last = bend - 1
    pad = 5 * FPS
    near_sp = np.array([bool(((sp_t >= a - pad) & (sp_t <= b + pad)).any()) for a, b in zip(bstart, be_last)])
    near_gl = np.array([bool(((goal_start >= a - pad) & (goal_start <= b + pad)).any()) for a, b in zip(bstart, be_last)])
    near_inv = np.array([bool(inv_long_t[max(a - pad, 0):b + pad + 1].any()) for a, b in zip(bstart, be_last)])
    contain_sp = np.array([bool(((sp_t >= a) & (sp_t <= b)).any()) for a, b in zip(bstart, be_last)])
    # random-placement baseline: same block lengths, random start inside a random graded half (200 draws)
    blen = bend - bstart
    base = {k_: [] for k_ in ("sp", "gl", "inv")}
    for _ in range(200):
        vals = {k_: [] for k_ in base}
        for L_ in blen:
            hh = HALVES[int(rng.integers(0, 2))]
            a0 = int(rng.integers(hh[0], hh[1] - L_))
            b0_ = a0 + L_ - 1
            vals["sp"].append(bool(((sp_t >= a0 - pad) & (sp_t <= b0_ + pad)).any()))
            vals["gl"].append(bool(((goal_start >= a0 - pad) & (goal_start <= b0_ + pad)).any()))
            vals["inv"].append(bool(inv_long_t[max(a0 - pad, 0):b0_ + pad + 1].any()))
        for k_ in base:
            base[k_].append(np.mean(vals[k_]))
    bs_ = {k_: f"{f(np.mean(v), 3)} +- {f(np.std(v), 3)}" for k_, v in base.items()}
    w("**(d) Proximity** (block extended by 5 s each side; last column = same-length blocks placed at random in the graded halves, 200 draws):\n")
    table(["blocks that contain / are within 5 s of ...", "blocks", "share", "random placement"],
          [["a SET PIECES window start (within 5 s)", int(near_sp.sum()), f(near_sp.mean(), 3), bs_["sp"]],
           ["a SET PIECES window start (contained in block)", int(contain_sp.sum()), f(contain_sp.mean(), 3), "-"],
           ["a GOALS window start", int(near_gl.sum()), f(near_gl.mean(), 3), bs_["gl"]],
           ["a broadcast homography-invalid run > 2 s (mapped)", int(near_inv.sum()), f(near_inv.mean(), 3), bs_["inv"]],
           ["any of the three", int((near_sp | near_gl | near_inv).sum()), f((near_sp | near_gl | near_inv).mean(), 3), "-"],
           ["none of the three", int((~(near_sp | near_gl | near_inv)).sum()), f((~(near_sp | near_gl | near_inv)).mean(), 3), "-"]])
    w(f"(Reference: {int(sp_t.size)} SET PIECES windows, {goal_start.size} GOALS windows, {int(long_inv.sum())} broadcast "
      f"invalid runs > 2 s of which {len(sel)} fully mapped.)")

    # ================================================================== C5
    w("\n## C5. Player speed vs ball availability")
    q = [10, 25, 50, 75, 90]
    rows = [["ball present (graded)"] + [f(v, 3) for v in np.nanpercentile(mps[G & P], q)] + [int((G & P & np.isfinite(mps)).sum())],
            ["ball missing (graded)"] + [f(v, 3) for v in np.nanpercentile(mps[G & ~P], q)] + [int((G & ~P & np.isfinite(mps)).sum())]]
    table(["per-frame mean tracked-player speed (m/s)", "p10", "p25", "p50", "p75", "p90", "frames"], rows)
    dec = np.full((bstart.size, 10), np.nan)
    for i, (a, b) in enumerate(zip(bstart, bend)):
        edges_ = np.linspace(a, b, 11).astype(int)
        for j in range(10):
            dec[i, j] = np.nanmean(mps[edges_[j]:max(edges_[j + 1], edges_[j] + 1)])
    w("Mean speed by tenth of the block (average over blocks of the per-block tenth means): " +
      ", ".join(f"{j + 1}: {f(np.nanmean(dec[:, j]), 3)}" for j in range(10)) + ".\n")
    es, ns = eta(mps, bstart, 10, 10, mask=G)
    ee, ne = eta(mps, bend, 10, 10, mask=G)
    w("Event-triggered mean speed (m/s) aligned at block START and at block END (END = first present frame after the "
      "block), 1 s bins from -10 s:\n")
    table(["offset (s)", "START", "n", "END", "n"], [[off, f(es[j], 3), int(ns[j]), f(ee[j], 3), int(ne[j])]
                                                      for j, off in enumerate(range(-10, 10))])
    inblk = np.zeros(T, bool)
    for a, b in zip(bstart, bend):
        inblk[a:b] = True
    w("Share of frames with mean speed below a threshold:\n")
    rows = []
    for thr in (1.0, 1.25, 1.5):
        rows.append([f"< {thr} m/s", f(np.nanmean(mps[inblk] < thr), 4), f(np.nanmean(mps[G & P] < thr), 4)])
    table(["threshold", f"frames in >= 5 s missing blocks (n={int(inblk.sum())})", f"ball-present graded frames (n={int((G & P).sum())})"], rows)

    # ================================================================== C6
    w("\n## C6. Semantics check on the public window only")
    spd_bins = [(-1, 0.8, "<0.8"), (0.8, 1.0, "0.8-1.0"), (1.0, 1.25, "1.0-1.25"), (1.25, 1.5, "1.25-1.5"),
                (1.5, 2.0, "1.5-2.0"), (2.0, 2.5, "2.0-2.5"), (2.5, 1e9, ">2.5")]
    rows = []
    for lo, hi, nme in spd_bins:
        m_ = (isD | isAl) & (mps >= lo) & (mps < hi) if lo >= 0 else (isD | isAl) & (mps < hi)
        n_ = int(m_.sum())
        rows.append([nme, n_, int((m_ & isD).sum()), f((m_ & isD).sum() / n_, 3) if n_ else "nan"])
    table(["mean speed bin (m/s)", "frames", "D frames", "P(D)"], rows)
    mo = (isD | isAl) & outside
    w(f"- P(D | ball outside the pitch rectangle): {int((mo & isD).sum())} of {int(mo.sum())} labelled frames "
      f"({f((mo & isD).sum() / max(mo.sum(), 1), 3)}); window base rate P(D) = {f(nD / (nD + nAl), 3)}.")
    pres_ad = [bool(P[t_]) for t_ in tr_ad]
    out2 = [bool(outside[t_ + 1:t_ + 1 + 2 * FPS].any()) for t_ in tr_ad]
    w(f"- alive->D transitions: {tr_ad.size}; ball present at the first D frame: {sum(pres_ad)}; ball outside the pitch "
      f"within the following 2 s: {sum(out2)}.")
    pres_da = [bool(P[t_]) for t_ in tr_da]
    af = [np.nanmean(bsp[t_:t_ + FPS]) if np.isfinite(bsp[t_:t_ + FPS]).any() else np.nan for t_ in tr_da]
    bf = [np.nanmean(bsp[t_ - FPS:t_]) if np.isfinite(bsp[t_ - FPS:t_]).any() else np.nan for t_ in tr_da]
    w(f"- D->alive transitions: {tr_da.size}; ball present at the first alive frame: {sum(pres_da)}; mean ball speed in "
      f"the second after (m/s): {f(np.nanmean(af), 2)} (defined for {int(np.isfinite(af).sum())}), in the second before: "
      f"{f(np.nanmean(bf), 2)} (defined for {int(np.isfinite(bf).sum())}).")

    w(f"\n_Runtime of this script: {f(time.time() - t0, 0)} s (a clean run that also decodes both audio tracks and recomputes the sync takes about 2 min 5 s; with caches warm about 1.5 min)._")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print("report written:", args.out, "lines:", len(OUT))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - never echo data through a traceback
        tb = exc.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(f"FAILED: {type(exc).__name__} at line {tb.tb_lineno} ({Path(tb.tb_frame.f_code.co_filename).name})")
        sys.exit(1)
