"""Aggregate-only report on the auxiliary streams (possession/aux_streams.py, target_fused.py).

Writes ``research/aux-report.md``. Prints only aggregates. R5 uses the public ground-truth window for a
semantics check only; nothing here sets a parameter. Run: ``uv run python scripts/aux_report.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from possession import sync_broadcast as sb  # noqa: E402
from possession.aux_streams import build_aux, SP_EDGES_S  # noqa: E402
from possession.match import H, A, D  # noqa: E402
from possession.paths import REPO  # noqa: E402
from possession.target import graded_mask, load_player_speed, load_target_match  # noqa: E402
from possession.target_fused import fill_stats, fused_ball_source  # noqa: E402

FPS = 25
SEC_BINS = [(0, 2, "<2 s"), (2, 5, "2-5 s"), (5, 10, "5-10 s"), (10, 30, "10-30 s"), (30, 60, "30-60 s"),
            (60, 1e12, ">60 s")]
BST = ["0 unmapped", "1 invalid homography", "2 valid, not live (replay)", "3 live, no ball", "4 live, ball"]
SPP = ["0 none", "1 [-10,-5)s", "2 [-5,0)s", "3 [0,2)s", "4 [2,4)s", "5 [4,6)s", "6 [6,9)s", "7 [9,14)s", "8 [14s,end]"]
SPT = ["none", "corner", "free kick", "goal kick", "KO after goal", "KO start", "throw-in", "other"]
OUT: list[str] = []


def w(s=""):
    OUT.append(s)


def table(h, rows):
    w("| " + " | ".join(h) + " |")
    w("|" + "---|" * len(h))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def runs(mask, G):
    """(start, end_exclusive) of True runs, not crossing the graded-block boundary (mask & G)."""
    m = (mask & G).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e


def run_hist(lengths):
    sec = lengths / FPS
    return [[n, int(((sec >= lo) & (sec < hi)).sum()), int(lengths[(sec >= lo) & (sec < hi)].sum())] for lo, hi, n in SEC_BINS]


def pq(a, qs=(10, 50, 90, 99), nd=2):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n=0"
    return f"n={a.size}; " + ", ".join(f"p{q}={np.percentile(a, q):.{nd}f}" for q in qs)


def main():
    aux = build_aux()
    m = load_target_match()
    T = m.T
    G = graded_mask(T)
    bs = aux["bstate"]
    tact = np.isfinite(m.bx) & np.isfinite(m.by)
    src = fused_ball_source()
    fused = src > 0
    ng = int(G.sum())
    w("# Auxiliary streams report (aggregates over graded frames; R5 is a public-window semantics check only)\n")
    w(f"Graded frames: {ng}. All shares are over graded frames unless stated.\n")

    # ---- R1
    w("## R1. Broadcast state\n")
    rows = []
    for k, nme in enumerate(BST):
        a = bs[G] == k
        rows.append([nme, int(a.sum()), f"{a.mean():.4f}", f"{(bs[G & tact] == k).mean():.4f}",
                     f"{(bs[G & ~tact] == k).mean():.4f}"])
    table(["bstate", "frames", "share overall", "share given tactical ball present", "share given tactical ball missing"], rows)
    w(f"Tactical ball present on {tact[G].mean():.4f} of graded frames (missing: {int((~tact[G]).sum())}).\n")
    st = aux["live_stat"][G]
    h = np.histogram(st[np.isfinite(st)], bins=10, range=(0, 1))[0]
    w("**LIVE statistic** (share of broadcast-tracked players with a same-team tactical player within 2 m, median "
      "over +-12 frames; valid-homography mapped graded frames with >= 4 players on both recordings, "
      f"n={int(h.sum())}).\n")
    table(["bin"] + [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)], [["frames"] + [int(x) for x in h],
          ["share"] + [f"{x / h.sum():.3f}" for x in h]])
    thr = float(aux["live_thr"])
    w(f"Threshold {thr:.2f}: Otsu's between-class-variance maximiser on a 100-bin histogram of the same statistic "
      f"over all mapped valid frames of the whole match (label-free). Valid frames in runs with no decided frame "
      f"(treated as not live): {int(aux['n_undecided_run'])}; frames decided by carrying the nearest decided value "
      f"within the run: {int(aux['n_decided_by_carry'])} (whole match).\n")
    s_, e_ = runs(bs == 2, G)
    ln = e_ - s_
    w(f"**bstate == 2 runs** (graded): count {ln.size}, total frames {int(ln.sum())}; length percentiles (s) "
      f"p10/p50/p90 = {np.percentile(ln / FPS, 10):.1f} / {np.percentile(ln / FPS, 50):.1f} / {np.percentile(ln / FPS, 90):.1f}.\n")

    # ---- R2
    w("## R2. Fill of the tactical ball from the live broadcast\n")
    fs = fill_stats()
    filled_g = int((src[G] == 2).sum())
    s2, _ = runs(src == 2, G)
    cand_g = int((~tact & (bs == 4) & G).sum())
    w(f"Whole match: candidates {fs['candidates']} in {fs['candidate_runs']} runs; filled {fs['filled']}; dropped by the "
      f"rectangle guard (pitch +3 m) {fs['dropped_rect']}, by the 40 m/s jump guard {fs['dropped_jump']}.\n")
    w(f"Graded: candidates {cand_g}, filled {filled_g} frames in {s2.size} runs, dropped {cand_g - filled_g}. "
      f"Ball-missing share before {(~tact[G]).mean():.4f}, after {(~fused[G]).mean():.4f}.\n")
    for nme, mk in (("tactical ball only (before)", tact), ("after fill", fused)):
        w(f"**{nme}** - missing runs / present runs (graded, within graded blocks)\n")
        a, b = runs(~mk, G)
        c, d = runs(mk, G)
        table(["run length", "missing runs", "frames", "present runs", "frames"],
              [x + y[1:] for x, y in zip(run_hist(b - a), run_hist(d - c))])
    S = sb.load_sources()
    mp = sb.load_sync()["mapping"]
    mb = np.where(mp >= 0, mp, 0)
    ok = (mp >= 0) & S.bP[mb]
    ex = S.bx[mb] - 0.5 * 0
    dist = np.hypot(S.bx[mb] - (m.bx + sb.PX / 2), S.by[mb] - (m.by + sb.PY / 2))
    dist = np.where(ok & tact, dist, np.nan)
    w("Distance (m) between the tactical ball and the mapped broadcast ball, graded frames where both exist "
      "(broadcast ball taken from the raw track, so also on not-live frames):\n")
    table(["bstate", "frames both present", "percentiles"],
          [[BST[k], int(np.isfinite(dist[G & (bs == k)]).sum()), pq(dist[G & (bs == k)])] for k in (4, 2, 3, 1)])

    # ---- R3
    w("## R3. Cross-tabs (row shares)\n")

    def xtab(rowv, rownames, colv, colnames, title):
        w(f"**{title}**\n")
        rows = []
        for i, nme in enumerate(rownames):
            sel = G & (rowv == i)
            n = int(sel.sum())
            rows.append([nme, n] + [f"{(colv[sel] == j).mean():.3f}" if n else "-" for j in range(len(colnames))])
        table(["row", "frames"] + list(colnames), rows)

    xtab(aux["sp_phase"], SPP, bs, BST, "sp_phase (rows) x bstate (columns)")
    xtab(aux["goal_win"], ["none", "goal first 20%", "goal rest"], bs, BST, "goal_win x bstate")
    xtab(aux["att_phase"], ["none", "home attacking", "away attacking", "both"], bs, BST, "att_phase x bstate")
    xtab(bs, BST, aux["att_phase"], ["none", "home", "away", "both"], "bstate (rows) x att_phase")

    # ---- R4
    w("## R4. Physical sanity (label-free)\n")
    hs, as_ = load_player_speed()
    sp = np.concatenate([hs, as_], 1).astype(np.float64)
    with np.errstate(all="ignore"):
        cnt = np.isfinite(sp).sum(1)
        msp = np.where(cnt > 0, np.nansum(sp, 1) / np.maximum(cnt, 1), np.nan)
    w("Mean tracked-player speed (m/s), mean of per-frame means.\n")
    table(["bstate", "frames", "mean speed"], [[BST[k], int((G & (bs == k)).sum()),
          f"{np.nanmean(msp[G & (bs == k)]):.3f}"] for k in range(5)])
    table(["sp_phase", "frames", "mean speed", "tactical ball present"],
          [[SPP[k], int((G & (aux['sp_phase'] == k)).sum()), f"{np.nanmean(msp[G & (aux['sp_phase'] == k)]):.3f}",
            f"{tact[G & (aux['sp_phase'] == k)].mean():.3f}"] for k in range(9)])

    # ---- R5
    w("## R5. SEMANTICS CHECK on the public window (understanding only; nothing fitted from this)\n")
    lab = m.label
    L = (lab != 3) & G
    w(f"Labelled frames: {int(L.sum())}.\n")
    streams = [("bstate", BST), ("sp_phase", SPP), ("sp_type", SPT), ("sp_team", ["none", "home", "away"]),
               ("goal_win", ["none", "first20", "rest"]), ("att_phase", ["none", "home", "away", "both"]),
               ("cc_win", ["none", "1st third", "2nd", "3rd"])]
    for nm, names in streams:
        v = aux[nm]
        rows = []
        for i, nme in enumerate(names):
            sel = L & (v == i)
            n = int(sel.sum())
            rows.append([nme, n] + ([f"{(lab[sel] == c).mean():.3f}" for c in (D, H, A)] if n else ["-"] * 3))
        w(f"**{nm}**\n")
        table(["value", "labelled frames", "P(D)", "P(H)", "P(A)"], rows)
    for nm, vals in (("att_phase", (1, 2)), ("sp_team", (1, 2))):
        v = aux[nm]
        sel = L & np.isin(v, vals) & np.isin(lab, (H, A))
        match = ((v == 1) & (lab == H)) | ((v == 2) & (lab == A))
        w(f"{nm}: share of labelled alive frames (H/A) with a single tagged team whose label equals the tagged team: "
          f"{match[sel].mean() if sel.any() else float('nan'):.3f} (n={int(sel.sum())}).\n")

    path = REPO.parent / "research" / "aux-report.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", path.name, "lines", len(OUT))


if __name__ == "__main__":
    main()
