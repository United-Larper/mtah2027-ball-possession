"""Label statistics, metric sensitivity and baseline calibration on the cached DFL matches.

    uv run scripts/dfl_label_report.py [--out ../research/dfl-label-report.md]

Reads cache/external/dfl_<id>.npz (+ .report.json) written by scripts/build_external.py.
Scoring convention: each match is one array = period 1 labels, one X frame, period 2 labels,
for BOTH ground truth and prediction, so the scorer's valid_runs() sees the periods as separate
runs. Pooled scores sum the 3x4 confusion matrices (macroF1) and the per-IoU TP/FP/FN spell counts
(spellF1) over matches; the mean of per-match spellF1 is also reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession.match import Match  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
FPS = 25.0
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------------------------------------------------------------------- scoring helpers

def join(parts):
    out = []
    for i, p in enumerate(parts):
        if i:
            out.append(np.array([X], dtype=np.int8))
        out.append(np.asarray(p, dtype=np.int8))
    return np.concatenate(out)


def score_arrays(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    sp = pm.spell_scores(gt, filled, runs)
    counts = {thr: (v["n_matched"], v["n_spurious"], v["n_missed"]) for thr, v in sp["per_iou"].items()}
    return cm3, counts, sp["mean_f1"]


def pool(results):
    cm3 = sum(r[0] for r in results)
    f1s = []
    for thr in pm.SPELL_IOUS:
        tp = sum(r[1][thr][0] for r in results)
        fp = sum(r[1][thr][1] for r in results)
        fn = sum(r[1][thr][2] for r in results)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"spell": float(np.mean(f1s)), "macro": pm.macro_f1(cm3),
            "mean_match_spell": float(np.mean([r[2] for r in results]))}


# ----------------------------------------------------------------------------- perturbations

def rle(a):
    b = np.concatenate(([0], np.flatnonzero(a[1:] != a[:-1]) + 1, [len(a)]))
    return [(int(b[i]), int(b[i + 1]), int(a[b[i]])) for i in range(len(b) - 1)]


def p_shift(a, s, rng):
    out = np.empty_like(a)
    out[s:] = a[:-s]
    out[:s] = a[0]
    return out


def p_erase_spells(a, minlen, rng):
    out = a.copy()
    prev = None
    for s0, s1, c in rle(a):
        if c == D:
            continue
        new = prev if (s1 - s0 < minlen and prev is not None) else c
        out[s0:s1] = new
        prev = new
    return out


def p_erase_D(a, minlen, rng):
    out = a.copy()
    segs = rle(a)
    for i, (s0, s1, c) in enumerate(segs):
        if c == D and s1 - s0 < minlen:
            if i > 0:
                out[s0:s1] = out[s0 - 1]
            elif len(segs) > 1:
                out[s0:s1] = segs[1][2]
    return out


def p_merge_same_team_D(a, _p, rng):
    out = a.copy()
    segs = rle(a)
    for i in range(1, len(segs) - 1):
        s0, s1, c = segs[i]
        if c == D and segs[i - 1][2] == segs[i + 1][2]:
            out[s0:s1] = segs[i - 1][2]
    return out


def p_jitter(a, J, rng):
    segs = rle(a)
    if len(segs) < 2:
        return a.copy()
    bounds = np.array([s[0] for s in segs[1:]])
    nb = np.clip(bounds + rng.integers(-J, J + 1, len(bounds)), 1, len(a) - 1)
    nb = np.maximum.accumulate(nb)
    edges = np.concatenate(([0], nb, [len(a)]))
    return np.repeat(np.array([s[2] for s in segs], dtype=np.int8), np.diff(edges))


def p_blips(a, frac, rng, blip=12, min_spell=100):
    out = a.copy()
    spells = [s for s in rle(a) if s[2] in (H, A) and s[1] - s[0] > min_spell]
    k = int(round(frac * len(spells)))
    for idx in rng.choice(len(spells), size=k, replace=False) if k else []:
        s0, s1, c = spells[idx]
        st = int(rng.integers(s0 + 1, s1 - blip - 1))
        out[st:st + blip] = 1 - c
    return out


PERTURB = {"shift": p_shift, "erase_spells": p_erase_spells, "erase_D": p_erase_D,
           "merge_D": p_merge_same_team_D, "jitter": p_jitter, "blips": p_blips}


def run_task(args):
    kind, param, seed, parts = args
    rng = np.random.default_rng(seed)
    pred_parts = [PERTURB[kind](p, param, rng) for p in parts]
    return score_arrays(join(parts), join(pred_parts))


def sensitivity(parts_by_match, ex):
    scen = []
    for s in (3, 8, 12, 25):
        scen.append((f"a. true labels shifted later by {s} frames ({s / FPS:.2f} s)", "shift", s, [0]))
    for sec in (1, 2, 4):
        scen.append((f"b. spells < {sec} s erased (relabelled to preceding alive spell)", "erase_spells", sec * 25, [0]))
    for sec in (2, 5):
        scen.append((f"c. D stretches < {sec} s relabelled as preceding label", "erase_D", sec * 25, [0]))
    scen.append(("d. all D stretches between same-team spells relabelled as that team", "merge_D", 0, [0]))
    for J in (5, 12, 25):
        scen.append((f"e. boundaries jittered uniformly in +-{J} frames (mean of 5 seeds)", "jitter", J, list(range(5))))
    for fr in (0.1, 0.3):
        scen.append((f"f. 0.5 s wrong-team blip in {int(fr * 100)}% of spells > 4 s (mean of 5 seeds)", "blips", fr, list(range(5))))
    rows = []
    base = pool([score_arrays(join(p), join(p)) for p in parts_by_match])
    rows.append(("identity (sanity)", base["spell"], 0.0, base["macro"], base["mean_match_spell"]))
    for name, kind, param, seeds in scen:
        per_seed = []
        for sd in seeds:
            res = list(ex.map(run_task, [(kind, param, 1000 * sd + i, parts) for i, parts in enumerate(parts_by_match)]))
            per_seed.append(pool(res))
        sp = [r["spell"] for r in per_seed]
        rows.append((name, float(np.mean(sp)), float(np.std(sp)) if len(sp) > 1 else 0.0,
                     float(np.mean([r["macro"] for r in per_seed])),
                     float(np.mean([r["mean_match_spell"] for r in per_seed]))))
        print("  done", name, flush=True)
    return rows


# ----------------------------------------------------------------------------- baseline

def nearest(m: Match):
    """Per frame: team (0 home / 1 away) of the nearest tracked player to the ball, and distance.
    team = -1 and dist = nan where the ball is missing or nobody is tracked."""
    px = np.concatenate([m.hx, m.ax], axis=1)
    py = np.concatenate([m.hy, m.ay], axis=1)
    d = np.hypot(px - m.bx[:, None], py - m.by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    idx = d.argmin(axis=1)
    dist = d[np.arange(len(d)), idx]
    ok = np.isfinite(dist) & ~np.isnan(m.bx)
    team = np.where(ok, (idx >= m.hx.shape[1]).astype(np.int8), -1).astype(np.int8)
    return team, np.where(ok, dist, np.nan)


def baseline_period(raw, W):
    """raw: -1 (no ball -> D), 0 home vote, 1 away vote. Centred majority vote +-W, ties to home."""
    hv = np.concatenate(([0], np.cumsum(raw == 0)))
    av = np.concatenate(([0], np.cumsum(raw == 1)))
    n = len(raw)
    i = np.arange(n)
    lo = np.clip(i - W, 0, n)
    hi = np.clip(i + W + 1, 0, n)
    h = hv[hi] - hv[lo]
    a = av[hi] - av[lo]
    out = np.where(h >= a, H, A).astype(np.int8)
    out[(h + a) == 0] = D
    return out


# ----------------------------------------------------------------------------- formatting

def md(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def pct(x, qs):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * len(qs)
    return list(np.percentile(x, qs))


def f(x, nd=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pc(x, nd=1):
    return f"{100 * x:.{nd}f}%"


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO.parent / "research" / "dfl-label-report.md"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted((REPO / "cache" / "external").glob("dfl_*.npz"))
    matches = [Match.load(p) for p in files]
    ids = [m.match_id.replace("DFL-MAT-", "") for m in matches]
    reports = {}
    for p, i in zip(files, ids):
        rp = p.with_suffix(".report.json")
        reports[i] = json.loads(rp.read_text()) if rp.exists() else {}
    print("matches:", ids, flush=True)

    parts_by_match = []
    for m in matches:
        sl = m.period_slices()
        parts_by_match.append([m.label[sl[p]] for p in sorted(sl)])
    gts = [join(p) for p in parts_by_match]
    runs_l = [pm.valid_runs(g) for g in gts]

    out = ["# DFL label report", "",
           f"Matches parsed: {', '.join(ids)} ({len(ids)} of 7). Label rule: `D if BallStatus==0 else (H if BallPossession==1 else A)`. "
           "Spells = maximal H or A runs per period (scorer `possession_spells`); frames at 25 fps. "
           "Pooled row = all matches pooled (spell lists concatenated, frame counts summed).", ""]

    # ---- data quality
    out += ["## Data quality and parsing", ""]
    dq = []
    for i in ids:
        r = reports[i]
        for p, v in sorted(r.get("periods", {}).items()):
            dq.append((i, p, v["T"], v["gaps_in_union"], v["duplicate_frame_records"],
                       v["ball_frames"], v["player_union_frames"], v["ball_not_in_players"],
                       v["players_not_in_ball"], v["n_ball_framesets"], v["n_player_framesets"],
                       r.get("parse_seconds")))
    out.append(md(["match", "period", "T", "gaps in N", "dup records", "ball frames", "player-union frames",
                   "ball not in players", "players not in ball", "ball FrameSets", "player FrameSets", "parse s"], dq))
    out.append("Home attacking direction (+1 = towards +x). Kickoff = mean x of each team over the first 5 s; "
               "deepest = side of the home team's per-frame extreme-x player over the whole period (keeper side), "
               "defence side inferred as the opposite of attack.\n")
    hd = []
    for i in ids:
        r = reports[i]
        for p, ev in sorted(r.get("home_dir_evidence", {}).items()):
            hd.append((i, p, ev["home_mean_x_first5s"], ev["away_mean_x_first5s"], ev["home_mean_min_x"],
                       ev["home_mean_max_x"], ev["kickoff_dir"], ev["deepest_player_dir"], ev["agree"]))
        if r:
            hd.append((i, "flips between halves", r.get("home_dir_flips_between_halves"), "", "", "", "", "", ""))
    out.append(md(["match", "period", "home mean x 5s", "away mean x 5s", "home mean min x", "home mean max x",
                   "kickoff dir", "deepest-player dir", "agree"], hd))
    out.append("Cross-tab BallPossession x BallStatus (frames):\n")
    ct = []
    for i in ids:
        tot = {}
        for p, c in reports[i].get("crosstab", {}).items():
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
        ct.append((i, tot.get("status1_poss1"), tot.get("status1_poss2"), tot.get("status0_poss1"),
                   tot.get("status0_poss2"), tot.get("other_or_missing")))
    out.append(md(["match", "alive & home(1)", "alive & away(2)", "dead & home(1)", "dead & away(2)", "other/missing"], ct))

    # ---- L1 / L2 / L3
    l1a, l1b, l1c = [], [], []
    l2, l3 = [], []
    pooled = {"spell_dur": [], "frames": np.zeros(4, dtype=np.int64), "per": [0, 0], "d_dur": [],
              "ho": 0, "ah": 0, "ad": 0, "da": 0, "d_same": 0, "d_int": 0}
    for m, gt, runs, i in zip(matches, gts, runs_l, ids):
        sl = m.period_slices()
        cnt = np.bincount(m.label, minlength=4)
        tot = cnt[:3].sum()
        spells = pm.possession_spells(gt, runs)
        dur = np.array([(b - a) / FPS for a, b, _ in spells])
        segs = pm.segments(gt, runs)
        d_dur = np.array([(b - a) / FPS for a, b, c in segs if c == D])
        alive_frames = cnt[0] + cnt[1]
        fr = [len(m.label[sl[p]]) if p in sl else 0 for p in (1, 2)]
        l1a.append((i, fr[0], fr[1], pc(cnt[0] / tot), pc(cnt[1] / tot), pc(cnt[2] / tot), len(spells),
                    f(dur.mean()), f(np.median(dur))))
        l1b.append((i, *[f(v) for v in pct(dur, [5, 25, 50, 75, 95])]))
        sh = [np.mean(dur < t) for t in (1, 2, 4)]
        shf = [np.sum(np.array([b - a for a, b, _ in spells])[dur < t]) / alive_frames for t in (1, 2, 4)]
        l1c.append((i, *[pc(v) for v in sh], *[pc(v) for v in shf]))
        l2.append((i, len(d_dur), *[f(v) for v in pct(d_dur, [5, 25, 50, 75, 95])],
                   pc(np.mean(d_dur < 2)), pc(np.mean(d_dur < 5))))
        ho = ah = ad = da = same = inte = 0
        for (a0, a1, c0), (b0, b1, c1) in zip(segs[:-1], segs[1:]):
            if a1 != b0:
                continue
            if c0 == H and c1 == A: ho += 1
            elif c0 == A and c1 == H: ah += 1
            elif c0 in (H, A) and c1 == D: ad += 1
            elif c0 == D and c1 in (H, A): da += 1
        for k in range(1, len(segs) - 1):
            if segs[k][2] == D and segs[k - 1][1] == segs[k][0] and segs[k][1] == segs[k + 1][0]:
                inte += 1
                same += segs[k - 1][2] == segs[k + 1][2]
        l3.append((i, ho, ah, ad, da, inte, same, pc(same / inte) if inte else "nan"))
        pooled["spell_dur"].append((dur, np.array([b - a for a, b, _ in spells])))
        pooled["frames"] += cnt
        pooled["d_dur"].append(d_dur)
        for k, v in (("ho", ho), ("ah", ah), ("ad", ad), ("da", da), ("d_same", same), ("d_int", inte)):
            pooled[k] += v
        pooled["per"][0] += fr[0]; pooled["per"][1] += fr[1]
    dur_all = np.concatenate([d for d, _ in pooled["spell_dur"]])
    len_all = np.concatenate([l for _, l in pooled["spell_dur"]])
    cnt = pooled["frames"]; tot = cnt[:3].sum(); alive_all = cnt[0] + cnt[1]
    l1a.append(("**pooled**", pooled["per"][0], pooled["per"][1], pc(cnt[0] / tot), pc(cnt[1] / tot), pc(cnt[2] / tot),
                len(dur_all), f(dur_all.mean()), f(np.median(dur_all))))
    l1b.append(("**pooled**", *[f(v) for v in pct(dur_all, [5, 25, 50, 75, 95])]))
    l1c.append(("**pooled**", *[pc(np.mean(dur_all < t)) for t in (1, 2, 4)],
                *[pc(len_all[dur_all < t].sum() / alive_all) for t in (1, 2, 4)]))
    dd = np.concatenate(pooled["d_dur"])
    l2.append(("**pooled**", len(dd), *[f(v) for v in pct(dd, [5, 25, 50, 75, 95])], pc(np.mean(dd < 2)), pc(np.mean(dd < 5))))
    l3.append(("**pooled**", pooled["ho"], pooled["ah"], pooled["ad"], pooled["da"], pooled["d_int"], pooled["d_same"],
               pc(pooled["d_same"] / pooled["d_int"])))
    out += ["## L1. Possession spells", "", md(["match", "frames P1", "frames P2", "H", "A", "D", "spells", "mean spell s", "median spell s"], l1a)]
    out += ["Spell duration percentiles (s):\n", md(["match", "p5", "p25", "p50", "p75", "p95"], l1b)]
    out += ["Short spells: share of spells, and share of all alive frames they cover:\n",
            md(["match", "spells <1s", "<2s", "<4s", "alive frames in <1s", "<2s", "<4s"], l1c)]
    out += ["## L2. Dead-ball (D) stretches", "",
            "Includes stretches touching a period edge (kick-off / final whistle).\n",
            md(["match", "count", "p5 s", "p25 s", "p50 s", "p75 s", "p95 s", "<2s", "<5s"], l2)]
    out += ["## L3. Transitions", "",
            "Counted between adjacent segments inside a period. D stretches: only those with an alive spell on both sides.\n",
            md(["match", "H->A", "A->H", "alive->D", "D->alive", "interior D stretches", "same team before/after", "share same"], l3)]

    # ---- L4 / L5 / L6 (frame-level arrays)
    l4a, l4b, l5a, l5b, l6 = [], [], [], [], []
    acc = {k: [] for k in ("v_alive", "v_dead", "pv_alive", "pv_dead", "cnt_h", "cnt_a", "dist", "nlab", "bz_alive")}
    acc.update({"nan": 0, "T": 0, "dead_out": 0, "dead_n": 0, "dead_slow": 0, "dead_slow_n": 0})
    near_cache = []
    for m, i in zip(matches, ids):
        alive = (m.label == H) | (m.label == A)
        dead = m.label == D
        L, W = m.pitch
        sl = m.period_slices()
        v = np.full(m.T, np.nan)
        pv = np.full((m.T, m.hx.shape[1] + m.ax.shape[1]), np.nan, dtype=np.float32)
        px = np.concatenate([m.hx, m.ax], axis=1); py = np.concatenate([m.hy, m.ay], axis=1)
        for p, s in sl.items():
            a, b = s.start, s.stop
            v[a + 1:b] = np.hypot(np.diff(m.bx[a:b]), np.diff(m.by[a:b])) * FPS
            pv[a + 1:b] = np.hypot(np.diff(px[a:b], axis=0), np.diff(py[a:b], axis=0)) * FPS
        mean_pv = np.nanmean(pv, axis=1)
        nan_ball = np.isnan(m.bx)
        bz_ok = alive & ~np.isnan(m.bz)
        out_dead = dead & ~nan_ball & ((np.abs(m.bx) > L / 2) | (np.abs(m.by) > W / 2))
        dn = (dead & ~nan_ball).sum()
        slow = dead & ~np.isnan(v) & (v < 0.5)
        dn2 = (dead & ~np.isnan(v)).sum()
        q_a = pct(v[alive], [50, 90, 99]); q_d = pct(v[dead], [50, 90, 99])
        l4a.append((i, pc(nan_ball.mean(), 2), pc((m.bz[bz_ok] > 1).mean(), 2) if bz_ok.any() else "nan",
                    pc((m.bz[bz_ok] > 2).mean(), 2) if bz_ok.any() else "nan",
                    pc(out_dead.sum() / dn) if dn else "nan", pc(slow.sum() / dn2) if dn2 else "nan"))
        l4b.append((i, *[f(x) for x in q_a], *[f(x) for x in q_d]))
        nh = (~np.isnan(m.hx)).sum(1); na = (~np.isnan(m.ax)).sum(1)
        l5a.append((i, nh.min(), int(np.median(nh)), nh.max(), na.min(), int(np.median(na)), na.max()))
        l5b.append((i, *[f(x) for x in pct(mean_pv[alive], [10, 50, 90])], *[f(x) for x in pct(mean_pv[dead], [10, 50, 90])]))
        team, dist = nearest(m)
        near_cache.append(team)
        ok = alive & (team >= 0)
        eq = team[ok] == m.label[ok]
        close = ok & (dist < 1.5)
        eqc = team[close] == m.label[close]
        l6.append((i, pc(eq.mean()), pc(eqc.mean()), pc(close.sum() / ok.sum()),
                   *[f(x) for x in pct(dist[ok], [50, 90])]))
        acc["v_alive"].append(v[alive]); acc["v_dead"].append(v[dead])
        acc["pv_alive"].append(mean_pv[alive]); acc["pv_dead"].append(mean_pv[dead])
        acc["cnt_h"].append(nh); acc["cnt_a"].append(na)
        acc["bz_alive"].append(m.bz[bz_ok])
        acc["nan"] += int(nan_ball.sum()); acc["T"] += m.T
        acc["dead_out"] += int(out_dead.sum()); acc["dead_n"] += int(dn)
        acc["dead_slow"] += int(slow.sum()); acc["dead_slow_n"] += int(dn2)
        acc.setdefault("eq", [0, 0, 0, 0]); acc["eq"][0] += int(eq.sum()); acc["eq"][1] += int(ok.sum())
        acc["eq"][2] += int(eqc.sum()); acc["eq"][3] += int(close.sum())
        acc["dist"].append(dist[ok])
    cat = lambda k: np.concatenate(acc[k])
    bz = cat("bz_alive")
    l4a.append(("**pooled**", pc(acc["nan"] / acc["T"], 2), pc((bz > 1).mean(), 2), pc((bz > 2).mean(), 2),
                pc(acc["dead_out"] / acc["dead_n"]), pc(acc["dead_slow"] / acc["dead_slow_n"])))
    l4b.append(("**pooled**", *[f(x) for x in pct(cat("v_alive"), [50, 90, 99])], *[f(x) for x in pct(cat("v_dead"), [50, 90, 99])]))
    nh_all, na_all = cat("cnt_h"), cat("cnt_a")
    l5a.append(("**pooled**", nh_all.min(), int(np.median(nh_all)), nh_all.max(), na_all.min(), int(np.median(na_all)), na_all.max()))
    l5b.append(("**pooled**", *[f(x) for x in pct(cat("pv_alive"), [10, 50, 90])], *[f(x) for x in pct(cat("pv_dead"), [10, 50, 90])]))
    e = acc["eq"]
    l6.append(("**pooled**", pc(e[0] / e[1]), pc(e[2] / e[3]), pc(e[3] / e[1]), *[f(x) for x in pct(cat("dist"), [50, 90])]))
    out += ["## L4. Ball", "",
            "Speed = planar displacement between consecutive frames x 25 (m/s), not the file's S field. "
            "bz shares are over alive frames with a non-NaN bz. Dead-ball shares are over dead frames with a ball position "
            "(slow: with a defined speed).\n",
            md(["match", "ball NaN", "bz>1 m (alive)", "bz>2 m (alive)", "dead & outside pitch", "dead & speed<0.5 m/s"], l4a),
            "Ball planar speed percentiles (m/s):\n",
            md(["match", "alive p50", "alive p90", "alive p99", "dead p50", "dead p90", "dead p99"], l4b)]
    out += ["## L5. Players", "",
            "Players tracked = non-NaN positions per frame. Player speed = per-player displacement x 25, averaged over tracked "
            "players of both teams per frame; percentiles are over frames.\n",
            md(["match", "home min", "home median", "home max", "away min", "away median", "away max"], l5a),
            md(["match", "alive p10", "alive p50", "alive p90", "dead p10", "dead p50", "dead p90"], l5b)]
    out += ["## L6. Nearest player vs label (alive frames)", "",
            md(["match", "nearest team == label", "same, nearest < 1.5 m", "share of frames with nearest < 1.5 m",
                "nearest dist p50 m", "p90 m"], l6)]

    # ---- L7 / L8
    print("sensitivity ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = sensitivity(parts_by_match, ex)
    out += ["## L7. Metric sensitivity (truth vs perturbed copy)", "",
            "Pooled over matches (summed spell TP/FP/FN per IoU threshold; summed confusion for macroF1). "
            "'mean of per-match' is the unpooled mean of per-match spellF1. Perturbations act per period. "
            "Shift fills the leading frames with the first label. Spell erasure is a single pass using original durations; "
            "a short spell with no preceding alive spell is left unchanged. Blip = 12 frames at a uniform position inside the spell. "
            "Random rows are the mean over 5 seeds (sd of pooled spellF1 across seeds shown).\n",
            md(["perturbation", "spellF1 (pooled)", "sd over seeds", "macroF1 (pooled)", "spellF1 (mean of per-match)"],
               [(r[0], f(r[1], 4), f(r[2], 4), f(r[3], 4), f(r[4], 4)) for r in rows])]

    base_rows = []
    pred_by_w = {}
    for Wn in (0, 25, 50, 125):
        res, preds0 = [], []
        for m, parts, nt in zip(matches, parts_by_match, near_cache):
            sl = m.period_slices()
            pp = [baseline_period(nt[sl[p]], Wn) for p in sorted(sl)]
            res.append(score_arrays(join(parts), join(pp)))
            preds0.append(join(pp))
        pld = pool(res)
        pred_by_w[Wn] = preds0
        npred = sum(len(pm.possession_spells(pr, pm.valid_runs(g))) for pr, g in zip(preds0, gts))
        ngt = sum(len(pm.possession_spells(g, pm.valid_runs(g))) for g in gts)
        base_rows.append((Wn, f(pld["spell"], 4), f(pld["macro"], 4), f(pld["mean_match_spell"], 4), ngt, npred,
                          ", ".join(f(r[2], 3) for r in res)))
    # cross-check with the scorer's own evaluate()
    chk = pm.evaluate(gts[0], pred_by_w[50][0])["spell_f1_mean"]
    mine = score_arrays(gts[0], pred_by_w[50][0])[2]
    out += ["## L8. Organisers' baseline on clean DFL", "",
            "Nearest tracked player to the ball -> team; ball NaN -> D; centred majority vote over +-W frames (ties to home; "
            "no votes -> D). Reference: same baseline scores spellF1 0.2433 at W=50 on the target match "
            "(tracking there is noisy and the ball is missing on ~41% of frames).\n",
            md(["W", "spellF1 (pooled)", "macroF1 (pooled)", "spellF1 (mean of per-match)", "GT spells", "pred spells", "per-match spellF1"], base_rows),
            f"Cross-check: scorer `evaluate()` spell_f1_mean on the first match at W=50 = {chk:.6f}; this script = {mine:.6f}.\n"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
