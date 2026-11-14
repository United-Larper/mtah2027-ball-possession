"""Leave-one-match-out cross-validation on the external (DFL) matches.

    uv run python scripts/cv_external.py oof              # train folds, save out-of-fold probabilities
    uv run python scripts/cv_external.py tune             # tune decoders on the OOF, write the report
    uv run python scripts/cv_external.py ablate           # feature-group ablation (3 grouped folds, degraded)

Held-out matches are predicted both clean and degraded (held-out seed, mid-range degradation
parameters). Nothing here touches the competition match.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import canon, evalutil, model  # noqa: E402
from possession.decode import DecoderParams, decode  # noqa: E402
from possession.degrade import default_params, degrade  # noqa: E402
from possession.external_list import external_ids, load_external  # noqa: E402
from possession.features import extract  # noqa: E402

OOF = ROOT / "cache" / "oof"
REPORT = ROOT.parent / "research" / "cv-report.md"
SEED = 11
K_COPIES = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def heldout_versions(m, i):
    """(clean, degraded) canonical versions of a held-out match."""
    return {"clean": m, "deg": degrade(m, np.random.default_rng([999, i]), default_params(), mid=True)}


def build_all(matches):
    t = time.time()
    per_match, names = [], None
    for i, m in enumerate(matches):
        sets, names = model.make_training_sets([m], seed=SEED * 100 + i, k_copies=K_COPIES)
        per_match.append(sets)
        log(f"training sets for match {i} built ({sum(len(s.X) for s in sets)} rows)")
    log(f"all training sets {time.time() - t:.0f}s")
    return per_match, names


def predict_match(models, m):
    X, _ = extract(m)
    keep = m.period > 0
    ps, pt = models.predict(X)
    return ps.astype(np.float32), pt.astype(np.float32)


def phase_oof():
    ids = external_ids()
    matches = [canon.canonicalise(m) for m in load_external()]
    per_match, names = build_all(matches)
    OOF.mkdir(parents=True, exist_ok=True)
    timings = {}
    # --- number of trees by early stopping on one training match (match 1 validates fold 0); resumable
    tf = OOF / "trees.json"
    if tf.exists():
        st = json.loads(tf.read_text())
    else:
        t = time.time()
        tr = [s for j, ss in enumerate(per_match) if j not in (0, 1) for s in ss]
        mm = model.fit(tr, names, 1000, 1000, valid=per_match[1])
        st = dict(n_state=int(mm.best[0] * 1.15) + 10, n_trans=int(mm.best[1] * 1.15) + 10, early_stopping_fit_s=time.time() - t, fold_fit_s={})
        tf.write_text(json.dumps(st))
    n_state, n_trans = st["n_state"], st["n_trans"]
    timings["early_stopping_fit_s"] = st["early_stopping_fit_s"]
    for i, m in enumerate(matches):
        if (OOF / f"{ids[i]}_deg.npz").exists():
            continue
        t = time.time()
        tr = [s for j, ss in enumerate(per_match) if j != i for s in ss]
        mdl = model.fit(tr, names, n_state, n_trans)
        st["fold_fit_s"][ids[i]] = time.time() - t
        for cond, mv in heldout_versions(m, i).items():
            ps, pt = predict_match(mdl, mv)
            np.savez_compressed(OOF / f"{ids[i]}_{cond}.npz", p_state=ps, p_trans=pt)
        tf.write_text(json.dumps(st))
        log(f"fold {i} ({ids[i]}) fit {st['fold_fit_s'][ids[i]]:.0f}s")
    timings["fold_fit_s"] = list(st["fold_fit_s"].values())
    (OOF / "meta.json").write_text(json.dumps(dict(n_state=n_state, n_trans=n_trans, timings=timings, ids=ids, n_features=len(names),
                                                   k_copies=K_COPIES)))


# ------------------------------------------------------------------ tuning
def load_oof():
    ids = external_ids()
    matches = [canon.canonicalise(m) for m in load_external()]
    out = {"clean": [], "deg": []}
    for i, m in enumerate(matches):
        vers = heldout_versions(m, i)
        for cond in out:
            z = np.load(OOF / f"{ids[i]}_{cond}.npz")
            out[cond].append((vers[cond], z["p_state"], z["p_trans"]))
    return ids, out


def score_params(items, prm, fast=True, use_trans=True):
    s = []
    for m, ps, pt in items:
        pred = decode(ps, pt if use_trans else None, m.period, prm)
        gt, pr = evalutil.eval_arrays(m, pred)
        s.append(evalutil.fast_spell_f1(gt, pr))
    return float(np.mean(s))


def tune_constant(items):
    best, bp = -1, None
    grid = [3, 6, 10, 15, 22, 30, 45, 65]
    for c1 in grid:
        for c2 in grid:
            for c3 in grid:
                prm = DecoderParams(alpha=1.0, beta=(0, 0, 0), c=(c1, c2, c3))
                s = score_params(items, prm, use_trans=False)
                if s > best:
                    best, bp = s, prm
    return bp, best


def tune_learned(items, init: DecoderParams, n_random=300, seed=0):
    rng = np.random.default_rng(seed)
    best_p, best = init, score_params(items, init)
    log(f"learned-transition start from constant optimum: {best:.4f}")
    cand = lambda p: DecoderParams(**p)  # noqa: E731
    for it in range(n_random):
        loc = it > n_random // 3
        if not loc:
            prm = DecoderParams(alpha=float(rng.choice([0.5, 1, 2])), beta=tuple(rng.uniform(0, 2.5, 3).round(2)),
                                c=tuple(rng.uniform(0, 60, 3).round(1)), L=float(rng.choice([4, 6, 8, 10])))
        else:
            b = best_p
            prm = DecoderParams(alpha=float(np.clip(b.alpha * np.exp(rng.normal(0, .15)), .25, 4)),
                                beta=tuple(np.clip(np.array(b.beta) + rng.normal(0, .25, 3), 0, 4).round(2)),
                                c=tuple(np.clip(np.array(b.c) + rng.normal(0, 4, 3), 0, 100).round(1)), L=b.L)
        s = score_params(items, prm)
        if s > best:
            best, best_p = s, prm
            log(f"  it {it} {best:.4f} {best_p}")
    return best_p, best


def systems_table(items, prm_c, prm_l, ids):
    rows = {}
    for name in ("baseline", "argmax", "viterbi_const", "viterbi_learned"):
        res = []
        for m, ps, pt in items:
            if name == "baseline":
                pred = evalutil.baseline_labels(m)
            elif name == "argmax":
                pred = np.where(m.period > 0, ps.argmax(1), 3).astype(np.int8)
            elif name == "viterbi_const":
                pred = decode(ps, None, m.period, prm_c)
            else:
                pred = decode(ps, pt, m.period, prm_l)
            gt, pr = evalutil.eval_arrays(m, pred)
            res.append((evalutil.full_eval(gt, pr), pred))
        rows[name] = res
    return rows


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def phase_tune():
    ids, oof = load_oof()
    meta = json.loads((OOF / "meta.json").read_text())
    t = time.time()
    items = oof["deg"]
    prm_c, s_c = tune_constant(items)
    log(f"constant-cost optimum (degraded OOF): {s_c:.4f} {prm_c}")
    prm_l, s_l = tune_learned(items, prm_c)
    t_tune = time.time() - t
    log(f"learned optimum {s_l:.4f} {prm_l}  ({t_tune:.0f}s)")
    prm_l.to_json(ROOT / "models" / "decoder.json") if (ROOT / "models").exists() else None
    (ROOT / "models").mkdir(exist_ok=True)
    prm_l.to_json(ROOT / "models" / "decoder.json")
    prm_c.to_json(OOF / "decoder_const.json")

    out = ["# Cross-validation report (external DFL matches, leave-one-match-out)\n",
           f"{len(ids)} matches: {', '.join(ids)}. Training per fold: the other matches x {meta['k_copies']} degraded copies x 2 (team swap). "
           f"{meta['n_features']} features; trees state/transition {meta['n_state']}/{meta['n_trans']}. All scores from `possession_metrics.evaluate`, "
           "averaged over held-out matches (each scored as one match; halves separated by a gap).\n",
           "Held-out 'degraded' = degraded copy with a held-out seed and mid-range degradation parameters. Decoders were tuned on the degraded OOF "
           "predictions only, so the degraded numbers are mildly optimistic (about 10 free parameters, 7 matches).\n"]
    summary = {}
    for cond in ("clean", "deg"):
        rows = systems_table(oof[cond], prm_c, prm_l, ids)
        out.append(f"\n## Pooled ({cond})\n")
        tab = []
        for name, res in rows.items():
            sp = np.mean([r[0]["spell_f1_mean"] for r in res]); mf = np.mean([r[0]["macro_f1"] for r in res])
            npred = np.mean([r[0]["n_pred_spells"] for r in res]); ngt = np.mean([r[0]["n_gt_spells"] for r in res])
            tab.append([name, f"{sp:.4f}", f"{mf:.4f}", f"{npred:.0f} / {ngt:.0f}"])
            summary[(cond, name)] = (sp, mf)
        out.append(md_table(["system", "spellF1", "macroF1", "pred / true possessions"], tab))
        out.append(f"\n\nPer match spellF1 / macroF1 ({cond}):\n")
        out.append(md_table(["match"] + list(rows), [[ids[i]] + [f"{rows[n][i][0]['spell_f1_mean']:.3f} / {rows[n][i][0]['macro_f1']:.3f}" for n in rows] for i in range(len(ids))]))
        best = rows["viterbi_learned"]
        res = [r[0] for r in best]
        out.append(f"\n\n### Best decoder ({cond}): per-IoU F1\n")
        out.append(md_table(["IoU"] + [str(k) for k in res[0]["spell_f1"]], [["F1"] + [f"{np.mean([r['spell_f1'][k]['f1'] for r in res]):.3f}" for k in res[0]["spell_f1"]]]))
        trows = []
        for tol in (12, 25):
            trows.append([str(tol)] + [f"{np.mean([r['transitions'][tol][q] for r in res]):.3f}" for q in ("precision", "recall")])
        out.append("\n\nTransition precision / recall (matched on destination):\n\n" + md_table(["tol (frames)", "precision", "recall"], trows))
        pc = {c: [np.mean([r["per_class"][c][q] for r in res]) for q in ("precision", "recall")] for c in ("H", "A", "D")}
        out.append("\n\nPer-class precision / recall: " + ", ".join(f"{c} {v[0]:.3f}/{v[1]:.3f}" for c, v in pc.items()))
        # accuracy by ball presence
        acc = {"present": [0, 0], "missing": [0, 0]}
        for (m, ps, pt), (_, pred) in zip(oof[cond], best):
            g = (m.period > 0)
            pres = np.isfinite(m.bx)
            for k, sel in (("present", g & pres), ("missing", g & ~pres)):
                acc[k][0] += (pred[sel] == m.label[sel]).sum(); acc[k][1] += sel.sum()
        out.append("\n\nFrame accuracy, learned Viterbi: " + ", ".join(f"ball {k}: {a / max(b, 1):.3f} ({b} frames)" for k, (a, b) in acc.items()))
        # alive-only team accuracy by presence
        acc2 = {}
        for k in ("present", "missing"):
            a_ = b_ = 0
            for (m, ps, pt), (_, pred) in zip(oof[cond], best):
                g = (m.period > 0) & (m.label < 2) & (pred < 2)
                sel = g & (np.isfinite(m.bx) if k == "present" else ~np.isfinite(m.bx))
                a_ += (pred[sel] == m.label[sel]).sum(); b_ += sel.sum()
            acc2[k] = a_ / max(b_, 1)
        out.append("\n\nH-vs-A accuracy on frames both alive: " + ", ".join(f"ball {k}: {v:.3f}" for k, v in acc2.items()) + "\n")
    out.append("\n## Tuned decoder parameters (degraded OOF)\n")
    out.append(f"- constant-cost Viterbi: c(team, alive->D, D->alive) = {prm_c.c}, alpha=1 (spellF1 {s_c:.4f} on the tuning pool)\n")
    out.append(f"- learned-transition Viterbi: alpha={prm_l.alpha}, beta={prm_l.beta}, c={prm_l.c}, L={prm_l.L} (spellF1 {s_l:.4f} on the tuning pool)\n")
    out.append(f"- gain from learned transitions on the tuning pool: {s_l - s_c:+.4f}\n")
    out.append(f"\n## Timings\n\n- early-stopping fit {meta['timings']['early_stopping_fit_s']:.0f}s; per-fold fit "
               f"{np.mean(meta['timings']['fold_fit_s']):.0f}s mean; decoder tuning {t_tune:.0f}s\n")
    REPORT.parent.mkdir(exist_ok=True)
    existing = REPORT.read_text() if REPORT.exists() else ""
    marker = "<!-- CALIBRATION/ABLATION APPENDIX -->"
    tail = existing[existing.index(marker):] if marker in existing else ""
    REPORT.write_text("\n".join(out) + "\n" + tail)
    log("report written", REPORT)
    print({f"{k[0]}:{k[1]}": tuple(round(x, 4) for x in v) for k, v in summary.items()})


# ------------------------------------------------------------------ ablation
GROUPS = {"touch": ["touch_"], "carry": ["carry_"], "shape": ["shape_"], "context": ["ctx_"], "none": []}


def phase_ablate():
    ids = external_ids()
    matches = [canon.canonicalise(m) for m in load_external()]
    meta = json.loads((OOF / "meta.json").read_text())
    per_match, names = build_all(matches)
    prm = DecoderParams.from_json(ROOT / "models" / "decoder.json")
    fold_of = [i % 3 for i in range(len(matches))]
    res = {}
    for g, prefixes in GROUPS.items():
        scores, t = [], time.time()
        for f in range(3):
            tr = [s for j, ss in enumerate(per_match) if fold_of[j] != f for s in ss]
            mdl = model.fit(tr, names, meta["n_state"] // 2, meta["n_trans"] // 2, drop_cols=prefixes)
            for i in [j for j in range(len(matches)) if fold_of[j] == f]:
                mv = heldout_versions(matches[i], i)["deg"]
                X, _ = extract(mv)
                ps, pt = mdl.predict(X)
                pred = decode(ps, pt, mv.period, prm)
                gt, pr = evalutil.eval_arrays(mv, pred)
                r = evalutil.full_eval(gt, pr)
                scores.append((r["spell_f1_mean"], r["macro_f1"]))
        res[g] = tuple(np.mean(scores, 0))
        log(f"ablate drop {g}: spell {res[g][0]:.4f} macro {res[g][1]:.4f} ({time.time() - t:.0f}s)")
    (OOF / "ablation.json").write_text(json.dumps(res))


if __name__ == "__main__":
    {"oof": phase_oof, "tune": phase_tune, "ablate": phase_ablate}[sys.argv[1]]()
