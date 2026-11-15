"""v1 grouped 5-fold cross-validation over the 10 external matches (7 DFL, 3 Metrica sample games).

    uv run python scripts/cv_external.py oof     # train folds, save out-of-fold probabilities (resumable)
    uv run python scripts/cv_external.py tune    # tune the decoder on the pooled degraded OOF, write the report

Deterministic folds hold out two matches each, the three Metrica games in different folds. Held-out matches are
predicted clean and degraded (held-out seed, mid-range degradation parameters). Nothing here touches the
competition match.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import canon, evalutil, model  # noqa: E402
from possession.decode import DecoderParams, decode  # noqa: E402
from possession.degrade import default_params, degrade  # noqa: E402
from possession.external_list import external_ids, load_external  # noqa: E402
from possession.features import extract  # noqa: E402

OOF = ROOT / "cache" / "oof_v1"
REPORT = ROOT.parent / "research" / "cv-report-v1.md"
SEED = 11
K_COPIES = 2
FOLDS = [(0, 7), (1, 8), (2, 9), (3, 4), (5, 6)]       # indices into external_ids(): 0-6 DFL, 7-9 Metrica
ES_HOLD = (1, 8)                                        # early-stopping validation matches (fold 0 trains without them)
V0 = {"clean": {"baseline": 0.3081, "argmax": 0.4882, "viterbi_const": 0.7135, "viterbi_learned": 0.7191},
      "deg": {"baseline": 0.2989, "argmax": 0.4498, "viterbi_const": 0.6503, "viterbi_learned": 0.6562}}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def heldout_versions(m, i):
    return {"clean": m, "deg": degrade(m, np.random.default_rng([999, i]), default_params(), mid=True)}


def phase_oof():
    ids = external_ids()
    matches = [canon.canonicalise(m) for m in load_external()]
    t = time.time()
    per_match, names = [], None
    for i, m in enumerate(matches):
        sets, names = model.make_training_sets([m], seed=SEED * 100 + i, k_copies=K_COPIES)
        per_match.append(sets)
    log(f"training sets built {time.time() - t:.0f}s ({len(names)} features, {len(model.ad_columns(names))} in the restricted set)")
    OOF.mkdir(parents=True, exist_ok=True)
    tf = OOF / "trees.json"
    if tf.exists():
        st = json.loads(tf.read_text())
    else:
        t = time.time()
        hold = {0, 7, *ES_HOLD}
        tr = [s for j, ss in enumerate(per_match) if j not in hold for s in ss]
        va = [s for j in ES_HOLD for s in per_match[j]]
        mm = model.fit(tr, names, 1000, valid=va)
        st = dict(trees={k: int(v * 1.15) + 10 for k, v in mm.best.items()}, best=mm.best,
                  early_stopping_fit_s=time.time() - t, fold_fit_s={})
        tf.write_text(json.dumps(st))
        log("early stopping", st["best"], f"{st['early_stopping_fit_s']:.0f}s")
    nt = st["trees"]
    for fi, fold in enumerate(FOLDS):
        if all((OOF / f"{ids[i]}_deg.npz").exists() for i in fold):
            continue
        t = time.time()
        tr = [s for j, ss in enumerate(per_match) if j not in fold for s in ss]
        mdl = model.fit(tr, names, nt)
        fit_s = time.time() - t
        ad_all = None
        if fi == 0:                         # honesty check: AD with all features
            ad_all, _ = model.train_one("ad_all", tr, names, nt["ad"])
        for i in fold:
            for cond, mv in heldout_versions(matches[i], i).items():
                X, _ = extract(mv)
                ps, pt = mdl.predict(X)
                extra = {} if ad_all is None else {"p_alive_all": ad_all.predict(X, num_threads=model.NUM_THREADS).astype(np.float32)}
                np.savez_compressed(OOF / f"{ids[i]}_{cond}.npz", p_state=ps.astype(np.float32), p_trans=pt.astype(np.float32), **extra)
        st["fold_fit_s"][str(fi)] = fit_s
        tf.write_text(json.dumps(st))
        log(f"fold {fi} {[ids[i] for i in fold]} fit {fit_s:.0f}s total {time.time() - t:.0f}s")
    (OOF / "meta.json").write_text(json.dumps(dict(trees=nt, ids=ids, n_features=len(names), n_ad_features=len(model.ad_columns(names)),
                                                   k_copies=K_COPIES, folds=FOLDS, early_stopping_fit_s=st["early_stopping_fit_s"],
                                                   fold_fit_s=list(st["fold_fit_s"].values()))))


# ------------------------------------------------------------------ tuning / scoring
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


def score_params(items, prm, use_trans=True):
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


def tune_learned(items, init, n_random=300, seed=0):
    rng = np.random.default_rng(seed)
    best_p, best = init, score_params(items, init)
    log(f"learned-transition start from constant optimum: {best:.4f}")
    for it in range(n_random):
        if it <= n_random // 3:
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


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


SYSTEMS = ("baseline", "argmax", "viterbi_const", "viterbi_learned")


def predict_system(name, m, ps, pt, prm_c, prm_l):
    if name == "baseline":
        return evalutil.baseline_labels(m)
    if name == "argmax":
        return np.where(m.period > 0, ps.argmax(1), 3).astype(np.int8)
    if name == "viterbi_const":
        return decode(ps, None, m.period, prm_c)
    return decode(ps, pt, m.period, prm_l)


def ad_counts(items, p_alive=None):
    """Dead-class precision/recall, accuracy and frame count, split by ball present / missing (raw AD, threshold 0.5)."""
    c = {k: np.zeros(5) for k in ("present", "missing")}      # tp, fp, fn, correct, n
    for idx, (m, ps, pt) in enumerate(items):
        pa = (1 - ps[:, 2]) if p_alive is None else p_alive[idx]
        g = (m.period > 0)
        dead, pd_ = (m.label == 2), pa < 0.5
        pres = np.isfinite(m.bx)
        for k, sel in (("present", g & pres), ("missing", g & ~pres)):
            c[k] += [(dead & pd_ & sel).sum(), (~dead & pd_ & sel).sum(), (dead & ~pd_ & sel).sum(), ((dead == pd_) & sel).sum(), sel.sum()]
    return {k: dict(precision=v[0] / max(v[0] + v[1], 1), recall=v[0] / max(v[0] + v[2], 1), acc=v[3] / max(v[4], 1), n=int(v[4]),
                    dead_share=(v[0] + v[2]) / max(v[4], 1)) for k, v in c.items()}


def team_acc(items):
    r = {"present": [0, 0], "missing": [0, 0]}
    for m, ps, pt in items:
        sel0 = (m.period > 0) & (m.label < 2)
        pres = np.isfinite(m.bx)
        ok = (ps[:, 0] > ps[:, 1]) == (m.label == 0)
        for k, sel in (("present", sel0 & pres), ("missing", sel0 & ~pres)):
            r[k][0] += ok[sel].sum(); r[k][1] += sel.sum()
    return {k: (a / max(b, 1), b) for k, (a, b) in r.items()}


def phase_tune():
    ids, oof = load_oof()
    meta = json.loads((OOF / "meta.json").read_text())
    sources = ["metrica" if i.startswith("metrica") else "dfl" for i in ids]
    t = time.time()
    items = list(oof["deg"])
    prm_c, s_c = tune_constant(items)
    log(f"constant-cost optimum (degraded OOF, 10 matches): {s_c:.4f} {prm_c}")
    prm_l, s_l = tune_learned(items, prm_c)
    t_tune = time.time() - t
    log(f"learned optimum {s_l:.4f} {prm_l}  ({t_tune:.0f}s)")
    (ROOT / "models").mkdir(exist_ok=True)
    prm_l.to_json(ROOT / "models" / "decoder.json")
    prm_c.to_json(OOF / "decoder_const.json")

    groups = (("pooled (10)", list(range(len(ids)))), ("DFL (7)", [i for i, s in enumerate(sources) if s == "dfl"]),
              ("Metrica (3)", [i for i, s in enumerate(sources) if s == "metrica"]))
    out = ["# Cross-validation report v1 (grouped 5-fold over 10 external matches)\n",
           f"Matches: {', '.join(ids)} (7 DFL with labels coarsened by merge_short_spells sandwich 1.0 s; 3 Metrica sample games with the "
           f"ball filled across stoppages). Folds hold out two matches each: {[[ids[i] for i in f] for f in meta['folds']]}. "
           f"Per fold: the other 8 matches x {meta['k_copies']} degraded copies x 2 (team swap). {meta['n_features']} features, of which "
           f"{meta['n_ad_features']} in the restricted AD/A2D/D2A set. Trees (early stopping on matches {[ids[i] for i in ES_HOLD]}, x1.15+10): {meta['trees']}. "
           "Scores from `possession_metrics.evaluate`, averaged over held-out matches. Held-out 'degraded' = held-out seed, mid-range degradation parameters. "
           "The decoder was tuned on the pooled degraded OOF predictions (about 10 free parameters, 10 matches), so degraded decoded numbers are mildly optimistic.\n",
           "**v1 is expected to score lower than v0.** v0's 0.719 / 0.656 (7 DFL matches, raw labels, leave-one-match-out) benefited from the frozen-ball "
           "artefact (ball motion or proximity identifies alive play with certainty in DFL), which does not exist in the competition match. v1's AD model "
           "cannot see that artefact. Labels are also coarsened in v1 (fewer, longer possessions), so numbers are not strictly comparable.\n"]
    summary = {}
    rows_cache = {}
    for cond in ("clean", "deg"):
        res = {n: [] for n in SYSTEMS}
        for m, ps, pt in oof[cond]:
            for n in SYSTEMS:
                pred = predict_system(n, m, ps, pt, prm_c, prm_l)
                gt, pr = evalutil.eval_arrays(m, pred)
                res[n].append((evalutil.full_eval(gt, pr), pred))
        rows_cache[cond] = res
        for grp, sel in groups:
            out.append(f"\n## {grp}, {cond}\n")
            tab = []
            for n in SYSTEMS:
                rr = [res[n][i][0] for i in sel]
                sp, mf = np.mean([r["spell_f1_mean"] for r in rr]), np.mean([r["macro_f1"] for r in rr])
                npred, ngt = np.sum([r["n_pred_spells"] for r in rr]), np.sum([r["n_gt_spells"] for r in rr])
                v0 = V0[cond][n] if grp.startswith("DFL") else None
                tab.append([n, f"{sp:.4f}", f"{mf:.4f}", f"{npred} / {ngt}", "-" if v0 is None else f"{v0:.4f}"])
                summary[(cond, grp, n)] = (sp, mf)
            out.append(md_table(["system", "spellF1", "macroF1", "pred / true possessions (sum)", "v0 spellF1 (DFL-only LOMO, raw labels)"], tab))
        out.append(f"\n\nPer match spellF1 / macroF1 ({cond}):\n")
        out.append(md_table(["match"] + list(SYSTEMS), [[ids[i]] + [f"{res[n][i][0]['spell_f1_mean']:.3f} / {res[n][i][0]['macro_f1']:.3f}" for n in SYSTEMS] for i in range(len(ids))]))
        for grp, sel in groups:
            best = [res["viterbi_learned"][i][0] for i in sel]
            out.append(f"\n\n### Learned-transition Viterbi ({cond}, {grp}): per-IoU F1\n")
            out.append(md_table(["IoU"] + [str(k) for k in best[0]["spell_f1"]], [["F1"] + [f"{np.mean([r['spell_f1'][k]['f1'] for r in best]):.3f}" for k in best[0]["spell_f1"]]]))
            trows = [[str(tol)] + [f"{np.mean([r['transitions'][tol][q] for r in best]):.3f}" for q in ("precision", "recall")] for tol in (12, 25)]
            out.append("\n\nTransition precision / recall (matched on destination):\n\n" + md_table(["tol (frames)", "precision", "recall"], trows))
            pc = {c: [np.mean([r["per_class"][c][q] for r in best]) for q in ("precision", "recall")] for c in ("H", "A", "D")}
            out.append("\n\nPer-class precision / recall: " + ", ".join(f"{c} {v[0]:.3f}/{v[1]:.3f}" for c, v in pc.items()) + "\n")

    out.append("\n## AD (alive vs dead) and TEAM models, held-out DEGRADED, raw per-frame (threshold 0.5)\n")
    for grp, sel in groups:
        its = [oof["deg"][i] for i in sel]
        a, tm = ad_counts(its), team_acc(its)
        out.append(f"\n**{grp}**\n")
        out.append(md_table(["ball", "frames", "true dead share", "dead precision", "dead recall", "AD accuracy", "H-vs-A accuracy on alive"],
                            [[k, str(a[k]["n"]), f"{a[k]['dead_share']:.3f}", f"{a[k]['precision']:.3f}", f"{a[k]['recall']:.3f}", f"{a[k]['acc']:.3f}", f"{tm[k][0]:.3f}"] for k in ("present", "missing")]))
    out.append("\n\n### Decoded (learned Viterbi, degraded): frame accuracy by ball presence\n")
    acc = {"present": [0, 0], "missing": [0, 0]}
    for (m, ps, pt), (_, pred) in zip(oof["deg"], rows_cache["deg"]["viterbi_learned"]):
        g = m.period > 0
        pres = np.isfinite(m.bx)
        for k, sel in (("present", g & pres), ("missing", g & ~pres)):
            acc[k][0] += (pred[sel] == m.label[sel]).sum(); acc[k][1] += sel.sum()
    out.append(", ".join(f"ball {k}: {a / max(b, 1):.3f} ({b} frames)" for k, (a, b) in acc.items()) + "\n")

    f0 = meta["folds"][0]
    out.append("\n## Cost of removing the frozen-ball artefact: AD with ALL features vs restricted (fold 0 held-out: " + ", ".join(ids[i] for i in f0) + ")\n")
    for cond in ("clean", "deg"):
        its = [oof[cond][i] for i in f0]
        pal = [np.load(OOF / f"{ids[i]}_{cond}.npz")["p_alive_all"] for i in f0]
        ra, rb = ad_counts(its), ad_counts(its, pal)
        out.append(f"\n**{cond}**\n")
        out.append(md_table(["AD model", "ball", "frames", "dead precision", "dead recall", "accuracy"],
                            [[nm, k, str(r[k]["n"]), f"{r[k]['precision']:.3f}", f"{r[k]['recall']:.3f}", f"{r[k]['acc']:.3f}"]
                             for nm, r in (("restricted", ra), ("all features", rb)) for k in ("present", "missing")]))
    out.append("\n(Fold 0 holds one DFL and one Metrica game; the all-feature model gains most on the DFL game, where the ball is frozen when dead.)\n")

    out.append("\n## Tuned decoder parameters (pooled degraded OOF)\n")
    out.append(f"- constant-cost Viterbi: c(team, alive->D, D->alive) = {prm_c.c}, alpha=1 (spellF1 {s_c:.4f} on the tuning pool)\n")
    out.append(f"- learned-transition Viterbi: alpha={prm_l.alpha}, beta={prm_l.beta}, c={prm_l.c}, L={prm_l.L} (spellF1 {s_l:.4f} on the tuning pool)\n")
    out.append(f"- gain from learned transitions on the tuning pool: {s_l - s_c:+.4f}\n")
    out.append(f"\n## Timings\n\n- early-stopping fit {meta['early_stopping_fit_s']:.0f}s; per-fold fit (5 models) {np.mean(meta['fold_fit_s']):.0f}s mean; decoder tuning {t_tune:.0f}s\n")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(out))
    log("report written", REPORT)
    print({f"{k[0]}:{k[1]}:{k[2]}": tuple(round(x, 4) for x in v) for k, v in summary.items() if k[2] in ("viterbi_learned", "baseline")})


if __name__ == "__main__":
    {"oof": phase_oof, "tune": phase_tune}[sys.argv[1]]()
