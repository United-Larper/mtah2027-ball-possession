"""Run the v1 system on the competition match and write outputs/RickRolled.csv (+ outputs/variants/v1_noaux.csv).

    uv run python scripts/run.py [--refit-aux]

Pipeline: fused target (tactical ball filled from the live broadcast ball) -> canonicalise -> features -> hierarchical
models -> optional auxiliary fusion (tables in models/aux_tables.json; --refit-aux recomputes and rewrites them) ->
Viterbi decode with the kick-off rule. Prints aggregates only. The public-window numbers are a SANITY check of
alignment and label meaning; they are never used to choose anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import aux_fit, canon, evalutil, fusion  # noqa: E402
from possession.aux_streams import build_aux  # noqa: E402
from possession.decode import DecoderParams, decode  # noqa: E402
from possession.features import extract  # noqa: E402
from possession.io_tags import load_tags  # noqa: E402
from possession.model import Models  # noqa: E402
from possession.target import load_target_match  # noqa: E402
from possession.target_fused import load_target_match_fused  # noqa: E402

pm = evalutil.pm
OUT = ROOT / "outputs" / "RickRolled.csv"
VAR = ROOT / "outputs" / "variants"
TABLES = ROOT / "models" / "aux_tables.json"
LETTERS = np.array(["H", "A", "D", "X"])
N_FRAMES = 199319
HARD = -1.0e3          # hard-constraint emission penalty


def write_csv(path: Path, pred: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("frame,label\n" + "\n".join(f"{i},{LETTERS[c]}" for i, c in enumerate(pred)) + "\n", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_codes(path: Path) -> np.ndarray:
    return pm.label_codes(pm.read_label_csv(path))


def kickoff_mask(m) -> tuple[np.ndarray, dict]:
    """Frames of each graded half before the start of that half's 'Kick Off (Start)' SET PIECES window (dead ball)."""
    tg = load_tags()
    ko = tg[(tg.code.str.upper() == "SET PIECES") & (tg.tags.map(lambda d: d.get("Type")) == "Kick Off (Start)")]
    mask = np.zeros(m.T, bool)
    info = {}
    for p, sl in m.period_slices().items():
        cand = ko[(ko.start_frame < sl.stop) & (ko.end_frame > sl.start)].sort_values("start_frame")
        if len(cand):
            s = int(cand.iloc[0].start_frame)
            lo = sl.start
            if s > lo:
                mask[lo:s] = True
            info[p] = int(mask[sl].sum())
        else:
            info[p] = 0
    return mask, info


def possession_stats(pred, m):
    parts = []
    for p, sl in m.period_slices().items():
        sp = pm.possession_spells(pred, [(sl.start, sl.stop)])
        dur = [(e - s) / 25 for s, e, _ in sp]
        parts.append(f"half {p}: {len(sp)} poss, mean {np.mean(dur):.1f}s median {np.median(dur):.1f}s")
    return "; ".join(parts)


def report(name, pred, m, tact_present, bstate):
    g = m.period > 0
    table = pm.read_label_csv(VAR / f"{name}.csv" if name != "v1" else OUT)
    reason = pm.validate_submission(table, N_FRAMES)
    print(f"[{name}] validation: {'OK' if reason is None else 'FAIL ' + str(reason)}; shares H/A/D {[round(float((pred[g] == k).mean()), 3) for k in range(3)]}")
    print(f"[{name}] {possession_stats(pred, m)}")
    print(f"[{name}] D share: tactical ball present {float((pred[g & tact_present] == 2).mean()):.3f} | missing {float((pred[g & ~tact_present] == 2).mean()):.3f}"
          f" | by bstate {[(v, round(float((pred[g & (bstate == v)] == 2).mean()), 3)) for v in range(5)]}")


def sanity(name, pred, raw, tact_present):
    gt = raw.label.astype(np.int8)
    r = pm.evaluate(gt, pred)
    print(f"SANITY ONLY [{name}] public window: macroF1 {r['macro_f1']:.4f}, spellF1 {r['spell_f1_mean']:.4f}, pred/true possessions {r['n_pred_spells']}/{r['n_gt_spells']}")
    pub = gt < 3
    for k, sel in (("ball present", pub & tact_present), ("ball missing", pub & ~tact_present)):
        cm = np.array([[int(((gt == a) & (pred == b) & sel).sum()) for b in range(4)] for a in range(3)])
        print(f"   confusion {k} (rows true H/A/D, cols pred H/A/D/X): {cm.tolist()}")
    print("   transitions " + "; ".join(f"tol {tol}: matched {tr['n_matched']}/{tr['n_gt']} P {tr['precision']:.2f} R {tr['recall']:.2f}" for tol, tr in r["transitions"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-aux", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    raw = load_target_match()
    m = canon.canonicalise(load_target_match_fused())
    tact_present = np.isfinite(raw.bx)
    t1 = time.time()
    X, names = extract(m)
    t2 = time.time()
    mdl = Models.load(ROOT / "models")
    assert mdl.names == names, "feature set differs from the trained models"
    ps, pt = mdl.predict(X)
    t3 = time.time()
    prm = DecoderParams.from_json(ROOT / "models" / "decoder.json")
    g = m.period > 0

    ko, ko_info = kickoff_mask(m)
    extra_ko = np.zeros((m.T, 3))
    extra_ko[ko, 0] = extra_ko[ko, 1] = HARD
    p_free = decode(ps, pt, m.period, prm)
    print(f"kick-off rule: frames before each half's 'Kick Off (Start)' window {ko_info}; share the decoder labels D WITHOUT the rule: "
          f"{float((p_free[ko] == 2).mean()):.3f}")
    pred_noaux = decode(ps, pt, m.period, prm, extra=extra_ko)

    aux = build_aux()
    streams = aux_fit.make_streams(aux)
    if a.refit_aux or not TABLES.exists():
        tables = aux_fit.fit_aux(ps, m.period, streams)
        fusion.tables_to_json(tables, TABLES)
        print("aux tables refitted and written;", f"{tables['conf_mask_share']:.3f} of graded frames pass the confidence mask")
    else:
        tables = fusion.tables_from_json(TABLES)
    print("aux tables (rows H/A/D; columns = stream values), occupancy", np.round(tables["occupancy"], 3).tolist(),
          "| e_scale", tables["e_scale"], "gamma", tables["gamma"], "clip", tables["clip"])
    for nm in aux_fit.STREAM_NAMES:
        print(f"  {nm}:", np.round(np.array(tables["streams"][nm]["table"]), 3).tolist())
    la = aux_fit.log_aux(tables, streams, m.period)
    pred_v1 = decode(ps, pt, m.period, prm, extra=extra_ko + la)
    t4 = time.time()

    h_noaux = write_csv(VAR / "v1_noaux.csv", pred_noaux)
    h_v1 = write_csv(OUT, pred_v1)
    print(f"sha256 v1 {h_v1}, v1_noaux {h_noaux}")
    bstate = aux["bstate"]
    report("v1_noaux", pred_noaux, m, tact_present, bstate)
    report("v1", pred_v1, m, tact_present, bstate)

    v0 = read_codes(VAR / "v0.csv")
    cands = {"v0": v0, "v1_noaux": pred_noaux, "v1": pred_v1}
    ks = list(cands)
    print("pairwise graded-frame agreement: " + ", ".join(f"{x}~{y} {float((cands[x][g] == cands[y][g]).mean()):.3f}" for i, x in enumerate(ks) for y in ks[i + 1:]))
    print(f"[v0] shares H/A/D {[round(float((v0[g] == k).mean()), 3) for k in range(3)]}; D share ball present {float((v0[g & tact_present] == 2).mean()):.3f} missing {float((v0[g & ~tact_present] == 2).mean()):.3f}; {possession_stats(v0, m)}")
    for nm, pr in cands.items():
        sanity(nm, pr, raw, tact_present)
    print(f"timings: load {t1 - t0:.1f}s features {t2 - t1:.1f}s predict {t3 - t2:.1f}s aux+decode {t4 - t3:.1f}s total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
