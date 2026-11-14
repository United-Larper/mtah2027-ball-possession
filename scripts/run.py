"""Run the trained system on the competition match and write outputs/RickRolled.csv.

    uv run python scripts/run.py

Prints aggregates only. The public-window score is a sanity check of alignment and label meaning;
it is never used to choose anything.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import canon, evalutil  # noqa: E402
from possession.decode import DecoderParams, decode  # noqa: E402
from possession.features import extract  # noqa: E402
from possession.model import Models  # noqa: E402
from possession.target import load_target_match  # noqa: E402

pm = evalutil.pm
OUT = ROOT / "outputs" / "RickRolled.csv"
LETTERS = np.array(["H", "A", "D", "X"])
N_FRAMES = 199319


def main():
    t0 = time.time()
    raw = load_target_match()
    m = canon.canonicalise(raw)
    t1 = time.time()
    X, names = extract(m)
    t2 = time.time()
    mdl = Models.load(ROOT / "models")
    assert mdl.names == names, "feature set differs from the trained models"
    ps, pt = mdl.predict(X)
    t3 = time.time()
    prm = DecoderParams.from_json(ROOT / "models" / "decoder.json")
    pred = decode(ps, pt, m.period, prm)
    t4 = time.time()

    OUT.parent.mkdir(exist_ok=True)
    text = "frame,label\n" + "\n".join(f"{i},{LETTERS[c]}" for i, c in enumerate(pred)) + "\n"
    OUT.write_text(text, newline="\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]

    table = pm.read_label_csv(OUT)
    reason = pm.validate_submission(table, N_FRAMES)
    print("validation:", "OK" if reason is None else f"FAIL {reason}", "| sha256", digest)

    g = m.period > 0
    n = g.sum()
    print("graded frames", int(n), "shares H/A/D:", [round(float((pred[g] == k).mean()), 3) for k in range(3)])
    runs = [(sl.start, sl.stop) for sl in m.period_slices().values()]
    for p, (a, b) in zip(m.period_slices(), runs):
        sp = pm.possession_spells(pred, [(a, b)])
        dur = [(e - s) / 25 for s, e, _ in sp]
        print(f"half {p}: {len(sp)} possessions, mean {np.mean(dur):.1f}s, median {np.median(dur):.1f}s")
    pres = np.isfinite(m.bx)
    for k, sel in (("ball present", g & pres), ("ball missing", g & ~pres)):
        print(f"{k}: {int(sel.sum())} frames, shares H/A/D", [round(float((pred[sel] == c).mean()), 3) for c in range(3)])

    gt = raw.label.astype(np.int8)                      # public window only; X elsewhere
    r = pm.evaluate(gt, pred)
    print(f"SANITY (public window, not used for any choice): macroF1 {r['macro_f1']:.4f}, spellF1 {r['spell_f1_mean']:.4f}, "
          f"predicted/true possessions {r['n_pred_spells']}/{r['n_gt_spells']}")
    for tol, tr in r["transitions"].items():
        print(f"  transitions tol {tol}: P {tr['precision']:.2f} R {tr['recall']:.2f} matched {tr['n_matched']}/{tr['n_gt']}, "
              f"median latency {tr['median_latency']}, mean abs {tr['mean_abs_latency']}, p90 abs {tr['p90_abs_latency']}")
    print(f"timings: load {t1 - t0:.1f}s features {t2 - t1:.1f}s predict {t3 - t2:.1f}s decode {t4 - t3:.2f}s total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
