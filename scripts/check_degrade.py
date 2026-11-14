"""Compare degraded external matches with the target tracking (aggregates only).

    uv run python scripts/check_degrade.py [--margin -3.0] [--seeds 3]

Run lengths are counted per match in frames; DFL values are averaged over the external matches
present, then shown per match (the target is one match, so compare per-match counts).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import calib, canon  # noqa: E402
from possession.degrade import default_params, degrade  # noqa: E402
from possession.external_list import load_external  # noqa: E402


def mean_summaries(sums):
    """Average numeric leaves of a list of summarise() dicts."""
    def avg(vals):
        if isinstance(vals[0], dict):
            return {k: avg([v[k] for v in vals]) for k in vals[0]}
        if isinstance(vals[0], list):
            return [avg([v[i] for v in vals]) for i in range(len(vals[0]))]
        return float(np.mean(vals))
    return avg(sums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    prm = default_params()
    if a.margin is not None:
        prm = replace(prm, fov_margin=a.margin)
    matches = load_external()
    rows = {}
    t0 = time.time()
    clean, deg = [], []
    for k, m in enumerate(matches):
        c = canon.canonicalise(m)
        clean.append(calib.summarise(c))
        for s in range(a.seeds):
            deg.append(calib.summarise(degrade(c, np.random.default_rng(1000 + 17 * k + s), prm)))
    rows["DFL clean"] = mean_summaries(clean)
    rows["DFL degraded"] = mean_summaries(deg)
    try:
        from possession.target import load_target_match
        tm = canon.canonicalise(load_target_match())
        rows["target"] = calib.summarise(tm)
    except Exception as e:  # noqa: BLE001
        print("target unavailable:", type(e).__name__)
    print(calib.format_comparison(rows))
    print(f"\n(margin {prm.fov_margin}, {len(matches)} matches, {a.seeds} seeds each, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
