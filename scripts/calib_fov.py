"""Sweep the field-of-view margin and compare tracked-count histograms and 'deepest player tracked'
shares (by view-centre x) with the target statistics. Aggregates only."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import calib, canon  # noqa: E402
from possession.degrade import _camera_centre_x, default_params, degrade  # noqa: E402
from possession.external_list import load_external  # noqa: E402

STATS = json.loads((Path(__file__).resolve().parents[1] / "possession" / "assets" / "target_stats.json").read_text())


def deepest_shares(clean, deg, prm):
    """share of frames in which the clean deepest player of each team is still tracked, by view-centre bin."""
    bins = [(-99, -15), (-15, 15), (15, 99)]
    acc = {(t, b): [] for t in ("neg", "pos") for b in range(3)}
    for p, sl in clean.period_slices().items():
        dead = clean.label[sl] == 2
        cx = _camera_centre_x(clean.bx[sl], dead, prm)
        for team, (X, Xd) in (("neg", (clean.hx[sl], deg.hx[sl])), ("pos", (clean.ax[sl], deg.ax[sl]))):
            xs = np.where(np.isfinite(X), X, np.nan)
            ok = np.isfinite(xs).any(1)
            idx = np.nanargmin(xs[ok], 1) if team == "neg" else np.nanargmax(xs[ok], 1)
            tracked = np.isfinite(Xd[ok][np.arange(ok.sum()), idx])
            for b, (lo, hi) in enumerate(bins):
                sel = (cx[ok] >= lo) & (cx[ok] < hi)
                acc[(team, b)].append(tracked[sel])
    return {k: float(np.concatenate(v).mean()) for k, v in acc.items()}


if __name__ == "__main__":
    ms = [canon.canonicalise(m) for m in load_external()]
    tgt = STATS["fov"]["deepest_tracked_share_by_view_centre"]
    tgt_hist_h = np.array(STATS["players_tracked_hist"]["home"], float)
    print("target deepest-tracked share  neg: ", [round(v["team_defending_neg_x"], 2) for v in tgt.values()],
          " pos: ", [round(v["team_defending_pos_x"], 2) for v in tgt.values()])
    print("target mean tracked home %.2f" % ((tgt_hist_h * np.arange(17)).sum() / tgt_hist_h.sum()))
    for arg in sys.argv[1:] or ['0,0.5']:
        margin, qq = (float(v) for v in arg.split(','))
        prm = replace(default_params(), fov_margin=margin, fov_persist_q=qq, mask_blocks=False)
        res, cnt, h11 = [], [], []
        for k, m in enumerate(ms):
            d = degrade(m, np.random.default_rng(k), prm)
            res.append(deepest_shares(m, d, prm))
            cnt.append(np.isfinite(d.hx).sum(1).mean()); h11.append((np.isfinite(d.hx).sum(1) == 11).mean())
        r = {k: np.mean([x[k] for x in res]) for k in res[0]}
        print(f"margin {margin:5.1f} q {qq}: tracked home {np.mean(cnt):.2f} share==11 {np.mean(h11):.2f} | neg {[round(r[('neg', b)], 2) for b in range(3)]} pos {[round(r[('pos', b)], 2) for b in range(3)]}")
