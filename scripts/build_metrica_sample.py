"""Build cache/external/metrica_g<1|2|3>.npz (Match format + event-derived labels) from the Metrica sample data.

    uv run scripts/build_metrica_sample.py [--root DIR] [--games 1 2 3]
Default root: <project>/../external/metrica-sample/data. Label rules: possession/external/metrica_sample.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from possession.external import metrica_sample as ms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ms.DEFAULT_ROOT))
    ap.add_argument("--games", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    out = REPO / "cache" / "external"
    for g in args.games:
        t0 = time.time()
        m = ms.load_metrica_sample(g, root=args.root)
        m.save(out / f"metrica_g{g}.npz")
        cnt = np.bincount(m.label, minlength=4)
        # spells per period (maximal H/A runs)
        n_sp = 0
        for p, sl in m.period_slices().items():
            lab = m.label[sl]
            ch = np.flatnonzero(np.diff(lab) != 0) + 1
            seg = np.split(lab, ch)
            n_sp += sum(1 for s in seg if s[0] in (0, 1))
        print(f"game {g}: T={m.T} periods={ {p: int((m.period == p).sum()) for p in (1, 2)} } "
              f"cols H/A={m.hx.shape[1]}/{m.ax.shape[1]} home_dir={m.home_dir} "
              f"H/A/D/X={np.round(cnt / m.T, 3).tolist()} ball_nan={np.isnan(m.bx).mean():.3f} "
              f"spells={n_sp} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
