"""Touch-event rate per minute of ball-present time: degraded DFL vs target (aggregates only)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import canon, features
from possession.degrade import degrade
from possession.external_list import load_external
from possession.target import load_target_match


def rate(m):
    X, n = features.extract(m)
    sel = m.period > 0
    th = X[sel][:, n.index("touch_since_h")] == 0
    ta = X[sel][:, n.index("touch_since_a")] == 0
    pres = np.isfinite(m.bx[sel])
    mins = pres.sum() / m.fps / 60
    return (th & pres).sum() / mins, (ta & pres).sum() / mins


if __name__ == "__main__":
    for k, m in enumerate(load_external()):
        c = canon.canonicalise(m)
        print("DFL %d clean %.1f/%.1f  degraded %.1f/%.1f" % (k, *rate(c), *rate(degrade(c, np.random.default_rng(k)))))
    print("target %.1f/%.1f" % rate(canon.canonicalise(load_target_match())))
