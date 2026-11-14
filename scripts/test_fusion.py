"""Synthetic test of possession/fusion.py (no match data involved). Run: ``uv run python scripts/test_fusion.py``."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from possession import fusion as fu  # noqa: E402

T_HALF = 80_000
HOLD = 25                      # auxiliary values are redrawn every HOLD frames (piecewise constant, like tags)
TRUE_AD = dict(alive=np.array([0.40, 0.30, 0.20, 0.10]), dead=np.array([0.10, 0.20, 0.30, 0.40]))
SWAP = np.array([0, 2, 1, 4, 3])
TRUE_TEAM = dict(home=np.array([0.20, 0.40, 0.10, 0.20, 0.10]), dead=np.array([0.40, 0.15, 0.15, 0.15, 0.15]))


def sim_chain(rng, n, mean_dur=200):
    s = np.empty(n, np.int8)
    t, cur = 0, rng.integers(3)
    while t < n:
        d = max(25, int(rng.exponential(mean_dur)))
        s[t:t + d] = cur
        t += d
        cur = rng.choice([x for x in range(3) if x != cur])
    return s


def sim_blocks(rng, n, mean_weak=500, mean_strong=750):
    m = np.zeros(n, bool)
    t, weak = 0, rng.random() < 0.4
    while t < n:
        d = int(rng.exponential(mean_weak if weak else mean_strong)) + 50
        m[t:t + d] = weak
        t += d
        weak = not weak
    return m


def smooth_noise(rng, shape, tau=15):
    z = rng.standard_normal(shape)
    k = np.exp(-np.arange(0, 4 * tau) / tau)
    k /= np.sqrt((k ** 2).sum())
    return np.stack([np.convolve(z[:, i], k, mode="same") for i in range(shape[1])], 1)


def draw_stream(rng, state, tab_by_state):
    n = state.size
    v = np.empty(n, np.int64)
    for a in range(0, n, HOLD):
        s = state[a]
        v[a:a + HOLD] = rng.choice(tab_by_state.shape[1], p=tab_by_state[s])
    return v


def classifier(rng, state, weak, mu_strong=2.0, mu_weak=0.3, d_bias=0.0):
    n = state.size
    oh = np.zeros((n, 3))
    oh[np.arange(n), state] = 1.0
    mu = np.where(weak, mu_weak, mu_strong)[:, None]
    le = mu * oh + smooth_noise(rng, (n, 3))
    le[weak, 2] -= d_bias        # under-predicts D on the weak part
    return le


def acc(g, state, mask):
    return float((g.argmax(1)[mask] == state[mask]).mean())


def run(d_bias, rng):
    n = 2 * T_HALF
    state = np.concatenate([sim_chain(rng, T_HALF), sim_chain(rng, T_HALF)])
    period = np.repeat(np.array([1, 2], np.int8), T_HALF)
    weak = sim_blocks(rng, n)
    le = classifier(rng, state, weak, d_bias=d_bias)
    tab_ad = np.stack([TRUE_AD["alive"], TRUE_AD["alive"], TRUE_AD["dead"]])
    tab_tm = np.stack([TRUE_TEAM["home"], TRUE_TEAM["home"][SWAP], TRUE_TEAM["dead"]])
    streams = [dict(name="ad", values=draw_stream(rng, state, tab_ad), n_values=4, sym="alive_dead"),
               dict(name="tm", values=draw_stream(rng, state, tab_tm), n_values=5, sym="team", swap=SWAP)]
    raw = None
    g0, _ = fu.posterior(le, None, period)
    m_all = np.ones(n, bool)
    print(f"\n=== classifier D-bias on weak part = {d_bias} (aux values held {HOLD} frames) ===")
    print(f"{'frame accuracy':<40}{'all':>8}{'weak':>8}{'strong':>8}{'Drec weak':>10}")
    rowf = lambda name, g: print(f"{name:<40}{acc(g, state, m_all):8.3f}{acc(g, state, weak):8.3f}"  # noqa: E731
                                 f"{acc(g, state, ~weak):8.3f}{float((g.argmax(1)[weak & (state == 2)] == 2).mean()):10.3f}")
    rowf("classifier argmax (no HMM)", le)
    rowf("classifier + HMM smoothing", g0)
    err = {}
    for gam, esc in ((1.0, 1.0), (0.3, 1.0), (1.0, 0.3), (0.3, 0.3), (0.3, 0.1), (1.0 / HOLD, 0.1)):
        t0 = time.time()
        tables = fu.fit_aux_tables(le, period, streams, gamma=gam, e_scale=esc)
        t_fit = time.time() - t0
        e = {"ad": float(np.abs(np.array(tables["streams"]["ad"]["table"]) - tab_ad).max()),
             "tm": float(np.abs(np.array(tables["streams"]["tm"]["table"]) - tab_tm).max())}
        err[(gam, esc)] = e
        _, _, g1 = fu.fuse(le, period, streams, tables=tables)
        rowf(f"fused EM gamma={gam:.2f} e_scale={esc:.1f} ({len(tables['loglik'])}it)", g1)
        print(f"{'':<6}max|table err| ad {e['ad']:.3f} tm {e['tm']:.3f}; occ {np.round(tables['occupancy'], 3)}; "
              f"{t_fit / len(tables['loglik']):.2f}s/iter")
        oracle = dict(stay=0.995, gamma=gam, clip=3.0, occupancy=np.bincount(state, minlength=3) / n,
                      streams={"ad": dict(table=tab_ad), "tm": dict(table=tab_tm)})
        g2, _ = fu.posterior(le, fu.apply_aux_tables(oracle, streams, period), period)
        rowf(f"   oracle tables gamma={gam:.2f}", g2)
    return err


if __name__ == "__main__":
    fu._fb(np.zeros((5, 3)), 0.99)   # compile
    rng = np.random.default_rng(7)
    e0 = run(0.0, rng)
    e1 = run(1.5, rng)
