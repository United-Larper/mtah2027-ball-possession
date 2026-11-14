"""Two LightGBM models: a 3-class frame-state model (H, A, D) and a 4-class transition model
(none, team change, alive->D, D->alive; positive within +-2 frames of a transition)."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import canon
from .degrade import degrade, default_params
from .features import extract
from .match import Match

NUM_THREADS = 16
TRANS_HALF_WINDOW = 2
NEAR = 25
REST_FRACTION = 0.10

BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=63,
            deterministic=True, force_row_wise=True, num_threads=NUM_THREADS, seed=7,
            feature_fraction_seed=7, bagging_seed=7, data_random_seed=7, verbose=-1)


def transition_labels(label: np.ndarray, period: np.ndarray):
    """(y [n] in 0..3, near [n] bool): per-frame transition class and a mask within +-NEAR frames of any transition."""
    y = np.zeros(len(label), np.int8)
    near = np.zeros(len(label), bool)
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        a0 = int(idx[0])
        lab = label[idx[0]:idx[-1] + 1]
        for t in np.flatnonzero(lab[1:] != lab[:-1]) + 1:
            a, b = lab[t - 1], lab[t]
            k = 1 if (a < 2 and b < 2) else (2 if b == 2 else 3)
            lo, hi = a0 + max(0, t - TRANS_HALF_WINDOW), a0 + min(len(lab), t + TRANS_HALF_WINDOW + 1)
            seg = y[lo:hi]
            seg[seg == 0] = k
            near[a0 + max(0, t - NEAR):a0 + min(len(lab), t + NEAR + 1)] = True
    return y, near


@dataclass
class TrainSet:
    X: np.ndarray            # float32 [r, F]
    y_state: np.ndarray      # int8 [r]
    y_trans: np.ndarray      # int8 [r]
    is_state: np.ndarray     # bool: row belongs to the state subsample
    is_trans: np.ndarray     # bool: row belongs to the transition sample
    w_trans: np.ndarray      # float32 weights undoing the sampling
    match_id: str = ""


def make_set(m: Match, rng: np.random.Generator, state_stride: int = 4) -> tuple[TrainSet, list[str]]:
    """Features + sampled rows of a (canonical, possibly degraded) match."""
    X, names = extract(m)
    graded = (m.period > 0) & (m.label < 3)
    y_tr, near = transition_labels(m.label, m.period)
    t = np.arange(m.T)
    is_state = graded & ((t + int(rng.integers(0, state_stride))) % state_stride == 0)
    rest = graded & ~near & (rng.random(m.T) < REST_FRACTION)
    is_trans = graded & near | rest
    w = np.where(near, 1.0, 1.0 / REST_FRACTION).astype(np.float32)
    keep = is_state | is_trans
    return TrainSet(X[keep], m.label[keep].astype(np.int8), y_tr[keep], is_state[keep], is_trans[keep], w[keep], m.match_id), names


def make_training_sets(canon_matches, seed: int, k_copies: int = 2, swaps: bool = True, state_stride: int = 4):
    """K degraded copies (randomised parameters) of each canonical match, each also team-swapped."""
    sets, names = [], None
    prm = default_params()
    for i, m in enumerate(canon_matches):
        for c in range(k_copies):
            rng = np.random.default_rng([seed, i, c])
            d = degrade(m, rng, prm)
            for sw in ((False, True) if swaps else (False,)):
                mm = canon.swap_teams(d) if sw else d
                s, names = make_set(mm, np.random.default_rng([seed, i, c, int(sw)]), state_stride)
                sets.append(s)
    return sets, names


def _stack(sets, kind):
    if kind == "state":
        X = np.concatenate([s.X[s.is_state] for s in sets])
        y = np.concatenate([s.y_state[s.is_state] for s in sets])
        return X, y, None
    X = np.concatenate([s.X[s.is_trans] for s in sets])
    y = np.concatenate([s.y_trans[s.is_trans] for s in sets])
    w = np.concatenate([s.w_trans[s.is_trans] for s in sets])
    return X, y, w


@dataclass
class Models:
    state: lgb.Booster
    trans: lgb.Booster
    names: list

    def predict(self, X: np.ndarray):
        return (self.state.predict(X, num_threads=NUM_THREADS), self.trans.predict(X, num_threads=NUM_THREADS))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.state.save_model(str(folder / "state.txt"))
        self.trans.save_model(str(folder / "trans.txt"))
        (folder / "features.json").write_text(json.dumps(self.names))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        return cls(lgb.Booster(model_file=str(folder / "state.txt")), lgb.Booster(model_file=str(folder / "trans.txt")),
                   json.loads((folder / "features.json").read_text()))


def _train(kind, sets, names, n_trees, valid=None, drop_cols=None, early=50):
    X, y, w = _stack(sets, kind)
    cols = np.array([i for i, n in enumerate(names) if not (drop_cols and any(n.startswith(p) for p in drop_cols))])
    params = dict(BASE, objective="multiclass", num_class=3 if kind == "state" else 4)
    dtr = lgb.Dataset(X[:, cols], y, weight=w, feature_name=[names[i] for i in cols], free_raw_data=True)
    kw = {}
    if valid is not None:
        Xv, yv, wv = _stack(valid, kind)
        dv = lgb.Dataset(Xv[:, cols], yv, weight=wv, reference=dtr)
        kw = dict(valid_sets=[dv], callbacks=[lgb.early_stopping(early, verbose=False)])
    b = lgb.train(params, dtr, num_boost_round=n_trees, **kw)
    return b, cols


class MaskedModels(Models):
    """Models trained on a column subset (feature-group ablation)."""
    cols: np.ndarray = None

    def predict(self, X):
        return super().predict(X[:, self.cols])


def fit(sets, names, n_state=500, n_trans=500, valid=None, drop_cols=None):
    """Train both models. With `valid` sets, early stopping is used and best iterations are returned in .best."""
    bs, cols = _train("state", sets, names, n_state, valid, drop_cols)
    bt, _ = _train("trans", sets, names, n_trans, valid, drop_cols)
    mm = MaskedModels(bs, bt, [names[i] for i in cols])
    mm.cols = cols
    mm.best = (bs.best_iteration or n_state, bt.best_iteration or n_trans)
    return mm
