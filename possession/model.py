"""v1 hierarchical models (LightGBM, binary each):

  AD    alive vs dead, trained on all graded frames, RESTRICTED feature set (allow-list below);
  TEAM  home vs away given alive, alive frames only, all features;
  TC    team change within +-2 frames; frames whose +-2 neighbourhood is all alive or holds a team change; all features;
  A2D   alive->dead within +-2 frames, restricted features;
  D2A   dead->alive within +-2 frames, restricted features.

Combined: p(H) = p(alive) p(H|alive), p(A) = p(alive) (1 - p(H|alive)), p(D) = 1 - p(alive); transitions are
returned as [none, team change, alive->D, D->alive] for the decoder.

Why the restriction: in the external datasets the ball is frozen (DFL) or absent (Metrica) during every stoppage,
so ball speed / proximity / touch features identify "alive" with certainty there but not in the competition
match, where the tracked ball keeps moving during stoppages. AD therefore only sees team shape/motion, ball
availability, instantaneous ball position and position-only carry-over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
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

KINDS = ("ad", "team", "tc", "a2d", "d2a")
RESTRICTED = ("ad", "a2d", "d2a")

# ---- documented allow-list of the restricted (AD / A2D / D2A) feature set
_POS = ["x", "y", "near", "far", "goal", "dtl"]
AD_EXACT = (
    # ball availability
    ["avail_present", "avail_since_present", "avail_until_present", "avail_run_len", "avail_run_pos", "avail_run_rem"]
    # instantaneous ball position (when present)
    + ["ball_x", "ball_y", "ball_absy", "ball_d_touchline", "ball_d_goalline", "ball_out_near", "ball_out_far",
       "ball_out_goal", "ball_box_opp", "ball_box_own"]
    # position-only carry-over at missing frames
    + [f"carry_last_{k}" for k in _POS] + [f"carry_next_{k}" for k in _POS] + ["carry_ago_s", "carry_ahead_s"]
)
AD_PREFIX = (
    "shape_",                                            # all team shape / motion features
    "ctx_present_", "ctx_pspeed_", "ctx_vx_h_", "ctx_vx_a_",   # temporal context of availability and team motion only
)
# groups that must NOT be in the restricted set
AD_FORBIDDEN_PREFIX = ("ball_speed", "ball_acc", "ball_head", "geo_", "touch_",
                       "carry_last_speed", "carry_last_acc", "carry_last_ns", "carry_last_d1", "carry_last_dd",
                       "carry_next_ns", "carry_next_d1", "carry_next_dd", "carry_last_touch",
                       "ctx_ns_", "ctx_dd_", "ctx_d1min_", "ctx_bspeed_", "ctx_bacc_", "ctx_touch_")


def ad_columns(names: list[str]) -> list[int]:
    cols = [i for i, n in enumerate(names) if n in AD_EXACT or n.startswith(AD_PREFIX)]
    miss = [n for n in AD_EXACT if n not in names]
    assert not miss, f"allow-listed features missing from the extractor: {miss}"
    bad = [names[i] for i in cols if names[i].startswith(AD_FORBIDDEN_PREFIX)]
    assert not bad, f"forbidden features in the restricted set: {bad}"
    return cols


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
    y_label: np.ndarray      # int8 [r]  0 H, 1 A, 2 D
    y_trans: np.ndarray      # int8 [r]  0 none, 1 team change, 2 alive->D, 3 D->alive
    nbr_dead: np.ndarray     # bool [r]  a dead frame within +-2 frames
    is_state: np.ndarray     # bool: row belongs to the state subsample
    is_trans: np.ndarray     # bool: row belongs to the transition sample
    w_trans: np.ndarray      # float32 weights undoing the sampling
    match_id: str = ""


def make_set(m: Match, rng: np.random.Generator, state_stride: int = 4) -> tuple[TrainSet, list[str]]:
    """Features + sampled rows of a (canonical, possibly degraded) match."""
    X, names = extract(m)
    graded = (m.period > 0) & (m.label < 3)
    y_tr, near = transition_labels(m.label, m.period)
    dead = (m.label == 2).astype(np.int8)
    k = TRANS_HALF_WINDOW
    nd = np.zeros(m.T, bool)
    for s in range(-k, k + 1):
        nd |= np.roll(dead, s) > 0
    t = np.arange(m.T)
    is_state = graded & ((t + int(rng.integers(0, state_stride))) % state_stride == 0)
    rest = graded & ~near & (rng.random(m.T) < REST_FRACTION)
    is_trans = graded & near | rest
    w = np.where(near, 1.0, 1.0 / REST_FRACTION).astype(np.float32)
    keep = is_state | is_trans
    return TrainSet(X[keep], m.label[keep].astype(np.int8), y_tr[keep], nd[keep], is_state[keep], is_trans[keep], w[keep], m.match_id), names


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
    """(X, y, w) of the rows used by one model."""
    Xs, ys, ws = [], [], []
    for s in sets:
        if kind in ("ad", "ad_all"):
            sel = s.is_state
            y = (s.y_label < 2)
            w = None
        elif kind == "team":
            sel = s.is_state & (s.y_label < 2)
            y = (s.y_label == 0)
            w = None
        else:
            sel = s.is_trans
            if kind == "tc":
                sel = sel & ((s.y_trans == 1) | ((s.y_trans == 0) & ~s.nbr_dead))
                y = (s.y_trans == 1)
            elif kind == "a2d":
                y = (s.y_trans == 2)
            else:
                y = (s.y_trans == 3)
            w = s.w_trans
        Xs.append(s.X[sel]); ys.append(y[sel].astype(np.int8))
        if w is not None:
            ws.append(w[sel])
    return np.concatenate(Xs), np.concatenate(ys), (np.concatenate(ws) if ws else None)


def _cols(kind, names):
    return np.array(ad_columns(names) if kind in RESTRICTED else range(len(names)))


def train_one(kind, sets, names, n_trees, valid=None, early=50):
    X, y, w = _stack(sets, kind)
    cols = _cols(kind, names)
    dtr = lgb.Dataset(X[:, cols], y, weight=w, feature_name=[names[i] for i in cols], free_raw_data=True)
    kw = {}
    if valid is not None:
        Xv, yv, wv = _stack(valid, kind)
        dv = lgb.Dataset(Xv[:, cols], yv, weight=wv, reference=dtr)
        kw = dict(valid_sets=[dv], callbacks=[lgb.early_stopping(early, verbose=False)])
    b = lgb.train(dict(BASE, objective="binary"), dtr, num_boost_round=n_trees, **kw)
    return b, cols


@dataclass
class Models:
    boosters: dict          # kind -> lgb.Booster
    cols: dict              # kind -> np.ndarray of column indices
    names: list
    best: dict = None

    def predict_parts(self, X: np.ndarray) -> dict:
        out = {}
        for k in KINDS:
            out[k] = self.boosters[k].predict(X[:, self.cols[k]], num_threads=NUM_THREADS)
        return out

    def predict(self, X: np.ndarray):
        """(p_state [n,3] H/A/D, p_trans [n,4] none/team/alive->D/D->alive)."""
        p = self.predict_parts(X)
        pa, ph = p["ad"], p["team"]
        ps = np.stack([pa * ph, pa * (1 - ph), 1 - pa], 1)
        pt = np.stack([np.zeros(len(pa)), p["tc"], p["a2d"], p["d2a"]], 1)
        pt[:, 0] = np.clip(1 - pt[:, 1:].sum(1), 0, 1)
        return ps, pt

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for k in KINDS:
            self.boosters[k].save_model(str(folder / f"{k}.txt"))
        (folder / "features.json").write_text(json.dumps(self.names))
        (folder / "ad_features.json").write_text(json.dumps([self.names[i] for i in self.cols["ad"]]))

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        names = json.loads((folder / "features.json").read_text())
        ad = [names.index(n) for n in json.loads((folder / "ad_features.json").read_text())]
        assert ad == list(ad_columns(names)), "stored AD feature set differs from the allow-list"
        cols = {k: (np.array(ad) if k in RESTRICTED else np.arange(len(names))) for k in KINDS}
        return cls({k: lgb.Booster(model_file=str(folder / f"{k}.txt")) for k in KINDS}, cols, names)


def fit(sets, names, n_trees: dict | int = 400, valid=None):
    """Train the five models. With `valid` sets, early stopping is used and the best iterations are in .best."""
    nt = n_trees if isinstance(n_trees, dict) else {k: int(n_trees) for k in KINDS}
    boosters, cols, best = {}, {}, {}
    for k in KINDS:
        b, c = train_one(k, sets, names, nt[k], valid)
        boosters[k], cols[k], best[k] = b, c, (b.best_iteration or nt[k])
    return Models(boosters, cols, names, best)
