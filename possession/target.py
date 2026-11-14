"""The competition match (tactical recording) as a common `Match`.

Index = tactical video frame (T = 199,319). Coordinates are metres, pitch-centred. The label is X
everywhere except the public ground-truth window (sanity scoring only, never for fitting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import games
from .io_atd import load_atd
from .match import Match, H, A, D, X
from .paths import REPO, cache_dir

GAME = "BAR-ATH"
PITCH = games.GAMES[GAME]["pitch"]
_PUBLIC_GT = REPO / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
_LAB = {"H": H, "A": A, "D": D, "X": X}


def graded_blocks() -> list[tuple[int, int, int]]:
    """[(period, first_video_frame, last_video_frame)] inclusive."""
    return list(games.half_video_blocks(GAME))


def graded_mask(T: int = 199319) -> np.ndarray:
    g = np.zeros(T, bool)
    for _, a, b in graded_blocks():
        g[a:b + 1] = True
    return g


def infer_home_dir(m: Match, min_players: int = 8) -> dict:
    """+1 if home attacks towards +x, per period, from the data: the team whose deepest tracked
    player is (median over the period) nearer x = -L/2 defends that goal."""
    out = {}
    for p, sl in m.period_slices().items():
        med = []
        for x in (m.hx[sl], m.ax[sl]):
            n = np.isfinite(x).sum(1)
            ok = n >= min_players
            med.append(np.median(np.nanmin(x[ok], axis=1)))
        out[p] = 1 if med[0] < med[1] else -1  # home deepest at -x => home defends -x => attacks +x
    return out


def _public_labels(T: int) -> np.ndarray:
    lab = np.full(T, X, np.int8)
    gt = pd.read_csv(_PUBLIC_GT)
    lab[gt["frame"].to_numpy()] = np.array([_LAB[s] for s in gt["label"]], np.int8)
    return lab


def _build() -> tuple[Match, np.ndarray, np.ndarray]:
    atd = load_atd("tactical")
    T = atd.T
    L, W = PITCH
    hm, am = atd.team == 0, atd.team == 1

    def xy(mask):
        return (((atd.px[:, mask].astype(np.float64) - 0.5) * L).astype(np.float32),
                ((atd.py[:, mask].astype(np.float64) - 0.5) * W).astype(np.float32))

    hx, hy = xy(hm)
    ax, ay = xy(am)
    bx = ((atd.bx - 0.5) * L).astype(np.float32)
    by = ((atd.by - 0.5) * W).astype(np.float32)
    period = np.zeros(T, np.int8)
    for p, a, b in graded_blocks():
        period[a:b + 1] = p
    m = Match(source="atd", match_id=GAME + "_tactical", fps=float(atd.fps), pitch=(L, W),
              hx=hx, hy=hy, ax=ax, ay=ay, bx=bx, by=by,
              bz=np.full(T, np.nan, np.float32), label=_public_labels(T), period=period,
              home_dir={1: 1, 2: -1}, frame_id=np.arange(T, dtype=np.int32))
    m.home_dir = infer_home_dir(m)
    hs = atd.ps[:, hm].astype(np.float32)
    as_ = atd.ps[:, am].astype(np.float32)
    return m, hs, as_


_SPEED_CACHE: dict = {}


def load_target_match(use_cache: bool = True) -> Match:
    cache = cache_dir() / "target_match.npz"
    sp = cache_dir() / "target_speed.npz"
    if use_cache and cache.exists() and sp.exists():
        return Match.load(cache)
    m, hs, as_ = _build()
    m.save(cache)
    np.savez_compressed(sp, hs=hs, as_=as_)
    return m


def load_player_speed(use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """ATD player speed field (m/s) aligned with Match.hx / Match.ax."""
    sp = cache_dir() / "target_speed.npz"
    if not (use_cache and sp.exists() and (cache_dir() / "target_match.npz").exists()):
        load_target_match(use_cache=False)
    with np.load(sp) as z:
        return z["hs"], z["as_"]
