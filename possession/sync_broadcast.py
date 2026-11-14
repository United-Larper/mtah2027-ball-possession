"""Synchronise the broadcast recording to the tactical timeline (tactical_frame = broadcast_frame + delta).

Everything is discovered from the data: the ball track of each ATD is cut into chunks, each chunk is
matched against the whole tactical ball track by masked FFT correlation, the best offset is refined
to single-frame resolution, and chunks with a confident, consistent offset are merged into plateaus.
Boundaries between plateaus with different offsets are located with ball + player evidence.

`load_mapping()` returns an int32 array of length 199319: broadcast frame for each tactical frame or -1.
The result is cached under ``cache/``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_atd import load_atd
from .io_homography import load_homography
from .paths import cache_dir
import games

PX, PY = games.GAMES["BAR-ATH"]["pitch"]
T_TAC = games.GAMES["BAR-ATH"]["n_video_frames"]
CHUNK = 1500
STRIDE = 750
D_COARSE = 8.0       # metres, scale of the coarse score N - SSD / D^2
NFFT = 1 << 18       # >= 199319 + 1500
MAXD = 2.0           # metres, "same ball" threshold
CACHE_VERSION = 4


@dataclass
class Sources:
    tx: np.ndarray; ty: np.ndarray; tP: np.ndarray       # tactical ball (metres), present
    bx: np.ndarray; by: np.ndarray; bP: np.ndarray       # broadcast ball (metres), present (unflipped)
    tpx: np.ndarray; tpy: np.ndarray; tteam: np.ndarray  # tactical players (T,S) metres, team
    bpx: np.ndarray; bpy: np.ndarray; bteam: np.ndarray
    bvalid: np.ndarray                                    # broadcast homography valid
    Tb: int = 0


def load_sources() -> Sources:
    ta, ba = load_atd("tactical"), load_atd("broadcast")
    _, _, bvalid = load_homography("broadcast")
    tx, ty = ta.bx * PX, ta.by * PY
    bx, by = ba.bx * PX, ba.by * PY
    return Sources(tx=tx, ty=ty, tP=np.isfinite(tx) & np.isfinite(ty),
                   bx=bx, by=by, bP=np.isfinite(bx) & np.isfinite(by),
                   tpx=ta.px.astype(np.float64) * PX, tpy=ta.py.astype(np.float64) * PY, tteam=ta.team,
                   bpx=ba.px.astype(np.float64) * PX, bpy=ba.py.astype(np.float64) * PY, bteam=ba.team,
                   bvalid=bvalid, Tb=ba.T)


def apply_flip(S: Sources, fx: bool, fy: bool):
    """Broadcast ball and player coordinates (metres) under a candidate axis flip."""
    f = lambda a, fl, L: (L - a) if fl else a  # noqa: E731
    return (f(S.bx, fx, PX), f(S.by, fy, PY), f(S.bpx, fx, PX), f(S.bpy, fy, PY))


class _TacFFT:
    """FFTs of the tactical masked moments, reused for every chunk."""

    def __init__(self, S: Sources):
        m = S.tP.astype(np.float64)
        x = np.where(S.tP, S.tx, 0.0)
        y = np.where(S.tP, S.ty, 0.0)
        self.Fm = np.fft.rfft(m, NFFT)
        self.Fx = np.fft.rfft(x, NFFT)
        self.Fy = np.fft.rfft(y, NFFT)
        self.Fr = np.fft.rfft(x * x + y * y, NFFT)


def chunk_scores(tf: _TacFFT, bx, by, bP, b0: int, b1: int) -> tuple[np.ndarray, np.ndarray]:
    """Coarse score for every delta for broadcast chunk [b0,b1). Returns (deltas, score).

    score(delta) = N - SSD / D^2 over frames where both balls are present, with N the number of such
    frames and SSD the sum of squared distances (all computed with 5 FFT correlations).
    """
    m = bP[b0:b1].astype(np.float64)
    x = np.where(bP[b0:b1], bx[b0:b1], 0.0)
    y = np.where(bP[b0:b1], by[b0:b1], 0.0)
    r = x * x + y * y
    Fm, Fx, Fy, Fr = (np.conj(np.fft.rfft(a, NFFT)) for a in (m, x, y, r))
    # corr[k] = sum_i a[i] c[i+k]
    N = np.fft.irfft(Fm * tf.Fm, NFFT)
    ssd = np.fft.irfft(Fr * tf.Fm - 2 * Fx * tf.Fx - 2 * Fy * tf.Fy + Fm * tf.Fr, NFFT)
    L = b1 - b0
    ks = np.arange(-(L - 1), T_TAC)               # tactical start index of the chunk
    idx = ks % NFFT
    score = N[idx] - ssd[idx] / D_COARSE ** 2
    deltas = ks - b0
    return deltas, score


def exact_count(S: Sources, bx, by, b0: int, b1: int, delta: int, thr: float = MAXD):
    """Number of broadcast frames in [b0,b1) with both balls present within thr metres at tactical = b+delta,
    plus the number of frames with both present and the median distance."""
    b = np.arange(b0, b1)
    t = b + delta
    ok = (t >= 0) & (t < T_TAC)
    b, t = b[ok], t[ok]
    both = S.bP[b] & S.tP[t]
    if not both.any():
        return 0, 0, np.nan
    d = np.hypot(bx[b[both]] - S.tx[t[both]], by[b[both]] - S.ty[t[both]])
    return int((d < thr).sum()), int(both.sum()), float(np.median(d))


def search_chunks(S: Sources, bx, by, chunks: list[tuple[int, int]]):
    """For each chunk (None if < 100 ball frames): coarse delta, best score s1, best score >= 50 frames away s2, refined delta."""
    tf = _TacFFT(S)
    out = []
    for b0, b1 in chunks:
        if S.bP[b0:b1].sum() < 100:
            out.append(None)
            continue
        deltas, sc = chunk_scores(tf, bx, by, S.bP, b0, b1)
        i1 = int(np.argmax(sc))
        d1, s1 = int(deltas[i1]), float(sc[i1])
        far = np.abs(deltas - d1) > 50
        s2 = float(sc[far].max()) if far.any() else -np.inf
        # refine to single-frame resolution: among deltas within +-8 frames of the coarse peak, the one with
        # the smallest median ball distance (needs >= 100 joint-present frames); n2 counts frames within 2 m
        best = (np.inf, d1)
        for dd in range(d1 - 8, d1 + 9):
            dist = ball_dist(S, bx, by, b0, b1, dd)
            if dist.size >= 100 and np.median(dist) < best[0]:
                best = (float(np.median(dist)), dd)
        n2, nb, md = exact_count(S, bx, by, b0, b1, best[1])
        out.append(dict(b0=b0, b1=b1, delta=best[1], coarse_delta=d1, s1=s1, s2=s2, n2=n2, nboth=nb, med=md))
    return out


def make_chunks(Tb: int) -> list[tuple[int, int]]:
    return [(b0, min(b0 + CHUNK, Tb)) for b0 in range(0, Tb - CHUNK // 2, STRIDE)]


def is_confident(r: dict | None) -> bool:
    """Confidence: >=100 frames within 2 m at the refined delta, and coarse best score at least twice the
    best score at any delta >= 50 frames away (or that one is non-positive)."""
    if r is None:
        return False
    return r["n2"] >= 100 and r["s1"] >= 2.0 * max(r["s2"], 1.0)


def player_support(S: Sources, bpx, bpy, bs: np.ndarray, delta: float, thr: float = MAXD) -> np.ndarray:
    """Per broadcast frame in bs: fraction of broadcast-tracked players with a same-team tactical player
    within thr metres at tactical frame b + delta (integer delta). Only frames with a valid broadcast
    homography are evaluated; NaN where invalid, out of range, or no broadcast player is tracked."""
    out = np.full(bs.size, np.nan)
    t = bs + int(delta)
    ok = (t >= 0) & (t < T_TAC) & S.bvalid[bs]
    bi, ti = bs[ok], t[ok]
    n_slots = bpx.shape[1]
    hit = np.zeros((bi.size, n_slots), bool)
    for team in (0, 1):
        b_slots = np.flatnonzero(S.bteam == team)
        t_slots = np.flatnonzero(S.tteam == team)
        for a in range(0, bi.size, 20000):
            sl = slice(a, a + 20000)
            dx = bpx[bi[sl]][:, b_slots][:, :, None] - S.tpx[ti[sl]][:, t_slots][:, None, :]
            dy = bpy[bi[sl]][:, b_slots][:, :, None] - S.tpy[ti[sl]][:, t_slots][:, None, :]
            d = np.hypot(dx, dy)
            h = (np.where(np.isfinite(d), d, np.inf).min(2) < thr)
            sub = hit[sl]
            sub[:, b_slots] = h
            hit[sl] = sub
    present = np.isfinite(bpx[bi]) & np.isfinite(bpy[bi])
    n = present.sum(1)
    out[ok] = np.where(n > 0, (hit & present).sum(1) / np.maximum(n, 1), np.nan)
    return out


def ball_support(S: Sources, bx, by, bs: np.ndarray, delta: int, thr: float = MAXD) -> np.ndarray:
    """1.0 if both balls present within thr at delta, 0.0 if both present but farther, NaN otherwise."""
    t = bs + int(delta)
    ok = (t >= 0) & (t < T_TAC)
    out = np.full(bs.size, np.nan)
    both = ok.copy()
    both[ok] = S.bP[bs[ok]] & S.tP[t[ok]]
    d = np.hypot(bx[bs[both]] - S.tx[t[both]], by[bs[both]] - S.ty[t[both]])
    out[both] = (d < thr).astype(float)
    return out


def ball_dist(S: Sources, bx, by, b0: int, b1: int, delta: float) -> np.ndarray:
    """Distances (m) between broadcast ball at b and the tactical ball at b + delta (linear interpolation
    between the two neighbouring tactical frames when delta is fractional), where both are present."""
    b = np.arange(b0, b1)
    fl = int(np.floor(delta))
    fr = float(delta - fl)
    t0 = b + fl
    ok = (t0 >= 0) & (t0 + 1 < T_TAC)
    b, t0 = b[ok], t0[ok]
    if fr == 0.0:
        m = S.bP[b] & S.tP[t0]
        return np.hypot(bx[b[m]] - S.tx[t0[m]], by[b[m]] - S.ty[t0[m]])
    m = S.bP[b] & S.tP[t0] & S.tP[t0 + 1]
    b, t0 = b[m], t0[m]
    x = (1 - fr) * S.tx[t0] + fr * S.tx[t0 + 1]
    y = (1 - fr) * S.ty[t0] + fr * S.ty[t0 + 1]
    return np.hypot(bx[b] - x, by[b] - y)


def chunk_delta_frac(S: Sources, bx, by, b0: int, b1: int, centre: int, half_range: float = 3.0, step: float = 0.25):
    """Fractional delta in centre +- half_range minimising the median ball distance over [b0,b1); NaN if <200 pairs."""
    best = (np.inf, np.nan)
    for s in np.arange(-half_range, half_range + 1e-9, step):
        d = ball_dist(S, bx, by, b0, b1, centre + s)
        if d.size < 200:
            return np.nan
        v = float(np.median(d))
        if v < best[0]:
            best = (v, centre + s)
    return best[1]


# ------------------------------------------------------------------------------------------------
UNMAPPED = -(1 << 30)
MIN_PLATEAU_CHUNKS = 3
SUPPORT_WIN = 251          # frames, window for plateau-extent evidence
SUPPORT_MIN_VALID = 25     # valid-homography frames needed inside the window
SUPPORT_THR = 0.35         # mean same-team-within-2m fraction needed (null level is ~0.03)


@dataclass
class SyncResult:
    flip: tuple
    flip_stats: dict
    chunks: list
    plateaus: list            # dicts: delta, delta_frac, b0, b1 (inclusive extent), t0, t1, n_chunks, chunk_ids
    dropped_groups: list      # confident chunk groups with < MIN_PLATEAU_CHUNKS chunks (replays / outliers)
    mapping: np.ndarray       # (T_TAC,) int32 broadcast frame or -1
    delta_of_b: np.ndarray    # (Tb,) int32, UNMAPPED where no plateau applies


def choose_flip(S: Sources, sample: int = 40, seed: int = 0):
    """Score the four axis flips on a seeded sample of chunks; return (best flip, per-flip stats)."""
    chunks = make_chunks(S.Tb)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(chunks), size=min(sample, len(chunks)), replace=False))
    stats = {}
    for fx in (False, True):
        for fy in (False, True):
            bx, by, _, _ = apply_flip(S, fx, fy)
            res = [r for r in search_chunks(S, bx, by, [chunks[i] for i in idx]) if r is not None]
            stats[(fx, fy)] = dict(n=len(res), confident=sum(is_confident(r) for r in res),
                                   median_n2=float(np.median([r["n2"] for r in res])) if res else 0.0)
    best = max(stats, key=lambda k: (stats[k]["confident"], stats[k]["median_n2"]))
    return best, stats


def group_chunks(chunks: list, tol: int = 12) -> list[list[dict]]:
    """Cluster confident chunks by delta (1-D, a new cluster starts where sorted deltas jump by more than tol)."""
    conf = sorted((r for r in chunks if is_confident(r)), key=lambda r: r["delta"])
    groups: list[list[dict]] = []
    for r in conf:
        if groups and r["delta"] - groups[-1][-1]["delta"] <= tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    for g in groups:
        g.sort(key=lambda r: r["b0"])
    return groups


def _rolling_nanmean(x: np.ndarray, win: int, min_n: int) -> np.ndarray:
    ok = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    h = win // 2
    lo = np.clip(np.arange(x.size) - h, 0, x.size)
    hi = np.clip(np.arange(x.size) + h + 1, 0, x.size)
    cnt = n[hi] - n[lo]
    return np.where(cnt >= min_n, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


MAX_GAP = 10000   # frames; unsupported gaps up to this length inside one delta are bridged (invalid stretches)


def build_plateaus(S: Sources, flip, chunks: list):
    """Plateaus = (delta, supported broadcast segments). One row per segment."""
    bx, by, bpx, bpy = apply_flip(S, *flip)
    groups = group_chunks(chunks)
    keep = [g for g in groups if len(g) >= MIN_PLATEAU_CHUNKS]
    dropped = [dict(delta=int(np.median([c["delta"] for c in g])), b0=g[0]["b0"], b1=g[-1]["b1"], n_chunks=len(g))
               for g in groups if len(g) < MIN_PLATEAU_CHUNKS]
    allb = np.arange(S.Tb)
    segs = []
    for g in keep:
        med = int(np.round(np.median([c["delta"] for c in g])))
        # pooled single-frame delta: smallest median ball distance over all frames of the group's chunks
        best = (np.inf, med)
        for dd in range(med - 4, med + 5):
            dist = np.concatenate([ball_dist(S, bx, by, c["b0"], c["b1"], dd) for c in g])
            if np.median(dist) < best[0]:
                best = (float(np.median(dist)), dd)
        delta = best[1]
        ps = player_support(S, bpx, bpy, allb, delta)
        w = _rolling_nanmean(ps, SUPPORT_WIN, SUPPORT_MIN_VALID)
        idx = np.flatnonzero((w >= SUPPORT_THR) & np.isfinite(ps) & (ps >= 0.5))
        if idx.size == 0:
            continue
        cut = np.flatnonzero(np.diff(idx) > MAX_GAP)
        starts = np.concatenate([[0], cut + 1])
        ends = np.concatenate([cut, [idx.size - 1]])
        for s_, e_ in zip(starts, ends):
            b0_, b1_ = int(idx[s_]), int(idx[e_])
            nc = sum(1 for c in g if c["b0"] >= b0_ and c["b1"] <= b1_ + 1)
            segs.append(dict(delta=delta, b0=b0_, b1=b1_, n_chunks=nc, n_chunks_group=len(g)))
    segs.sort(key=lambda p: p["b0"])
    for a_, b_ in zip(segs[:-1], segs[1:]):          # no overlaps in broadcast time
        if a_["b1"] >= b_["b0"]:
            cut = (a_["b1"] + b_["b0"]) // 2
            a_["b1"], b_["b0"] = cut, cut + 1
    for p in segs:
        p["t0"], p["t1"] = p["b0"] + p["delta"], p["b1"] + p["delta"]
    return segs, dropped


def mapping_from_plateaus(S: Sources, plateaus: list):
    delta_of_b = np.full(S.Tb, UNMAPPED, np.int32)
    mapping = np.full(T_TAC, -1, np.int32)
    for p in plateaus:
        b = np.arange(p["b0"], p["b1"] + 1)
        delta_of_b[b] = p["delta"]
        t = b + p["delta"]
        ok = (t >= 0) & (t < T_TAC)
        mapping[t[ok]] = b[ok]
    return delta_of_b, mapping


def run_sync() -> SyncResult:
    S = load_sources()
    flip, fstats = choose_flip(S)
    bx, by, _, _ = apply_flip(S, *flip)
    chunks = search_chunks(S, bx, by, make_chunks(S.Tb))
    plateaus, dropped = build_plateaus(S, flip, chunks)
    delta_of_b, mapping = mapping_from_plateaus(S, plateaus)
    return SyncResult(flip, fstats, chunks, plateaus, dropped, mapping, delta_of_b)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def _save(res: SyncResult) -> None:
    import json
    meta = json.dumps(_jsonable(dict(flip=list(res.flip), flip_stats=res.flip_stats, plateaus=res.plateaus,
                                     dropped=res.dropped_groups, chunks=res.chunks)))
    np.savez(cache_dir() / "sync_mapping.npz", version=CACHE_VERSION, mapping=res.mapping,
             delta_of_b=res.delta_of_b, meta=np.array(meta))


def load_sync(use_cache: bool = True) -> dict:
    """Cached sync products: dict(mapping, delta_of_b, flip, flip_stats, plateaus, dropped, chunks)."""
    import json
    path = cache_dir() / "sync_mapping.npz"
    if not (use_cache and path.exists()) or int(np.load(path, allow_pickle=False)["version"]) != CACHE_VERSION:
        _save(run_sync())
    z = np.load(path, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    return dict(mapping=z["mapping"], delta_of_b=z["delta_of_b"], flip=tuple(m["flip"]), flip_stats=m["flip_stats"],
                plateaus=m["plateaus"], dropped=m["dropped"], chunks=m["chunks"])


def load_mapping(use_cache: bool = True) -> np.ndarray:
    """int32 array (199319,): broadcast frame for each tactical frame, or -1. Cached under cache/."""
    return load_sync(use_cache)["mapping"]
