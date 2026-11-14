"""Homography loader and vectorised pitch<->pixel helpers (normalised pitch 0-1 <-> 1920x1080 px)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import asset_paths

_COLS = [f"h{r}{c}" for r in range(3) for c in range(3)]


def load_homography(source: str = "tactical") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H (T,3,3) float64, reprojection_error (T,), valid (T,) bool). Invalid rows are NaN matrices."""
    t = pd.read_csv(asset_paths(source)["homography"])
    return (t[_COLS].to_numpy(np.float64).reshape(-1, 3, 3),
            t["reprojection_error"].to_numpy(np.float64),
            t["valid"].to_numpy().astype(bool))


def _apply(M: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-frame (or single) 3x3 matrices M to x, y (broadcast over leading axes)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if M.ndim == 2:
        M = M[None]
    w = M[..., 2, 0] * x + M[..., 2, 1] * y + M[..., 2, 2]
    return ((M[..., 0, 0] * x + M[..., 0, 1] * y + M[..., 0, 2]) / w,
            (M[..., 1, 0] * x + M[..., 1, 1] * y + M[..., 1, 2]) / w)


def pitch_to_pixel(H: np.ndarray, x, y):
    return _apply(H, x, y)


def pixel_to_pitch(H: np.ndarray, u, v):
    """Inverse mapping (pixels -> normalised pitch). NaN/singular matrices give NaN."""
    H = np.asarray(H, np.float64)
    single = H.ndim == 2
    if single:
        H = H[None]
    inv = np.full_like(H, np.nan)
    ok = np.isfinite(H).all(axis=(-1, -2))
    ok[ok] = np.abs(np.linalg.det(H[ok])) > 1e-18
    inv[ok] = np.linalg.inv(H[ok])
    x, y = _apply(inv, u, v)
    return (x[0], y[0]) if single else (x, y)
