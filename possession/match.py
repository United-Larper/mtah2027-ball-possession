"""Common in-memory match format: tracking positions in metres plus per-frame possession labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

H, A, D, X = 0, 1, 2, 3  # label codes, same as evaluation/possession_metrics.py


@dataclass
class Match:
    source: str                      # "dfl", "atd", ...
    match_id: str
    fps: float                       # 25.0
    pitch: tuple                     # (length_m, width_m)
    hx: np.ndarray                   # float32 [T, Nh], metres, pitch-centred, NaN = not tracked
    hy: np.ndarray
    ax: np.ndarray                   # float32 [T, Na]
    ay: np.ndarray
    bx: np.ndarray                   # float32 [T], NaN where missing
    by: np.ndarray
    bz: np.ndarray                   # float32 [T], all-NaN if the source has no height
    label: np.ndarray                # int8 [T], 0=H 1=A 2=D 3=X
    period: np.ndarray               # int8 [T], 1 or 2 (halves concatenated, no filler)
    home_dir: dict = field(default_factory=dict)   # {1: +1|-1, 2: +1|-1}
    frame_id: object = None          # optional int32 [T], source frame number

    @property
    def T(self) -> int:
        return len(self.label)

    def period_slices(self) -> dict:
        out = {}
        for p in (1, 2):
            idx = np.flatnonzero(self.period == p)
            if len(idx):
                out[p] = slice(int(idx[0]), int(idx[-1]) + 1)
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            source=np.array(self.source), match_id=np.array(self.match_id),
            fps=np.array(self.fps, dtype=np.float64),
            pitch=np.array(self.pitch, dtype=np.float64),
            hx=self.hx.astype(np.float32), hy=self.hy.astype(np.float32),
            ax=self.ax.astype(np.float32), ay=self.ay.astype(np.float32),
            bx=self.bx.astype(np.float32), by=self.by.astype(np.float32),
            bz=self.bz.astype(np.float32),
            label=self.label.astype(np.int8), period=self.period.astype(np.int8),
            home_dir=np.array([self.home_dir[1], self.home_dir[2]], dtype=np.int8),
        )
        if self.frame_id is not None:
            arrays["frame_id"] = np.asarray(self.frame_id).astype(np.int32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Match":
        with np.load(Path(path), allow_pickle=False) as z:
            hd = z["home_dir"]
            return cls(
                source=str(z["source"]), match_id=str(z["match_id"]), fps=float(z["fps"]),
                pitch=(float(z["pitch"][0]), float(z["pitch"][1])),
                hx=z["hx"], hy=z["hy"], ax=z["ax"], ay=z["ay"],
                bx=z["bx"], by=z["by"], bz=z["bz"],
                label=z["label"], period=z["period"],
                home_dir={1: int(hd[0]), 2: int(hd[1])},
                frame_id=z["frame_id"] if "frame_id" in z.files else None,
            )


def load(path) -> Match:
    return Match.load(path)
