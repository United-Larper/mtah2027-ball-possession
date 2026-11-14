"""Locate the match data, cache directory and per-source asset paths."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # games.py is vendored at the repo root
    sys.path.insert(0, str(REPO))

import games  # noqa: E402

CACHE_DIR = REPO / "cache"


def data_dir() -> Path:
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        return Path(env)
    if (REPO / "data" / "tactical").is_dir():
        return REPO / "data"
    return REPO.parent / "data"


def asset_paths(source: str = "tactical") -> dict[str, Path]:
    """Asset name -> path. Accepts videos named `<stem>-001.mp4` as well as `<stem>.mp4`."""
    paths = {k: Path(v) for k, v in games.asset_paths(data_dir=str(data_dir()), source=source).items()}
    video = paths["video"]
    if not video.exists():
        alt = video.with_name(video.stem + "-001" + video.suffix)
        if alt.exists():
            paths["video"] = alt
    return paths


def cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR
