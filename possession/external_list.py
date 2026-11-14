"""Locate and load the cached external (DFL) matches."""
from __future__ import annotations

from .match import Match
from .paths import cache_dir


def external_ids() -> list[str]:
    return sorted(p.stem.removeprefix("dfl_") for p in (cache_dir() / "external").glob("dfl_*.npz"))


def load_external(ids=None) -> list[Match]:
    ids = ids or external_ids()
    return [Match.load(cache_dir() / "external" / f"dfl_{i}.npz") for i in ids]
