"""Parse the Metrica FIFA/EPTS ATD (metadata XML + raw text) into numpy arrays, cached as .npz.

Raw line: ``<frame_id>:<player>;<player>;...:<ball>``; each group is comma separated. The number of
fields per group and the slot order are read from the metadata's DataFormatSpecification and
checked against the raw file. Row index 0 is video frame 0 (frame_id 1).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import asset_paths, cache_dir
import games


def _nm(e: ET.Element) -> str:
    return e.tag.split("}")[-1]


@dataclass
class ATD:
    px: np.ndarray      # (T, S) float32, normalised x, NaN missing
    py: np.ndarray
    ps: np.ndarray      # player speed field (kph per metadata)
    bx: np.ndarray      # (T,) float64 (kept in double so second differences are exact)
    by: np.ndarray
    bextra: np.ndarray  # (T, K) float64, ball fields beyond x,y
    team: np.ndarray    # (S,) int8, 0 home 1 away
    fps: float
    pitch_m: tuple[float, float]
    team_names: tuple[str, str]
    slot_ids: list[str]
    ball_fields: list[str]
    player_fields: list[str]
    meta: dict[str, str]

    @property
    def T(self) -> int:
        return self.px.shape[0]


def _parse_meta(path: Path) -> dict:
    root = ET.parse(path).getroot()
    channels = {}
    for e in root.iter():
        if _nm(e) == "Channel":
            channels[e.get("id")] = e.findtext("Name") or e.get("id")
    pch = {e.get("id"): (e.get("playerId"), e.get("channelId"))
           for e in root.iter() if _nm(e) == "PlayerChannel"}
    teams = [e for e in root.iter() if _nm(e) == "Team"]
    team_ids = [t.get("id") for t in teams]
    team_names = [(t.findtext("Name") or t.get("id")) for t in teams]
    players = {e.get("id"): e.get("teamId") for e in root.iter() if _nm(e) == "Player"}
    spec = next(e for e in root.iter() if _nm(e) == "DataFormatSpecification")
    regs = [c for c in spec if _nm(c) == "SplitRegister"]
    player_reg, ball_reg = regs[0], regs[1]
    slot_ids: list[str] = []
    player_fields: list[str] = []
    for i, grp in enumerate(player_reg):
        refs = [pch[r.get("playerChannelId")] for r in grp]
        slot_ids.append(refs[0][0])
        assert all(r[0] == refs[0][0] for r in refs)
        if i == 0:
            player_fields = [channels[r[1]] for r in refs]
    ball_fields = [channels[r.get("channelId")] for r in ball_reg.iter() if _nm(r) == "BallChannelRef"]
    meta = {}
    for pp in root.iter():
        if _nm(pp) == "ProviderParameter":
            n, v = pp.findtext("Name"), pp.findtext("Value")
            if n in ("attack_direction_first_half", "tracking_source", "data_type", "version"):
                meta[n] = v or ""
    score = next(e for e in root.iter() if _nm(e) == "Score")
    return dict(
        fps=float(next(e.text for e in root.iter() if _nm(e) == "FrameRate")),
        pitch_m=(float(next(e.text for e in root.iter() if _nm(e) == "Width")),
                 float(next(e.text for e in root.iter() if _nm(e) == "Height"))),
        team_ids=team_ids, team_names=team_names, home_id=score.get("idLocalTeam"),
        team=np.array([0 if players[s] == team_ids[0] else 1 for s in slot_ids], np.int8),
        slot_ids=slot_ids, player_fields=player_fields, ball_fields=ball_fields,
        meta=meta, n_spec_frames=int(spec.get("endFrame")))


def _parse_raw(path: Path, n_slots: int, n_pf: int, n_bf: int, T: int):
    pa = np.full((T, n_slots * n_pf), np.nan, np.float32)
    ba = np.full((T, n_bf), np.nan, np.float64)
    ids = np.empty(T, np.int64)
    n = 0
    with open(path, "r") as f:
        for line in f:
            if n >= T:
                raise ValueError("more lines than expected")
            parts = line.rstrip("\r\n").split(":")
            ids[n] = int(parts[0])
            pv = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float32)
            if pv.size == pa.shape[1]:
                pa[n] = pv
            else:
                raise ValueError("unexpected player field count")
            if len(parts) > 2 and parts[2]:
                bv = np.fromstring(parts[2].replace(";", ","), sep=",", dtype=np.float64)
                ba[n, :min(bv.size, n_bf)] = bv[:n_bf]
            n += 1
    if n != T:
        raise ValueError("line count mismatch")
    return pa, ba, ids


def load_atd(source: str = "tactical", use_cache: bool = True) -> ATD:
    paths = asset_paths(source)
    cache = cache_dir() / f"atd_{source}.npz"
    expected = games.source_frames(source)
    m = _parse_meta(paths["atd_meta"])
    S, pf, bf = len(m["slot_ids"]), len(m["player_fields"]), len(m["ball_fields"])
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        px, py, ps = z["px"], z["py"], z["ps"]
        bx, by, bextra = z["bx"], z["by"], z["bextra"]
    else:
        pa, ba, ids = _parse_raw(paths["atd_raw"], S, pf, bf, m["n_spec_frames"])
        assert len(ids) == expected, f"T={len(ids)} != {expected}"
        assert ids[0] == 1 and np.all(np.diff(ids) == 1), "frame ids not contiguous from 1"
        pa = pa.reshape(len(ids), S, pf)
        px, py = np.ascontiguousarray(pa[..., 0]), np.ascontiguousarray(pa[..., 1])
        ps = np.ascontiguousarray(pa[..., 2]) if pf > 2 else np.full_like(px, np.nan)
        bx, by, bextra = ba[:, 0].copy(), ba[:, 1].copy(), ba[:, 2:].copy()
        np.savez(cache, px=px, py=py, ps=ps, bx=bx, by=by, bextra=bextra)
    return ATD(px=px, py=py, ps=ps, bx=bx, by=by, bextra=bextra, team=m["team"], fps=m["fps"],
               pitch_m=m["pitch_m"], team_names=tuple(m["team_names"][:2]), slot_ids=m["slot_ids"],
               ball_fields=m["ball_fields"], player_fields=m["player_fields"], meta=m["meta"])
