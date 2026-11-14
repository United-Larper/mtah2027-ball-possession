"""Parser for the open DFL/IDSSE tracking data (Bassek et al. 2025, CC BY 4.0).

load_dfl_match(positions_xml, matchinfo_xml) -> Match.
load_dfl_match_with_report(...) -> (Match, diagnostics dict).
"""
from __future__ import annotations

import array
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from possession.match import Match, H, A, D, X  # noqa: E402

FPS = 25.0
SECTIONS = {"firstHalf": 1, "secondHalf": 2}


def parse_matchinfo(path) -> dict:
    root = ET.parse(str(path)).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    teams = {}
    for team in root.iter("Team"):
        role = team.get("Role")
        if role in ("home", "guest"):
            teams[role] = {
                "team_id": team.get("TeamId"), "name": team.get("TeamName"),
                "persons": [p.get("PersonId") for p in team.iter("Player")],
            }
    if set(teams) != {"home", "guest"}:
        raise ValueError(f"match info lacks home/guest Team roles: {sorted(teams)}")
    if general is not None and general.get("HomeTeamId") not in (None, teams["home"]["team_id"]):
        raise ValueError("General/@HomeTeamId disagrees with Team Role=home")
    pitch = (float(env.get("PitchX", 105.0)), float(env.get("PitchY", 68.0))) if env is not None else (105.0, 68.0)
    return {"match_id": general.get("MatchId") if general is not None else None,
            "home": teams["home"], "away": teams["guest"], "pitch": pitch}


def _fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _stream(positions_xml, home_id, away_id):
    """Single streaming pass. Returns (players, ball, header_pitch).

    players[(section, 'H'/'A', person)] -> (N, X, Y)
    ball[section] -> list of dicts of arrays (one per FrameSet that carries ball frames)
    Referees and anything else are skipped.
    """
    players, ball = {}, {}
    cur = None
    header_pitch = None
    for ev, el in ET.iterparse(str(positions_xml), events=("start", "end")):
        tag = el.tag
        if ev == "start":
            if tag == "Frame":
                if cur is None:
                    continue
                at = el.attrib
                if "BallStatus" in at:
                    b = cur["ball"]
                    b["N"].append(int(at["N"])); b["X"].append(_fnum(at.get("X")))
                    b["Y"].append(_fnum(at.get("Y"))); b["Z"].append(_fnum(at.get("Z")))
                    b["P"].append(int(at.get("BallPossession", -1)))
                    b["S"].append(int(at.get("BallStatus", -1)))
                elif cur["team"] is not None:
                    cur["N"].append(int(at["N"]))
                    cur["X"].append(_fnum(at.get("X"))); cur["Y"].append(_fnum(at.get("Y")))
            elif tag == "FrameSet":
                at = el.attrib
                tid = at.get("TeamId")
                team = "H" if tid == home_id else ("A" if tid == away_id else None)
                cur = {"section": at.get("GameSection"), "person": at.get("PersonId"),
                       "team": team, "N": array.array("q"), "X": array.array("d"),
                       "Y": array.array("d"),
                       "ball": {"N": array.array("q"), "X": array.array("d"), "Y": array.array("d"),
                                "Z": array.array("d"), "P": array.array("b"), "S": array.array("b")}}
            elif tag == "PitchSize":
                header_pitch = (float(el.get("X")), float(el.get("Y")))
        else:
            if tag == "Frame":
                el.clear()
            elif tag == "FrameSet":
                if cur is not None:
                    if len(cur["ball"]["N"]):
                        ball.setdefault(cur["section"], []).append(
                            {k: np.array(v) for k, v in cur["ball"].items()})
                    if cur["team"] is not None and len(cur["N"]):
                        players[(cur["section"], cur["team"], cur["person"])] = (
                            np.array(cur["N"]), np.array(cur["X"]), np.array(cur["Y"]))
                cur = None
                el.clear()
    return players, ball, header_pitch


def _extreme_means(px, step):
    """Mean over sampled frames of the per-frame min and max x of a team."""
    sub = px[::step]
    ok = ~np.all(np.isnan(sub), axis=1)
    sub = sub[ok]
    if len(sub) == 0:
        return np.nan, np.nan
    return float(np.mean(np.nanmin(sub, axis=1))), float(np.mean(np.nanmax(sub, axis=1)))


def _home_dir(hx, ax, fps):
    """+1 if home attacks towards +x. Kickoff-half mean x, cross-checked with deepest player."""
    n = min(len(hx), int(5 * fps))
    h5 = float(np.nanmean(hx[:n])); a5 = float(np.nanmean(ax[:n]))
    kick = 1 if h5 < a5 else -1  # home starts on -x half => defends -x, attacks +x
    mn, mx = _extreme_means(hx, max(1, int(fps)))
    deepest_side = -1 if abs(mn) > abs(mx) else 1  # side of home's deepest player (keeper)
    gk = -deepest_side
    return kick, {"home_mean_x_first5s": round(h5, 2), "away_mean_x_first5s": round(a5, 2),
                  "home_mean_min_x": round(mn, 1), "home_mean_max_x": round(mx, 1),
                  "kickoff_dir": kick, "deepest_player_dir": gk, "agree": kick == gk}


def load_dfl_match_with_report(positions_xml, matchinfo_xml):
    info = parse_matchinfo(matchinfo_xml)
    players, ball, header_pitch = _stream(positions_xml, info["home"]["team_id"],
                                          info["away"]["team_id"])
    pitch = info["pitch"]
    rep = {"match_id": info["match_id"], "pitch": pitch, "header_pitch": header_pitch,
           "pitch_matches_header": header_pitch is None or tuple(header_pitch) == tuple(pitch),
           "periods": {}, "crosstab": {}, "home_dir_evidence": {}}

    # Columns per team: order of first appearance, first half first.
    cols = {"H": [], "A": []}
    for (sec, team, pid) in sorted(players, key=lambda k: SECTIONS.get(k[0], 9)):
        if sec in SECTIONS and pid not in cols[team]:
            cols[team].append(pid)
    col_idx = {t: {p: i for i, p in enumerate(cols[t])} for t in cols}

    parts = {k: [] for k in ("hx", "hy", "ax", "ay", "bx", "by", "bz", "label", "period", "fid")}
    home_dir = {}
    for sec, pnum in SECTIONS.items():
        keys = [k for k in players if k[0] == sec]
        bsets = ball.get(sec, [])
        if not keys and not bsets:
            continue
        all_ns = [players[k][0] for k in keys] + [b["N"] for b in bsets]
        union = np.unique(np.concatenate(all_ns))
        lo, hi = int(union[0]), int(union[-1])
        T = hi - lo + 1
        dup_total = int(sum(len(n) - len(np.unique(n)) for n in all_ns))
        ball_n = np.unique(np.concatenate([b["N"] for b in bsets])) if bsets else np.array([], np.int64)
        p_union = np.unique(np.concatenate([players[k][0] for k in keys])) if keys else np.array([], np.int64)
        rep["periods"][pnum] = {
            "N_range": (lo, hi), "T": T, "gaps_in_union": int(T - len(union)),
            "duplicate_frame_records": dup_total, "n_ball_framesets": len(bsets),
            "ball_frames": int(len(ball_n)), "player_union_frames": int(len(p_union)),
            "ball_not_in_players": int(len(np.setdiff1d(ball_n, p_union))),
            "players_not_in_ball": int(len(np.setdiff1d(p_union, ball_n))),
            "n_player_framesets": len(keys)}
        arr = {t: [np.full((T, len(cols[t])), np.nan, np.float32) for _ in range(2)] for t in "HA"}
        for key in keys:
            n, x, y = players[key]
            j = col_idx[key[1]][key[2]]
            arr[key[1]][0][n - lo, j] = x
            arr[key[1]][1][n - lo, j] = y
        bx = np.full(T, np.nan, np.float32); by = bx.copy(); bz = bx.copy()
        poss = np.full(T, -1, np.int8); stat = np.full(T, -1, np.int8)
        for b in bsets:
            i = b["N"] - lo
            bx[i], by[i], bz[i], poss[i], stat[i] = b["X"], b["Y"], b["Z"], b["P"], b["S"]
        label = np.full(T, X, np.int8)
        label[stat == 0] = D
        label[(stat == 1) & (poss == 1)] = H
        label[(stat == 1) & (poss == 2)] = A
        ct = {f"status{sv}_poss{pv}": int(((stat == sv) & (poss == pv)).sum())
              for sv in (0, 1) for pv in (1, 2)}
        ct["other_or_missing"] = int(T - sum(ct.values()))
        rep["crosstab"][pnum] = ct
        d, ev = _home_dir(arr["H"][0], arr["A"][0], FPS)
        home_dir[pnum] = d
        rep["home_dir_evidence"][pnum] = ev
        parts["hx"].append(arr["H"][0]); parts["hy"].append(arr["H"][1])
        parts["ax"].append(arr["A"][0]); parts["ay"].append(arr["A"][1])
        parts["bx"].append(bx); parts["by"].append(by); parts["bz"].append(bz)
        parts["label"].append(label)
        parts["period"].append(np.full(T, pnum, np.int8))
        parts["fid"].append(np.arange(lo, hi + 1, dtype=np.int32))
    if set(home_dir) == {1, 2}:
        rep["home_dir_flips_between_halves"] = home_dir[1] == -home_dir[2]
    rep["n_player_columns"] = {"H": len(cols["H"]), "A": len(cols["A"])}
    m = Match(source="dfl", match_id=info["match_id"], fps=FPS, pitch=pitch,
              hx=np.concatenate(parts["hx"]), hy=np.concatenate(parts["hy"]),
              ax=np.concatenate(parts["ax"]), ay=np.concatenate(parts["ay"]),
              bx=np.concatenate(parts["bx"]), by=np.concatenate(parts["by"]),
              bz=np.concatenate(parts["bz"]), label=np.concatenate(parts["label"]),
              period=np.concatenate(parts["period"]), home_dir=home_dir,
              frame_id=np.concatenate(parts["fid"]))
    return m, rep


def load_dfl_match(positions_xml, matchinfo_xml) -> Match:
    return load_dfl_match_with_report(positions_xml, matchinfo_xml)[0]
