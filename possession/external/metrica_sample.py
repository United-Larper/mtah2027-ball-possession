"""Metrica Sports public sample games (3 anonymised matches, 25 fps) -> common Match format with
per-frame possession labels derived from the event data.

Source: github.com/metrica-sports/sample-data (no LICENSE file; the README asks users to "be
responsible with the use of this data" and to acknowledge the source if used for anything public).
Local copy: <project>/../external/metrica-sample/data (outside the project, never committed).

    load_metrica_sample(game) -> Match      game in {1, 2, 3}
    load_tracking(game), load_events(game)  raw pieces; derive_labels(tr, ev, ...) the label rules

Formats. Games 1, 2: CSV tracking (home / away files; header rows: team, jersey, names), CSV events.
Game 3: EPTS tracking (frame:players;...:ball, player-to-slot mapping changes with substitutions,
read from the 11 DataFormatSpecification blocks of the metadata) + JSON events. All three are 1-based
frame numbers at exactly 25 fps, periods concatenated (period 1 then 2), so no resampling is needed.
Coordinates: normalised 0-1, origin top-left, pitch 105 x 68 m -> metres, pitch-centred, x right,
y UP (y is flipped). Match.frame_id = source frame number. home_dir per period is decided from the
data exactly as possession/external/dfl.py does (mean x of both teams over the first 5 s, cross-checked
with the deepest player). Home = "Home" (games 1, 2) / the EPTS local team (game 3).

Frame alignment: an event with Start Frame f refers to tracking Frame f (array index f-1, no
offset): the passer is the nearest player to the ball at PASS start frames 97.6 / 98.8 / 92.1 %
(games 1/2/3) at offset 0; offset +1 is equal within 0.1 pt (97.4 / 98.8 / 92.2 %), -1 is 2-3 pt
lower, +-3 is 5-8 pt lower (see check_alignment). Offset 0 is used, i.e. array index = frame - 1.

LABEL RULES (derive_labels)
1. Dead ball (D) = event-derived "ball not in play" flag (event_dead_mask), NOT raw ball-NaN:
   a stoppage starts the frame after the END of a stopper event: BALL OUT; SHOT whose subtype has OUT
   or GOAL; FAULT RECEIVED; CHALLENGE with FAULT; BALL LOST with FORCED / OFFSIDE / END HALF /
   HAND BALL; PASS with OFFSIDE. It lasts until the frame BEFORE the next restart event in that
   period (first SET PIECE, or RECOVERY / PASS / CARRY / SHOT at or after the stoppage start; the
   goal kick is a RECOVERY + PASS GOAL KICK, not a SET PIECE). Several stoppers before one restart
   form one stoppage that starts after the latest of their ends. Extra rule: every SET PIECE that
   is not preceded by a dead frame is taken to restart a dead ball that started when the previous
   event ended (keeper delay / unlogged stoppage). Frames before the first event of a period
   (pre-kick-off) and after the end of its last event (after the whistle) are D (the competition
   also marks the 2,934 pre-kick-off frames inside its graded second half as D). So no X remains.
   Raw ball-NaN is the alternative (dead='nan'); the two agree on 96.7-99.7 % of frames (report M2/M3/M6).
2. Team in possession on alive frames = team of the most recent control-implying event: PASS, CARRY
   (game 3 only), SHOT, RECOVERY, SET PIECE, and the start of a BALL LOST / BALL OUT (their team had
   the ball). CHALLENGE (won or lost), CARD and FAULT RECEIVED alone confer nothing. Possession
   therefore switches at the START frame of the other team's first control event, typically its
   RECOVERY; a loose ball between a BALL LOST end and the next RECOVERY stays with the team that
   lost it (the definition: the last team keeps a loose ball until the other gains control).
   Same-frame ties: BALL LOST is applied before the other events of that frame.
Alternatives kept for the sensitivity analysis: dead in {'nan','nan_long','union'},
switch='lostend' (a BALL LOST hands the ball to the opponent at its END frame), challenge_won=True.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from possession.match import Match, H, A, D, X  # noqa: E402

FPS = 25.0
PITCH = (105.0, 68.0)
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "external" / "metrica-sample" / "data"


def _game_dir(game, root):
    return Path(root) / f"Sample_Game_{game}"


def _to_m(nx, ny):
    """Normalised (origin top-left, y down) -> metres, pitch-centred, x right, y UP."""
    return (nx - 0.5) * PITCH[0], (0.5 - ny) * PITCH[1]


# --------------------------------------------------------------------------------------- tracking

def _read_csv_team(path):
    raw = pd.read_csv(path, skiprows=2, header=None, low_memory=False)
    names = [str(v) for v in raw.iloc[0].tolist()]
    body = raw.iloc[1:].reset_index(drop=True)
    period = body[0].astype(int).to_numpy()
    frame = body[1].astype(int).to_numpy()
    time = body[2].astype(float).to_numpy()
    vals = body.iloc[:, 3:].astype(float).to_numpy()
    n_pl = (vals.shape[1] - 2) // 2
    pnames = [names[3 + 2 * i] for i in range(n_pl)]
    x = vals[:, 0:2 * n_pl:2]
    y = vals[:, 1:2 * n_pl:2]
    return period, frame, time, pnames, x, y, vals[:, -2], vals[:, -1]


def _tracking_csv(game, root):
    d = _game_dir(game, root)
    ph, fh, th, nh, hx, hy, bxh, byh = _read_csv_team(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    pa, fa, ta, na, ax, ay, bxa, bya = _read_csv_team(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    if not (np.array_equal(ph, pa) and np.array_equal(fh, fa)):
        raise ValueError("home/away tracking files are not frame aligned")
    bx = np.where(np.isnan(bxh), bxa, bxh)
    by = np.where(np.isnan(byh), bya, byh)
    return dict(period=ph, frame=fh, time=th, hnames=nh, anames=na, hx=hx, hy=hy, ax=ax, ay=ay, bx=bx, by=by)


def _tracking_epts(game, root):
    d = _game_dir(game, root)
    meta = ET.parse(str(d / f"Sample_Game_{game}_metadata.xml")).getroot()
    tag = lambda e: e.tag.split("}")[-1]
    fps = float([e.text for e in meta.iter() if tag(e) == "FrameRate"][0])
    pid_team = {}
    teams = {}
    for e in meta.iter():
        if tag(e) == "Player":
            pid_team[e.get("id")] = e.get("teamId")
        if tag(e) == "Score":
            teams = {"H": e.get("idLocalTeam"), "A": e.get("idVisitingTeam")}
    chan_pid = {e.get("id"): e.get("playerId") for e in meta.iter() if tag(e) == "PlayerChannel"}
    params = {}
    for e in meta.iter():
        if tag(e) == "ProviderParameter":
            n = e.find("Name")
            v = e.find("Value")
            if n is not None and v is not None and n.text in ("first_half_start", "first_half_end",
                                                              "second_half_start", "second_half_end"):
                params[n.text] = int(v.text)
    specs = []
    for e in meta.iter():
        if tag(e) == "DataFormatSpecification":
            order = []
            for r in e.iter():
                if tag(r) == "PlayerChannelRef":
                    cid = r.get("playerChannelId")
                    if cid.endswith("_x"):
                        order.append(chan_pid[cid])
            specs.append((int(e.get("startFrame")), int(e.get("endFrame")), order))
    team_of = lambda p: "H" if pid_team[p] == teams["H"] else "A"
    cols = {"H": [], "A": []}
    for _, _, order in specs:
        for pid in order:
            if pid not in cols[team_of(pid)]:
                cols[team_of(pid)].append(pid)
    ci = {t: {p: i for i, p in enumerate(cols[t])} for t in cols}
    slot_maps = [(s, e_, [(team_of(p), ci[team_of(p)][p]) for p in order]) for s, e_, order in specs]
    n_frames = specs[-1][1]
    arr = {t: (np.full((n_frames, len(cols[t])), np.nan), np.full((n_frames, len(cols[t])), np.nan)) for t in cols}
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    sm_i = 0
    with open(d / f"Sample_Game_{game}_tracking.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            fr_s, pl_s, ba_s = line.rstrip("\n").split(":")
            fr = int(fr_s)
            while slot_maps[sm_i][1] < fr:
                sm_i += 1
            smap = slot_maps[sm_i][2]
            toks = np.array(pl_s.replace(";", ",").split(","), dtype=float)
            for k, (t, c) in enumerate(smap):
                arr[t][0][fr - 1, c] = toks[2 * k]
                arr[t][1][fr - 1, c] = toks[2 * k + 1]
            b = ba_s.split(",")
            bx[fr - 1] = float(b[0])
            by[fr - 1] = float(b[1])
    frame = np.arange(1, n_frames + 1)
    period = np.where(frame <= params["first_half_end"], 1, 2).astype(int)
    return dict(period=period, frame=frame, time=frame / fps, hnames=cols["H"], anames=cols["A"],
                hx=arr["H"][0], hy=arr["H"][1], ax=arr["A"][0], ay=arr["A"][1], bx=bx, by=by, fps=fps)


def load_tracking(game, root=DEFAULT_ROOT):
    """Raw (normalised) tracking dict for a game; keys period, frame, time, hnames, anames, hx.. by."""
    t = _tracking_csv(game, root) if game in (1, 2) else _tracking_epts(game, root)
    t.setdefault("fps", float(np.round(1.0 / np.median(np.diff(t["time"])), 3)))
    return t


# --------------------------------------------------------------------------------------- events

EV_COLS = ["team", "type", "sub", "period", "sf", "ef", "frm", "to", "sx", "sy", "ex", "ey"]


def load_events(game, root=DEFAULT_ROOT):
    """Unified events DataFrame. team in {H, A}; sub = '|'-joined sorted subtype tokens;
    sx..ey in metres (pitch-centred, y up); frames are the source's own (1-based)."""
    d = _game_dir(game, root)
    if game in (1, 2):
        e = pd.read_csv(d / f"Sample_Game_{game}_RawEventsData.csv")
        sub = e["Subtype"].fillna("").map(lambda s: "|".join(sorted(t for t in s.split("-") if t)))
        sx, sy = _to_m(e["Start X"].to_numpy(float), e["Start Y"].to_numpy(float))
        ex, ey = _to_m(e["End X"].to_numpy(float), e["End Y"].to_numpy(float))
        out = pd.DataFrame(dict(team=e["Team"].map({"Home": "H", "Away": "A"}), type=e["Type"], sub=sub,
                                period=e["Period"].astype(int), sf=e["Start Frame"].astype(int),
                                ef=e["End Frame"].astype(int), frm=e["From"], to=e["To"],
                                sx=sx, sy=sy, ex=ex, ey=ey))
    else:
        js = json.load(open(d / f"Sample_Game_{game}_events.json", encoding="utf-8"))["data"]
        meta = ET.parse(str(d / f"Sample_Game_{game}_metadata.xml")).getroot()
        sc = [e for e in meta.iter() if e.tag.split("}")[-1] == "Score"][0]
        home_id = sc.get("idLocalTeam")
        rows = []
        for x in js:
            st = x["subtypes"]
            toks = [] if st is None else ([y["name"] for y in st] if isinstance(st, list) else [st["name"]])
            g = lambda o, k: np.nan if o is None or o.get(k) is None else o[k]
            sx, sy = _to_m(g(x["start"], "x"), g(x["start"], "y"))
            ex, ey = _to_m(g(x["end"], "x"), g(x["end"], "y"))
            rows.append(dict(team="H" if x["team"]["id"] == home_id else "A", type=x["type"]["name"],
                             sub="|".join(sorted(toks)), period=int(x["period"]), sf=int(x["start"]["frame"]),
                             ef=int(x["end"]["frame"]), frm=None if x["from"] is None else x["from"]["id"],
                             to=None if x["to"] is None else x["to"]["id"], sx=sx, sy=sy, ex=ex, ey=ey))
        out = pd.DataFrame(rows)
    return out[EV_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------------------- labels
CONTROL_TYPES = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE")
RESTART_TYPES = ("SET PIECE", "RECOVERY", "PASS", "CARRY", "SHOT")
BL_STOP_SUBS = ("FORCED", "OFFSIDE", "END HALF", "HAND BALL")


def _is_stopper(r):
    sub = set(r["sub"].split("|")) if r["sub"] else set()
    t = r["type"]
    if t in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if t == "SHOT" and ({"OUT", "GOAL"} & sub):
        return True
    if t == "BALL LOST" and (set(BL_STOP_SUBS) & sub):
        return True
    if t == "PASS" and "OFFSIDE" in sub:
        return True
    if t == "CHALLENGE" and "FAULT" in sub:
        return True
    return False


def event_dead_mask(tr, ev, restart_types=RESTART_TYPES, setpiece_rule=True):
    """Boolean per tracking frame: True where the event data say the ball is not in play.

    A stoppage starts the frame after the END of a stopper event (BALL OUT; SHOT that goes OUT or
    scores; FAULT RECEIVED; CHALLENGE...FAULT; BALL LOST FORCED / OFFSIDE / END HALF / HAND BALL;
    PASS OFFSIDE) and lasts until the frame before the next restart event (SET PIECE, or the
    goal-kick style RECOVERY, or any other PASS/CARRY/SHOT) in the same period. Frames of a period
    before its first event (pre-kick-off) and after its last event's end (after the whistle) are dead.
    """
    f0 = int(tr["frame"][0])
    T = len(tr["frame"])
    dead = np.zeros(T, dtype=bool)
    for p in (1, 2):
        pidx = np.flatnonzero(tr["period"] == p)
        if not len(pidx):
            continue
        lo, hi = int(pidx[0]), int(pidx[-1])            # array indices, inclusive
        e = ev[ev.period == p].sort_values("sf", kind="stable").reset_index(drop=True)
        if not len(e):
            dead[lo:hi + 1] = True
            continue
        sf = e.sf.to_numpy() - f0
        ef = np.maximum(e.ef.to_numpy(), e.sf.to_numpy()) - f0
        dead[lo:min(sf.min(), hi + 1)] = True            # before the first event
        dead[max(ef.max() + 1, lo):hi + 1] = True        # after the last event
        stop = np.array([_is_stopper(r) for _, r in e.iterrows()])
        types = e["type"].to_numpy()
        restart = np.isin(types, restart_types)
        k = 0
        n = len(e)
        while k < n:
            if not stop[k]:
                k += 1
                continue
            start = ef[k] + 1
            cand = np.flatnonzero(restart & (sf >= start))
            end = (sf[cand[0]] if len(cand) else hi + 1)
            j = k
            while j + 1 < n and sf[j + 1] < end:
                j += 1
                if stop[j]:
                    start = max(start, ef[j] + 1)
            if start < end:
                dead[start:end] = True
            k = j + 1
        # a SET PIECE always restarts a dead ball: if no stopper event explains it, the ball was
        # dead from the end of the previous event (e.g. keeper delay, an unlogged stoppage)
        if setpiece_rule:
            setp = np.flatnonzero(types == "SET PIECE")
            for k in setp:
                if k == 0 or sf[k] - 1 < lo or dead[sf[k] - 1]:
                    continue
                prev_end = int(ef[:k].max()) + 1
                if prev_end < sf[k]:
                    dead[prev_end:sf[k]] = True
    return dead


def event_team(tr, ev, switch="recovery", challenge_won=False, ball_lost_ctrl=True, recovery_ctrl=True):
    """Per tracking frame, the team (0=H, 1=A, -1=none yet) of the most recent control event.

    switch='recovery': every PASS/CARRY/SHOT/RECOVERY/SET PIECE/BALL LOST/BALL OUT sets the team
    at its start frame.  switch='lostend': as above, but a BALL LOST additionally hands the ball
    to the OTHER team at its END frame (unless FORCED/OFFSIDE/END HALF/HAND BALL/REFEREE HIT).
    challenge_won=True: a CHALLENGE ... WON also confers possession on its team (sensitivity only).
    """
    f0 = int(tr["frame"][0])
    T = len(tr["frame"])
    items = []   # (frame_idx, priority, team)
    for _, r in ev.iterrows():
        t = 0 if r.team == "H" else 1
        s = int(r.sf) - f0
        sub = set(r["sub"].split("|")) if r["sub"] else set()
        if r["type"] == "RECOVERY" and not recovery_ctrl:
            continue
        if r["type"] in CONTROL_TYPES or r["type"] in ("BALL LOST", "BALL OUT"):
            if r["type"] == "BALL LOST" and not ball_lost_ctrl:
                pass
            else:
                items.append((s, 0 if r["type"] == "BALL LOST" else 1, t))
            if switch == "lostend" and r["type"] == "BALL LOST" and not (
                    sub & {"FORCED", "OFFSIDE", "END HALF", "HAND BALL", "REFEREE HIT"}):
                items.append((max(int(r.ef), int(r.sf)) - f0, 2, 1 - t))
        elif challenge_won and r["type"] == "CHALLENGE" and "WON" in sub:
            items.append((s, 1, t))
    items = [it for it in items if 0 <= it[0] < T]
    items.sort(key=lambda it: (it[0], it[1]))
    idx = np.full(T, -1, dtype=np.int64)
    for k, (s, _, _) in enumerate(items):
        idx[s] = k
    idx = np.maximum.accumulate(idx)
    teams = np.array([it[2] for it in items] + [-1])
    return np.where(idx >= 0, teams[idx], -1).astype(np.int8)


def derive_labels(tr, ev, dead="event", switch="recovery", challenge_won=False, nan_min_dead=1,
                  setpiece_rule=True, ball_lost_ctrl=True, recovery_ctrl=True):
    """Per-frame H/A/D/X codes. dead: 'event' | 'nan' (raw ball NaN) | 'nan_long' (NaN runs of at
    least nan_min_dead frames only) | 'union' (event OR raw NaN)."""
    T = len(tr["frame"])
    nan = np.isnan(tr["bx"])
    if dead == "event":
        dm = event_dead_mask(tr, ev, setpiece_rule=setpiece_rule)
    elif dead == "nan":
        dm = nan
    elif dead == "nan_long":
        dm = np.zeros(T, bool)
        d = np.diff(np.concatenate(([0], nan.astype(int), [0])))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= nan_min_dead:
                dm[a:b] = True
    elif dead == "union":
        dm = event_dead_mask(tr, ev, setpiece_rule=setpiece_rule) | nan
    else:
        raise ValueError(dead)
    team = event_team(tr, ev, switch=switch, challenge_won=challenge_won, ball_lost_ctrl=ball_lost_ctrl,
                       recovery_ctrl=recovery_ctrl)
    lab = np.where(team == 0, H, np.where(team == 1, A, X)).astype(np.int8)
    lab[dm] = D
    return lab


# --------------------------------------------------------------------------------------- Match
def check_alignment(game, root=DEFAULT_ROOT, offsets=range(-5, 6), tr=None, ev=None):
    """Hit rate of 'passer is the nearest tracked player to the ball' at PASS start frame + offset.
    Returns {offset: (hit_rate, n)}."""
    tr = tr or load_tracking(game, root)
    ev = ev if ev is not None else load_events(game, root)
    names = list(tr["hnames"]) + list(tr["anames"])
    nm = {n: i for i, n in enumerate(names)}
    px = np.concatenate([tr["hx"], tr["ax"]], axis=1)
    py = np.concatenate([tr["hy"], tr["ay"]], axis=1)
    P = ev[(ev.type == "PASS") & ev.frm.isin(nm)]
    f0 = int(tr["frame"][0])
    out = {}
    for off in offsets:
        hit = tot = 0
        for sf, who in zip(P.sf.to_numpy(), P.frm.to_numpy()):
            f = int(sf) - f0 + off
            if not 0 <= f < len(px) or np.isnan(tr["bx"][f]):
                continue
            d = np.hypot(px[f] - tr["bx"][f], py[f] - tr["by"][f])
            if np.all(np.isnan(d)):
                continue
            tot += 1
            hit += int(np.nanargmin(d) == nm[who])
        out[off] = (hit / max(tot, 1), tot)
    return out


def load_metrica_sample(game, root=DEFAULT_ROOT, dead="event", switch="recovery", challenge_won=False,
                        with_extras=False):
    """Match for sample game 1, 2 or 3 with derive_labels() labels (see module docstring)."""
    from possession.external.dfl import _home_dir
    tr = load_tracking(game, root)
    ev = load_events(game, root)
    if abs(tr["fps"] - FPS) > 1e-6:
        raise NotImplementedError(f"game {game} is {tr['fps']} fps; resampling not implemented")
    hx, hy = _to_m(tr["hx"], tr["hy"])
    ax, ay = _to_m(tr["ax"], tr["ay"])
    bx, by = _to_m(tr["bx"], tr["by"])
    home_dir = {}
    for p in (1, 2):
        s = tr["period"] == p
        home_dir[p] = int(_home_dir(hx[s], ax[s], FPS)[0])
    lab = derive_labels(tr, ev, dead=dead, switch=switch, challenge_won=challenge_won)
    m = Match(source="metrica_sample", match_id=f"metrica_g{game}", fps=FPS, pitch=PITCH,
              hx=hx.astype(np.float32), hy=hy.astype(np.float32), ax=ax.astype(np.float32),
              ay=ay.astype(np.float32), bx=bx.astype(np.float32), by=by.astype(np.float32),
              bz=np.full(len(bx), np.nan, np.float32), label=lab, period=tr["period"].astype(np.int8),
              home_dir=home_dir, frame_id=tr["frame"].astype(np.int32))
    return (m, tr, ev) if with_extras else m
