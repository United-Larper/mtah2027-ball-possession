"""Tactical camera position from the per-frame homographies (plane-based pinhole decomposition).

World frame: pitch-centred metres, x along the pitch length, y across, z up. The homography maps
normalised pitch (0-1) to pixels; it is converted to metres first. Principal point is the image
centre, pixels are square, the focal length is unknown per frame.
"""
from __future__ import annotations

import numpy as np

import games
from .io_homography import load_homography

PITCH = games.GAMES["BAR-ATH"]["pitch"]
CX, CY = 960.0, 540.0


def _graded_idx(T: int) -> np.ndarray:
    idx = [np.arange(a, b + 1) for _, a, b in games.half_video_blocks("BAR-ATH")]
    return np.concatenate(idx)


def decompose(Hn: np.ndarray, pitch=PITCH) -> dict:
    """Vectorised decomposition of (N,3,3) normalised-pitch->pixel homographies.

    Returns dict of arrays: f, C (N,3), ok (N,), f_a, f_b (the two focal estimates), R (N,3,3)."""
    L, W = pitch
    Hn = np.asarray(Hn, np.float64)
    N = len(Hn)
    S = np.array([[1 / L, 0, 0.5], [0, 1 / W, 0.5], [0, 0, 1.0]])
    Pc = np.array([[1, 0, -CX], [0, 1, -CY], [0, 0, 1.0]])
    G = Pc @ Hn @ S                         # metres -> centred pixels
    g1, g2, g3 = G[:, :, 0], G[:, :, 1], G[:, :, 2]
    # unknown u = 1/f^2 ; h = K^-1 g -> (gx/f, gy/f, gz)
    p1 = g1[:, 0] * g2[:, 0] + g1[:, 1] * g2[:, 1]
    q1 = g1[:, 2] * g2[:, 2]
    p2 = g1[:, 0] ** 2 + g1[:, 1] ** 2 - g2[:, 0] ** 2 - g2[:, 1] ** 2
    q2 = g1[:, 2] ** 2 - g2[:, 2] ** 2
    with np.errstate(all="ignore"):
        u = -(p1 * q1 + p2 * q2) / (p1 ** 2 + p2 ** 2)
        ua, ub = -q1 / p1, -q2 / p2
        fa, fb = 1 / np.sqrt(ua), 1 / np.sqrt(ub)
        f = 1 / np.sqrt(u)
    fin = np.isfinite(Hn).all(axis=(1, 2)) & np.isfinite(f) & (u > 0)
    ok = fin & (ua > 0) & (ub > 0) & (np.abs(np.log(fa / fb)) < 0.3)
    f = np.where(fin, f, np.nan)
    Kinv = np.zeros((N, 3, 3))
    Kinv[:, 0, 0] = Kinv[:, 1, 1] = 1 / f
    Kinv[:, 2, 2] = 1
    h = Kinv @ G
    h1, h2, h3 = h[:, :, 0], h[:, :, 1], h[:, :, 2]
    lam = 0.5 * (np.linalg.norm(h1, axis=1) + np.linalg.norm(h2, axis=1))
    s = np.sign(h3[:, 2])                    # plane origin must lie in front of the camera
    s = np.where(s == 0, 1, s)
    sc = (s / lam)[:, None]
    r1, r2, t = h1 * sc, h2 * sc, h3 * sc
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=2)
    R = np.where(np.isfinite(R).all(axis=(1, 2))[:, None, None], R, np.eye(3))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -np.einsum("nji,nj->ni", R, t)       # -R^T t
    # r3 = r1 x r2 points DOWN for this data (image v grows with pitch y, so (x, y, up) is
    # left-handed); flip z so that "up" is positive. The x, y of the centre are unaffected.
    C[:, 2] = -C[:, 2]
    return dict(f=f, C=C, ok=ok, f_a=fa, f_b=fb, R=R)


def estimate_camera_centre(n_frames: int = 4000, seed: int = 0) -> dict:
    """Median camera centre over a spread of graded frames, with spread statistics."""
    Hh, _, valid = load_homography("tactical")
    idx = _graded_idx(len(Hh))
    idx = idx[valid[idx]]
    sel = idx[np.linspace(0, len(idx) - 1, n_frames).astype(int)]
    d = decompose(Hh[sel])
    ok = d["ok"] & np.isfinite(d["C"]).all(axis=1)
    C = d["C"][ok]
    q = lambda a, qs: [float(v) for v in np.percentile(a, qs, axis=0)]  # noqa: E731
    return dict(
        centre=[float(v) for v in np.median(C, axis=0)],
        iqr=[float(v) for v in np.subtract(*np.percentile(C, [75, 25], axis=0))],
        q25=q(C, 25), q75=q(C, 75), p05=q(C, 5), p95=q(C, 95),
        rejected_share=float(1 - ok.mean()), n_frames=int(len(sel)), n_ok=int(ok.sum()),
        focal_px_p10_50_90=[float(v) for v in np.percentile(d["f"][ok], [10, 50, 90])],
    )


def project_to_ground(x, y, z, cam) -> tuple[np.ndarray, np.ndarray]:
    """Where a ball at (x, y, z) appears when its image is back-projected onto the ground plane."""
    cx, cy, cz = (cam["centre"] if isinstance(cam, dict) else cam)
    x, y, z = (np.asarray(v, np.float64) for v in (x, y, z))
    z = np.minimum(z, cz - 1.0)
    k = cz / (cz - z)
    return cx + (x - cx) * k, cy + (y - cy) * k


def height_for_overshoot(overshoot, cam, y_line: float) -> np.ndarray:
    """Ball height whose ground-plane image lies `overshoot` m further from the camera than a ball
    on the touchline y = y_line (only meaningful for the touchline FAR from the camera)."""
    cx, cy, cz = cam["centre"]
    o = np.asarray(overshoot, np.float64)
    d = abs(y_line - cy)
    return o * cz / (d + o)


def plausibility_check(cam: dict | None = None) -> dict:
    """Aggregate-only: ball-outside-pitch shares on ball-present graded frames and overshoot sizes.

    The camera sits on the +y side for this match (see estimate_camera_centre), so the y = -W/2
    touchline is the one far from the camera, where airborne balls would overshoot."""
    from .target import graded_mask, load_target_match
    cam = cam or estimate_camera_centre()
    m = load_target_match()
    L, W = m.pitch
    g = graded_mask(m.T) & np.isfinite(m.bx) & np.isfinite(m.by)
    bx, by = m.bx[g].astype(np.float64), m.by[g].astype(np.float64)
    pc = [50, 90, 99]
    pos, neg = by > W / 2, by < -W / 2
    ov_pos = np.percentile(by[pos] - W / 2, pc)
    ov_neg = np.percentile(-W / 2 - by[neg], pc)
    return dict(
        n_present_graded=int(g.sum()), camera_side="+y" if cam["centre"][1] > 0 else "-y",
        share_beyond_pos_y_touchline=float(pos.mean()), share_beyond_neg_y_touchline=float(neg.mean()),
        share_goal_line_neg_x=float((bx < -L / 2).mean()), share_goal_line_pos_x=float((bx > L / 2).mean()),
        pos_y_overshoot_m_p50_90_99=[float(v) for v in ov_pos],
        neg_y_overshoot_m_p50_90_99=[float(v) for v in ov_neg],
        height_explaining_neg_y_overshoot_m_p50_90_99=[
            float(v) for v in height_for_overshoot(ov_neg, cam, -W / 2)],
        height_explaining_pos_y_overshoot_m_p50_90_99=[
            float(v) for v in height_for_overshoot(ov_pos, cam, W / 2)],  # wrong sign if camera is on +y
    )
