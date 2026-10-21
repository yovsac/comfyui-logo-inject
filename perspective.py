# Made by Yovan Vujacic
"""Measured badge perspective (2026-10-02). The model drew every painted-spot badge front-on, even on surfaces turned
away from the camera (DS grille, Peugeot tailgate, Alfa hubs - Yovan: "the badges come out frontal"), and a wider crop
did not fix it (ws-test12). Here the angle is MEASURED: a plane is fitted to Depth Anything 3's depth under the painted
spot, and the flat logo is laid on that plane and projected with DA3's camera intrinsics -> one homography per spot.
Pure numpy/cv2."""
import json

import cv2
import numpy as np


def parse_intrinsics(text):
    """DA3 intrinsics_json -> 3x3 K (first image)."""
    d = json.loads(text)["intrinsics"][0]
    return np.array(next(iter(d.values())), np.float64)


def fit_plane(depth, K, mask, iters=3):
    """Least-squares plane through the 3D points under mask (camera coords), refitted without outliers.
    Returns (unit normal facing the camera, centroid)."""
    ys, xs = np.nonzero(mask)
    if len(xs) > 40000:
        k = np.random.default_rng(0).choice(len(xs), 40000, replace=False); ys, xs = ys[k], xs[k]
    z = depth[ys, xs].astype(np.float64)
    P = np.column_stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z])
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        c = P[keep].mean(0)
        n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][-1]
        r = np.abs((P - c) @ n)
        keep = r <= max(2.0 * r[keep].std(), 1e-9)
    if n @ c > 0:  # face the camera
        n = -n
    return n / np.linalg.norm(n), c


def fit_plane_ransac(depth, K, mask, trials=400, tol=0.004, seed=0):
    """Dominant plane among the 3D points under mask: RANSAC (inlier tolerance = tol x median depth), then a
    least-squares refit on the inliers. A painted spot often straddles two faces (Alfa lip: the ledge top and its
    front face); the largest planar patch around it is the panel the badge sits on. Returns (normal, centroid, share)."""
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    if len(xs) > 30000:
        k = rng.choice(len(xs), 30000, replace=False); ys, xs = ys[k], xs[k]
    z = depth[ys, xs].astype(np.float64)
    P = np.column_stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z])
    t = tol * float(np.median(z)); best = None
    for _ in range(trials):
        a, b_, c_ = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b_ - a, c_ - a); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn; inl = np.abs((P - a) @ n) < t
        if best is None or inl.sum() > best.sum():
            best = inl
    c = P[best].mean(0); n = np.linalg.svd(P[best] - c, full_matrices=False)[2][-1]
    if n @ c > 0:
        n = -n
    return n / np.linalg.norm(n), c, float(best.mean())


def clamp_pitch(n, max_deg=35.0):
    """Keep the measured left/right turn, cap the up/down tilt. Badges sit on near-vertical faces (grille, tailgate,
    hubs); on small ledges DA3 reads the face as pointing up (Alfa lip: +67 deg -> logo squashed flat, 2026-10-02)."""
    p = np.degrees(np.arctan2(-n[1], np.hypot(n[0], n[2])))
    if abs(p) <= max_deg:
        return n
    h = np.array([n[0], 0.0, n[2]]); h /= np.linalg.norm(h)
    q = np.radians(np.clip(p, -max_deg, max_deg))
    return np.cos(q) * h + np.sin(q) * np.array([0.0, -1.0, 0.0])


def project(K, X):
    p = (K @ X.T).T
    return p[:, :2] / p[:, 2:3]


def logo_homography(K, n, c, center_px, logo_w, logo_h, target_px):
    """Homography logo pixels -> image pixels: the logo lies flat on the plane (n, c), upright (its vertical = the
    plane direction that projects straight up in the image), centred on the camera ray through center_px, scaled so its projected size
    (larger side) is target_px."""
    P0, u, v = plane_axes(K, n, c, center_px)
    corners = np.array([[0, 0], [logo_w, 0], [logo_w, logo_h], [0, logo_h]], np.float64)

    def img_pts(s):
        a = (corners[:, 0] - logo_w / 2) * s; b = (corners[:, 1] - logo_h / 2) * s
        return project(K, P0 + a[:, None] * u - b[:, None] * v)

    s = np.linalg.norm(P0) * 1e-4
    for _ in range(3):  # projection is near-linear in s at badge sizes: rescale twice to hit the target
        q = img_pts(s); ext = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))
        s *= target_px / max(ext, 1e-6)
    q = img_pts(s)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), q.astype(np.float32)), q


def plane_axes(K, n, c, center_px):
    """(P0, u, v): the point on plane (n, c) under center_px, and the logo's right / up directions on the plane."""
    ray = np.array([(center_px[0] - K[0, 2]) / K[0, 0], (center_px[1] - K[1, 2]) / K[1, 1], 1.0])
    P0 = ray * (n @ c) / (n @ ray)  # ray-plane intersection
    # logo's up = the direction on the plane that projects straight up in the image at the badge centre (vertical
    # edges stay vertical with no camera roll). The camera's up projected onto the plane swung the logo by up to ~20
    # deg whenever the measured up/down tilt was off (Alfa lip, Peugeot shield, ws-test13).
    v = np.cross(n, np.array([P0[2], 0.0, -P0[0]]))
    if np.linalg.norm(v) < 1e-9:
        v = np.array([0.0, -1.0, 0.0]) - n[1] * n
    v /= np.linalg.norm(v)
    if project(K, (P0 + 0.01 * np.linalg.norm(P0) * v)[None])[0, 1] > project(K, P0[None])[0, 1]:
        v = -v  # must point up in the image
    u = np.cross(v, n); u /= np.linalg.norm(u)
    if project(K, np.array([P0 + 0.01 * np.linalg.norm(P0) * u]))[0, 0] < project(K, P0[None])[0, 0]:
        u = -u  # logo's +x must run image-right
    return P0, u, v


def yaw_pitch(n):
    """(left/right turn, up/down tilt) of a camera-facing normal, degrees."""
    return (float(np.degrees(np.arctan2(n[0], -n[2]))), float(np.degrees(np.arctan2(-n[1], np.hypot(n[0], n[2])))))


def tilt_reliable(n, max_deg=35.0):
    """False when the measured up/down tilt is beyond what a badge surface has (thin ledges read as facing up)."""
    return abs(np.degrees(np.arctan2(-n[1], np.hypot(n[0], n[2])))) <= max_deg


def edge_angle(img, center_px, half, max_deg=40.0):
    """Dominant near-horizontal straight edge around the spot (image degrees, y down), length-weighted; None if none.
    On a ledge or crease this is the surface's own horizontal line."""
    x0, y0 = int(max(0, center_px[0] - half)), int(max(0, center_px[1] - half))
    g = cv2.cvtColor(img[y0:int(center_px[1] + half), x0:int(center_px[0] + half)], cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    L = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=40, minLineLength=int(0.8 * half), maxLineGap=6)
    if L is None:
        return None
    ang, w = [], []
    for x1, y1, x2, y2 in L.reshape(-1, 4):  # OpenCV 5 returns (N, 4), 4.x (N, 1, 4)
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        a = (a + 90) % 180 - 90
        if abs(a) <= max_deg:
            ang.append(a); w.append(np.hypot(x2 - x1, y2 - y1))
    if not ang:
        return None
    o = np.argsort(ang); cw = np.cumsum(np.array(w)[o])
    return float(np.array(ang)[o][np.searchsorted(cw, cw[-1] / 2)])


def normal_from_edge(K, c, center_px, theta, pitch_deg, yaw_hint=0.0):
    """Surface at the given up/down tilt whose left/right turn makes the logo's horizontal axis run along the image
    angle theta at center_px. Used when the measured tilt is implausible (Alfa lip: ledge read as facing up): the
    tilt is capped, and the noisy measured turn combined with it rotated the logo ~30 deg (ws-test15); the ledge line
    fixes the turn instead. Near-equal fits: the one closest to yaw_hint (the measured turn)."""
    best = None; pr = np.radians(pitch_deg)
    for yaw in np.arange(-80.0, 80.5, 0.5):
        r = np.radians(yaw)
        n = np.cos(pr) * np.array([np.sin(r), 0.0, -np.cos(r)]) + np.sin(pr) * np.array([0.0, -1.0, 0.0])
        P0, u, _ = plane_axes(K, n, c, center_px)
        q = project(K, np.array([P0, P0 + 0.01 * np.linalg.norm(P0) * u]))
        a = np.degrees(np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]))
        d = abs((a - theta + 90) % 180 - 90) + 0.01 * abs(yaw - yaw_hint)
        if best is None or d < best[0]:
            best = (d, n, yaw)
    return best[1], best[2], best[0]


def normal_nearest_edge(K, c, center_px, theta, yaw0, pitch0, max_tilt=35.0, w=0.01):
    """The surface normal NEAREST the measured one (yaw0, pitch0) whose logo horizontal runs along image angle theta.
    normal_from_edge only turns (yaw): on the Peugeot tailgate the light bar is level because the badge sits near the
    camera's horizon height, so matching it by turning alone threw away the measured 46 deg turn and drew the shield
    front-on (ws-test24 offline). Adjusting turn AND tilt keeps the foreshortening and levels the badge."""
    best = None
    for yaw in np.arange(-80.0, 81.0, 1.0):
        r = np.radians(yaw)
        for pitch in np.arange(-max_tilt, max_tilt + 0.5, 1.0):
            pr = np.radians(pitch)
            n = np.cos(pr) * np.array([np.sin(r), 0.0, -np.cos(r)]) + np.sin(pr) * np.array([0.0, -1.0, 0.0])
            P0, u, _ = plane_axes(K, n, c, center_px)
            q = project(K, np.array([P0, P0 + 0.01 * np.linalg.norm(P0) * u]))
            a = np.degrees(np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]))
            err = abs((a - theta + 90) % 180 - 90)
            d = err + w * (abs(yaw - yaw0) + abs(pitch - pitch0))
            if best is None or d < best[0]:
                best = (d, n, yaw, pitch, err)
    return best[1], best[2], best[3], best[4]
