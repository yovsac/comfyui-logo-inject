# Made by Yovan Vujacic
"""Logo Inject v2 geometry (deterministic, CPU) - ported verbatim from the offline harness
agent-skills/scripts/.runs/li_v2/geom.py (2026-10-07, 36-spot test set). Per painted spot: centre, pose, size -> the
homography that maps the logo onto the car.
- centre: the old emblem inside the spot (native SAM3 masks), else the spot's centre
- pose: round old emblem = perspective gauge (its ellipse); hub = the wheel's outline; else Depth Anything 3 plane
  under the spot (RANSAC) with the horizontal snapped to the nearest straight edge
- size: the old emblem's area (lettering: its letter height), then shrunk until it crosses no long strong edge
site_geometry(c, j, spot, lw, lh, body): c = dict(car, dep, K, sam=[bool HxW], wheels=[bool HxW]) at full resolution."""
import cv2
import numpy as np

from . import perspective as PP

MAX_TILT = 35.0
WIDE_LEAN = 0.6  # share of the band's lean a wide logo keeps (calibrated on Yovan's drawn Opel axes, 2026-10-08)
SYM_K = 2.5  # a symbol replacing lettering stands this many letter heights tall (real Audi rings ~2.7x V70 letters)
HUB_SIGHT = True  # hub pose relative to the line of sight to the hub (round 11)
HUB_NOLEDGE = True  # hubs ignore the ledge line - an edge inside the cap turned the Honda H (round 12)
EDGE_THROUGH = True  # lines the old badge sat on are no limit (geom.py round 23)
MIN_PAINT_FRAC = 0.45  # a badge never shrinks below this share of the painted spot (round 23)
CAP_FILL = 0.75  # a badge on a blank hub's plain cap fills this share of the cap (round 18)
CAP_INK = 0.82  # ...and its ink reaches at most this share of the cap radius (round 19)
RING_CAP_F = 0.88  # ring logo on a cap: its ring onto the cap's rim line, 0.88 of the cap (round 12 ghost ring)
V6 = True  # v6 rules (panel-band plane, level badge, keep the turn, outline size, hub caps) - see geom.py
LIP_BAND = True  # thin lip with the same slope as the glass above: follow it (geom.py round 13)
LIP_LEAN = 0.35  # share of such a lip's lean kept: = Yovan's approved round-8 Alfa lip (lean 23.6 of 52)
LIP_FILL = 0.8  # badge height on such a lip, share of the lip's height (round 8: 0.81)
LEDGE = "panel"  # implausible tilt + ledge line: the panel round the spot, logo along the line on it (geom.py round 9)
SPOT_FILL = 0.9  # blank panel (no old badge): the badge spans this share of the painted spot (Yovan drew the DS
# ~0.9-1.0 of his 91 px paint, 2026-10-12; was 0.6 - the DS came out 54 px and lost its detail)
SPOT_UP_TILT = 52.0  # blank spot whose panel faces up (bonnet top / nose): the badge lies on it, level, with the
# depth's turn and at most this tilt (Yovan's perspective square on the Infiniti bonnet nose, 2026-10-12 = turn -44 /
# tilt 52; the depth read -50 / 67 -> a sliver; r15c's damped turn -10 / 56 leaned only 30 deg - 'still not good')


def logo_parts(logo_rgba):
    """alpha of the logo's marks, the logo's solid body (hull) and the cropped RGB, cropped to the hull."""
    a = logo_rgba[..., 3] if logo_rgba.shape[2] == 4 else None
    rgb = logo_rgba[..., :3]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if a is None or a.min() == 255:  # logo on a flat background: marks = the minority tone
        g2 = 255 - g if np.median(g) > 128 else g
        a = g2
    ys, xs = np.nonzero(a > 40)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    body = np.zeros_like(g); cv2.fillConvexPoly(body, hull, 255)
    x, y, w, h = cv2.boundingRect(hull)
    return a[y:y + h, x:x + w].astype(np.float32) / 255, body[y:y + h, x:x + w], rgb[y:y + h, x:x + w]


def strong_edges(car, exclude, min_len, through=None, through_limit=None):
    """Long straight strong edges (light bar, panel gaps, rims) as a binary map; the old emblem's own edges excluded.
    through: a line that runs INTO this area (the old badge / the painted spot) is the surface the badge sits on, not
    a limit (round 23: the green centre line under the DS, the bonnet crease under the Abarth wedge, the grille mesh
    round the E-3008 shield shrank the new badges to 30 / 17 / 42 px).
    through_limit (cx, cy, r): a blank painted spot - a line through it still IS a limit when it runs well off the spot's
    centre (>= 0.4 r) and is a real boundary (step or ridge >= 50 in Lab): the grille bar's edge, the boot lip, the light
    bar and the plate recess under 2x badges (round 24); weak trim lines and lines through the centre stay under the badge."""
    g = cv2.GaussianBlur(cv2.cvtColor(car, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 60, 160)
    e[exclude] = 0
    out = np.zeros_like(e)
    L = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=max(20, int(min_len * 0.6)), minLineLength=int(min_len), maxLineGap=4)
    if L is not None:
        for x1, y1, x2, y2 in L[:, 0]:
            if through is not None:
                seg = np.zeros_like(e); cv2.line(seg, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3)
                if (seg.astype(bool) & through).any() and not (through_limit is not None and _boundary(car, (x1, y1, x2, y2), through_limit)):
                    continue
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    return out > 0


def _boundary(car, seg, lim, k=6, gate=50.0, off_min=0.4):
    """Is this segment a panel boundary well off the spot centre? (see strong_edges, round 24)"""
    x1, y1, x2, y2 = (float(v) for v in seg); cx, cy, r = lim
    d = np.array([x2 - x1, y2 - y1]); ln = float(np.hypot(*d))
    if ln < 1 or abs(d[0] * (cy - y1) - d[1] * (cx - x1)) / ln < off_min * r:
        return False
    H, W = car.shape[:2]; u = d / ln; nrm = np.array([-u[1], u[0]]); st, ri = [], []
    for t in np.linspace(0.1, 0.9, 15):
        p = np.array([x1, y1]) + t * d; pts = [p + s * k * nrm for s in (-1, 0, 1)]
        if all(0 <= q[0] < W and 0 <= q[1] < H for q in pts):
            a, m, b = (cv2.cvtColor(car[int(q[1]):int(q[1]) + 1, int(q[0]):int(q[0]) + 1], cv2.COLOR_RGB2LAB)[0, 0].astype(float) for q in pts)
            st.append(np.linalg.norm(a - b)); ri.append(np.linalg.norm(m - (a + b) / 2))
    return bool(st) and max(float(np.median(st)), float(np.median(ri))) >= gate


def triu(w, h, a):
    """Same as li_core._triu: maps the unit circle onto the ellipse with the x axis kept level and the y axis onto the
    conjugate diameter (the disc-in-perspective convention Yovan approved 2026-09-30)."""
    t = np.deg2rad(a); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    S = R @ np.diag([(w / 2) ** 2, (h / 2) ** 2]) @ R.T
    c = np.sqrt(S[1, 1]); b = S[0, 1] / c
    return np.array([[np.sqrt(max(S[0, 0] - b * b, 1e-6)), b], [0, c]])


def quad_mask(q, shape):
    m = np.zeros(shape[:2], np.uint8); cv2.fillConvexPoly(m, q.astype(np.int32), 1); return m > 0


def panel_plane(c, spot, old, ctr, ext, strip):
    """The plane of the panel the emblem is mounted on: the band between the nearest long edges above and below it
    (Opel Y: the ring plane caught the sloped hood above, pitch 26 -> squashed, rotated badge; the band reads -8), or
    for lettering a box around the word (Peugeot tailgate: the thin strip under the light bar gave noise; a box reads
    the 42 deg turn the whole tailgate has). None when neither is a plausible badge surface."""
    car, dep, K = c["car"], c["dep"], c["K"]; H, W = car.shape[:2]
    tx, ty = ctr; ex = cv2.dilate((spot | old).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    e = strong_edges(car, ex, max(12, 0.5 * ext))
    x0, x1 = int(max(0, tx - ext)), int(min(W, tx + ext))
    col = e[:, x0:x1].any(1)
    oy = np.nonzero((spot | old).any(1))[0]; top, bot = int(oy.min()), int(oy.max())
    up = [y for y in range(top, -1, -1) if col[y]]; dn = [y for y in range(bot, H) if col[y]]
    ya = up[0] if up else max(0, int(top - 2 * ext)); yb = dn[0] if dn else min(H - 1, int(bot + 2 * ext))
    cands = []
    if yb - ya >= max(24, 0.6 * (bot - top + 1)):
        band = np.zeros((H, W), bool); band[ya + 3:yb - 2, int(max(0, tx - 2.5 * ext)):int(min(W, tx + 2.5 * ext))] = True
        band &= ~ex
        if band.sum() >= 400:
            cands.append(("band", band))
    if strip:
        box = np.zeros((H, W), bool); box[int(max(0, ty - ext)):int(min(H, ty + 1.5 * ext)), int(max(0, tx - 1.5 * ext)):int(min(W, tx + 1.5 * ext))] = True
        box &= ~ex
        cands.append(("box", box))
    for name, m in cands:
        n_, c_, sh = PP.fit_plane_ransac(dep, K, m)
        if PP.tilt_reliable(n_, MAX_TILT):
            return n_, c_, sh, name
    return None


def nearest_edge_keep_turn(K, c_, ctr, th, yaw0, pitch0, max_tilt, w_yaw=0.05, w_pitch=0.01):
    """Like PP.normal_nearest_edge, but the measured TURN costs 5x more to change than the tilt: levelling the Peugeot
    shield with the light bar turned it from 42 to 23 deg and it looked front-on (Yovan, v5 review)."""
    best = None
    for yaw in np.arange(-80.0, 81.0, 1.0):
        r = np.radians(yaw)
        for pitch in np.arange(-max_tilt, max_tilt + 0.5, 1.0):
            pr = np.radians(pitch)
            n = np.cos(pr) * np.array([np.sin(r), 0.0, -np.cos(r)]) + np.sin(pr) * np.array([0.0, -1.0, 0.0])
            P0, u, _ = PP.plane_axes(K, n, c_, ctr)
            q = PP.project(K, np.array([P0, P0 + 0.01 * np.linalg.norm(P0) * u]))
            a = np.degrees(np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]))
            err = abs((a - th + 90) % 180 - 90)
            d = err + w_yaw * abs(yaw - yaw0) + w_pitch * abs(pitch - pitch0)
            if best is None or d < best[0]:
                best = (d, n, yaw, pitch, err)
    return best[1], best[2], best[3], best[4]


def fit_ring_to_cap(Hm, lw, lh, cap_ell, a_marks=None):
    """A RINGED open logo on a hub with its own round cap: the logo's ring becomes the cap's rim (Yovan, round 4: the
    Mercedes on the black cap with its ring drawn inside the cap's rim = 'not a pass', two rings). Scales the
    homography about the cap centre so the logo's larger projected side = the cap's mean diameter."""
    (ex, ey), (ea, eb), _ = cap_ell
    if a_marks is None or not round_ring(a_marks, lw, lh):
        return Hm, None  # only a truly ROUND ring (Mercedes, Opel); the Honda's rounded-square frame stays as approved
    q = cv2.perspectiveTransform(np.array([[[0, 0], [lw, 0], [lw, lh], [0, lh]]], np.float64), Hm)[0]
    ctr = Hm @ np.array([lw / 2, lh / 2, 1.0]); ctr = ctr[:2] / ctr[2]
    # onto the cap's chrome rim LINE, not its outer edge (round 12, Mercedes 'ghosting'): at the outer edge the finish took
    # our ring for the cap's rim and drew the logo's own ring + a smaller star inside it; the lock then put ours back on
    # top -> two rings, two stars. At 0.88 (the rim line on both black caps) it draws one ring and keeps the star's size
    s = RING_CAP_F * (ea + eb) / 2 / max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))
    S = np.array([[s, 0, ex - s * ctr[0]], [0, s, ey - s * ctr[1]], [0, 0, 1.0]])
    return S @ Hm, round(float(s), 2)


def round_ring(a_marks, lw, lh):
    """The logo is a truly ROUND ring (Mercedes, Opel) - not the Honda's rounded-square frame, not a ringless mark."""
    pts, kind = cap_shape(a_marks, lw, lh)
    if kind != "ring filled":
        return False
    p = pts.astype(np.float32)[:, None, :]; (_, _), (ra, rb), _ = cv2.fitEllipse(p)
    return min(ra, rb) / max(ra, rb) >= 0.97 and cv2.contourArea(p) >= 0.99 * np.pi * ra * rb / 4


def cap_shape(a, lw, lh):
    """Centre cap for an OPEN logo on a wheel hub (Yovan, v5 review: the Opel on the front wheel 'looks like it's
    floating' - the plate was erased and nothing held the badge; his approved showcase hubs all sit on a disc).
    Only RINGED logos (Opel, Citroen oval) get one: the ring filled in, so the ring is the cap's rim. v6 (2026-10-08):
    a separate round cap under a logo got its own chrome rim from the finish LoRA -> Opel came out with two or three
    rings (the double ring Yovan rejected on t06), DS turned into an X in a ring, FIAT got a green plaque; unringed
    logos keep v5 (no cap). The Opel ring has small gaps where the bolt crosses it (closed first) and the bolt tips
    stick out of it (an opening strips them before the ellipse test). Returns (logo-space polygon N x 2, "ring filled")
    or (None, None)."""
    kc = max(3, int(0.06 * min(lw, lh))) | 1
    # work on a zero-padded copy: the logo is cropped to its ring, and a closing that reaches the image border keeps
    # border-hugging blocks -> four rectangular tabs stuck out past the ring, which the finish drew as a pale block
    # beside the Mercedes ring (n06 "ghost ring edge", robust2 + round 3, 2026-10-09)
    pd = kc + 2
    mk = np.pad((a > 0.3).astype(np.uint8), pd)
    mc = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    inv = np.pad(1 - mc, 1, constant_values=1).astype(np.uint8)
    cv2.floodFill(inv, None, (0, 0), 2)
    filled = (inv[1:-1, 1:-1] != 2).astype(np.uint8)
    ko = max(3, int(0.15 * min(lw, lh))) | 1
    op = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    cs = cv2.findContours(op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if cs:
        big = max(cs, key=cv2.contourArea)
        if len(big) >= 5:
            (ex, ey), (ew, eh), ea = cv2.fitEllipse(big)
            el = np.zeros_like(mk); cv2.ellipse(el, ((ex, ey), (ew, eh), ea), 1, -1)
            ar = cv2.contourArea(big)
            # an ellipse (not a D bowl or a letter O), most of the filled logo, holding most of the marks
            if ar >= 0.85 * np.pi * ew * eh / 4 and ar >= 0.6 * filled.sum() and (mk & el).sum() >= 0.75 * mk.sum():
                return big[:, 0, :].astype(np.float64) - pd, "ring filled"
    return None, None


def thicken(a, size_px, min_px=3.5, max_share=0.06):
    """OPEN logos: strokes that come out as wire at the placed size get bolder (Yovan, round 3: the Opel 'too thin and
    fragile' = 2.2 px strokes at 69 px; the DS on a 23 px hub had 0.7 px strokes and its S faded; a 30 px Mercedes ring
    under 1 px went missing). Median stroke (2 x area / perimeter) raised to min_px native px (2.5 barely moved the Opel), never past max_share of
    the logo. Big badges keep their artwork (Bentley wings 3.9 px, spare-wheel Mercedes 4.3 px). Returns (marks, info)."""
    ext = float(max(a.shape)); mk = (a > 0.5).astype(np.uint8)
    cs = cv2.findContours(mk, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    per = sum(cv2.arcLength(c_, True) for c_ in cs)
    if per <= 0 or size_px <= 0:
        return a, None
    w = 2 * mk.sum() / per  # logo px
    want = min(min_px * ext / size_px, max_share * ext)
    r = int(round((want - w) / 2))
    if r < 1:
        return a, None
    return cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))), \
        dict(stroke_px=round(w * size_px / ext, 2), now_px=round((w + 2 * r) * size_px / ext, 2))


def ink_centred(Hm, a, tgt):
    """OPEN logos on a hub: the design's ink centroid onto tgt (the cap's centre), not its bounding box - a lopsided
    design (DS: heavy D bar, thin S, ink 12% up-left of its box) read 'not centred' on both hubs (round 3). A small image
    translation after the homography. Returns (Hm, shift px)."""
    ys, xs = np.nonzero(a > 0.3); wt = a[ys, xs]
    p = Hm @ np.array([(xs * wt).sum() / wt.sum(), (ys * wt).sum() / wt.sum(), 1.0]); p = p[:2] / p[2]
    d = (float(tgt[0] - p[0]), float(tgt[1] - p[1]))
    return np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]]) @ Hm, [round(d[0], 1), round(d[1], 1)]


def band_between_creases(c, old, ctr):
    """The sloped band an emblem sits on, measured between the two creases that bound it (Yovan, v6 review
    2026-10-08: the red car's nose badges 'wrong perspective and orientation' three times; his drawn axes: horizontal
    along the band, up leaning ~35 deg with the surface). The depth on the strip between the creases reads the slope
    (cit nose 47 deg); the wider band v6 used also took the upright face below (~0 deg) and stood the badge up.
    Returns (n, c, line angle deg, info) or None when there is no clear crease pair or the strip isn't sloped."""
    car, dep, K = c["car"], c["dep"], c["K"]; H, W = car.shape[:2]
    oy, ox = np.nonzero(old); tx, ty = int(round(ctr[0])), int(round(ctr[1])); h = int(np.ptp(oy)) + 1
    g = cv2.GaussianBlur(cv2.cvtColor(car, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5))
    cols = np.r_[max(0, tx - 3 * h):max(0, tx - h), min(W, tx + h):min(W, tx + 3 * h)]
    if len(cols) < 10:
        return None
    prof = gy[:, cols].mean(1); base = float(np.median(prof)) + 1e-3
    m = max(2, int(0.1 * h))  # the old emblem may touch a crease (opel Y's arms reach the upper one)
    a0, a1 = max(0, oy.min() - 2 * h), min(H, oy.min() + m); b0, b1 = max(0, oy.max() - m), min(H, oy.max() + 2 * h)
    if a1 - a0 < 3 or b1 - b0 < 3:
        return None
    up = a0 + int(np.argmax(prof[a0:a1])); dn = b0 + int(np.argmax(prof[b0:b1]))
    if prof[up] < 2.5 * base or prof[dn] < 2.5 * base or dn - up < 1.05 * h:  # 2.5: the 1360 px red car's upper crease is 2.8x
        return None
    # each crease's image angle: per column, the strongest edge row near the crease -> line fit
    angs = []
    for r in (up, dn):
        w = max(3, int(0.15 * h)); pts = []
        for x in cols:
            y0 = max(0, r - w); k = int(np.argmax(gy[y0:r + w + 1, x])); pts.append((x, y0 + k))
        p = np.array(pts, float)
        if np.ptp(p[:, 0]) > 0:
            angs.append(float(np.degrees(np.arctan(np.polyfit(p[:, 0], p[:, 1], 1)[0]))))
    th = float(np.median(angs)) if angs else 0.0
    ex = cv2.dilate(old.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    q = max(3, (dn - up) // 8)
    strip = np.zeros((H, W), bool); strip[up + q:dn - q, max(0, tx - 6 * h):min(W, tx + 6 * h)] = True; strip &= ~ex
    if strip.sum() < 400:
        return None
    n, c_, share = PP.fit_plane_ransac(dep, K, strip)
    yaw, pitch = PP.yaw_pitch(n)
    if share < 0.6 or not 15 <= pitch <= 65:  # upright or facing down: the v6 pose already fits
        return None
    # a real chamfer steps the slope down: top surface above it faces up more, the face below it less (red car nose:
    # 77 / 48 / -1). The Alfa lip read 52 / 53 / 12: a thin lip the depth can't tell from the glass above it. Yovan once
    # passed it level (v6) but asked for it to follow the slope (round 13) - his approved round-8 badge leaned 23.6 deg
    bh = dn - up; side = []
    for y0, y1 in ((up - max(bh, h), up - 3), (dn + 3, dn + max(bh, h))):
        m2 = np.zeros((H, W), bool); m2[max(0, y0):min(H, y1), max(0, tx - 6 * h):min(W, tx + 6 * h)] = True; m2 &= ~ex
        side.append(PP.yaw_pitch(PP.fit_plane_ransac(dep, K, m2)[0])[1] if m2.sum() >= 400 else None)
    if side[0] is None or side[1] is None or pitch < side[1] + 15:
        return None
    lip = LIP_BAND and abs(side[0] - pitch) < 3  # same slope as above: a lip, kept with LIP_LEAN of its lean
    if side[0] < pitch + 3 and not lip:
        return None
    info = dict(creases=[int(up), int(dn)], band_slope=round(pitch, 1), band_line=round(th, 1), band_share=round(share, 2),
                band_steps=[round(side[0]), round(pitch), round(side[1])])
    if lip:
        info["lip"] = True
    return n, c_, th, info


def homography_on_surface(K, n, c, center_px, lw, lh, target_px, th):
    """Like PP.logo_homography, but the logo's horizontal runs along the panel line (image angle th at the centre) ON
    the plane and its up is the surface direction across that line (v = n x u): on a sloped band the badge leans with
    the surface. (PP.logo_homography keeps the logo's up straight up in the picture - right for upright panels, and
    the reason every badge on the red car's sloped nose stood square to the camera.)"""
    P0, _, _ = PP.plane_axes(K, n, c, center_px)
    e = 0.01 * np.linalg.norm(P0)
    d = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
    ray = np.array([(center_px[0] + 5 * d[0] - K[0, 2]) / K[0, 0], (center_px[1] + 5 * d[1] - K[1, 2]) / K[1, 1], 1.0])
    u = ray * (n @ P0) / (n @ ray) - P0; u /= np.linalg.norm(u)
    if PP.project(K, (P0 + e * u)[None])[0, 0] < PP.project(K, P0[None])[0, 0]:
        u = -u
    v = np.cross(n, u); v /= np.linalg.norm(v)
    if PP.project(K, (P0 + e * v)[None])[0, 1] > PP.project(K, P0[None])[0, 1]:
        v = -v  # up in the picture
    corners = np.array([[0, 0], [lw, 0], [lw, lh], [0, lh]], np.float64)

    def img_pts(s):
        a = (corners[:, 0] - lw / 2) * s; b = (corners[:, 1] - lh / 2) * s
        return PP.project(K, P0 + a[:, None] * u - b[:, None] * v)

    s = np.linalg.norm(P0) * 1e-4
    for _ in range(3):
        q = img_pts(s); ext = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))
        s *= target_px / max(ext, 1e-6)
    q = img_pts(s)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), q.astype(np.float32)), q


def rim_centre(car, wheel, spot, ctr):
    """Centre of a side-on wheel's bright rim (Otsu inside the SAM3 wheel, the component under the cap, ellipse on its
    hull), or None unless it is a clean, near-round rim well clear of the frame, at most 0.4 spot radii from ctr.
    (Checked on every hub in the test set: turned wheels and dark rims fit badly -> they keep the old cap's centre.)"""
    g = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY); H, W = g.shape
    ys, xs = np.nonzero(spot); sr = max(np.ptp(xs), np.ptp(ys)) / 2
    t, _ = cv2.threshold(g[wheel].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.morphologyEx((wheel & (g > t)).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    k, lab, st, cen = cv2.connectedComponentsWithStats(b)
    i = lab[int(round(ctr[1])), int(round(ctr[0]))]
    if i == 0:
        d = [np.hypot(cen[m][0] - ctr[0], cen[m][1] - ctr[1]) if m and st[m, cv2.CC_STAT_AREA] > 50 else 1e9 for m in range(k)]
        i = int(np.argmin(d))
        if d[i] >= 1e9:
            return None
    cs = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    if len(hull) < 5:
        return None
    (rx, ry), (ra, rb), _ = cv2.fitEllipse(hull)
    fill = cv2.contourArea(hull) / (np.pi * ra * rb / 4 + 1e-6)
    x, y, w, h = cv2.boundingRect(hull)
    if (min(ra, rb) / max(ra, rb) < 0.85 or fill < 0.9 or min(ra, rb) < 6 * sr or x <= 1 or y <= 1 or x + w >= W - 1
            or y + h >= H - 1 or np.hypot(rx - ctr[0], ry - ctr[1]) > 0.4 * sr):
        return None
    return float(rx), float(ry)


def cap_disc_centre(c, spot, ctr, badge):
    """Centre of the round centre cap the emblem sits in: the smallest round SAM3 disc holding the whole spot, 1.2-4x the
    badge (not the emblem, not the whole wheel), or None. The wheel's rim centre is off the cap on a dished wheel seen at
    an angle: the Alfa on the wet cap sat 20 px right of the cap (Yovan, round 8: 'not centered')."""
    best = None
    for m in c["sam"]:
        if m.sum() < 30 or (m & spot).sum() < 0.95 * spot.sum():
            continue
        cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        big = max(cs, key=cv2.contourArea)
        if len(big) < 5:
            continue
        (ex, ey), (ea, eb), ang = cv2.fitEllipse(big)
        el = np.zeros(m.shape, np.uint8); cv2.ellipse(el, ((ex, ey), (ea, eb), ang), 1, -1)
        iou = (m & (el > 0)).sum() / max(1, (m | (el > 0)).sum())
        d = min(ea, eb)
        if iou >= 0.92 and 1.2 * badge <= d <= 4.0 * badge and np.hypot(ex - ctr[0], ey - ctr[1]) <= 0.25 * d:
            if best is None or d < best[0]:
                best = (d, (float(ex), float(ey)))
    return best[1] if best else None


def wordmark_parts(marks, lw, lh):
    """(is the logo a wordmark?, its letter height / its box height). Wordmark = 3+ separate marks in a wide row (Jeep,
    ORION, VELTRA); the Audi rings and the Dacia mark are one piece -> symbols. Letter height = median letter box."""
    if marks is None:
        return lw / lh > 2.2, 1.0
    m = marks > 0.5
    k, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    hs = [st[i, cv2.CC_STAT_HEIGHT] for i in range(1, k) if st[i, cv2.CC_STAT_AREA] >= 0.02 * m.sum()]
    if not (lw / lh > 2.2 and len(hs) >= 3):
        return False, 1.0
    return True, float(np.median(hs)) / lh  # per letter, like the old letters (a T or L column is just its bar)


def lettering_frame(c, main):
    """The old lettering as the gauge (Yovan, robust2 2026-10-09: the Dacia over ASTON MARTIN 'not following the
    surface', his drawn verticals 24 deg): baseline = the words' long axis; letter height = median height of the letters'
    ink across the baseline (SAM3's word boxes run 1.2-1.4x taller); plane = depth on a band 4 letter heights deep along
    the words (n10: 25 deg lean; wider bands reach the creases and flatten it to 17). That plane is
    used only when the letters' own slant agrees with it - VERVE's upright letters vs a -20 deg depth lean - else the
    usual pose stays. Returns dict(base, letter_h, span, plane=(n, c) | None, lean, lean_info)."""
    car, dep, K = c["car"], c["dep"], c["K"]; H, W = car.shape[:2]
    my, mx = np.nonzero(main)
    (lx, ly), (rw, rh), ra = cv2.minAreaRect(np.column_stack([mx, my]).astype(np.float32))
    base = ra if rw >= rh else ra + 90; base = (base + 90) % 180 - 90; t = np.radians(base)
    rot = lambda x, y: ((x - lx) * np.cos(t) + (y - ly) * np.sin(t), -(x - lx) * np.sin(t) + (y - ly) * np.cos(t))

    def col_h(x, y):
        u, v = rot(x, y); ui = np.round(u - u.min()).astype(int)
        lo = np.full(ui.max() + 1, np.inf); hi = np.full(ui.max() + 1, -np.inf)
        np.minimum.at(lo, ui, v); np.maximum.at(hi, ui, v); ok = np.isfinite(lo)
        return float(np.median((hi - lo + 1)[ok])), float(np.ptp(u)) + 1
    mask_h, span = col_h(mx, my)
    e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(car, cv2.COLOR_RGB2GRAY), (3, 3), 0), 40, 120) > 0
    e &= cv2.dilate(main.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ey, ex = np.nonzero(e)
    k, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)))
    hs = [np.ptp(rot(*np.nonzero(lab == i)[::-1])[1]) + 1 for i in range(1, k) if st[i, cv2.CC_STAT_AREA] >= 15]
    letter_h = min(mask_h, float(np.median(hs))) if len(hs) >= 3 else mask_h  # per-letter ink height (the boxes run taller)
    # the letters' own slant: shear that stacks the most edge ink into the fewest columns (+ = top leans right)
    u, v = rot(ex, ey) if len(ex) else (np.zeros(1), np.zeros(1))
    sc = []
    for s in np.linspace(-0.6, 0.6, 121):
        cl = np.round(u - s * v).astype(int); sc.append(((np.bincount(cl - cl.min()).astype(float) ** 2).sum(), s))
    lean_l = base - float(np.degrees(np.arctan(max(sc)[1])))
    out = dict(base=base, letter_h=letter_h, span=span, plane=None, lean=None, lean_info=f"letters {lean_l:.0f}")
    bm = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(bm, cv2.boxPoints(((lx, ly), (span + 2 * letter_h, 4 * letter_h), base)).astype(np.int32), 1)
    bm = (bm > 0) & ~(cv2.dilate(main.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    if bm.sum() < 200:
        return out
    n, c_, share = PP.fit_plane_ransac(dep, K, bm)
    if share < 0.5 or not PP.tilt_reliable(n, MAX_TILT):
        out["lean_info"] += f", band plane unusable (share {share:.2f}, pitch {PP.yaw_pitch(n)[1]:.0f})"
        if abs(lean_l) >= 8:  # no surface to follow, but the letters themselves lean clearly: their slant is the gauge
            # (Yovan, round 10: ORION over the Vantage's ASTON MARTIN, letters at 24 deg, 'does not follow the surface')
            out["lean_shear"] = lean_l; out["lean_info"] += f" -> letters' lean {lean_l:.0f}"
        return out
    # the words' middle ON THE SURFACE: on a turned panel the far letters are smaller, so the middle of the letters in
    # the photo sits off the panel's middle (Yovan, round 3: the Jeep over RANGER "not centred" - 21 px right of the
    # tailgate's axis). Both ends of the words onto the band plane, their 3D midpoint projected back.
    uu = (mx - lx) * np.cos(t) + (my - ly) * np.sin(t)
    ends = []
    for s_ in (uu.min(), uu.max()):
        ray = np.array([(lx + s_ * np.cos(t) - K[0, 2]) / K[0, 0], (ly + s_ * np.sin(t) - K[1, 2]) / K[1, 1], 1.0])
        ends.append(ray * (n @ c_) / (n @ ray))
    out["mid"] = tuple(float(x) for x in PP.project(K, ((ends[0] + ends[1]) / 2)[None])[0])
    Hm, _ = homography_on_surface(K, n, c_, (lx, ly), 100, 100, 100, base)
    p = [Hm @ np.array([50, yy, 1.0]) for yy in (50, 0)]; d = p[1][:2] / p[1][2] - p[0][:2] / p[0][2]
    lean_d = float(np.degrees(np.arctan2(d[0], -d[1])))
    out["lean_info"] += f", band plane {lean_d:.0f}"
    # trusted only when the letters clearly lean the same way (n10: letters 18 / plane 25; VERVE: its upright letters
    # read -10 here, the plane -20 - neither is right, so the usual pose stays)
    if abs(lean_d) < 5 or (abs(lean_l) >= 8 and abs(lean_d - lean_l) <= 8):
        out.update(plane=(n, c_), lean=lean_d)
    elif abs(lean_l) >= 5 and abs(lean_d) >= 5 and np.sign(lean_l) == np.sign(lean_d):
        # same direction, different amount (VERVE: letters -10, plane -20): the surface DOES lean - Yovan, round 4,
        # drew the VELTRA verticals at -16 ("not following the slope") -> the plane, its lean set to the mean of the two
        out.update(plane=(n, c_), lean=lean_d, lean_target=(lean_l + lean_d) / 2)
        out["lean_info"] += f" -> lean {(lean_l + lean_d) / 2:.0f}"
    return out


def shear_to(Hm, lw, lh, lean):
    """Shear the placed logo about its centre so its verticals lean `lean` deg in the image (+ = top leans right)."""
    ctr = Hm @ np.array([lw / 2, lh / 2, 1.0]); ctr = ctr[:2] / ctr[2]
    dn = Hm @ np.array([lw / 2, lh / 2 + 1.0, 1.0]); dv = dn[:2] / dn[2] - ctr
    k = (-np.tan(np.radians(lean)) * dv[1] - dv[0]) / dv[1]
    Hm = np.array([[1, k, -k * ctr[1]], [0, 1, 0], [0, 0, 1.0]]) @ Hm
    q = Hm @ np.array([[0, 0, 1], [lw, 0, 1], [lw, lh, 1], [0, lh, 1]], float).T
    return Hm, (q[:2] / q[2]).T


def edge_hits(edges, qs):
    """Strong-edge pixels inside each quad (eroded 1 px), counted on one local window round all of them."""
    a = np.vstack(qs); H, W = edges.shape
    x0, y0 = max(0, int(a[:, 0].min()) - 3), max(0, int(a[:, 1].min()) - 3)
    x1, y1 = min(W, int(a[:, 0].max()) + 4), min(H, int(a[:, 1].max()) + 4)
    if x1 <= x0 or y1 <= y0:
        return [0] * len(qs)
    e = edges[y0:y1, x0:x1]; out = []
    for q in qs:
        m = np.zeros(e.shape, np.uint8); cv2.fillConvexPoly(m, (q - [x0, y0]).astype(np.int32), 1)  # same pixels as quad_mask
        out.append(int((e & (cv2.erode(m, np.ones((3, 3), np.uint8)) > 0)).sum()))
    return out


def site_geometry(c, j, spot, lw, lh, body, marks=None):
    car, dep, K = c["car"], c["dep"], c["K"]
    H, W = car.shape[:2]
    ys, xs = np.nonzero(spot); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; ext = max(x1 - x0, y1 - y0)
    # old-emblem marks: masks mostly inside the spot, minus whole-hub / whole-panel detections (wheel s2 got the hub disc)
    on_hub = any((w & spot).sum() >= 0.8 * spot.sum() for w in c["wheels"])
    # the whole-hub-disc guard (> 45% of the spot) only on hubs: a user painting tightly round a grille emblem makes the
    # emblem most of the spot (t01/t04/t05/t06/t08, 2026-10-07: every old emblem was dropped -> no size, no pose gauge)
    # >= 90% inside: a VERVE detection that ran on into the GDI badge beside it (85% inside) doubled the lettering length
    cand = [m for m in c["sam"] if (m & spot).sum() >= 0.9 * m.sum() and (not on_hub or m.sum() <= 0.45 * spot.sum())]
    geo = dict(spot=int(j), hub=on_hub)
    old = np.zeros_like(spot)
    # only the part near the paint: one van detection held both hub plates (90% of it inside spot 1, a few px of the
    # far plate 2900 px away) -> the hull spanned the car and read as lettering (2026-10-07)
    rn = max(3, int(0.15 * ext))
    near = cv2.dilate(spot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rn + 1, 2 * rn + 1))) > 0
    for m in cand:
        old |= m & near
    body_area_frac = float((body > 0).mean())  # logo body area / its bounding box
    if old.sum() > 0:
        oy, ox = np.nonzero(old)
        tx, ty = float(ox.mean()), float(oy.mean())
        hull = cv2.convexHull(np.column_stack([ox, oy]).astype(np.int32))
        area = float(cv2.contourArea(hull))
        # badge AREA = old emblem's hull area: side of the logo's bounding square scaled so body area matches
        target = float(np.sqrt(area / max(body_area_frac, 1e-3) * max(lw, lh) / min(lw, lh)))
        # specks (< 5% of the marks) don't decide the shape: a sliver of the spoiler in one ASTON MARTIN mask made the
        # words read as one emblem (n10 Dacia, robust2 2026-10-09)
        k_, lab_, st_, _ = cv2.connectedComponentsWithStats(old.astype(np.uint8))
        main = np.isin(lab_, [i for i in range(1, k_) if st_[i, cv2.CC_STAT_AREA] >= 0.05 * old.sum()])
        my_, mx_ = np.nonzero(main)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([mx_, my_]).astype(np.float32))
        # lettering = a level strip made of SEVERAL separate marks; one flat oval emblem seen at a grazing angle is
        # just as elongated but is one piece (Renault EV nose, 2026-10-07)
        # (or so long that no disc would look like that: ASTON MARTIN came back from SAM3 as 2 merged blobs)
        asp = max(rw, rh) / max(1.0, min(rw, rh))
        strip = asp > 4 or (asp > 2.2 and cv2.connectedComponents(main.astype(np.uint8))[0] - 1 >= 3)
        obox = (ox.max() - ox.min() + 1, oy.max() - oy.min() + 1)
        if V6 and not strip and not on_hub:  # an upright letter (Y) has little area: the badge spans its outline
            target = max(target, 0.9 * max(obox))  # (Citroen / Opel came out half the showcase size, alfa lip too small)
        if strip:  # lettering: centre of the word, not of its ink (VERVE's wide letters pulled the mean right)
            (tx, ty), _, _ = cv2.minAreaRect(np.column_stack([mx_, my_]).astype(np.float32)); tx, ty = float(tx), float(ty)
        geo.update(site="lettering" if strip else "emblem", n_marks=len(cand), old_area=round(area))
    else:
        tx, ty = cx, cy; target = (0.6 if on_hub else SPOT_FILL) * ext; geo["site"] = "spot"; strip = True; obox = None; main = None  # hubs keep 0.6 (approved sizes)
    # pose: plane under the spot
    # plane under the emblem + a thin ring around it (the panel it is mounted on); the whole spot often straddles a
    # crease (Citroen Y under the hood edge read as the hood top, pitch 35)
    if old.sum() > 0 and not strip:
        rr = max(5, int(0.35 * np.sqrt(old.sum())))
        pm = (cv2.dilate(old.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))) > 0) & spot
    else:
        pm = spot
    n_, c_, share = PP.fit_plane_ransac(dep, K, pm); n0 = n_.copy()
    if V6 and not on_hub:
        alt = panel_plane(c, spot, old, (tx, ty), ext, strip)
        if alt is not None:
            n_, c_, share, geo["plane_src"] = alt; n0 = n_.copy()
    geo["plane_share"] = round(share, 2)
    th = PP.edge_angle(car, (tx, ty), max(ext, 24))
    if V6 and th is None and not on_hub:  # no panel line found: the badge stays level (Citroen Y: rotated 10 deg)
        th = 0.0; geo["level"] = "no edge -> level"
    y0_, p0_ = PP.yaw_pitch(n_); ledge = None
    if not PP.tilt_reliable(n_, MAX_TILT):
        if th is not None:
            if V6 and LEDGE == "keepturn":  # keep the measured TURN, only the tilt gives: turning alone to match a level
                # ledge flipped the alfa tailgate badge to +18 deg on a -35 deg tailgate (Yovan, round 8: 'not following the surface')
                n_, _, _, _ = nearest_edge_keep_turn(K, c_, (tx, ty), th, y0_, float(np.clip(p0_, -MAX_TILT, MAX_TILT)), MAX_TILT)
            elif V6 and LEDGE in ("surface", "panel"):
                n_ = None
                if LEDGE == "panel":  # the panel round the spot (then below it) when its tilt is believable
                    for ya, yb, kx in ((-1.0, 1.5, 1.5), (0.5, 2.5, 3.0)):
                        m = np.zeros((H, W), bool)
                        m[int(max(0, ty + ya * ext)):int(min(H, ty + yb * ext)), int(max(0, tx - kx * ext)):int(min(W, tx + kx * ext))] = True
                        nn, _, _ = PP.fit_plane_ransac(dep, K, m & ~spot)
                        if PP.tilt_reliable(nn, MAX_TILT):
                            n_ = nn; geo["ledge_plane"] = [round(v, 1) for v in PP.yaw_pitch(nn)]; break
                if n_ is None:
                    r_, pr_ = np.radians(y0_), np.radians(float(np.clip(p0_, -MAX_TILT, MAX_TILT)))
                    n_ = np.cos(pr_) * np.array([np.sin(r_), 0.0, -np.cos(r_)]) + np.sin(pr_) * np.array([0.0, -1.0, 0.0])
                ledge = th
            else:
                n_, _, _ = PP.normal_from_edge(K, c_, (tx, ty), th, float(np.clip(p0_, -MAX_TILT, MAX_TILT)), y0_)
            geo["pose"] = "edge (tilt implausible)"
        else:
            n_ = PP.clamp_pitch(n_, MAX_TILT); geo["pose"] = "clamped"
    elif th is not None and not on_hub:
        if V6:
            n_, _, _, err = nearest_edge_keep_turn(K, c_, (tx, ty), th, y0_, p0_, MAX_TILT)
        else:
            n_, _, _, err = PP.normal_nearest_edge(K, c_, (tx, ty), th, y0_, p0_, MAX_TILT)
        geo["pose"] = f"plane + edge {th:.1f} (err {err:.1f})"
    else:
        geo["pose"] = "plane"
    lie = False
    # blank panel facing up (bonnet top / nose) that no believable plane explained: the badge lies on it, level (the
    # 35 deg fallback stood it up on the grille face under the bonnet nose, r15 b2). Not when a panel band was believable
    # (merc front wing read 39 deg under the spot but its band 0 - a fender badge stays upright, r15 b3)
    if geo["site"] == "spot" and not on_hub and geo["pose"] in ("edge (tilt implausible)", "clamped"):
        n_s, c_s, _ = PP.fit_plane_ransac(dep, K, spot); ys_, ps_ = PP.yaw_pitch(n_s)
        if ps_ > MAX_TILT:
            r_, p_ = np.radians(ys_), np.radians(min(ps_, SPOT_UP_TILT))
            n_ = np.cos(p_) * np.array([np.sin(r_), 0.0, -np.cos(r_)]) + np.sin(p_) * np.array([0.0, -1.0, 0.0]); c_ = c_s
            lie, ledge = True, None; geo["pose"] = f"lies on the panel (depth {ys_:.0f}/{ps_:.0f})"
    if on_hub:  # a hub faces the way its wheel does: the wheel's outline (SAM3) is a far steadier gauge than the depth
        # under a small recessed cap (modern EV front hub: plane said yaw 49 for a near-round cap)
        wm = max((w for w in c["wheels"] if (w & spot).sum() >= 0.8 * spot.sum()), key=lambda w: (w & spot).sum())
        cs = cv2.findContours(wm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        (wx, wy), (ea, eb), wang = cv2.fitEllipse(max(cs, key=cv2.contourArea))
        # (no snap to the wheel ellipse's centre: on a close-up the SAM3 wheel is cut by the frame and its centre fell
        # 40 px off the cap - red DS close-up, 2026-10-07; the paint is the better centre)
        yw = float(np.degrees(np.arccos(np.clip(min(ea, eb) / max(ea, eb), 0, 1))))
        if V6 and yw >= 20:
            # the outline holds the tyre's TREAD on a turned wheel, so it reads rounder than the wheel face (Yovan,
            # 2026-10-08: the Opel on the front wheel "too frontal" - outline 32 deg, face ~50). The tread also shifts
            # the outline's centre off the hub by half its visible width: take it off -> the face's own angle
            a_ = np.radians(wang if ea < eb else wang + 90)  # minor-axis direction
            d = abs((tx - wx) * np.cos(a_) + (ty - wy) * np.sin(a_))
            yw = float(np.degrees(np.arccos(np.clip((min(ea, eb) - 2 * d) / max(ea, eb), 0, 1))))
            geo["hub_tread_px"] = round(float(d), 1)
        # the outline's squash is seen along the LINE OF SIGHT to the hub, not along the camera axis (Yovan, round 11:
        # the red hatch's far front hub 'too squashed' - 61 deg from the axis read as ~70 that far right in the frame)
        phi = float(np.degrees(np.arctan((tx - K[0, 2]) / K[0, 0]))) if HUB_SIGHT else 0.0
        yw *= (1 if y0_ + phi >= 0 else -1)
        r_ = np.radians(yw - phi); n_ = np.array([np.sin(r_), 0.0, -np.cos(r_)])
        geo["pose"] = f"wheel outline (yaw {yw:.0f})"
        if HUB_NOLEDGE and ledge is not None:  # a hub badge stands upright on its wheel: the 'ledge' line was an edge
            # inside the cap (Honda MPV front: its chrome cap's mirrored horizon) and turned the H ~10 deg -> 'mangled'
            geo["ledge_dropped"] = round(float(ledge), 1); ledge = None
        if phi:
            geo["hub_sight"] = round(phi, 1)
        cd = cap_disc_centre(c, spot, (tx, ty), target) if V6 else None
        rc = cd or (rim_centre(car, wm, spot, (tx, ty)) if V6 else None)
        if rc is not None:  # side-on wheel: the badge goes to the rim's centre, not the old cap's (Honda MPV: the
            # generated caps sat 7 px high on both wheels, Yovan 2026-10-09 "not centered in the wheel")
            geo["hub_centre"] = f"{'cap' if cd else 'rim'} ({rc[0] - tx:+.0f},{rc[1] - ty:+.0f} px)"; tx, ty = rc
    band = None
    if V6 and old.sum() > 0 and not strip and not on_hub:
        band = band_between_creases(c, old, (tx, ty))
        if band is not None:
            n_, c_ = band[0], band[1]; geo.update(band[3]); geo["pose"] = f"band between creases (slope {band[3]['band_slope']:.0f})"
    let = lettering_frame(c, main) if V6 and strip and main is not None and not on_hub else None
    if let is not None:
        geo.update(letter_h=round(let["letter_h"], 1), letter_base=round(let["base"], 1), letter_span=round(let["span"]),
                   letter_lean=let["lean_info"])
        if let["plane"] is not None:
            n_, c_ = let["plane"]; geo["pose"] = f"lettering band (lean {let['lean']:.0f})"
        if let.get("mid") is not None and np.hypot(let["mid"][0] - tx, let["mid"][1] - ty) <= 0.25 * let["span"]:
            geo["letter_mid"] = f"{let['mid'][0] - tx:+.0f},{let['mid'][1] - ty:+.0f} px"; tx, ty = let["mid"]
    yaw, pitch = PP.yaw_pitch(n_); geo.update(yaw=round(yaw), pitch=round(pitch))
    # clamp: shrink until the badge crosses no long strong edge (old emblem's own edges excluded)
    excl = cv2.dilate(old.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0 if old.sum() else np.zeros_like(spot)
    if on_hub:  # the painted cap/plate is erased anyway: its own rim is no limit (van plates' top edge shrank the badge)
        excl |= spot
    through = through_lim = None
    if EDGE_THROUGH and not on_hub and not (strip and old.sum()):  # lettering keeps its edge clamp (Dacia over ASTON MARTIN: the tail-light strip runs into the words)
        if old.sum():
            through = cv2.dilate(old.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        else:  # no old emblem found: the painted spot is where the badge goes - what crosses it lies under it
            excl |= spot; through = cv2.dilate(spot.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            sy_, sx_ = np.nonzero(spot); through_lim = (sx_.mean(), sy_.mean(), max(np.ptp(sx_), np.ptp(sy_)) / 2)
    # clearance: the badge keeps a small gap to every long edge (the Peugeot shield sat ON the light bar's edge)
    edges = strong_edges(car, excl, max(12, 0.5 * target), through, through_lim); raw_edges = edges
    gap = max(2, int(round(0.06 * target)))
    edges = cv2.dilate(edges.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))) > 0
    # a round old emblem is the best perspective gauge there is: a disc on the panel, seen as this ellipse. The depth
    # plane can't see a bonnet top at a grazing angle (tilt read as implausible, clamped to 35 deg -> the Renault diamond
    # came out front-on over a flat oval, Yovan 2026-10-07)
    ell = None
    if old.sum() > 0 and not strip:
        hull = cv2.convexHull(np.column_stack(np.nonzero(old)[::-1]).astype(np.int32))
        if len(hull) >= 5:
            (ex, ey), (ew, eh), ea = cv2.fitEllipse(hull)
            fillr = cv2.contourArea(hull) / (np.pi * ew * eh / 4 + 1e-6)
            # a truly OVAL emblem (Infiniti grille, van nose) is not a disc in perspective: when the depth plane is
            # trustworthy, a disc on it would look rounder than this -> keep the plane pose, don't stretch the logo
            v = np.linalg.solve(K, np.array([ex, ey, 1.0])); v /= np.linalg.norm(v)
            disc = abs(float(n0 @ v)) if PP.tilt_reliable(n0, MAX_TILT) else None
            meas = min(ew, eh) / max(ew, eh)
            # ...and a disc is squashed ALONG the panel's tilt: Infiniti oval is wide while the yawed grille would make a
            # disc tall (t04) -> a real oval even though its aspect alone looked plausible
            if disc is not None and disc < 0.9 and meas < 0.9:
                exp_minor = np.degrees(np.arctan2(n0[1], n0[0])); got_minor = ea if ew < eh else ea + 90
                dang = abs((got_minor - exp_minor + 90) % 180 - 90)
                if dang > 35:
                    disc = 1.0
            # real oval badges are WIDE with a level long axis (Infiniti, van); a tilted long axis is a disc seen at an
            # angle the depth plane didn't catch (DS hatch: disc stands proud of the grille the plane was fitted to)
            major = (ea + 90 if ew < eh else ea) % 180
            if abs((major + 90) % 180 - 90) > 20:
                disc = None
            if 0.88 < fillr < 1.12 and min(ew, eh) >= 8 and (disc is None or meas >= 0.9 * disc):
                ell = (ex, ey, ew, eh, ea)
                geo.update(pose=f"old round emblem as gauge (ellipse {ew:.0f}x{eh:.0f} @ {ea:.0f})", ellipse_fill=round(fillr, 2))
            elif 0.88 < fillr < 1.12:
                geo["oval"] = f"real oval {meas:.2f} vs disc {disc:.2f}"

    def make_h(t):
        Hm, q = make_h0(t)
        if let is not None and let["plane"] is None and let.get("lean_shear") is not None:
            Hm, q = shear_to(Hm, lw, lh, let["lean_shear"])
        return Hm, q

    def make_h0(t):
        if let is not None and let["plane"] is not None:  # along the words, up = the surface across them (full lean)
            Hm, q = homography_on_surface(K, n_, c_, (tx, ty), lw, lh, t, let["base"])
            if let.get("lean_target") is not None:  # shear the verticals to the agreed lean (+ = top leans right)
                ctr = Hm @ np.array([lw / 2, lh / 2, 1.0]); ctr = ctr[:2] / ctr[2]
                dn = Hm @ np.array([lw / 2, lh / 2 + 1.0, 1.0]); dv = dn[:2] / dn[2] - ctr
                k = (-np.tan(np.radians(let["lean_target"])) * dv[1] - dv[0]) / dv[1]
                Hm = np.array([[1, k, -k * ctr[1]], [0, 1, 0], [0, 0, 1.0]]) @ Hm
                q = Hm @ np.array([[0, 0, 1], [lw, 0, 1], [lw, lh, 1], [0, lh, 1]], float).T; q = (q[:2] / q[2]).T
            return Hm, q
        if ell is None and band is not None:
            Hm, q = homography_on_surface(K, n_, c_, (tx, ty), lw, lh, t, band[2])
            keep = LIP_LEAN if band[3].get("lip") else WIDE_LEAN if band_wide else 1.0
            if keep < 1:  # a WIDE logo's full lean reads as a turn (Opel at 25 deg: ring looked tilted, "yellow is the correct
                # one") and no lean reads flat ("my badge is not in perspective"); Yovan drew 14.6 deg -> keep 60% of the lean
                ctr = Hm @ np.array([lw / 2, lh / 2, 1.0]); ctr = ctr[:2] / ctr[2]
                dn = Hm @ np.array([lw / 2, lh / 2 + 1.0, 1.0]); dv = dn[:2] / dn[2] - ctr
                k = -(1 - keep) * dv[0] / dv[1]; Hm = np.array([[1, k, -k * ctr[1]], [0, 1, 0], [0, 0, 1.0]]) @ Hm
                q = Hm @ np.array([[0, 0, 1], [lw, 0, 1], [lw, lh, 1], [0, lh, 1]], float).T; q = (q[:2] / q[2]).T
            return Hm, q
        if lie:
            return homography_on_surface(K, n_, c_, (tx, ty), lw, lh, t, 0.0)
        if ell is None and ledge is not None:  # a ledge read as facing up: logo along the ledge line ON the surface
            return homography_on_surface(K, n_, c_, (tx, ty), lw, lh, t, ledge)
        if ell is None:
            return PP.logo_homography(K, n_, c_, (tx, ty), lw, lh, t)
        ex, ey, ew, eh, ea = ell; f = t / max(ew, eh)
        sc = 2.0 / max(lw, lh)
        L = np.array([[sc, 0, -sc * lw / 2], [0, sc, -sc * lh / 2], [0, 0, 1.0]])
        # centred on the old emblem's BOX, not its ellipse: a shield's wide top pulls the fitted ellipse up and the new
        # shield sat 5 px high on the 308 SW grille (2026-10-07); for a round emblem the two are the same point
        M = np.eye(3); M[:2, :2] = triu(ew * f, eh * f, ea); M[:2, 2] = [(ox.min() + ox.max()) / 2, (oy.min() + oy.max()) / 2]
        if geo.get("hub_centre"):  # a hub with a measured rim centre: the badge goes THERE, not on the old emblem's box
            M[:2, 2] = [tx, ty]  # (Yovan, round 4: the Alfa on the wet cap 'not centred' - the lion sat 30 px off the cap's centre)
        Hm = M @ L
        q = (Hm @ np.array([[0, 0, 1], [lw, 0, 1], [lw, lh, 1], [0, lh, 1]], float).T)[:2].T
        return Hm, q

    if ell is not None:
        target = 0.95 * max(ell[2], ell[3])
    wordmark, own = wordmark_parts(marks, lw, lh) if strip and old.sum() > 0 else (False, 1.0)
    sym_let = False
    if strip and old.sum() > 0 and lw / lh > 2.2 and (wordmark or let is None):
        # wordmark replacing lettering: same letter height, never wider
        # letter height = median vertical ink extent per column (the min side of the word's box also holds its slant:
        # ORION came out 1.7x the ASTON MARTIN letters). The NEW logo's letters are measured the same way, per letter
        # box (Yovan, robust2 2026-10-09: Jeep over RAM "too small" - its 'eep' was sized as if all capitals)
        cols = [np.ptp(np.nonzero(old[:, x])[0]) + 1 for x in range(ox.min(), ox.max() + 1) if old[:, x].any()]
        letter_h = float(np.median(cols))
        target = min(1.05 * letter_h / own * lw / lh, 1.05 * max(rw, rh)); geo.update(letter_h=round(letter_h, 1), own_letter=round(own, 2))
    elif let is not None:
        sym_let = True
        # a SYMBOL replacing lettering (Audi rings over V O L V O "too small", 37 x 10 px): SYM_K letter heights tall,
        # at most 60% of the words' length, measured across / along the baseline after the perspective
        want_h, max_w = SYM_K * let["letter_h"], 0.6 * let["span"]
        b = np.radians(let["base"]); ub = np.array([np.cos(b), np.sin(b)]); nb = np.array([-np.sin(b), np.cos(b)])
        # ...centred in the free band between the nearest long edges above and below the words, and no taller than it
        # (V70: chrome strip 12 px under the letters, room above - the edge clamp kept the rings letter-sized)
        reach = int(4 * let["letter_h"]); ups, dns = [], []
        for x in range(int(tx - 0.25 * let["span"]), int(tx + 0.25 * let["span"]) + 1, 2):
            yc = int(round(ty + (x - tx) * np.tan(b)))
            if not (0 <= x < W and reach <= yc < H - reach):
                continue
            col = edges[yc - reach:yc + reach + 1, x]; a_, b_ = np.nonzero(col[:reach])[0], np.nonzero(col[reach + 1:])[0]
            if len(a_) and len(b_):
                ups.append(a_[-1] - reach); dns.append(b_[0] + 1)
        if len(ups) >= 5:
            u_, d_ = float(np.median(ups)), float(np.median(dns))
            ty += (u_ + d_) / 2; want_h = min(want_h, 0.9 * (d_ - u_) * np.cos(b))
            geo["letter_band"] = [round(u_), round(d_)]
        cs = cv2.findContours((body > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sil = np.vstack([cc[:, 0, :] for cc in cs]).astype(np.float64)[None]
        for _ in range(4):
            sp = cv2.perspectiveTransform(sil, make_h(target)[0])[0]
            target *= min(want_h / max(np.ptp(sp @ nb), 1e-6), max_w / max(np.ptp(sp @ ub), 1e-6))
        geo["want_h"] = round(want_h, 1)
    band_wide = band is not None and lw >= 1.2 * lh
    if band is not None and ell is None:
        # on a sloped band the squash made the badge look tiny when sized by its width (cit / white 'very small'): size it
        # by its VISIBLE height = 0.85 x the old emblem's, width at most 2x that height
        cs = cv2.findContours((body > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sil = np.vstack([cc[:, 0, :] for cc in cs]).astype(np.float64)[None]
        oh = float(np.ptp(oy) + 1)
        if band[3].get("lip"):  # a melted old emblem on a thin lip (Alfa: 58 px on a 92 px lip, 'too small' twice) -> fill
            cr = band[3]["creases"]; oh = max(oh, LIP_FILL / 0.85 * (cr[1] - cr[0]) * np.cos(np.radians(band[2])))
        for _ in range(3):
            sp = cv2.perspectiveTransform(sil, make_h(target)[0])[0]
            target *= min(0.85 * oh / max(np.ptp(sp[:, 1]), 1e-6), 2.0 * oh / max(np.ptp(sp[:, 0]), 1e-6))
    t0 = target; base_c = (tx, ty); slide = 0
    for _ in range(30):
        if sym_let:
            tx, ty = base_c
        Hm, q = make_h(target)
        qm = quad_mask(q, car.shape)
        hit = int((edges & cv2.erode(qm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum())
        if hit > 2 and sym_let:
            # slide across the words (<= 15% of the size) off a crease before shrinking: the Dacia over ASTON MARTIN
            # met the tail-light strip's lower edge with one corner and shrank 122 -> 63 px (Yovan, round 10: 'a bit small')
            sm = int(0.15 * target)
            offs = sorted((s_ for s_ in range(-sm, sm + 1) if s_), key=lambda v: (abs(v), v))
            ok = [s_ for s_, h_ in zip(offs, edge_hits(edges, [q + s_ * nb for s_ in offs])) if h_ <= 2] if offs else []
            slide = ok[0] if ok else 0
            if slide:
                tx, ty = base_c[0] + slide * nb[0], base_c[1] + slide * nb[1]
                Hm, q = make_h(target)
                qm = quad_mask(q, car.shape)
                hit = int((edges & cv2.erode(qm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).sum())
        qw, qh = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        # compact emblem: no bigger than its footprint, per axis (a tall shield on the van's wide oval hung below it)
        # (not on hubs: there the old mark is often a small logo on a bigger cap/plate, and the edge clamp below already
        # keeps the badge on the cap - the van's two equal plates came out 67 vs 77 px)
        if band is not None and ell is None:
            fits = True  # sized by the old emblem's height above; the creases and the edge clamp bound it
        elif V6 and obox is not None and obox[0] <= 1.3 * obox[1]:  # upright / round footprint: a square cap (a wide
            fits = strip or on_hub or max(qw, qh) <= 1.25 * max(obox)  # Citroen oval may be wider than the Y it replaces)
        else:
            fits = strip or on_hub or (qw <= 1.2 * obox[0] and qh <= 1.2 * obox[1])
        sy_, sx_ = np.nonzero(spot); floor = MIN_PAINT_FRAC * max(np.ptp(sx_), np.ptp(sy_)) if not (on_hub or strip) else 0
        if (hit <= 2 and fits) or target < 0.55 * t0 or target * 0.93 < floor:
            break
        target *= 0.93
    if slide:
        geo["slide_px"] = slide
    geo.update(size=round(target), size_first=round(t0), edge_hits=hit, centre=[round(tx), round(ty)])
    return Hm, q, geo, raw_edges


def hub_cap(c, spot, geo):
    """The round, neutral centre cap the old emblem sits on (hub spots only), as a cv2 ellipse in image px, or None.
    The erase LoRA removes a cap together with its emblem: the Honda MPV's chrome caps came back as bare hub and the H
    floated on the spokes (Yovan's robust2 review, 2026-10-09). Only a cap clearly bigger than the new badge counts -
    one about the badge's size is covered by the badge anyway (t03 Peugeot shields, approved without a cap)."""
    if not geo.get("hub") or "centre" not in geo:
        return None
    ys, xs = np.nonzero(spot); ext = max(np.ptp(xs), np.ptp(ys))
    rn = max(3, int(0.15 * ext))
    near = cv2.dilate(spot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rn + 1, 2 * rn + 1))) > 0
    tx, ty = geo["centre"]; size = float(geo["size"]); car = c["car"]
    best = None
    for m in c["sam"]:
        if m.sum() < 30 or (m & near).sum() < 0.9 * m.sum():
            continue
        cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        big = max(cs, key=cv2.contourArea)
        if len(big) < 5:
            continue
        (ex, ey), (ea, eb), ang = cv2.fitEllipse(big)
        el = np.zeros(m.shape, np.uint8); cv2.ellipse(el, ((ex, ey), (ea, eb), ang), 1, -1)
        iou = (m & (el > 0)).sum() / max(1, (m | (el > 0)).sum())
        px = car[m].astype(int)
        chroma = float(np.median(px.max(1) - px.min(1)))
        d = min(ea, eb)
        # a disc (a square plate fits its ellipse worse), neutral (coloured Alfa / Citroen plates still go), centred
        # under the badge (not the wheel boss), 1.2-2x the badge (not one blob of the old design, not the whole wheel)
        if iou >= 0.92 and chroma <= 20 and np.hypot(ex - tx, ey - ty) <= 0.1 * d and 1.2 * size <= d <= 2.0 * size:
            if best is None or m.sum() > best[0]:
                best = (int(m.sum()), ((float(ex), float(ey)), (float(ea), float(eb)), float(ang)))
    return best[1] if best else None


def hub_cap_image(c, spot, geo, n=72, gate=0.9):
    """The centre cap from the IMAGE when SAM3 returned none (round 12: the Mercedes front cap had no SAM3 mask, so the
    27 px ring was drawn inside the 39 px cap rim and the finish mangled the two - Yovan 'looks mangled'; the rear cap,
    found by SAM3, was fine). An ellipse with the wheel face's squash (yaw from the line of sight) and minor axis,
    centre within 0.4 badge of the paint centre, 1.2-2x the badge; score = mean radial gradient round it. The rim is
    often a thin bright ring: the outermost complete edge at the best centre is the cap's edge (agrees with SAM3's caps
    to 1 px / 1.4 px of diameter on the Mercedes rear and FIAT). Kept only when the edge is strong and runs round
    (>= 90% of the outline) - spokes and blur score on one side. Returns (cv2 ellipse | None, info)."""
    size = float(geo["size"]); car = c["car"]
    g = cv2.GaussianBlur(cv2.cvtColor(car, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    wm = max((w for w in c["wheels"] if (w & spot).sum() >= 0.8 * spot.sum()), key=lambda w: (w & spot).sum())
    cs = cv2.findContours(wm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    (_, _), (ea, eb), wang = cv2.fitEllipse(max(cs, key=cv2.contourArea))
    th = np.radians(wang if ea < eb else wang + 90)  # the wheel's minor-axis direction
    ratio = float(np.clip(np.cos(np.radians(geo.get("yaw", 0) + geo.get("hub_sight", 0))), 0.2, 1.0))
    ys, xs = np.nonzero(spot); pcx, pcy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False); u0, v0 = np.cos(t), np.sin(t)
    stp = max(0.5, size / 50); R = np.arange(0.6 * size, 1.0 * size + 1e-6, stp)
    off = np.arange(-0.4 * size, 0.4 * size + 1e-6, stp); CX, CY = np.meshgrid(pcx + off, pcy + off)

    def ring(r, cx, cy):  # |radial gradient| at n points round the ellipse of semi-major r centred (cx, cy)
        lu, lv = ratio * r * u0, r * v0
        nu, nv = u0 / (ratio * r), v0 / r; nn = np.hypot(nu, nv); nu, nv = nu / nn, nv / nn
        ndx, ndy = nu * np.cos(th) - nv * np.sin(th), nu * np.sin(th) + nv * np.cos(th)
        px = (np.asarray(cx)[..., None] + lu * np.cos(th) - lv * np.sin(th)).astype(np.float32)
        py = (np.asarray(cy)[..., None] + lu * np.sin(th) + lv * np.cos(th)).astype(np.float32)
        fx, fy = px.reshape(-1, n), py.reshape(-1, n)  # remap wants < 32767 rows
        sx = np.concatenate([cv2.remap(gx, fx[i:i + 20000], fy[i:i + 20000], cv2.INTER_LINEAR) for i in range(0, len(fx), 20000)])
        sy = np.concatenate([cv2.remap(gy, fx[i:i + 20000], fy[i:i + 20000], cv2.INTER_LINEAR) for i in range(0, len(fx), 20000)])
        return np.abs(sx.reshape(px.shape) * ndx + sy.reshape(px.shape) * ndy)

    best = None
    for r in R:
        sc = ring(r, CX, CY).mean(-1); k = np.unravel_index(np.argmax(sc), sc.shape)
        if best is None or sc[k] > best[0]:
            best = (float(sc[k]), float(CX[k]), float(CY[k]), float(r))
    s, ex, ey, r = best
    comp = float((ring(r, ex, ey) >= 0.4 * s).mean())
    for r2 in R[::-1]:  # the outermost complete edge, as strong as 60% of the best
        if r2 <= r:
            break
        rad = ring(r2, ex, ey)
        if rad.mean() >= 0.6 * s and (rad >= 0.4 * rad.mean()).mean() >= 0.8:
            r = float(r2); break
    info = dict(score=round(s, 1), round_share=round(comp, 2), centre=[round(ex, 1), round(ey, 1)], d=round(2 * r, 1))
    if comp < gate or s < 15:
        return None, info
    return ((ex, ey), (2 * r * ratio, 2 * r), float(np.degrees(th))), info


def blank_hub_cap(c, spot, geo):
    """The plain centre cap of a BLANK hub (no old emblem found: a studio render's bare cap), from the image. The badge
    size there is only a guess from the paint, so the cap is searched relative to the PAINT (diameter 0.66-1.1 x what
    was painted) and the roundness gate is 0.85 (a three-quarter-view cap measured 0.89; square plates score <= 0.85
    with an edge score < 15). Returns (cv2 ellipse | None, info)."""
    ys, xs = np.nonzero(spot); pd = float(max(np.ptp(xs), np.ptp(ys)) + 1)
    # 0.8: the sedan's near cap measured 0.849 with a clean 18.6 edge (round 18); square plates and spare covers stay
    # out on their edge score (< 15) or roundness (<= 0.74) across the 53-car test set
    return hub_cap_image(c, spot, dict(geo, size=0.55 * pd), gate=0.8)


def cap_is_plain(car, ell, inner=0.45, max_hp=8.0):
    """True when the cap's face holds nothing to erase: the high-pass texture inside `inner` of its radius, at a scale
    where the cap radius is 64 px, stays under max_hp (plain caps 1.3-5.8, caps with an old logo 12-71 in the 53-car
    test set; the rim and spokes sit outside 0.45 even on a 3/4-view cap). Returns (bool, hp)."""
    (ex, ey), (ea, eb), ang = ell
    r = max(ea, eb) / 2; s = 64.0 / max(r, 1.0); H, W = car.shape[:2]
    x0, y0, x1, y1 = int(max(0, ex - 1.2 * r)), int(max(0, ey - 1.2 * r)), int(min(W, ex + 1.2 * r + 1)), int(min(H, ey + 1.2 * r + 1))
    g = cv2.cvtColor(car[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (max(8, int((x1 - x0) * s)), max(8, int((y1 - y0) * s))), interpolation=cv2.INTER_AREA)
    m = np.zeros(g.shape, np.uint8)
    cv2.ellipse(m, (((ex - x0) * s, (ey - y0) * s), (ea * s * inner, eb * s * inner), ang), 1, -1)
    hp = g - cv2.GaussianBlur(g, (0, 0), 8)
    v = float(np.std(hp[m > 0])) if m.sum() >= 20 else 99.0
    return v < max_hp, round(v, 2)


def fit_badge_to_cap(Hm, lw, lh, cap_ell, fill=None, ink=None):
    """A badge on a blank hub's plain cap fills it like a real centre-cap badge (CAP_FILL of the cap's mean diameter,
    centred on the cap): sized from the paint it came out a 19-24 px speck on a 38 px cap (round 16)."""
    (ex, ey), (ea, eb), _ = cap_ell
    q = cv2.perspectiveTransform(np.array([[[0, 0], [lw, 0], [lw, lh], [0, lh]]], np.float64), Hm)[0]
    ctr = Hm @ np.array([lw / 2, lh / 2, 1.0]); ctr = ctr[:2] / ctr[2]
    s = (CAP_FILL if fill is None else fill) * (ea + eb) / 2 / max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))
    if ink is not None:  # a logo with corners (DS, Peugeot shield) reached the cap's rim at the box size (round 18):
        # its ink stays within CAP_INK of the cap radius; round/diamond logos (Alfa, Renault) keep the box size
        ys, xs = np.nonzero(ink > 0.3 * float(ink.max()))
        if len(xs):
            k = float(np.hypot(xs + 0.5 - lw / 2, ys + 0.5 - lh / 2).max()) / (max(lw, lh) / 2)
            s /= max(1.0, k * (CAP_FILL if fill is None else fill) / CAP_INK)
    S = np.array([[s, 0, ex - s * ctr[0]], [0, s, ey - s * ctr[1]], [0, 0, 1.0]])
    return S @ Hm, round(float(s), 2)


def cap_restore(crop, erased, ell, X0, Y0, k, rim=0.85, keep_rim=True):
    """Put the cap back into the erased crop (crop space, k = crop px per image px): its outline and rim band from the
    original, its face a smooth shade - a robust plane fit of the face's own luminance in the rim's tint - so the old
    mark on it is gone. keep_rim=False (a ring logo fitted to the cap): the face runs out to the cap's edge - the old
    chrome rim line sat ~2 px inside the new ring and the finish drew both (round 12, Mercedes front: 'ghosting').
    Returns (float crop, info)."""
    (ex, ey), (ea, eb), ang = ell
    cx, cy = (ex - X0) * k, (ey - Y0) * k; a, b = ea * k / 2, eb * k / 2; t = np.radians(ang)
    h, w = crop.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t); v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    crop = crop.astype(np.float32); erased = erased.astype(np.float32)
    lum = crop.mean(2)
    smp = (rho >= 0.5) & (rho < rim)
    A = np.column_stack([np.ones(smp.sum()), xx[smp] - cx, yy[smp] - cy]); y = lum[smp]
    wgt = np.ones_like(y)
    for _ in range(5):  # down-weight what sits far from the plane (bits of the old mark)
        cf = np.linalg.lstsq(A * wgt[:, None], y * wgt, rcond=None)[0]
        r_ = np.abs(y - A @ cf); s_ = 1.4826 * np.median(r_) + 1e-3
        wgt = 1 / np.maximum(1, r_ / (1.5 * s_))
    band = (rho >= rim) & (rho <= 1.0)
    tint = 1 + np.clip(np.median(crop[band], 0) / max(1e-3, np.median(lum[band])) - 1, -0.06, 0.06)
    face_l = cf[0] + cf[1] * (xx - cx) + cf[2] * (yy - cy)
    inner = rho < rim
    chrome = float(np.median(lum[inner])) >= 110
    if chrome:  # bright metal face: keep its horizontal reflection bands (row medians - the old mark averages out);
        # a plane read as satin paint on the Honda's mirror-chrome caps. Dark painted caps keep the plane: on the FIAT's
        # black cap the row medians left grey stripes where the old rings ran.
        prof = np.array([np.median(lum[y][inner[y]]) if inner[y].sum() >= 3 else np.nan for y in range(h)], np.float32)
        ok = ~np.isnan(prof)
        prof = np.interp(np.arange(h), np.nonzero(ok)[0], prof[ok]).astype(np.float32)
        prof = cv2.GaussianBlur(prof[:, None], (0, 0), max(1.0, 0.08 * min(a, b)))[:, 0]
        face_l = np.repeat(prof[:, None], w, 1)
    face = np.clip(face_l[..., None] * tint[None, None], 0, 255)
    fw = max(1.0, 0.04 * min(a, b))  # feather ~4% of the radius
    f_face = np.clip(((rim if keep_rim else 1.0) - rho) * min(a, b) / fw + 0.5, 0, 1)[..., None]
    f_cap = np.clip((1.0 - rho) * min(a, b) / fw + 0.5, 0, 1)[..., None]
    out = erased * (1 - f_cap) + (crop * (1 - f_face) + face * f_face) * f_cap
    return out, dict(cap_px=[round(ea, 1), round(eb, 1)], face="chrome rows" if chrome else [round(float(x), 2) for x in cf],
                     **({} if keep_rim else dict(rim="repainted")))
