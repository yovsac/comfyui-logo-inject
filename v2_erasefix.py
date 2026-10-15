# Made by Yovan Vujacic
# Verbatim copy of agent-skills/scripts/.runs/li_v2/erase_fix.py (harness) - keep in sync, edit there first.
"""Low-frequency erase fix. The base model's erase is great on texture (grille mesh) but on smooth reflective panels
and trim bars it leaves a round PLATE where the badge was (t01 grille star, t05 chrome bar). Keep the model's fine
texture, replace its large-scale light/dark with a directional interpolation across the hole from its own border:
rows when the surroundings run horizontally (bars, reflection bands), columns when vertical, both blended otherwise.
fix(crop, erased, hole) -> fixed, info"""
import cv2, numpy as np


def _interp(img, hole, axis):
    a = img if axis == 1 else img.transpose(1, 0, 2)
    h = hole if axis == 1 else hole.T
    out = a.copy(); ok = np.zeros(h.shape, bool)
    for y in range(a.shape[0]):
        xs = np.nonzero(h[y])[0]
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        if x0 < 3 or x1 > a.shape[1] - 4:
            continue
        L, R = a[y, x0 - 3:x0].mean(0), a[y, x1 + 1:x1 + 4].mean(0)
        t = np.linspace(0, 1, x1 - x0 + 1)[:, None]
        out[y, x0:x1 + 1] = L * (1 - t) + R * t; ok[y, x0:x1 + 1] = True
    return (out, ok) if axis == 1 else (out.transpose(1, 0, 2), ok.T)


def _bar_crosses(g, hole, d):
    """A long straight near-horizontal (or vertical) edge on the left AND right (top AND bottom) of the hole that line up."""
    ys, xs = np.nonzero(hole); cx, cy = xs.mean(), ys.mean()
    e = cv2.Canny(g.astype(np.uint8), 50, 150); e[cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    # .15d/.10d (round 24): at the bigger r23 crop the t05 chrome bar's left edges came out as 100 px pieces < .25d -> no
    # pair, no fix, a flat light plate behind the Opel
    L = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=int(0.10 * d), minLineLength=int(0.15 * d), maxLineGap=3)
    if L is None:
        return [], 0.0
    left, right, slopes = [], [], []
    for x1, y1, x2, y2 in L[:, 0]:
        if abs(y2 - y1) > 0.12 * abs(x2 - x1) + 1:
            continue
        yc = y1 + (y2 - y1) * (cx - x1) / ((x2 - x1) or 1)  # height where it would cross the hole centre
        if not (ys.min() - 2 <= yc <= ys.max() + 2):
            continue
        side = left if max(x1, x2) < cx else right if min(x1, x2) > cx else None
        if side is not None:
            side.append(yc); slopes.append((y2 - y1) / ((x2 - x1) or 1))
    # tolerance grows with the gap: the r06 Ferrari grille rod is slightly tilted, its two sides met 6 px apart over 430 px
    tol = max(4.0, 0.03 * d)
    pairs = sorted({round((a + b) / 2) for a in left for b in right if abs(a - b) <= tol})
    if not pairs:
        return [], 0.0
    # + the bar's other edges seen on one side only (r06: only the rod's top edge matched -> a 7 px band, rod half drawn)
    return sorted(set(pairs) | {round(y) for y in left + right if min(abs(y - q) for q in pairs) <= 0.15 * d}),         float(np.median(slopes))


def fix(crop, erased, hole):
    crop = crop.astype(np.float32); er = erased.astype(np.float32)
    ys, xs = np.nonzero(hole)
    if len(xs) == 0:
        return erased, dict(skipped=True)
    d = max(np.ptp(xs), np.ptp(ys)) + 1
    cx = float(xs.mean())
    # orientation of the surroundings: structure tensor on a ring around the hole (in the ORIGINAL crop)
    ring = (cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(d * 0.6) | 1,) * 2)) > 0) & ~hole
    g = cv2.GaussianBlur(cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx, jyy = float((gx * gx)[ring].sum()), float((gy * gy)[ring].sum())
    horiz = jyy / (jxx + jyy + 1e-6)  # strong vertical gradients = horizontal structures
    base = er.copy(); base[hole] = crop[hole]  # border values come from the erased image outside the hole = original
    rI, rok = _interp(er, hole, 1); cI, cok = _interp(er, hole, 0)
    w = np.clip((horiz - 0.35) / 0.3, 0, 1)  # 1 = rows, 0 = columns
    both = rok & cok
    est = np.where(both[..., None], w * rI + (1 - w) * cI, np.where(rok[..., None], rI, cI))
    have = rok | cok
    s = max(2.0, 0.06 * d)
    low_est = cv2.GaussianBlur(est, (0, 0), s); low_er = cv2.GaussianBlur(er, (0, 0), s)
    plate = float(np.abs(low_est - low_er)[hole & have].mean())
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    tex = float(hp[ring].std())  # grille mesh / textured surroundings: the model's erase is the better one there
    bar, slope = _bar_crosses(g, hole, d)
    # only where a straight bar/strip runs INTO the hole from both sides (t05 chrome bar). On a reflective panel with
    # irregular reflections (t01 building silhouettes) row interpolation smeared them into a streaky slab: keep the model
    if plate < 10 or tex > 12 or not bar:
        return erased, dict(horiz=round(horiz, 2), plate=round(plate, 1), tex=round(tex, 1), applied=False, bar=bool(bar))
    # only the rows of the bar (between its matched edges, +3 px): the bar continues, the model keeps the rest
    rows = np.zeros(hole.shape, bool); ysb = bar
    groups = [[ysb[0]]]
    for y in ysb[1:]:
        (groups[-1].append(y) if y - groups[-1][-1] <= 0.35 * d else groups.append([y]))
    for gr in groups:
        rows[max(0, gr[0] - 3):gr[-1] + 4] = True
    hb = hole
    if abs(slope) > 0.01:
        # a tilted bar (r06 rod, slope -0.06): interpolate ALONG it - sheared so the bar runs level, rows, sheared back.
        # Straight rows mixed its highlight with the dark grille 15 px off -> a dim grey rod across the gap
        hh, ww = hole.shape
        Ms = np.float32([[1, 0, 0], [-slope, 1, slope * cx]]); Mi = np.float32([[1, 0, 0], [slope, 1, -slope * cx]])
        her = cv2.warpAffine(er, Ms, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        hho = cv2.warpAffine(hole.astype(np.uint8), Ms, (ww, hh), flags=cv2.INTER_NEAREST)
        # the rod's ends are sampled 14 px past the hole: the erase's feathered edge had already half-faded them
        hho = cv2.dilate(hho, np.ones((1, 29), np.uint8)) > 0
        sI, sok = _interp(her, hho, 1)
        est = cv2.warpAffine(sI, Mi, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        have = cv2.warpAffine(sok.astype(np.uint8), Mi, (ww, hh), flags=cv2.INTER_NEAREST) > 0
        rows = cv2.warpAffine(rows.astype(np.uint8), Mi, (ww, hh), flags=cv2.INTER_NEAREST) > 0
        hb = cv2.warpAffine(hho.astype(np.uint8), Mi, (ww, hh), flags=cv2.INTER_NEAREST) > 0
    m = cv2.GaussianBlur((hb & have & rows).astype(np.float32), (0, 0), 1.5)[..., None]
    core = cv2.GaussianBlur(cv2.erode((hole & have).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
    out = er * (1 - m) + (est + core * (er - cv2.GaussianBlur(er, (0, 0), 2))) * m  # no grain near the rim: it carried the plate's outline
    return np.clip(out, 0, 255).astype(np.uint8), dict(horiz=round(horiz, 2), plate=round(plate, 1), tex=round(tex, 1), applied=True, bar=bar, slope=round(slope, 3))
