# Made by Yovan Vujacic
"""Logo Inject stage 2 ("badge detail pass") - pure numpy/cv2 core, testable outside ComfyUI.

Why: stage 1 (whole-image Qwen-Edit + Logo Inject LoRA at ~1 MP) draws each badge with only ~3-4 latent tokens
across, so lions, lettering and rings come out mangled. Stage 1's POSITION / ANGLE / PERSPECTIVE are usually right
(Yovan, 2026-09-28: "the current one" had the correct perspective); its DETAIL is not. So per badge:
  find the site (diff original vs stage 1) -> crop ~3.5x and enlarge -> [model erase] -> draw the clean brand logo
  with a homography fitted to the stage-1 badge (hub badges: snapped to the cap centre) -> [model chrome pass]
  -> shrink and paste back through the footprint mask.
Lessons baked in (all from 2026-09-28): partial denoise over the old badge copies it; erase + free redraw lets the
model pick its own size/angle; centre hub badges by area centroid, never bbox.
"""
import cv2
import numpy as np

CROP = 1024


def _u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def shift_diff(a, b, r=3, sigma=1.2):
    """Per-pixel change that tolerates a small misalignment: min over +-r px shifts of |a - shift(b)|.
    Stage 1 regenerates the WHOLE image, so every edge moves by a pixel or two; a plain diff lights up the entire
    frame (2026-09-28: 10 % of pixels, one giant blob, zero sites found)."""
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma); b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigma)
    best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.abs(a - np.roll(np.roll(b, dy, 0), dx, 1)).max(2)
            best = d if best is None else np.minimum(best, d)
    return best


def find_sites(original, stage1, max_sites=4, thresh=25, min_px=0.00006, max_px=0.02):
    """original: input image (any res); stage1: stage-1 output at its OWN resolution (~1 MP).
    Compares at stage-1 resolution, returns (sites in ORIGINAL pixel coords sorted by area, change map at original
    res). A site = compact blob of real change where stage 1 ADDED detail (removals are skipped)."""
    H, W = stage1.shape[:2]; Ho, Wo = original.shape[:2]; kx, ky = Wo / W, Ho / H
    o = cv2.resize(original, (W, H), interpolation=cv2.INTER_AREA)
    dm = shift_diff(o, stage1)
    d = (dm > thresh).astype(np.uint8) * 255
    # stage 1 also bends panel lines by a few px: thin changed LINES must go before blobs are joined, or a badge
    # merges with the line next to it into one sprawling blob that fails every shape test (cit, 2026-09-28)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(d)
    sites = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        frac = area / float(H * W)
        if frac < min_px or frac > max_px:
            continue
        if max(w, h) > 3 * min(w, h):  # thin strips = warped panel lines / edge noise, not a badge
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if area / float(w * h) < 0.3:
            continue
        ys_, xs_ = slice(y, y + h), slice(x, x + w)
        e1 = np.abs(cv2.Laplacian(cv2.cvtColor(stage1[ys_, xs_], cv2.COLOR_RGB2GRAY), cv2.CV_32F)).mean()
        e0 = np.abs(cv2.Laplacian(cv2.cvtColor(o[ys_, xs_], cv2.COLOR_RGB2GRAY), cv2.CV_32F)).mean()
        if e1 < 0.7 * e0:  # stage 1 smoothed something away here - a removal, not a badge
            continue
        sites.append(dict(cx=float((x + w / 2) * kx), cy=float((y + h / 2) * ky), w=int(round(w * kx)),
                          h=int(round(h * ky)), area=int(area), bbox=[int(x * kx), int(y * ky), int(w * kx), int(h * ky)]))
    sites.sort(key=lambda s: -s["area"])
    return sites[:max_sites], cv2.resize(dm, (Wo, Ho), interpolation=cv2.INTER_LINEAR)


def sites_from_detections(original, stage1, boxes, scores, sim_delta, min_score=0.3, min_delta=0.1,
                          strong_score=0.8, strong_change=0.2, max_aspect=3.0, max_sites=6):
    """Sites from a text-prompted logo detector (SAM3 'logo') run on stage 1, instead of the change map alone.
    Why: a stage-1 badge that REPLACES a similar mark (Opel bonnet Y -> Opel ring, hub 'OO' squares) barely changes
    the diff, so find_sites missed all 3 Opel badges, while tread markers / splitters that stage 1 redrew slightly
    became false sites (r1, 2026-09-28). The detector also fires on marks that were already there (Y, '5-01'), so a
    box is kept only if stage 1 made it look MORE like the reference logo (sim_delta = sim(stage1 crop, logo) -
    sim(original crop, logo), any image embedding) - or, for tiny badges an embedding can't read, the detector is
    sure AND the box really changed.
    boxes: [x0, y0, x1, y1] in stage-1 px; sim_delta: one float per box. Returns (sites in ORIGINAL px, change map
    at original res, per-box decisions)."""
    H, W = stage1.shape[:2]; Ho, Wo = original.shape[:2]; kx, ky = Wo / W, Ho / H
    o = cv2.resize(original, (W, H), interpolation=cv2.INTER_AREA)
    dm = shift_diff(o, stage1)
    kept, log = [], []
    for k_, (b, s, d) in sorted(enumerate(zip(boxes, scores, sim_delta)), key=lambda z: -z[1][1]):
        x0, y0, x1, y1 = [int(round(v)) for v in b]; w, h = x1 - x0, y1 - y0
        chg = float((dm[y0:y1, x0:x1] > 20).mean()) if w > 0 and h > 0 else 0.0
        why = None
        if s < min_score:
            why = "low score"
        elif max(w, h) > max_aspect * max(1, min(w, h)):
            why = "strip (plate text / wordmark)"
        elif any(_overlap(b, k["_b"]) > 0.5 for k in kept):
            why = "duplicate"
        # strong_change 0.2 (was 0.3): a shield replacing lettering at the same spot scored 0.93 / change 0.285 /
        # sim_delta 0.098 and was skipped, so stage 1's mirrored lion shipped (Peugeot SUV, WS test 2026-10-01)
        elif not (d >= min_delta or (s >= strong_score and chg >= strong_change)):
            why = "not more logo-like than the original"
        log.append(dict(det=k_, box=[x0, y0, w, h], score=round(float(s), 3), sim_delta=round(float(d), 3),
                        change=round(chg, 3), kept=why is None, why=why))
        if why is None:
            kept.append(dict(_b=b, det_index=k_, cx=(x0 + w / 2) * kx, cy=(y0 + h / 2) * ky, w=int(round(w * kx)),
                             h=int(round(h * ky)), area=int(w * h),
                             det_box=[x0 * kx, y0 * ky, w * kx, h * ky], bbox=[int(x0 * kx), int(y0 * ky),
                                                                          int(w * kx), int(h * ky)]))
    for k in kept:
        k.pop("_b")
    return kept[:max_sites], cv2.resize(dm, (Wo, Ho), interpolation=cv2.INTER_LINEAR), log


def stacked_duplicates(sites):
    """Indices of badges that sit directly BELOW another badge on the same face. Stage 1 sometimes places two (Alfa:
    tailgate centre + upper lip, node r2). Yovan's rule: rear badges on the upper rear only, front badges on the hood,
    never the bumper - so the upper one stays and the lower one is erased."""
    dup = set()
    for i, a in enumerate(sites):
        for j, b in enumerate(sites):
            if i == j:
                continue
            w = max(a["w"], b["w"]); h = max(a["h"], b["h"])
            if b["cy"] > a["cy"] + 0.5 * h and abs(b["cx"] - a["cx"]) < 1.5 * w and b["cy"] - a["cy"] < 6 * h:
                dup.add(j)
    return dup


def _overlap(a, b):
    """Intersection over the SMALLER box (a detector often returns a box and a sub-box of the same badge)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1]); sb = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(1e-6, min(sa, sb))


def erase_residual(stage1_crop, erased_crop, footprint, t=22):
    """Share of the stage-1 badge's contrast marks still present after the erase. r1: good erases 0.0-0.35, a failed
    erase that kept the old hub badge 0.49-0.6 (then a new logo lands beside it = double badge)."""
    def marks(a):
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return np.abs(g.astype(np.int16) - cv2.medianBlur(g, 61).astype(np.int16)) > t
    m = footprint > 0.5
    return float((marks(erased_crop) & m).sum()) / max(1, float((marks(stage1_crop) & m).sum()))


def metal_score(crop, stroke_mask):
    """How much a chrome pass looks like metal on the logo's strokes: spread of luminance + gradient energy there.
    A flat white sticker (cit r1) scores low; a bevelled reflective emblem scores high."""
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = stroke_mask > 0.5
    if m.sum() < 50:
        return 0.0
    gr = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float(g[m].std() + 0.25 * gr[m].mean())


def chrome_score(tri, guide, erased, solid, size=384):
    """Pick score for one chrome try (higher = better), and whether it passes the design gate.
    Tuned on 45 hand-labelled node sites (r2-r8, 2026-09-30; old metal_score picked right 29/45, this 43/45):
    - texture = luminance spread inside the badge's light parts (eroded). Flat white / flat grey tries are uniform.
    - open outline logos (Citroen, Opel): blown-white pixels don't count - a white sticker with one chrome edge
      (cit bonnet t1) scored high on spread alone.
    - solid badges (Alfa roundel, Peugeot shield): the dark design must stay dark; heavy chrome that greys the design
      out (alfa hubs) is rejected (mean design luminance > guide's + 10)."""
    g = lambda im: cv2.cvtColor(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gd, er, t = g(guide), g(erased), g(tri)
    stroke = (np.abs(gd - er) > 25).astype(np.uint8)
    hull = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    k3 = np.ones((3, 3), np.uint8)
    light = cv2.erode(((gd > 150) & hull).astype(np.uint8), k3) > 0
    dark = cv2.erode(((gd < 100) & hull).astype(np.uint8), k3) > 0
    if light.sum() < 30:
        return 0.0, True
    std = float(t[light].std())
    if not solid:
        return std * (1.0 - float((t[light] > 235).mean())), True
    ok = dark.sum() < 30 or float(t[dark].mean()) <= float(gd[dark].mean()) + 10
    return std, ok


def crop_site(stage1, original, site, change_map, crop_factor=3.5, size=CROP, min_share=0.1):
    """Square crop around the site, enlarged to size x size. Returns dict with crops, footprint + erase masks
    (float 0..1, crop space), box (x0, y0, side) in full-image pixels, and hub detection."""
    H, W = stage1.shape[:2]
    side = int(round(max(site["w"], site["h"]) * crop_factor))
    side = max(64, int(min_share * min(H, W)), side); side = min(side, H, W)  # small hub sites need context
    x0 = int(round(site["cx"] - side / 2)); y0 = int(round(site["cy"] - side / 2))
    x0 = min(max(0, x0), W - side); y0 = min(max(0, y0), H - side)
    s1 = cv2.resize(stage1[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_CUBIC)
    og = cv2.resize(original[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_CUBIC)
    k = size / side
    det = clip = None
    if "det_box" in site:  # detector box (full-image px): it, not the change map, picks the badge's marks
        bx, by, bw, bh = site["det_box"]
        det = np.zeros((size, size), np.uint8)
        cv2.ellipse(det, (int((bx + bw / 2 - x0) * k), int((by + bh / 2 - y0) * k)),
                    (int(bw * k * 0.6), int(bh * k * 0.6)), 0, 0, 360, 255, -1)
        chg = det.astype(np.float32)
        clip = np.zeros((size, size), np.uint8)  # the badge can't extend past the detector box (alfa crease, r2)
        cv2.rectangle(clip, (int((bx - x0 - 0.05 * bw) * k), int((by - y0 - 0.05 * bh) * k)),
                      (int((bx + 1.05 * bw - x0) * k), int((by + 1.05 * bh - y0) * k)), 255, -1)
    else:
        chg = cv2.resize(change_map[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_LINEAR)
    badge_px = max(site["w"], site["h"]) * size / side
    # light AND dark marks: light-only saw half the Alfa hub roundel / a third of the Opel ring and none of a dark
    # red badge (r2) -> logo drawn small or the site skipped. Light-only stays as the fallback.
    if site.get("det_mask") is not None:
        # The detector's own segmentation IS the stage-1 badge's outline (size, tilt, perspective). Contrast marks
        # can't tell a badge from a panel crease running through it (Alfa upper lip, node r2: logo drawn oversized
        # and front-on), so when the mask exists it defines the footprint and marks only feed the stroke fit.
        dmk = cv2.resize(site["det_mask"][y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_LINEAR) > 127
        foot = marks = None
        if dmk.sum() > 0.002 * size * size:
            ys_, xs_ = np.nonzero(dmk)
            foot = np.zeros((size, size), np.uint8)
            cv2.fillConvexPoly(foot, cv2.convexHull(np.column_stack([xs_, ys_]).astype(np.int32)), 255)
            g_ = cv2.GaussianBlur(cv2.cvtColor(s1, cv2.COLOR_RGB2GRAY), (0, 0), 1)
            v_ = g_.astype(np.int16); bg_ = cv2.medianBlur(g_, 61).astype(np.int16)
            marks = ((np.abs(v_ - bg_) > 22) & (cv2.erode(foot, np.ones((7, 7), np.uint8)) > 0)).astype(np.uint8) * 255
            if marks.sum() < 255 * 200:
                marks = cv2.Canny(foot, 50, 150)  # plain badge: fit the outline
    else:
        foot, marks = badge_marks(s1, chg, max(badge_px, 0.18 * size), contrast="abs", clip=clip)
        if foot is None:
            foot, marks = badge_marks(s1, chg, max(badge_px, 0.18 * size), clip=clip)
    if foot is None:
        return dict(valid=False, stage1_crop=s1, box=[x0, y0, side])
    ys, xs = np.nonzero(foot)
    fr = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    fcx, fcy = float(xs.mean()), float(ys.mean())
    # erase region: the badge's own footprint grown ~15 %, feathered. NOT the detector box - erasing the whole box
    # let the model redraw the plate under the badge (Opel square hub plates came back round, r2). The full
    # light+dark footprint already covers a badge's rim, which r1's light-only footprint left as a ghost ring.
    g = int(max(12, fr * 0.15)) | 1
    er = cv2.dilate(foot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    er = cv2.GaussianBlur(er, (0, 0), g / 3).astype(np.float32) / 255
    return dict(valid=True, stage1_crop=s1, original_crop=og, footprint=foot.astype(np.float32) / 255,
                marks=marks.astype(np.float32) / 255, erase_mask=er, box=[x0, y0, side], foot_r=fr, foot_c=[fcx, fcy])


def plate_under_badge(original_crop, footprint, min_cover=0.7, dev=30):
    """True when the ORIGINAL car already had a filled shape where stage 1 put the badge (a blank badge disc /
    plate), i.e. most of the footprint already differed from the panel around it. Stage 1 then draws the logo ON that
    plate and the detector's footprint is the whole plate: fitting the logo to it gave an oversized logo (DS on a blank
    grille disc, WS test 2026-10-01), and keeping the plate looked unnatural to Yovan ("strange disc behind the DS").
    A plain panel or an old open badge (Opel 'Y') covers far less of the footprint."""
    ft = footprint > 0.5
    ys, xs = np.nonzero(ft); fr = 0.5 * max(np.ptp(xs), np.ptp(ys))
    def ell(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    f8 = ft.astype(np.uint8)
    inner = cv2.erode(f8, ell(max(1, int(0.15 * fr)))) > 0
    ring = (cv2.dilate(f8, ell(max(2, int(0.3 * fr)))) > 0) & ~(cv2.dilate(f8, ell(max(1, int(0.1 * fr)))) > 0)
    if inner.sum() < 50 or ring.sum() < 50:
        return False
    panel = np.median(original_crop[ring].reshape(-1, 3), axis=0)
    d = np.abs(original_crop.astype(np.float32) - panel).max(axis=2)
    return float((d[inner] > dev).mean()) >= min_cover


def old_badge_area(original_crop, stage1_crop, footprint, foot_r, thresh=25):
    """What stage 1 changed AROUND the new badge - the remains of the car's own badge (Alfa red hub square, the orange
    tip of the original tailgate mark, the Peugeot lettering beside the shield). Since the output is the original with
    only badge areas pasted in, anything here that isn't pasted comes back (WS test 5, 2026-10-01). Solid change blobs
    touching the badge only: thin streaks (stage 1 blurring a panel line) are opened away, far-off changes ignored.
    Returns a float 0..1 mask in crop space."""
    b = (shift_diff(original_crop, stage1_crop) > thresh).astype(np.uint8) * 255
    k = max(3, int(0.12 * foot_r))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    near = cv2.dilate((footprint > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.6 * foot_r) + 1,) * 2))
    n, lab = cv2.connectedComponents(b)
    keep = np.zeros_like(b)
    for j in range(1, n):
        comp = lab == j
        if (comp & (near > 0)).any():
            keep[comp] = 255
    # the car's own small marks right beside the new badge (lettering, pieces of the old badge) count as old badge too -
    # stage 1 often leaves them half there, and the stitched output then shows them peeking out (Peugeot 'A..B', test 6).
    # Only compact marks: grille mesh, panel lines and creases are big or long and stay.
    g0 = cv2.GaussianBlur(cv2.cvtColor(original_crop, cv2.COLOR_RGB2GRAY), (0, 0), 1)
    mk = (np.abs(g0.astype(np.int16) - cv2.medianBlur(g0, 61).astype(np.int16)) > 22).astype(np.uint8) * 255
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fa = max(1, int((footprint > 0.5).sum()))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(mk)
    for j in range(1, n2):
        x, y, w, h, area = st2[j]
        if area > 1.2 * fa or max(w, h) > 2.2 * foot_r or max(w, h) > 4 * max(1, min(w, h)):
            continue
        if x == 0 or y == 0 or x + w >= mk.shape[1] or y + h >= mk.shape[0]:
            continue
        comp = lab2 == j
        if (comp & (near > 0)).any():
            keep[comp] = 255
    # ...and only close to it: a blurred panel edge that touches the badge ran the whole width of the Alfa lip (seed 7)
    keep[cv2.dilate((footprint > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.8 * foot_r) + 1,) * 2)) == 0] = 0
    g = max(5, int(0.08 * foot_r))
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2))
    return cv2.GaussianBlur(keep, (0, 0), g / 2).astype(np.float32) / 255


def shrink_footprint(footprint, share=0.8):
    """The footprint scaled to `share` about its centre (float 0..1). Plate sites: the logo is fitted to 80 % of the
    erased plate - the plate's full size let the DS run off the grille bar onto the bumper (WS test 2026-10-01)."""
    ft = (footprint > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(ft); cx, cy = float(xs.mean()), float(ys.mean())
    T = np.array([[share, 0, cx * (1 - share)], [0, share, cy * (1 - share)]], np.float32)
    return cv2.warpAffine(ft * 255, T, (ft.shape[1], ft.shape[0]), flags=cv2.INTER_NEAREST).astype(np.float32) / 255


def badge_marks(s1, chg_map, badge_px, min_marks=500, contrast="light", clip=None):
    """Segment the stage-1 badge by its own markings: pixels that stand out from the local background, grouped
    into clusters; the badge = the cluster that overlaps the change map most (the change map only PICKS the cluster -
    a badge replacing a similar mark changes only partly, cit wheel 2026-09-28). Returns (hull footprint, marks)
    uint8, or (None, None) when there is no badge-like cluster (a false site - never draw a logo there)."""
    size = s1.shape[0]; c = size / 2
    gray = cv2.cvtColor(s1, cv2.COLOR_RGB2GRAY)
    v = gray.astype(np.int16); bg = cv2.medianBlur(gray, 61).astype(np.int16)
    light = (v - bg) > 22 if contrast == "light" else np.abs(v - bg) > 22
    chg = (chg_map > 20)
    yy, xx = np.mgrid[0:size, 0:size]
    near = np.hypot(xx - c, yy - c) < 0.4 * size
    if clip is not None:
        near &= clip > 0
    marks = (light & near).astype(np.uint8) * 255
    marks = cv2.morphologyEx(marks, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cl = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(cl)
    best, bs = 0, 0.0
    for j in range(1, n):
        ov = int((chg & (lab == j)).sum())
        s = ov / (1 + np.hypot(cen[j][0] - c, cen[j][1] - c) / badge_px)
        if ov >= 30 and s > bs:
            best, bs = j, s
    if not best:
        return None, None
    keep = ((lab == best) & (marks > 0)).astype(np.uint8) * 255
    ys, xs = np.nonzero(keep)
    if len(xs) < min_marks or max(xs.max() - xs.min(), ys.max() - ys.min()) < 0.12 * size:
        return None, None  # too small to be a badge at this zoom (crop >= 3.5x the badge)
    foot = np.zeros_like(keep)
    cv2.fillConvexPoly(foot, cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32)), 255)
    return foot, keep


def detect_cap(img, centre, badge_r):
    """A wheel centre cap = a circle around the badge, radius 1.05-2.6x the badge radius, centred within 0.35 r.
    Returns dict(cx, cy, r) or None."""
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    for p2 in (60, 45, 35):
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, 40, param1=60, param2=p2,
                             minRadius=int(badge_r * 1.02), maxRadius=int(badge_r * 1.6))
        if c is None:
            continue
        c = [k for k in c[0] if np.hypot(k[0] - centre[0], k[1] - centre[1]) < 0.15 * k[2]]
        if c:
            k = min(c, key=lambda k: np.hypot(k[0] - centre[0], k[1] - centre[1]))
            return dict(cx=float(k[0]), cy=float(k[1]), r=float(k[2]))
    return None


def _logo_parts(logo):
    """logo: HxWx3 uint8, light mark on dark ground (the Logo Inject reference). Returns stroke alpha (float),
    filled hull mask (uint8) and hull contour, all at the logo's own resolution."""
    g = cv2.cvtColor(logo, cv2.COLOR_RGB2GRAY)
    if np.median(g) > 128:  # dark mark on light ground -> invert
        g = 255 - g
    alpha = g.astype(np.float32) / 255
    ys, xs = np.nonzero(g > 40)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    hm = np.zeros_like(g); cv2.fillConvexPoly(hm, hull, 255)
    return alpha, hm, hull


def _tri(w, h, a):
    t = np.deg2rad(a); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    S = R @ np.diag([(w / 2) ** 2, (h / 2) ** 2]) @ R.T
    p = np.sqrt(S[0, 0]); q = S[0, 1] / p
    return np.array([[p, 0], [q, np.sqrt(max(S[1, 1] - q * q, 1e-6))]])


def _triu(w, h, a):
    """Upper-triangular factor U of the ellipse (U U^T = its shape matrix): maps the unit circle onto the ellipse with the
    x axis kept level and the y axis onto the conjugate diameter."""
    t = np.deg2rad(a); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    S = R @ np.diag([(w / 2) ** 2, (h / 2) ** 2]) @ R.T
    c = np.sqrt(S[1, 1]); b = S[0, 1] / c
    return np.array([[np.sqrt(max(S[0, 0] - b * b, 1e-6)), b], [0, c]])


def _axes_deg(Hm, x, y):
    """(vertical-axis lean, horizontal-axis angle) in degrees of the logo's own axes at (x, y) after mapping."""
    q0 = Hm @ [x, y, 1.0]; qv = Hm @ [x, y + 10, 1.0]; qh = Hm @ [x + 10, y, 1.0]
    dv = qv[:2] / qv[2] - q0[:2] / q0[2]; dh = qh[:2] / qh[2] - q0[:2] / q0[2]
    return float(np.degrees(np.arctan2(-dv[0], dv[1]))), float(np.degrees(np.arctan2(dh[1], dh[0])))


def fit_homography(logo_alpha, logo_hull_mask, footprint, marks, max_tilt=35.0):
    """Homography logo -> crop laying the clean logo's strokes onto the stage-1 badge's strokes.
    Init = upright affine (verticals stay vertical) between the outline ellipses; refined by ECC on strokes;
    kept only if it beats the init on stroke overlap. Returns (H, overlap score).
    Solved at half resolution (3 inits x 3 ECC scales at 1024 px took minutes), then scaled back up."""
    full = footprint.shape[0]; size = full // 2
    k = size / logo_hull_mask.shape[0]
    hull = cv2.resize(logo_hull_mask, (size, size), interpolation=cv2.INTER_AREA)
    src = (cv2.resize(logo_alpha, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    tgt = ((cv2.resize(marks.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.25) * 255).astype(np.uint8)
    ftp = ((cv2.resize(footprint.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.5) * 255).astype(np.uint8)

    def ell(m):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cv2.fitEllipse(max(cs, key=cv2.contourArea))
    (sx, sy), (sw, sh), sa = ell(hull); (tx, ty), (tw, th), ta = ell(ftp)
    # Orientation = a disc seen in perspective, like stage 1 draws it: the badge's horizontal diameter stays level (panel
    # edges are level for a no-roll camera) and its vertical diameter is the CONJUGATE one - through the oval's top-most and
    # bottom-most points. For an oval rising to the right that leans top-right (Yovan's line on stage 1, 2026-09-30).
    # Rejected before: r4 stroke fit over-turned it, r5 kept vertical upright (sheared lettering), r6 levelled the oval,
    # r7 turned it rigidly (vertical leaned top-left = wrong direction).
    A = _triu(tw, th, ta) @ np.linalg.inv(_triu(sw, sh, sa))
    H0 = np.eye(3); H0[:2, :2] = A; H0[:2, 2] = np.array([tx, ty]) - A @ [sx, sy]
    tb = cv2.dilate(tgt, np.ones((5, 5), np.uint8)) > 0

    def score(Hm):
        w = cv2.warpPerspective(src, Hm, (size, size)) > 100
        return (w & tb).sum() / max(1, w.sum())
    def axes(cx, cy, w, h, a):
        r = np.deg2rad(a); u = np.array([np.cos(r), np.sin(r)]) * w / 2; v = np.array([-np.sin(r), np.cos(r)]) * h / 2
        return np.float32([[cx, cy] + u, [cx, cy] + v, [cx, cy] - u])
    inits = [H0]
    for flip in (False, True):  # rotated inits: a badge seen at an angle on a sloping panel (Citroen bonnet)
        dst = axes(tx, ty, tw, th, ta) if not flip else axes(tx, ty, th, tw, ta + 90)
        Ai = np.vstack([cv2.getAffineTransform(axes(sx, sy, sw, sh, sa), dst), [0, 0, 1]])
        if np.linalg.det(Ai[:2, :2]) > 0:
            inits.append(Ai)
    best, bs = H0, score(H0)
    def tilt(Hm):  # angle of the logo's vertical axis after mapping, at the logo centre
        p0 = np.array([sx, sy, 1.0]); p1 = np.array([sx, sy + 10, 1.0])
        q0 = Hm @ p0; q1 = Hm @ p1; d = q1[:2] / q1[2] - q0[:2] / q0[2]
        v = np.degrees(np.arctan2(-d[0], d[1]))
        p1 = np.array([sx + 10, sy, 1.0]); q1 = Hm @ p1; d = q1[:2] / q1[2] - q0[:2] / q0[2]
        h = np.degrees(np.arctan2(d[1], d[0]))  # both axes vs the perspective init's own axes
        return max(abs(v - v0), abs(h - h0))
    v0, h0 = _axes_deg(H0, sx, sy)
    for Hi in inits:
        if tilt(Hi) < max_tilt and score(Hi) > bs:
            best, bs = Hi, score(Hi)
        try:
            Winv = np.linalg.inv(Hi).astype(np.float32)
            for sig in (5, 2.5, 1.25):
                a_ = cv2.GaussianBlur(src, (0, 0), sig).astype(np.float32) / 255
                b_ = cv2.GaussianBlur(tgt, (0, 0), sig).astype(np.float32) / 255
                _, Winv = cv2.findTransformECC(b_, a_, Winv, cv2.MOTION_HOMOGRAPHY,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
            Hm = np.linalg.inv(Winv)
            ar = abs(np.linalg.det(Hm[:2, :2]) / np.linalg.det(H0[:2, :2]))
            if np.linalg.det(Hm[:2, :2]) > 0 and 0.6 < ar < 1.6 and tilt(Hm) < max_tilt and score(Hm) > bs:
                best, bs = Hm, score(Hm)
        except cv2.error:
            pass
    up = full / size
    S0, S1 = np.diag([k, k, 1.0]), np.diag([up, up, 1.0])
    return S1 @ best @ S0, float(bs), S1 @ H0 @ S0


def hub_similarity(logo_hull_mask, hull, cap, footprint, max_share=0.80):
    """Hub badge: no shear, upright, stage-1 size (clamped to the cap), hull CENTROID on the cap centre."""
    ys, xs = np.nonzero(footprint > 0.5)
    fr = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    hx, hy, hw, hh = cv2.boundingRect(hull)
    lr = 0.5 * max(hw, hh)
    s = min(fr / lr, max_share * cap["r"] / lr)
    M = cv2.moments(hull); mx, my = M["m10"] / M["m00"], M["m01"] / M["m00"]
    Hm = np.array([[s, 0, cap["cx"] - s * mx], [0, s, cap["cy"] - s * my], [0, 0, 1]], np.float64)
    return Hm


def badge_guide(erased_crop, footprint, marks, logo, body="open", hub_mode="auto", mark=235, dark=40, light=215,
                max_tilt=35.0):
    """Draw the clean logo, warped to the stage-1 badge, as a flat guide on the erased crop.
    body: 'open' = strokes only, paint shows through (Citroen, Opel); 'solid' = dark filled badge with light
    design (Peugeot shield, Alfa roundel). Returns guide (uint8 HxWx3), chrome mask (float), transform."""
    size = erased_crop.shape[0]
    alpha, hm, hull = _logo_parts(logo)
    ys, xs = np.nonzero(footprint > 0.5)
    fr = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    hub = None if hub_mode == "off" else detect_cap(erased_crop, (float(xs.mean()), float(ys.mean())), fr)
    fit, fallback, check = None, False, None
    if hub:
        Hm = hub_similarity(hm, hull, hub, footprint)
    else:
        Hm, fit, H0 = fit_homography(alpha, hm, footprint, marks, max_tilt=max_tilt)
        check = fit_check(hm, Hm, footprint)
        if check:  # degenerate fit (edge-on sliver beside the badge, alfa hubs r1) -> plain outline fit
            Hm, fallback = H0, True
            check = fit_check(hm, Hm, footprint)
            if check:
                return None, None, dict(hub=None, stroke_fit=fit, rejected=check)
    Hm = clamp_to_footprint(hm, Hm, footprint, grow_below=0 if hub else 0.87)  # hub size is cap-limited
    a = cv2.warpPerspective(alpha, Hm, (size, size), flags=cv2.INTER_AREA)
    out = erased_crop.astype(np.float32)
    if body == "solid":
        bm = cv2.warpPerspective(hm.astype(np.float32) / 255, Hm, (size, size), flags=cv2.INTER_AREA)
        out = out * (1 - bm[..., None]) + dark * bm[..., None]
        out = out * (1 - a[..., None]) + light * a[..., None]
        cover = np.maximum(a, bm)
    else:
        out = out * (1 - a[..., None]) + mark * a[..., None]
        cover = a
    m = cv2.dilate((cover > 0.1).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    m = cv2.GaussianBlur(m, (0, 0), 6).astype(np.float32) / 255
    return _u8(out), m, dict(H=Hm.tolist(), hub=hub, stroke_fit=fit, outline_fallback=fallback, stroke=a,
                             tilt=round(vertical_tilt(Hm, hull), 1), htilt=round(horizontal_tilt(Hm, hull), 1))


def vertical_tilt(Hm, hull):
    """Signed degrees the logo's own vertical axis leans after mapping (0 = upright, + = clockwise)."""
    M = cv2.moments(hull); cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    q0 = Hm @ [cx, cy, 1.0]; q1 = Hm @ [cx, cy + 10, 1.0]; d = q1[:2] / q1[2] - q0[:2] / q0[2]
    return float(np.degrees(np.arctan2(-d[0], d[1])))


def horizontal_tilt(Hm, hull):
    """Signed degrees the logo's own horizontal axis leans after mapping (0 = level, + = right side lower)."""
    M = cv2.moments(hull); cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    q0 = Hm @ [cx, cy, 1.0]; q1 = Hm @ [cx + 10, cy, 1.0]; d = q1[:2] / q1[2] - q0[:2] / q0[2]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _mapped_hull(hm, Hm, size):
    return cv2.warpPerspective(hm, Hm, (size, size), flags=cv2.INTER_NEAREST) > 127


def fit_check(hm, Hm, footprint):
    """None if the fitted logo sits like the stage-1 badge (similar area, same centre, not squashed flatter than
    the badge itself), else the reason."""
    size = footprint.shape[0]
    mh = _mapped_hull(hm, Hm, size); ft = footprint > 0.5
    if mh.sum() < 50:
        return "logo mapped to nothing"
    ys, xs = np.nonzero(ft); fr = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())
    area = mh.sum() / max(1, ft.sum())
    my, mx = np.nonzero(mh)
    off = np.hypot(mx.mean() - xs.mean(), my.mean() - ys.mean()) / fr
    def flat(ys_, xs_):
        (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([xs_, ys_]).astype(np.float32))
        return min(w, h) / max(1e-6, max(w, h))
    if not 0.45 < area < 1.6:
        return f"area x{area:.2f} of the badge"
    if off > 0.3:
        return f"centre off by {off:.2f} r"
    if flat(my, mx) < 0.6 * flat(ys, xs):
        return "squashed flatter than the badge"
    return None


def clamp_to_footprint(hm, Hm, footprint, slack=1.08, grow_below=0.87):
    """Logo the size of the stage-1 badge: never bigger (cit hub r1 spilled off its cap onto the plate), and never
    clearly smaller (alfa upper tailgate r2 came out ~35 % small) - grown until its tighter side matches."""
    size = footprint.shape[0]
    mh = _mapped_hull(hm, Hm, size); ft = footprint > 0.5
    if mh.sum() < 50:
        return Hm
    my, mx = np.nonzero(mh); fy, fx = np.nonzero(ft)
    r = min(np.ptp(fx) / max(1, np.ptp(mx)), np.ptp(fy) / max(1, np.ptp(my)))
    s = min(1.0, slack * r)
    if 1 / max(r, 1e-6) < grow_below:
        s = 0.97 * r
    if abs(s - 1.0) < 1e-3:
        return Hm
    cx, cy = mx.mean(), my.mean()
    T = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1]])
    return T @ Hm


def align_to(src, ref, valid, scale=0.5):
    """Warp src (HxWx3 float) onto ref by an affine ECC fit on luminance over the `valid` pixels. Returns src unchanged if
    the fit fails or moves anything more than 4 % of the crop (a wrong lock, not a re-render shift)."""
    h, w = ref.shape[:2]; sz = (max(16, int(w * scale)), max(16, int(h * scale)))
    g = lambda a: cv2.GaussianBlur(cv2.resize(cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY), sz, interpolation=cv2.INTER_AREA).astype(np.float32) / 255, (0, 0), 1.5)
    msk = cv2.resize(valid.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(g(ref), g(src), W, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), msk, 5)
    except cv2.error:
        return src
    W[:, 2] /= scale
    corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
    if np.abs(corners @ W.T - corners[:, :2]).max() > 0.04 * max(w, h):
        return src
    return cv2.warpAffine(src, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def restore_original(full, original, box, mask, ring=31):
    """Put the ORIGINAL image's pixels back over a removed badge (a duplicate stage 1 added where the car had plain
    panel). A model erase there drew a round recess where the badge had been (Alfa tailgate, node r3). Stage 1 shifts
    colours slightly, so the patch gets the mean colour offset measured on a ring just outside the mask."""
    x0, y0, side = box
    m = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)
    cur = full[y0:y0 + side, x0:x0 + side].astype(np.float32); org = original[y0:y0 + side, x0:x0 + side].astype(np.float32)
    inner = (m > 0.05).astype(np.uint8)
    # stage 1 redraws the whole car slightly shifted: pasted unaligned, the original's crease / light-bar / bumper edges
    # land beside stage 1's and smear (alfa tailgate r9, "distortion"). Align the original crop to stage 1 first (affine
    # ECC on luminance, the badge itself masked out).
    org = align_to(org, cur, cv2.dilate(inner, np.ones((9, 9), np.uint8)) == 0)
    band = (cv2.dilate(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring))) > 0) & (inner == 0)
    # offset = a PLANE fitted on the ring, not one mean colour: stage 1 re-shades the panel as a gradient, and a flat
    # offset left a round ghost + arc on the alfa tailgate (node r6)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32) / side
    if band.sum() > 50:
        X = np.column_stack([np.ones(band.sum()), xx[band], yy[band]])
        coef, *_ = np.linalg.lstsq(X, cur[band] - org[band], rcond=None)
        off = coef[0] + xx[..., None] * coef[1] + yy[..., None] * coef[2]
    else:
        off = np.zeros((side, side, 3), np.float32)
    out = full.astype(np.float32).copy()
    out[y0:y0 + side, x0:x0 + side] = cur * (1 - m[..., None]) + (org + off) * m[..., None]
    return _u8(out)


def tone_match(crop, ref, mask, clamp=30.0):
    """Shift the crop's tone to the reference (the original) with a per-channel plane fitted on a ring at the mask's
    edge, so a re-rendered area can't come out darker or lighter than the panel around it (Alfa seed-7 lip, test 6)."""
    ring = (mask > 0.05) & (mask < 0.6)
    if ring.sum() < 200:
        return crop
    ys, xs = np.nonzero(ring)
    if len(xs) > 20000:
        k = np.random.default_rng(0).choice(len(xs), 20000, replace=False); ys, xs = ys[k], xs[k]
    A = np.column_stack([np.ones(len(xs)), xs / mask.shape[1], ys / mask.shape[0]])
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    out = crop.astype(np.float32).copy()
    for ch in range(3):
        d = ref[ys, xs, ch].astype(np.float32) - crop[ys, xs, ch].astype(np.float32)
        coef = np.linalg.lstsq(A, d, rcond=None)[0]
        out[..., ch] += np.clip(coef[0] + coef[1] * xx / mask.shape[1] + coef[2] * yy / mask.shape[0], -clamp, clamp)
    return out


def paste_back(full, crop, box, mask):
    """Shrink the finished crop back into the full image through the (erase) mask, tone-matched to what it covers."""
    x0, y0, side = box
    out = full.astype(np.float32).copy()
    c = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)[..., None]
    c = tone_match(c, out[y0:y0 + side, x0:x0 + side], m[..., 0])
    out[y0:y0 + side, x0:x0 + side] = out[y0:y0 + side, x0:x0 + side] * (1 - m) + c * m
    return _u8(out)
