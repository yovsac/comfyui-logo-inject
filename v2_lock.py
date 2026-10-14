# Made by Yovan Vujacic
"""Logo Inject v2 locks (mechanical, after the finish LoRA) - ported from the offline harness
agent-skills/scripts/.runs/li_v2/design_lock.py (2026-10-07).
- lock_masks: push each pixel's brightness toward the EXACT logo (marks -> the badge's metal tone, field -> its own
  tone), keeping the model's shading as a residual; align=True first moves the marks onto the model's rendering
  (it can draw the design a few px off -> ghost copy); field_k damps the field correction (keeps enamel gloss);
  floor = minimum metal/field contrast (open logo rendered in the panel's tone).
- open_gaps: open logos stay open (panel back between the strokes).
- chroma_lock: mono reference -> no invented colours."""
import cv2, numpy as np


def masks_from_paste(paste, erased, solid):
    p, e = paste.astype(np.float32), erased.astype(np.float32)
    d = np.abs(p - e).max(2)
    pm = p.mean(2)
    changed = d > 6
    if solid:
        body = changed.astype(np.float32)
        marks = np.clip((pm - 35) / (215 - 35), 0, 1) * body
    else:
        body = None
        em = e.mean(2)
        marks = np.where(changed, np.clip((pm - em) / np.maximum(215 - em, 1), 0, 1), 0).astype(np.float32)
    return marks, body


def lock(out, paste, erased, solid, strength=0.85):
    marks, body = masks_from_paste(paste, erased, solid)
    return lock_masks(out, marks, body, strength)


def align_marks(Y, marks, body, max_shift=10):
    """The finish LoRA can render the design a few px off the paste (t03 hub: PEUGEOT drawn ~7 px lower) - locking to
    the unshifted marks then paints a ghost copy. Find the shift (ECC, translation, on gradients inside the badge) and
    move the marks/body onto the model's own rendering first."""
    reg = (body if body is not None else marks) > 0.3
    if reg.sum() < 50:
        return marks, body, (0.0, 0.0)
    ys, xs = np.nonzero(reg); pad = 12
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    g = lambda im: (lambda b: np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)))(cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.2))
    a, b = g(Y[y0:y1, x0:x1]), g(marks[y0:y1, x0:x1] * 255)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(a / (a.max() + 1e-6), b / (b.max() + 1e-6), W, cv2.MOTION_TRANSLATION,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error:
        return marks, body, (0.0, 0.0)
    dx, dy = float(W[0, 2]), float(W[1, 2])
    if np.hypot(dx, dy) > max_shift or np.hypot(dx, dy) < 0.5:
        return marks, body, (round(dx, 1), round(dy, 1))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])  # ECC warps the input onto the template with inverse mapping
    mv = lambda im: cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR)
    return mv(marks), (mv(body) if body is not None else None), (round(dx, 1), round(dy, 1))


def lock_masks(out, marks, body, strength=0.85, floor=0, align=False, field_k=1.0):
    solid = body is not None
    out = out.astype(np.float32)
    Y = out @ np.array([0.299, 0.587, 0.114], np.float32)
    shift = None
    if align:
        marks, body, shift = align_marks(Y, marks, body)
    region = body if solid else (cv2.dilate((marks > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.float32)
    inner = cv2.erode((region > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0  # keep the rim/bevel
    mk, fd = inner & (marks > 0.7), inner & (marks < 0.1)
    if mk.sum() < 20 or fd.sum() < 20:
        return out.astype(np.uint8), dict(skipped=True)
    Tm, Tf = np.percentile(Y[mk], 75), np.percentile(Y[fd], 30)
    if abs(Tm - Tf) < 25 and not floor:  # no clear metal/field contrast to restore
        return out.astype(np.uint8), dict(skipped=True, Tm=float(Tm), Tf=float(Tf))
    if floor and abs(Tm - Tf) < floor:  # open logo the finish rendered nearly the panel's tone (dark Renault hub diamond)
        # keep the model's own polarity (VELTRA: light chrome letters on a light-pink panel must not be forced dark)
        Tm = Tf + floor if (Tm > Tf or (Tm == Tf and Tf < 128)) else Tf - floor
    frac = np.clip((Y - Tf) / (Tm - Tf), 0, 1)
    want = cv2.GaussianBlur(marks, (0, 0), 0.6)
    dY = (want - frac) * (Tm - Tf)
    if field_k < 1:  # keep most of the model's gloss on the field: only damp it, marks still get their full lift
        dY = np.where((want < 0.5) & (dY < 0), dY * field_k, dY)
    w = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 1.5) * strength
    res = out + (dY * w)[..., None]
    wrong = float((np.abs(want - frac)[inner] > 0.5).mean())
    return np.clip(res, 0, 255).astype(np.uint8), dict(Tm=float(Tm), Tf=float(Tf), wrong_frac=round(wrong, 3), shift=shift)


def open_gaps(out, marks, paste):
    """Open logos stay open (2026-10-07): the finish LoRA fills the band between strokes with dark enamel (Renault hub
    diamond came out a filled dark diamond). Away from the strokes, put the erased panel back (paste == erased there);
    keep the model's pixels on the strokes and a thin bevel/contact-shadow margin around them."""
    mk = marks > 0.5
    if mk.sum() < 20:
        return out, dict(gaps="skipped")
    sw = 2 * float(np.percentile(cv2.distanceTransform(mk.astype(np.uint8), cv2.DIST_L2, 5)[mk], 90))
    m = float(np.clip(0.35 * sw, 1.5, 5))
    dist = cv2.distanceTransform((marks <= 0.05).astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip((dist - m) / 2, 0, 1)[..., None]
    res = out.astype(np.float32) * (1 - w) + paste.astype(np.float32) * w
    return np.clip(res, 0, 255).astype(np.uint8), dict(stroke_px=round(sw, 1), margin=round(m, 1))


def chroma_lock(out, region):
    """Mono reference -> no invented colours (Alfa came out with a red cross, the reference is black/white). Inside the
    badge, pull any colour the surroundings can't explain to the nearest tint between neutral and the scene's."""
    reg = region > 0.5
    if reg.sum() < 20:
        return out, dict(chroma="skipped")
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = (cv2.dilate(reg.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~(cv2.dilate(reg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    ab = lab[..., 1:3] - 128
    e = np.median(ab[ring], 0)
    envd = float(np.percentile(np.hypot(*(ab[ring] - e).T), 90))
    # allowed tints: anywhere between neutral metal and the scene's own tint (chrome mirrors it partly). VELTRA on a red
    # car is grey-to-red chrome; a red cross on a peach car is neither -> pulled to the nearest allowed tint
    e2 = 0.5 * e  # metal mirrors its scene's tint, never at full strength
    t = np.clip((ab @ e2) / max(float(e2 @ e2), 1e-6), 0, 1)[..., None]
    near = t * e2
    d = np.linalg.norm(ab - near, axis=2)
    soft = cv2.GaussianBlur(reg.astype(np.float32), (0, 0), 1.5)
    w = np.clip((d - envd - 4) / 12, 0, 1) * soft
    lab[..., 1:3] += (near - ab) * w[..., None]
    res = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return res, dict(env_chroma=round(envd, 1), recoloured=round(float((w > 0.5)[reg].mean()), 3))
