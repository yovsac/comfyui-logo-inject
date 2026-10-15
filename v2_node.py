# Made by Yovan Vujacic
"""Logo Inject v2 - ONE node, every painted spot (2026-10-07). Same pipeline as the offline harness that passed the
36-spot test set (agent-skills/scripts/.runs/li_v2/test_finish.py + lock_run.py):

  per painted spot:  geometry (v2_geom: old emblem / wheel outline / depth plane -> homography)
                  -> erase   (Qwen-Edit + Logo_Inject_Erase_v1): body spots = plain edit of the spot inside twice the
                              context, composited back through the grown erase mask; hubs = masked edit of the spot's own
                              crop (the wide edit left square hub plates behind)
                  -> paste   the exact logo, flat, in the measured pose (marks 215, solid body 35), drawn in crop space
                  -> finish  (Qwen-Edit + Logo_Inject_Finish_v1, masked): flat paste -> real badge; light-field
                              solid logos (white field = most of the badge: Alfa, Lamborghini, Ferrari refs) get
                              LIGHT_FIELD_CLAUSE added to the caption
                  -> polarity check (solid badges): an inverted finish (dark field turned to metal) is redone with
                              other seeds; light-field badges also retry seeds while the design lock disagrees with
                              the finish (ghost copy), keeping the closest
                  -> locks   (v2_lock): open logos stay open, no invented colours, design snapped back onto the logo
                  -> stitch  only the changed pixels, feathered, into the original (everything else stays untouched)

All erases run first, then all finishes, so each LoRA'd model is patched once per run.
Inputs from stock nodes: Depth Anything 3 (normalization Raw) depth + intrinsics, native SAM3_Detect masks for the old
emblems ('emblem:50' / 'badge:50' / 'logo:50') and the wheels ('wheel:20'), all on the ~1 MP image."""
import json

import cv2
import numpy as np
import torch

import comfy.samplers
import nodes

from .detail_pass import _n, _t, encode
from . import perspective as PP
from . import v2_geom as GM
from . import v2_lock as DL
from . import v2_erasefix as EF

SIDE = 1024
ERASE_CAP = ("b4dg3_3r4s3, remove every car emblem, badge, lettering and badge plate and continue the real surface "
             "underneath: the same paint, reflections, trim bar, panel line, grille mesh or wheel centre cap, straight and "
             "unbroken; keep everything else in the photo unchanged.")
FINISH_CAP = ("b4dg3_f1n1sh, turn the flat pasted logo shape into a real manufactured car badge mounted on the panel: polished "
              "metal and glossy enamel, bevelled edge, real thickness, reflections and shading matching the scene; keep the "
              "exact design, outline, size and position, and keep everything else in the photo unchanged.")
RETRY_SEEDS = (11, 23, 41)
# Light-field solid logos at full denoise: the finish draws its own famous badge a few px off ours and the design lock
# then leaves a ghost copy (Ferrari horse, LAMBORGHINI letters; Yovan, round 8). The lock's disagreement (wrong_frac)
# ranks the seeds right (Ferrari 7 .37 ghost, 11 .25, 23/41 .07 clean): keep the lowest, stop once <= GHOST_OK.
GHOST_OK = 0.12
GHOST_GAIN = 0.8  # a retry seed replaces the current finish only below 80% of its score: r07 Alfa cap switched for
# .258 -> .248 and its white field came out duller grey (Yovan, round 9: 'clear gain only')
# Dark-field solid badges: the finish can paint past the badge - a light glow, a notch, a wavy lobe just outside the rim
# (n06 Mercedes spare cover, round 13: the r11 sight-line fix moved the crop 1 px, seed 11 re-rolled into one). Score =
# share of a 5-11 px band outside the body (moved onto the finish's own rendering) that came out >25 levels lighter
# than the erased panel. Above HALO_MAX the remaining seeds are tried; one replaces the finish only when it isn't
# inverted and scores below HALO_GAIN x the current. Synthetic 1024 crops: glow .52, rim drawn 3 px wider all round .16,
# rim inside the body / badge drawn 5 px off 0.
HALO_MAX = 0.15
HALO_GAIN = 0.6
# Light-field solid logos: with the plain caption the finish LoRA redraws Alfa / Lamborghini as the real black-enamel
# badges on every seed (its training set has 24 light-paste -> dark-badge pairs). Denoise 0.6 kept the colours but left
# the flat paste ("looks like a cutout", Yovan 2026-10-10); this clause at 1.0 keeps the colours AND builds the material
# (test_finish_cap100: Alfa cap + Lamborghini pass).
LIGHT_FIELD_CLAUSE = (" Keep the logo's own colours: the light field stays bright white enamel, the dark lettering and "
                      "emblem stay dark, exactly as in the pasted shape.")


def _light_field(body, marks):
    """The reference's light marks cover most of the badge body (same test as _polarity's light-field branch)."""
    bb = body > 0.5
    return bool(bb.sum() >= 50 and ((marks > 0.5) & bb).sum() > 0.5 * bb.sum())


def _painted(spots, W, H):
    m = spots[0, ..., 0].cpu().numpy() if spots.ndim == 4 else spots[0].cpu().numpy()
    if m.mean() > 0.98 or m.max() < 0.5:
        return None
    return cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0


def _masks(*ms, W, H):
    out = []
    for m in ms:
        if m is None:
            continue
        a = m.cpu().numpy()
        a = a[None] if a.ndim == 2 else a
        for x in a:
            if x.max() >= 0.5:
                out.append(cv2.resize((x >= 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0)
    return out


def _decode(vae, samples):
    img = vae.decode(samples)
    if img.ndim == 5:
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    return _n(img)


def _polarity(fin, body, marks):
    """Dark-field (solid) badges: score of 'the reference's dark field stayed dark and its light marks stayed light',
    higher = better, and whether it's inverted. Field FAR from any mark, away from the rim (an inverted finish keeps
    chrome edges on the strokes). Same as .runs/li_v2/finish_retry.polarity v2."""
    y = cv2.cvtColor(fin, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bb = (body > 0.5).astype(np.uint8)
    if bb.sum() < 50:
        return None, False
    ys, xs = np.nonzero(bb); ext = max(np.ptp(xs), np.ptp(ys))
    inner = cv2.distanceTransform(bb, cv2.DIST_L2, 5) > ext / 12
    far = inner & (cv2.distanceTransform((marks < 0.3).astype(np.uint8), cv2.DIST_L2, 5) > ext / 25)
    core = inner & (cv2.erode((marks > 0.6).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    if ((marks > 0.5) & (body > 0.5)).sum() > 0.5 * (body > 0.5).sum():
        # light-field logo (Alfa / Ferrari / Lamborghini refs: the white field is most of the badge, the design is black):
        # the white field must come out clearly lighter than the design. Round 3 r07 Alfa: field rendered as dark metal,
        # 15 levels above the design, chrome edges on the strokes -> read as inverted; good ones sit 35-73 above
        # (2026-10-09). The dark-field rule below passed it (56).
        fld = inner & (cv2.erode((marks < 0.3).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if fld.sum() < 30 or core.sum() < 30:
            return None, False
        fm = float(np.median(y[core])); cf = fm - float(np.median(y[fld]))
        # ...and the white field must come out as LIGHT metal: r03 Lamborghini rendered it as black enamel (52) with a
        # darker bull (13) - 'lighter than the design' held, the badge read black, and the lock then half-forced the
        # white design over it -> doubled edges (2026-10-10). Light fields that look right sit above 60.
        return round(min(cf, 2 * (fm - 60) + 25), 1), bool(cf < 25 or fm < 60)
    if far.sum() < 30:
        return None, False
    ff = float(np.median(y[far]))
    cf = float(np.median(y[core]) - ff) if core.sum() > 500 else 0.0
    return round(cf - 2 * max(0.0, ff - 80), 1), bool(ff > 110 or cf < -15)


def _wall_cut(crop, em_c, covc):
    """A blank painted spot (no old emblem found) that runs over a long panel edge: the part beyond the edge from the new
    badge is not erased, and the erase's feathered rim stops there too. The painted circle under the Citroen oval overlapped
    the plate recess and the erase drew a dark smudge onto the plate's top edge (Yovan's r24 review, 2026-10-14).
    -> (cut: erase-mask pixels to drop, far: pixels the feathered composite must not reach) or (None, None)"""
    cov1 = covc > 0.05; emb = em_c > 8
    g_ = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    if not cov1.any():
        return None, None
    e_ = cv2.Canny(g_, 60, 160); wall = np.zeros(emb.shape, np.uint8)
    # gap 10: the plate's top edge steps a row every ~200 px and at gap 4 Hough broke it into pieces under the min length
    L_ = cv2.HoughLinesP(e_, 1, np.pi / 360, threshold=40, minLineLength=int(0.12 * SIDE), maxLineGap=10)
    # a panel edge runs past the erase on both ends; a straight stroke of an old emblem SAM missed (the Peugeot shield's
    # top edge on the E-3008, the DS 'D' upright) ends inside it and stayed un-erased as a bright line (r24c review)
    # 8 px, not a share of the erase: the E-3008 plate edge fades at the plate's rounded corner just past the erase rim
    out_ = cv2.dilate(emb.astype(np.uint8), np.ones((17, 17), np.uint8)) == 0
    # Hough splits one edge into collinear pieces (rc7's plate edge: left half + right half, each ending inside the
    # erase), so chain pieces on the same line (angle < 1 deg, offset < 4 px, gap < 30 px) before the both-ends test
    segs = []
    for x1, y1, x2, y2 in (L_[:, 0].astype(float) if L_ is not None else []):
        th = np.arctan2(y2 - y1, x2 - x1) % np.pi; d_ = np.array([np.cos(th), np.sin(th)])
        rho = -d_[1] * x1 + d_[0] * y1; t1, t2 = sorted((d_ @ (x1, y1), d_ @ (x2, y2)))
        for s in segs:
            dth = abs(s[0] - th); dth = min(dth, np.pi - dth)
            if dth < np.pi / 180 and abs(s[1] - rho) < 4 and t1 <= s[3] + 30 and t2 >= s[2] - 30:
                s[2], s[3] = min(s[2], t1), max(s[3], t2); break
        else:
            segs.append([th, rho, t1, t2])
    for th, rho, t1, t2 in segs:
        d_ = np.array([np.cos(th), np.sin(th)]); n_ = np.array([-d_[1], d_[0]])
        p1, p2 = d_ * t1 + n_ * rho, d_ * t2 + n_ * rho
        if all(out_[int(np.clip(p[1], 0, emb.shape[0] - 1)), int(np.clip(p[0], 0, emb.shape[1] - 1))] for p in (p1, p2)):
            cv2.line(wall, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1, 5)
    wall = wall > 0
    wall &= ~cov1

    def beyond(region):
        n_, lab_ = cv2.connectedComponents((region & ~wall).astype(np.uint8))
        own = np.unique(lab_[cov1 & region]); own = own[own > 0]
        return region & ~np.isin(lab_, own) & ~cov1
    cut = beyond(emb)
    if not cut.any():
        return None, None
    far = beyond(cv2.dilate(emb.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) | (wall & ~cov1 & cv2.dilate(cut.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool))
    return cut, far


def _halo(fin, erased, body, fm):
    """Solid badges: share of the band just outside the body (5-11 px, inside the finish mask) that the finish made
    >25 levels lighter than the erased panel - a glow or lobe past the rim. The body is first moved onto the finish's
    own rendering (same ECC shift as the lock), so a badge drawn a few px off doesn't count as a halo."""
    yf = cv2.cvtColor(fin, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ye = cv2.cvtColor(np.clip(erased, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bs = DL.align_marks(yf, (body > 0.3).astype(np.float32), None)[0]
    d = cv2.distanceTransform((bs < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    band = (d >= 5) & (d <= 11) & (fm > 127)
    if band.sum() < 50:
        return None
    return round(float(((yf - ye)[band] > 25).mean()), 3)


class LogoInjectV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "erase_model": ("MODEL", {"tooltip": "Qwen-Image-Edit 2511 + Logo_Inject_Erase_v1 LoRA (ModelSamplingAuraFlow 3.1 + CFGNorm applied)"}),
            "finish_model": ("MODEL", {"tooltip": "Qwen-Image-Edit 2511 + Logo_Inject_Finish_v1 LoRA (same sampling patches)"}),
            "clip": ("CLIP",), "vae": ("VAE",),
            "image": ("IMAGE", {"tooltip": "the car at full resolution"}),
            "logo": ("IMAGE", {"tooltip": "the logo: light design on a dark background (or the reverse) - the design is the minority tone"}),
            "spots": ("IMAGE", {"tooltip": "painted badge spots, white on black, any size"}),
            "depth": ("IMAGE", {"tooltip": "Depth Anything 3 raw depth of the ~1 MP image (normalization Raw)"}),
            "intrinsics": ("STRING", {"forceInput": True, "tooltip": "Depth Anything 3 intrinsics_json"}),
            # a STRING, not a combo: Workflow Studio feeds it from a PrimitiveString select (a combo can't take a link
            # without the ComfyLiterals node, which the v1 bundle needed)
            "badge_body": ("STRING", {"default": "open", "tooltip": "open = chrome strokes on the paint (Renault, Citroen, "
                           "Opel, DS); solid = a badge with a dark enamel field (Peugeot shield, Alfa roundel, Maserati oval)"}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {
            "emblems": ("MASK", {"tooltip": "SAM3_Detect individual masks, 'emblem:50' on the ~1 MP image"}),
            "emblems_2": ("MASK", {"tooltip": "SAM3_Detect 'badge:50'"}),
            "emblems_3": ("MASK", {"tooltip": "SAM3_Detect 'logo:50'"}),
            "wheels": ("MASK", {"tooltip": "SAM3_Detect 'wheel:20': a spot on a wheel takes the wheel's angle"}),
            "erase_seed": ("INT", {"default": 35, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta57"}),
            "erase_grow": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "the old emblem's mask is grown by this share of its radius, so its mount / boss goes too"}),
            "erase_context": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.25,
                                        "tooltip": "body spots: the erase sees the spot inside this much wider view"}),
            "polarity_retry": ("BOOLEAN", {"default": True, "tooltip": "solid badges: redo an inverted finish with other seeds"}),
            "max_tilt": ("FLOAT", {"default": 35.0, "min": 5.0, "max": 80.0, "step": 1.0}),
        }}
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "placement", "report")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    # --- one Qwen edit -------------------------------------------------------------------------------------------
    def _edit(self, model, img, prompt, seed, mask=None, denoise=1.0):
        t = _t(img)
        pos = encode(self.clip, self.vae, prompt, [t], [1.0]); neg = encode(self.clip, self.vae, "", [t], [1.0])
        lat = {"samples": self.vae.encode(t[:, :, :, :3])}
        if mask is not None:
            m = torch.from_numpy(mask.astype(np.float32) / 255.0)
            lat["noise_mask"] = m.reshape((-1, 1, m.shape[-2], m.shape[-1]))
        out = nodes.common_ksampler(model, seed, self.steps, self.cfg, self.sampler, self.scheduler, pos, neg, lat,
                                    denoise=denoise)[0]
        o = _decode(self.vae, out["samples"])
        return cv2.resize(o, (SIDE, SIDE), interpolation=cv2.INTER_LANCZOS4) if o.shape[:2] != (SIDE, SIDE) else o

    def _erase(self, model, jb, car, erase_seed, erase_context):
        """Hub: masked edit of the spot's own crop. Body: plain edit of the spot inside a wider view, composited back
        through the feathered erase mask."""
        H, W = car.shape[:2]
        crop, em_c, X0, Y0, side = jb["crop"], jb["em_c"], jb["X0"], jb["Y0"], jb["side"]
        if jb["geo"].get("hub"):
            er = self._edit(model, crop, ERASE_CAP, erase_seed, mask=em_c)
            m = (em_c.astype(np.float32) / 255)[..., None]
        else:
            s2 = min(int(round(erase_context * side)), W, H)
            A0 = int(np.clip(X0 + side / 2 - s2 / 2, 0, W - s2)); B0 = int(np.clip(Y0 + side / 2 - s2 / 2, 0, H - s2))
            big = cv2.resize(car[B0:B0 + s2, A0:A0 + s2], (SIDE, SIDE), interpolation=cv2.INTER_AREA if s2 > SIDE else cv2.INTER_LANCZOS4)
            er = self._edit(model, big, ERASE_CAP, erase_seed)
            if s2 != side:
                er = cv2.resize(er, (s2, s2), interpolation=cv2.INTER_LANCZOS4)[Y0 - B0:Y0 - B0 + side, X0 - A0:X0 - A0 + side]
                er = cv2.resize(er, (SIDE, SIDE), interpolation=cv2.INTER_LANCZOS4)
            m = cv2.GaussianBlur(cv2.dilate(em_c, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3)[..., None] / 255
            if jb.get("far") is not None:
                m = m * (1 - jb["far"][..., None])
        return (crop * (1 - m) + er * m).astype(np.uint8).astype(np.float32)

    def _finish(self, model, jb, sd):
        paste, fm = jb["paste"], jb["fm"]
        mf = cv2.GaussianBlur(fm.astype(np.float32) / 255, (0, 0), 1.5)[..., None]
        o = self._edit(model, paste, FINISH_CAP + (LIGHT_FIELD_CLAUSE if jb["light"] else ""), sd, mask=fm)
        return (paste * (1 - mf) + o * mf).astype(np.uint8)

    def run(self, erase_model, finish_model, clip, vae, image, logo, spots, depth, intrinsics, badge_body, seed,
            emblems=None, emblems_2=None, emblems_3=None, wheels=None, erase_seed=35, steps=25, cfg=4.0,
            sampler_name="euler", scheduler="beta57", erase_grow=0.35, erase_context=2.0, polarity_retry=True,
            max_tilt=35.0):
        self.clip, self.vae, self.steps, self.cfg, self.sampler, self.scheduler = clip, vae, steps, cfg, sampler_name, scheduler
        GM.MAX_TILT = float(max_tilt)
        car = _n(image); H, W = car.shape[:2]
        paint = _painted(spots, W, H)
        if paint is None:
            return (image, image, json.dumps({"spots": 0, "note": "nothing painted"}))
        dep = depth[0].cpu().numpy().astype(np.float32)
        dep = dep[..., 0] if dep.ndim == 3 else dep
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_LINEAR)
        K = PP.parse_intrinsics(intrinsics)
        K[0] *= W / (2 * K[0, 2]); K[1] *= H / (2 * K[1, 2])
        c = dict(car=car, dep=dep, K=K, sam=_masks(emblems, emblems_2, emblems_3, W=W, H=H), wheels=_masks(wheels, W=W, H=H))
        lg = _n(logo)
        a, body, _ = GM.logo_parts(lg); lh, lw = a.shape
        mono = np.percentile(lg.astype(int).max(2) - lg.astype(int).min(2), 99.5) < 20
        badge_body = "solid" if str(badge_body).strip().lower().startswith("solid") else "open"
        solid = badge_body == "solid"
        E = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        n, lab = cv2.connectedComponents(paint.astype(np.uint8))
        jobs, report = [], []
        placement = car.astype(np.float32).copy()
        for j in range(1, n):
            spot = lab == j
            if spot.sum() < 30:
                continue
            Hm, q, geo, _ = GM.site_geometry(c, j, spot, lw, lh, body, marks=a)
            am = a
            if not solid:  # open logos: no wire-thin strokes at the placed size (round 3: Opel 'too thin and fragile')
                am, geo["thicken"] = GM.thicken(a, geo["size"])
            cap_ell = GM.hub_cap(c, spot, geo)
            blank_cap = False
            if cap_ell is None and geo.get("hub") and not solid and GM.round_ring(am, lw, lh):
                # SAM3 missed the cap under a ring logo: the ring went inside the cap's rim, two rings (round 12, Mercedes)
                cap_ell, geo["cap_img"] = GM.hub_cap_image(c, spot, geo)
            if cap_ell is None and geo.get("hub") and geo.get("site") == "spot":  # (after the ring path: r02 untouched)
                # a BLANK hub (no old emblem found - a studio render's bare cap): the erase took the plain cap and its
                # chrome ring and the badge sat dark on the bare hub (round 16 4/4 fail; Yovan: fix before release).
                # Its cap from the image; a plain one is not erased and the badge fills it like a real centre-cap badge
                cap_ell, geo["cap_img"] = GM.blank_hub_cap(c, spot, geo)
                if cap_ell is not None:
                    blank_cap = True
                    geo["cap_plain"] = GM.cap_is_plain(car, cap_ell)
            if cap_ell is not None and geo.get("hub") and not solid:  # a round ring logo: its ring = the cap's rim
                Hm, geo["ring_to_cap"] = GM.fit_ring_to_cap(Hm, lw, lh, cap_ell, am)
            if blank_cap and geo["cap_plain"][0] and geo.get("ring_to_cap") is None:  # paint-sized: a 19-24 px speck on a 38 px cap
                Hm, geo["cap_fill"] = GM.fit_badge_to_cap(Hm, lw, lh, cap_ell, ink=am)
            if geo.get("hub") and not solid:  # open logo on a hub: its ink centred on the cap (round 3: DS 'not centred')
                Hm, geo["ink_shift"] = GM.ink_centred(Hm, am, cap_ell[0] if cap_ell is not None else geo["centre"])
            mw = cv2.warpPerspective(am, Hm, (W, H), flags=cv2.INTER_AREA)
            bw = cv2.warpPerspective((body > 0).astype(np.float32), Hm, (W, H), flags=cv2.INTER_AREA)
            cov = np.maximum(mw, bw if solid else 0)
            ys, xs = np.nonzero(cov > 0.3)
            if len(xs) == 0:
                report.append(dict(spot=j, skipped="badge fell outside the image")); continue
            bx, by = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2; bpx = max(np.ptp(xs), np.ptp(ys))
            sy, sx = np.nonzero(spot)
            side = int(max(256, 3.2 * bpx, 1.3 * max(np.ptp(sx), np.ptp(sy)))); side = min(side, W, H)
            X0 = int(np.clip(bx - side / 2, 0, W - side)); Y0 = int(np.clip(by - side / 2, 0, H - side))
            k = SIDE / side
            cr = lambda im, it=cv2.INTER_LANCZOS4: cv2.resize(im[Y0:Y0 + side, X0:X0 + side], (SIDE, SIDE), interpolation=it)
            crop = cr(car)
            # erase mask = the spot + every old emblem it overlaps, grown so the mount / boss goes too
            grow = spot.copy()
            for sm in c["sam"]:
                if (sm & spot).sum() >= 0.5 * sm.sum() and sm.sum() <= 1.5 * spot.sum():
                    r = max(2, int(erase_grow * np.sqrt(sm.sum() / np.pi)))
                    grow |= cv2.dilate(sm.astype(np.uint8), E(2 * r + 1)) > 0
            em = cv2.dilate(grow.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
            # the flat paste, drawn straight into the 1024 crop from the full-res logo (2x supersampled)
            S2 = np.array([[2 * k, 0, -2 * k * X0], [0, 2 * k, -2 * k * Y0], [0, 0, 1]]) @ Hm
            mwc = cv2.resize(cv2.warpPerspective(am, S2, (2 * SIDE, 2 * SIDE), flags=cv2.INTER_LINEAR), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
            bwc = cv2.resize(cv2.warpPerspective((body > 0).astype(np.float32), S2, (2 * SIDE, 2 * SIDE), flags=cv2.INTER_LINEAR),
                             (SIDE, SIDE), interpolation=cv2.INTER_AREA)
            sol = solid
            if geo.get("hub") and not solid:  # open logo on a hub: a centre cap under it, so it doesn't float
                pts, geo["cap"] = GM.cap_shape(am, lw, lh)
                if pts is not None:  # ringed logos only (v6 review)
                    pp = cv2.perspectiveTransform(pts[None].astype(np.float64), S2)[0]
                    cv = np.zeros((2 * SIDE, 2 * SIDE), np.uint8); cv2.fillPoly(cv, [np.round(pp).astype(np.int32)], 255)
                    bwc = np.maximum(cv2.resize(cv.astype(np.float32) / 255, (SIDE, SIDE), interpolation=cv2.INTER_AREA), mwc); sol = True
            placement = placement * (1 - mw[..., None]) + np.array([40, 120, 255], np.float32) * mw[..., None]
            em_c, far = cr(em, cv2.INTER_LINEAR), None
            if geo.get("site") == "spot" and not geo.get("hub"):
                cut, far = _wall_cut(crop, em_c, np.maximum(mwc, bwc if sol else 0))
                if cut is not None:
                    em_c = em_c.copy(); em_c[cut] = 0; far = far.astype(np.float32); geo["erase_wall_cut"] = int(cut.sum())
            jobs.append(dict(j=j, geo=geo, X0=X0, Y0=Y0, side=side, k=k, crop=crop, em=em, em_c=em_c, far=far,
                             mwc=mwc, bwc=bwc, solid=sol, cap_ell=cap_ell, keep_cap=bool(blank_cap and geo["cap_plain"][0]),
                             light=bool(solid and _light_field(bwc, mwc))))
        # 1) every erase (erase LoRA patched once)
        for jb in jobs:
            if jb["keep_cap"]:  # a plain cap on a blank hub: nothing to erase, the badge goes on the cap as it is
                erased = jb["crop"].astype(np.float32); jb["geo"]["erase"] = "skipped (plain cap)"
            else:
                erased = self._erase(erase_model, jb, car, erase_seed, erase_context)
            if not jb["geo"].get("hub"):  # a straight bar through the spot continues across it (r06 rod, t05 chrome bar)
                erased, jb["geo"]["erase_fix"] = EF.fix(jb["crop"], erased, jb["em_c"] > 8)
                erased = np.asarray(erased, np.float32)
            if jb["cap_ell"] is not None and not jb["keep_cap"]:  # the erase took the cap with the old emblem: put it back
                # a ring logo fitted to the cap is the cap's new rim: the old rim line goes too (round 12 ghost ring)
                erased, jb["geo"]["cap_kept"] = GM.cap_restore(jb["crop"], erased, jb["cap_ell"], jb["X0"], jb["Y0"], jb["k"],
                                                               keep_rim=jb["geo"].get("ring_to_cap") is None)
            mwc, bwc, solid = jb["mwc"], jb["bwc"], jb["solid"]
            paste = jb["erased"] = erased
            if solid:
                paste = paste * (1 - bwc[..., None]) + 35 * bwc[..., None]
            jb["paste"] = np.clip(paste * (1 - mwc[..., None]) + 215 * mwc[..., None], 0, 255).astype(np.uint8)
            covc = np.maximum(mwc, bwc if solid else 0)
            fm = cv2.dilate((covc > 0.05).astype(np.uint8) * 255, E(21))
            if not solid:  # open logos: wide finish mask except gaps that are wide inside the outline (they stay bare)
                mk = (covc > 0.05).astype(np.uint8)
                yy, xx = np.nonzero(mk); ext = max(np.ptp(xx), np.ptp(yy)) + 1
                hull = np.zeros_like(mk); cv2.fillConvexPoly(hull, cv2.convexHull(np.column_stack([xx, yy]).astype(np.int32)), 1)
                dist = cv2.distanceTransform((1 - mk).astype(np.uint8), cv2.DIST_L2, 5)
                rb = 0.07 * ext
                bigg = ((dist > rb) & (hull > 0)).astype(np.uint8)
                keep = (cv2.dilate(bigg, E(2 * int(rb) + 1)) > 0) & (dist > 5) & (hull > 0)
                fm[keep] = 0
                fm = np.maximum(fm, cv2.dilate(mk * 255, E(9)))
            jb["fm"] = fm
        # 2) every finish (finish LoRA), polarity retry, locks, stitch
        final = car.astype(np.float32).copy()
        for jb in jobs:
            paste, mwc, bwc, solid = jb["paste"], jb["mwc"], jb["bwc"], jb["solid"]
            finish = lambda sd: self._finish(finish_model, jb, sd)
            res = finish(seed); pol = None; tries = [seed]
            if solid and polarity_retry:
                bp, bad = _polarity(res, bwc, mwc); pol = [(seed, bp, bad)]
                for sd in RETRY_SEEDS:
                    if bp is None or not bad:
                        break
                    cand = finish(sd); p, pb = _polarity(cand, bwc, mwc); pol.append((sd, p, pb)); tries.append(sd)
                    if p is not None and (p > bp or (bad and not pb)):
                        res, bp, bad = cand, p, pb
            ghost = None
            if solid and jb["light"] and polarity_retry:
                def _wrong(fin):
                    o_ = DL.chroma_lock(fin, bwc)[0] if mono else fin
                    return DL.lock_masks(o_, mwc, bwc, floor=0, align=True, field_k=0.25)[1].get("wrong_frac")
                bw = _wrong(res); ghost = [(tries[-1] if len(tries) == 1 else "polarity pick", bw)]
                for sd in RETRY_SEEDS:
                    if bw is None or bw <= GHOST_OK:
                        break
                    if sd in tries:  # already finished by the polarity retry - not kept there, so not better
                        continue
                    cand = finish(sd); tries.append(sd); p, pb = _polarity(cand, bwc, mwc); w = _wrong(cand)
                    ghost.append((sd, w, p, pb))
                    if not pb and w is not None and w < GHOST_GAIN * bw:
                        res, bw = cand, w
            halo = None
            if solid and not jb["light"] and polarity_retry:
                hb = _halo(res, jb["erased"], bwc, jb["fm"]); halo = [(tries[-1] if len(tries) == 1 else "polarity pick", hb)]
                for sd in RETRY_SEEDS:
                    if hb is None or hb <= HALO_MAX:
                        break
                    if sd in tries:
                        continue
                    cand = finish(sd); tries.append(sd); p, pb = _polarity(cand, bwc, mwc); h = _halo(cand, jb["erased"], bwc, jb["fm"])
                    halo.append((sd, h, p, pb))
                    if not pb and h is not None and h < HALO_GAIN * hb:
                        res, hb = cand, h
            out = res
            info = {}
            if not solid:
                out, info["gaps"] = DL.open_gaps(out, mwc, paste)
            if mono:
                reg = bwc if solid else (cv2.dilate((mwc > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.float32)
                out, info["chroma"] = DL.chroma_lock(out, reg)
            lk, info["lock"] = DL.lock_masks(out, mwc, bwc if solid else None, floor=0 if solid else 80, align=True, field_k=0.25)
            # stitch exactly like the harness: the finished crop's changed pixels first, then what the locks changed
            X0, Y0, side, crop = jb["X0"], jb["Y0"], jb["side"], jb["crop"].astype(np.float32)
            ch = (np.abs(res.astype(np.float32) - crop).max(2) > 3).astype(np.uint8)
            ch = cv2.GaussianBlur(cv2.dilate(ch, np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 2)
            back = cv2.resize(res, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
            chb = cv2.resize(ch, (side, side), interpolation=cv2.INTER_AREA)[..., None]
            final[Y0:Y0 + side, X0:X0 + side] = final[Y0:Y0 + side, X0:X0 + side] * (1 - chb) + back * chb
            d = np.abs(lk.astype(np.float32) - res.astype(np.float32)).max(2) > 2
            if d.any():
                w = cv2.GaussianBlur(cv2.dilate(d.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
                lb = cv2.resize(lk, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
                wb = cv2.resize(w, (side, side), interpolation=cv2.INTER_AREA)[..., None]
                final[Y0:Y0 + side, X0:X0 + side] += (lb - back) * wb
            report.append(dict(spot=jb["j"], crop=[X0, Y0, side], geo=jb["geo"], polarity=pol, light_field=jb["light"], ghost=ghost, halo=halo,
                               lock={k_: (v_ if not isinstance(v_, dict) else {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                                                                                for kk, vv in v_.items()}) for k_, v_ in info.items()}))
        rep = json.dumps({"spots": len(jobs), "badge_body": badge_body, "mono_logo": bool(mono), "sites": report}, default=str)
        return (_t(np.clip(final, 0, 255).astype(np.uint8)), _t(np.clip(placement, 0, 255).astype(np.uint8)), rep)
