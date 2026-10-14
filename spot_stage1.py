# Made by Yovan Vujacic
"""Stage 1 per painted spot, the way Workflow Studio's Qwen Edit does masked edits (InpaintCropImproved ->
edit the crop -> InpaintStitchImproved, 2026-10-01).

Whole-car stage 1 renders the car at ~1 MP, so a badge area pasted into a 4160 px original was enlarged ~3x: soft,
warped lettering and a notched lip around the badge (Alfa seed 7, Peugeot seed 123, ws-test7). Here, for each
painted spot: crop around it with context, enlarge the crop to 1 MP, let the Logo Inject model change ONLY the painted
area (noise mask), shrink back and blend it in with a soft edge. Pixels outside the paint stay exactly the original.

Nothing painted (Workflow Studio's empty.png): the whole-car stage 1 (lazy input) is used unchanged, so the tool
behaves exactly as before. Painted: the whole-car stage 1 is never run.
"""
import math

import cv2
import numpy as np
import torch

import comfy.samplers
import comfy.utils
import nodes

from .detail_pass import CHROME, _n, _t, _u8, encode
from . import li_core as C
from . import perspective as PP


ERASE_SPOT = ("Remove the emblem, badge, lettering or badge plate completely. Continue the surface underneath - paint, "
              "grille mesh, trim bar, panel crease or shut line, or the wheel's centre - straight and unbroken through "
              "the spot where it was, with matching colour, lighting and reflections. Keep everything else in the image "
              "exactly unchanged.")


DESIGN_HINT = (" The faint badge already drawn on the surface marks exactly where the new badge goes, its outline, size, "
               "angle and design: render that same badge as a real emblem, with the reference logo's exact design, "
               "lettering and proportions - do not invent another animal, shape or text.")


CHAIN_HINT = (" Picture 3 shows this same badge already placed elsewhere on this car: make this one the identical object - "
              "the same colours, the same finish, the same plate or no plate - only seen from this spot's angle.")


MATERIAL_HINT = (" The badge keeps the reference's own colours and finish - its light parts light, its dark design dark, "
                 "a thin polished metal rim - and every badge on this car is the same object in the same material. "
                 "Repaint the old emblem's whole plate so none of it is left around the new badge.")


OPEN_HINT = (" The emblem is an open chrome outline mounted directly on the surface: no disc, plate or background shape "
             "behind it - the surface shows through between its strokes.")


def painted_mask(spots):
    """Painted area as uint8 0/255 at the spot image's size, or None when nothing is painted."""
    m = spots[0, ..., 0].cpu().numpy()
    if m.mean() > 0.98 or m.max() < 0.5:
        return None
    return (m > 0.5).astype(np.uint8) * 255


class LogoInjectSpotStage1:
    wired = {"sites_a", "depth"}  # lazy inputs the bundle wires (ws-test24: SAM3 sites + DA3 depth)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "stage-1 model: Qwen-Image-Edit + Logo Inject LoRA, sampling patches applied"}),
            "clip": ("CLIP",), "vae": ("VAE",),
            "original": ("IMAGE", {"tooltip": "the car at full resolution"}),
            "small": ("IMAGE", {"tooltip": "the car at the size stage 1 / the detector work at (~1 MP)"}),
            "logo": ("IMAGE",),
            "spots": ("IMAGE", {"tooltip": "painted badge spots: white on black (empty / all white = whole car)"}),
            "whole_car_stage1": ("IMAGE", {"lazy": True, "tooltip": "whole-car stage 1: only run when nothing is painted"}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta57"}),
            "context": ("FLOAT", {"default": 2.5, "min": 1.2, "max": 6.0, "step": 0.1,
                                  "tooltip": "crop side = this x the spot's size: the model must see which panel / hub it is on"}),
            "grow": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.05, "step": 0.005,
                               "tooltip": "grow each painted spot by this share of the image's short side"}),
            "blend": ("INT", {"default": 32, "min": 0, "max": 128,
                              "tooltip": "soft stitch edge in working-resolution px (WS Qwen Edit: mask_blend_pixels 32)"}),
        }, "optional": {
            "badge_body": (["open", "solid"], {"default": "solid", "tooltip": "open = strokes only (DS, Citroen, Opel): "
                           "the draw step is told to put no disc/plate behind it (ws-test10: DS drawn on a grey disc)"}),
            "depth": ("IMAGE", {"lazy": True, "tooltip": "Depth Anything 3 raw depth of the original (normalization Raw): "
                                "with intrinsics, the badge's angle is MEASURED - the model drew every badge front-on "
                                "(Yovan 2026-10-02)"}),
            "intrinsics": ("STRING", {"lazy": True, "forceInput": True, "tooltip": "Depth Anything 3 intrinsics_json"}),
            "chrome_tries": ("INT", {"default": 3, "min": 1, "max": 6}),
            "erase_passes": ("INT", {"default": 2, "min": 1, "max": 3}),
            "chrome_denoise": ("FLOAT", {"default": 0.65, "min": 0.3, "max": 1.0, "step": 0.01}),
            "max_tilt": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 90.0, "step": 1.0,
                                   "tooltip": "cap on the measured up/down tilt (small ledges read as facing up)"}),
            "badge_size": ("FLOAT", {"default": 0.7, "min": 0.3, "max": 1.0, "step": 0.05,
                                     "tooltip": "badge size = this share of the painted spot's larger side"}),
            "sites_a": ("MASK", {"lazy": True, "tooltip": "SAM3Grounding masks on the ~1 MP car ('emblem'): the old emblem's exact "
                                 "outline inside each painted spot sets the new badge's centre, size and perspective"}),
            "sites_b": ("MASK", {"lazy": True, "tooltip": "SAM3Grounding masks ('badge')"}),
            "sites_c": ("MASK", {"lazy": True, "tooltip": "SAM3Grounding masks ('logo')"}),
            "finish_denoise": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05,
                                         "tooltip": "footprint path: the Logo Inject model finishes the pinned flat logo at this denoise"}),
            "chrome_after": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "footprint path: after the Logo Inject finish, a chrome pass (base model) over the "
                                                  "badge only at this denoise; 0 = off. The LoRA finish alone keeps the flat guide "
                                                  "(Peugeot shield = matte sticker at 0.7 and 0.85, ws-test17c); at 0.55 it "
                                                  "greyed the shield's black field out (ws-test17f) - off by default"}),
            "pin": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.1,
                              "tooltip": "footprint path: how hard the exact logo's pixels are pasted before the finish. -1 = auto "
                                         "(ws-test18: solid badges 0 - the model draws them well; open logos 1 - drawn free, the "
                                         "DS came back on a silver disc, pinned at finish 0.7 it is a clean chrome DS). 1 = hard "
                                         "paste (finish_denoise applies; came out as flat stickers - Yovan 2026-10-05); 0 = the "
                                         "Logo Inject model DRAWS the badge (denoise 1.0) inside the footprint only, from the "
                                         "logo reference, so position / size / perspective are forced by the mask and the "
                                         "material is the model's (ws-test8 look)"}),
            "logo_ref_mp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "footprint path: size (MP) the logo reference is encoded at for the draw / finish; "
                                                 "0.25 let the model invent a wrong lion (Peugeot ws-test19)"}),
            "fill": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.2, "step": 0.05,
                               "tooltip": "footprint path: new badge size as a share of the old emblem's outline"}),
            "erase_model": ("MODEL", {"tooltip": "BASE Qwen-Image-Edit (no LoRA): erases whatever is in each painted spot "
                                                 "before the logo is drawn. Without it the logo was drawn onto the old "
                                                 "emblem (DS: faint logo on a silver copy of the old disc, ws-test8)"}),
        }}
    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("stage1", "stage1_full", "painted", "erased")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def check_lazy_status(self, spots=None, whole_car_stage1=None, depth=None, intrinsics=None, erase_model=None,
                          sites_a=None, sites_b=None, sites_c=None, **kw):
        if spots is None:
            return []
        if painted_mask(spots) is None:
            return ["whole_car_stage1"] if whole_car_stage1 is None else []
        if erase_model is None:  # draw-only bundle (local CM 2026-10-02): depth isn't wired, asking for it errors the run
            return []
        want = []
        if "sites_a" in self.wired:  # footprint path: SAM3 outlines of the old emblems (centre + size)
            want += [k for k, v in (("sites_a", sites_a), ("sites_b", sites_b), ("sites_c", sites_c)) if v is None]
        if "depth" in self.wired:  # + DA3 plane (orientation / perspective, ws-test24)
            want += [k for k, v in (("depth", depth), ("intrinsics", intrinsics)) if v is None]
        return want

    def run(self, model, clip, vae, original, small, logo, spots, prompt, seed, steps, cfg, sampler_name, scheduler,
            context, grow, blend, whole_car_stage1=None, erase_model=None, badge_body="solid", depth=None, intrinsics=None, chrome_tries=3,
            chrome_denoise=0.65, max_tilt=35.0, badge_size=0.7, erase_passes=2, sites_a=None, sites_b=None, sites_c=None,
            finish_denoise=0.5, fill=0.9, chrome_after=0.55, pin=-1.0, logo_ref_mp=1.0):
        if pin < 0:  # auto: soft pin (ws-test20) - the exact design is in the guide, the model draws the material
            pin = 0.5
        self._chain = {}
        pm = painted_mask(spots)
        if pm is None:  # nothing painted: exactly the old path (detail pass enlarges stage 1 itself). Same frame as the
            # old ImageCompositeMasked(small <- stage 1 at 0,0): the decode is a few px smaller than 'small' (1368x760 vs
            # 1371x765), and that frame shifted the detector's crops enough to change the badge render (ws-test8)
            out = small.clone()
            h, w = min(out.shape[1], whole_car_stage1.shape[1]), min(out.shape[2], whole_car_stage1.shape[2])
            out[:, :h, :w] = whole_car_stage1[:, :h, :w]
            return (out, out, False, original)
        o = _n(original); Ho, Wo = o.shape[:2]
        pm = cv2.resize(pm, (Wo, Ho), interpolation=cv2.INTER_NEAREST)
        g = int(grow * min(Ho, Wo))
        if g:
            pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        n, lab, st, _ = cv2.connectedComponentsWithStats((pm > 127).astype(np.uint8))
        cur = o.copy(); side_w = 1024; ers = o.copy()
        site_masks = []
        for sm_ in (sites_a, sites_b, sites_c):
            if sm_ is None:
                continue
            for mk in sm_.reshape(-1, sm_.shape[-2], sm_.shape[-1]).cpu().numpy():
                mk = cv2.resize(mk.astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_LINEAR) > 0.5
                if mk.sum() > 0:
                    site_masks.append(mk)
        footprint = erase_model is not None and (sites_a is not None or sites_b is not None or sites_c is not None)
        have_plane = depth is not None and bool(intrinsics)
        measured = erase_model is not None and have_plane and not footprint
        if have_plane and footprint:
            dep = cv2.resize(depth[0, ..., 0].cpu().numpy().astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_LINEAR)
            K = PP.parse_intrinsics(intrinsics)
            K[0] *= Wo / (2 * K[0, 2]); K[1] *= Ho / (2 * K[1, 2])
        if footprint:
            lg = _n(logo); alpha, hm, hull = C._logo_parts(lg)
            lx, ly, lw, lh = cv2.boundingRect(hull); alpha = alpha[ly:ly + lh, lx:lx + lw]; hm = hm[ly:ly + lh, lx:lx + lw]
            self.geometry = []
            cs_ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            (sx, sy), (sw, sh), sa = cv2.fitEllipse(max(cs_, key=cv2.contourArea))
        if measured:
            dep = cv2.resize(depth[0, ..., 0].cpu().numpy().astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_LINEAR)
            K = PP.parse_intrinsics(intrinsics)
            K[0] *= Wo / (2 * K[0, 2]); K[1] *= Ho / (2 * K[1, 2])  # DA3 reports K at its output size: rescale to ours
            lg = _n(logo); alpha, hm, hull = C._logo_parts(lg)
            lx, ly, lw, lh = cv2.boundingRect(hull); alpha = alpha[ly:ly + lh, lx:lx + lw]; hm = hm[ly:ly + lh, lx:lx + lw]
            self.geometry = []
        text = prompt + (OPEN_HINT if badge_body == "open" else "")
        comps = sorted([j for j in range(1, n) if st[j, 4] >= 0.00002 * Ho * Wo], key=lambda j: -st[j, 4])  # biggest first
        pbar = comfy.utils.ProgressBar(max(1, len(comps)))
        for j in comps:
            x, y, w, h = st[j, :4]
            side = min(Ho, Wo, max(int(round(context * max(w, h))), 256))
            cx, cy = x + w / 2, y + h / 2
            x0 = int(min(max(0, round(cx - side / 2)), Wo - side)); y0 = int(min(max(0, round(cy - side / 2)), Ho - side))
            crop = cur[y0:y0 + side, x0:x0 + side]
            m = (lab[y0:y0 + side, x0:x0 + side] == j).astype(np.float32)
            px = cv2.resize(crop, (side_w, side_w), interpolation=cv2.INTER_CUBIC if side < side_w else cv2.INTER_AREA)
            mw = cv2.resize(m, (side_w, side_w), interpolation=cv2.INTER_LINEAR)

            def edit(mdl, src, text, refs, sd):  # one masked Qwen edit of the working-res crop (inside the spot only)
                cond = encode(clip, vae, text, [_t(src)] + refs, [1.0] * (1 + len(refs)))
                neg = encode(clip, vae, "", [_t(src)] + refs, [1.0] * (1 + len(refs)))
                latent = {"samples": vae.encode(_t(src)), "noise_mask": torch.from_numpy(mw).reshape(1, 1, side_w, side_w)}
                out = nodes.common_ksampler(mdl, sd, steps, cfg, sampler_name, scheduler, cond, neg, latent,
                                            denoise=1.0)[0]["samples"]
                img = vae.decode(out)
                if img.ndim == 5:
                    img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
                return _u8(src * (1 - mw[..., None]) + _n(img) * mw[..., None])  # outside the spot: the crop as it was

            keep_plate = False
            if footprint and badge_body == "solid":  # a solid badge sits ON the old plate / disc (ws-test8 Alfa look):
                spot_m = lab == j  # erasing the plate first left a dark blob twice its size (ws-test17); lettering is erased
                cand_ = [mk for mk in site_masks if (mk & spot_m).sum() >= 0.8 * mk.sum()]
                if cand_:
                    u_ = np.zeros_like(spot_m)
                    for q in cand_:
                        u_ |= q
                    (_, _), (uw, uh), ua = cv2.minAreaRect(np.column_stack(np.nonzero(u_)[::-1]).astype(np.float32))
                    # compact only: the elongated old lip emblem kept under the new roundel left its black tail and orange
                    # tip visible (ws-test24); hub squares (aspect ~1.2-1.3) keep their plate
                    keep_plate = max(uw, uh) / max(1.0, min(uw, uh)) < 1.4
            if erase_model is not None and not keep_plate:  # 1) clear the spot (old emblem, plate, lettering)
                for e_ in range(1 if footprint else erase_passes):  # 2nd pass never helped (Alfa far hub, ws-test16)
                    px = edit(erase_model, px, ERASE_SPOT, [], seed + 35 + e_)
            if footprint:  # 2) the exact logo pinned on the OLD emblem's outline, 3) the Logo Inject model finishes it
                spot_m = lab == j
                cand = [mk for mk in site_masks if (mk & spot_m).sum() >= 0.8 * mk.sum()]
                geo = dict(spot=int(j))
                if cand:  # every detected mark inside the paint (a wordmark = one mask per letter, Peugeot ws-test17)
                    mk = np.zeros_like(spot_m)
                    for q in cand:
                        mk |= q
                    pts_ = np.column_stack(np.nonzero(mk)[::-1]).astype(np.float32)
                    # the ellipse INSCRIBED in the emblem's min-area box: a square plate's hull-fit ellipse was 1.41x the
                    # plate (Alfa hub roundels filled the whole erased blob, ws-test17); for a disc both agree
                    (tx, ty), (tw, th), ta = cv2.minAreaRect(cv2.convexHull(pts_))
                    asp = max(tw, th) / max(1.0, min(tw, th))
                    long_ang = (ta + 90) % 180 if tw < th else ta % 180
                    if asp > 2.2 and min(long_ang, 180 - long_ang) < 25:  # level wordmark strip: size by letter height
                        short = min(tw, th) * 4.0; lasp = sw / sh  # 4x the letter height (min-area box; 3x read small, ws-test17f)
                        tw, th = (short * lasp, short) if tw < th else (short, short * lasp)
                        geo["site"] = "strip"
                    else:
                        tw, th = tw * fill, th * fill; geo["site"] = "outline"
                else:  # no detected emblem in the paint: an upright badge filling 60% of the painted blob
                    tx, ty = cx, cy; tw, th = 0.6 * w, 0.6 * w * sh / sw; ta = 0.0; geo["site"] = "paint"
                if have_plane:  # the badge's shape = the LOGO's own shape on the panel's plane (ws-test24): the old emblem's
                    # outline only gives centre + size. Mapping a roundel onto the old lip emblem's oval melted it (ws-test22);
                    # a level lettering strip gave a frontal Peugeot shield on a receding tailgate (ws-test21)
                    n_, c_, share = PP.fit_plane_ransac(dep, K, lab == j)
                    geo["plane_share"] = round(float(share), 2)
                    if not PP.tilt_reliable(n_, max_tilt):
                        th_ = PP.edge_angle(o, (cx, cy), max(w, h))
                        if th_ is not None:
                            y0_, p0_ = PP.yaw_pitch(n_)
                            n_, yaw_, err_ = PP.normal_from_edge(K, c_, (cx, cy), th_, float(np.clip(p0_, -max_tilt, max_tilt)), y0_)
                            geo.update(fallback="edge", edge_deg=round(float(th_), 1))
                        else:
                            n_ = PP.clamp_pitch(n_, max_tilt); geo["fallback"] = "clamp"
                    elif not keep_plate:  # the logo's horizontal follows the panel's own straight edge: the depth plane's
                        # turn alone tilted the Peugeot shield against the level light bar (ws-test24). Not on hubs: the
                        # nearest edges there are diagonal spokes
                        th_ = PP.edge_angle(o, (cx, cy), max(w, h))
                        if th_ is not None:
                            y0_, p0_ = PP.yaw_pitch(n_)
                            n_, yaw_, ptc_, err_ = PP.normal_nearest_edge(K, c_, (cx, cy), th_, float(y0_), float(p0_), max_tilt)
                            geo.update(edge="aligned", edge_deg=round(float(th_), 1), fit_err=round(float(err_), 1))
                    yaw_v, pitch_v = PP.yaw_pitch(n_); geo.update(yaw=round(float(yaw_v)), pitch=round(float(pitch_v)))
                    Hm, _ = PP.logo_homography(K, n_, c_, (tx, ty), lw, lh, float(max(tw, th)))
                else:  # same convention as the detail pass (Yovan 2026-09-30): horizontal diameter level, vertical = conjugate
                    A = C._triu(tw, th, ta) @ np.linalg.inv(C._triu(sw, sh, sa))
                    Hm = np.eye(3); Hm[:2, :2] = A; Hm[:2, 2] = np.array([tx, ty]) - A @ [sx, sy]
                k_ = side_w / side
                Hc = np.array([[k_, 0, -x0 * k_], [0, k_, -y0 * k_], [0, 0, 1.0]]) @ Hm
                a1 = cv2.warpPerspective(alpha, Hc, (side_w, side_w), flags=cv2.INTER_AREA)
                b1 = cv2.warpPerspective(hm.astype(np.float32) / 255, Hc, (side_w, side_w), flags=cv2.INTER_AREA)

                def paste(op):  # the exact logo laid on the (erased / kept) surface at opacity op
                    g_ = px.astype(np.float32)
                    if badge_body == "solid":
                        g_ = g_ * (1 - b1[..., None] * op) + 40 * b1[..., None] * op
                    g_ = g_ * (1 - a1[..., None] * op) + (215 if badge_body == "solid" else 235) * a1[..., None] * op
                    return _u8(g_)
                guide = paste(pin); sg = paste(1.0)  # sg: hard paste, only for scoring (where the strokes are)
                # the area the finish may change: the badge's footprint ellipse (+ margin), in crop px - not the strokes
                fp = cv2.dilate((b1 > 0.1).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
                if cand and (pin < 1 or keep_plate):  # the whole old plate is repainted with the badge (red plate edges stayed, ws-test18/24)
                    pm_ = cv2.resize(mk[y0:y0 + side, x0:x0 + side].astype(np.uint8) * 255, (side_w, side_w), interpolation=cv2.INTER_NEAREST)
                    fp = np.maximum(fp, cv2.dilate(pm_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))))
                if keep_plate:
                    cm = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
                    cm = cv2.GaussianBlur(cm, (0, 0), 6).astype(np.float32) / 255 * mw
                else:  # a free badge is redrawn only inside its own warped outline: with the loose footprint the redraw grew
                    # the Peugeot shield, turned it front-on and ran it into the light bar (ws-test25)
                    cm = cv2.dilate((b1 > 0.1).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                    cm = cv2.GaussianBlur(cm, (0, 0), 1.5).astype(np.float32) / 255 * mw
                group = "plate" if keep_plate else "free"
                chained = group in self._chain  # the pin < 1 gate meant the pinned runs never chained: mixed hubs (ws-test22/24)
                refs_ = [logo] + ([self._chain[group]] if chained else [])
                ftext = text + (MATERIAL_HINT if pin < 1 else "") + (DESIGN_HINT if 0 < pin < 1 else "") + (CHAIN_HINT if chained else "")
                cond = encode(clip, vae, ftext, [_t(guide)] + refs_, [1.0, logo_ref_mp] + [logo_ref_mp] * (len(refs_) - 1))
                neg = encode(clip, vae, "", [_t(guide)] + refs_, [1.0, logo_ref_mp] + [logo_ref_mp] * (len(refs_) - 1))
                lat = vae.encode(_t(guide)); tries = []
                for t_ in range(max(1, min(chrome_tries, 2))):
                    latent = {"samples": lat, "noise_mask": torch.from_numpy(cm).reshape(1, 1, side_w, side_w)}
                    out = nodes.common_ksampler(model, seed + t_, steps, cfg, sampler_name, scheduler, cond, neg,
                                                latent, denoise=1.0 if pin < 1 else finish_denoise)[0]["samples"]
                    img = vae.decode(out)
                    if img.ndim == 5:
                        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
                    tries.append(_u8(guide * (1 - cm[..., None]) + _n(img) * cm[..., None]))
                scored = [(C.chrome_score(t, sg, px, badge_body == "solid"), t) for t in tries]
                ok = [x for x in scored if x[0][1]] or scored
                gen = max(ok, key=lambda x: x[0][0])[1]
                guide = sg
                if not chained:  # later spots of the same kind see this badge: one material (biggest spot first)
                    ys_, xs_ = np.nonzero(fp > 0); m_ = 48
                    yb0, yb1 = max(0, ys_.min() - m_), min(side_w, ys_.max() + m_); xb0, xb1 = max(0, xs_.min() - m_), min(side_w, xs_.max() + m_)
                    self._chain[group] = _t(gen[yb0:yb1, xb0:xb1])
                if chrome_after > 0:  # 4) chrome pass on the finished badge only (base model, no LoRA)
                    ccond = encode(clip, vae, CHROME.format(where="on the car"), [_t(gen), logo], [1.0, 0.25])
                    cneg = encode(clip, vae, "", [_t(gen), logo], [1.0, 0.25])
                    clat = vae.encode(_t(gen)); ctries = []
                    for t_ in range(2):
                        latent = {"samples": clat, "noise_mask": torch.from_numpy(cm).reshape(1, 1, side_w, side_w)}
                        out = nodes.common_ksampler(erase_model, seed + 10 + t_, steps, cfg, sampler_name, scheduler, ccond,
                                                    cneg, latent, denoise=chrome_after)[0]["samples"]
                        img = vae.decode(out)
                        if img.ndim == 5:
                            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
                        ctries.append(_u8(gen * (1 - cm[..., None]) + _n(img) * cm[..., None]))
                    cscored = [(C.chrome_score(t, guide, px, badge_body == "solid"), t) for t in ctries]
                    cok = [x for x in cscored if x[0][1]] or cscored
                    gen = max(cok, key=lambda x: x[0][0])[1]
                geo.update(centre=[round(float(tx)), round(float(ty))], axes=[round(float(tw)), round(float(th))],
                           ang=round(float(ta), 1))
                self.geometry.append(geo)
            elif measured:  # 2) the exact logo laid on the MEASURED plane, 3) a light chrome pass on the logo only
                n_, c_, share = PP.fit_plane_ransac(dep, K, lab == j)
                geo = dict(spot=int(j), plane_share=round(share, 2))
                if not PP.tilt_reliable(n_, max_tilt):  # thin ledge read as facing up (Alfa lip): upright surface turned
                    th = PP.edge_angle(o, (cx, cy), max(w, h))  # so the logo's horizontal follows the ledge line
                    if th is not None:
                        y0_, p0_ = PP.yaw_pitch(n_)
                        n_, yaw, err = PP.normal_from_edge(K, c_, (cx, cy), th, float(np.clip(p0_, -max_tilt, max_tilt)), y0_)
                        geo.update(fallback="edge", edge_deg=round(th, 1), yaw=float(yaw), fit_err=round(float(err), 1))
                    else:
                        n_ = PP.clamp_pitch(n_, max_tilt); geo["fallback"] = "clamp"
                Hm, _ = PP.logo_homography(K, n_, c_, (cx, cy), lw, lh, badge_size * max(w, h))
                k_ = side_w / side
                Hc = np.array([[k_, 0, -x0 * k_], [0, k_, -y0 * k_], [0, 0, 1.0]]) @ Hm
                a = cv2.warpPerspective(alpha, Hc, (side_w, side_w), flags=cv2.INTER_AREA)
                guide = px.astype(np.float32)
                if badge_body == "solid":
                    bm = cv2.warpPerspective(hm.astype(np.float32) / 255, Hc, (side_w, side_w), flags=cv2.INTER_AREA)
                    guide = guide * (1 - bm[..., None]) + 40 * bm[..., None]
                    guide = guide * (1 - a[..., None]) + 215 * a[..., None]; cover = np.maximum(a, bm)
                else:
                    guide = guide * (1 - a[..., None]) + 235 * a[..., None]; cover = a
                guide = _u8(guide)
                cm = cv2.dilate((cover > 0.1).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
                cm = cv2.GaussianBlur(cm, (0, 0), 6).astype(np.float32) / 255 * mw
                cond = encode(clip, vae, CHROME.format(where="on the car"), [_t(guide), logo], [1.0, 0.25])
                neg = encode(clip, vae, "", [_t(guide), logo], [1.0, 0.25])
                lat = vae.encode(_t(guide)); tries = []
                for t_ in range(chrome_tries):  # one at a time: 3 batched at 1 MP spilled out of VRAM (DS: 11+ min, ws-test14)
                    latent = {"samples": lat, "noise_mask": torch.from_numpy(cm).reshape(1, 1, side_w, side_w)}
                    out = nodes.common_ksampler(erase_model, seed + t_, steps, cfg, sampler_name, scheduler, cond, neg,
                                                latent, denoise=chrome_denoise)[0]["samples"]
                    img = vae.decode(out)
                    if img.ndim == 5:
                        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
                    tries.append(_u8(guide * (1 - cm[..., None]) + _n(img) * cm[..., None]))
                scored = [(C.chrome_score(t, guide, px, badge_body == "solid"), t) for t in tries]
                ok = [x for x in scored if x[0][1]] or scored
                gen = max(ok, key=lambda x: x[0][0])[1]
                geo["normal"] = [round(float(v), 3) for v in n_]; self.geometry.append(geo)
            else:
                gen = edit(model, px, text, [logo], seed)
            # stitch: blend only inside the (feathered) painted spot, at working res, then shrink into place
            k = blend | 1 if blend else 0
            sm = cv2.GaussianBlur(mw, (0, 0), blend / 3) if k else mw
            sm = np.minimum(sm * 1.5, 1.0) * (cv2.dilate(mw, np.ones((k or 1, k or 1), np.uint8)) > 0)
            gen_s = cv2.resize(gen, (side, side), interpolation=cv2.INTER_AREA if side < side_w else cv2.INTER_CUBIC)
            sm_s = cv2.resize(sm.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)[..., None]
            cur[y0:y0 + side, x0:x0 + side] = _u8(crop * (1 - sm_s) + gen_s * sm_s)
            px_s = cv2.resize(px, (side, side), interpolation=cv2.INTER_AREA if side < side_w else cv2.INTER_CUBIC)
            ers[y0:y0 + side, x0:x0 + side] = _u8(ers[y0:y0 + side, x0:x0 + side] * (1 - sm_s) + px_s * sm_s)
            pbar.update(1)
        hs, ws = small.shape[1], small.shape[2]
        sm_img = cv2.resize(cur, (ws, hs), interpolation=cv2.INTER_AREA)
        return (_t(sm_img), _t(cur), True, _t(ers))
