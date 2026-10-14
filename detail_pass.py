# Made by Yovan Vujacic
"""Logo Inject stage 2 as ONE ComfyUI node: every badge stage 1 drew is found, erased, redrawn from the clean logo at
stage 1's angle, turned to chrome and pasted back - in a single job, any number of badges.

Same logic as the round-3 test driver (agent-skills/scripts/.runs/li3_run.py + li4_run.py, 11/11 badges on the 4 test
cars, 2026-09-30), plus the speed-ups:
  * the model passes run on a smaller crop (crop_mp, default 0.5 MP) instead of 1 MP;
  * reference images are fed at the crop size (badge crop) and 0.25 MP (logo) instead of the stock encoder's fixed
    1 MP each - the reference tokens were most of every render's cost;
  * the chrome tries are ONE batched sampler call instead of separate jobs.
All geometry (li_core) still works on 1024 px crops, so its calibrated thresholds are unchanged.

Inputs come from stock nodes: SAM3Grounding ('logo' on the stage-1 image) gives the candidate boxes; a CLIP vision
model scores how logo-like each box / render is.
"""
import json
import math
import time

import cv2
import numpy as np
import torch

import comfy.samplers
import comfy.utils
import node_helpers
import nodes

from . import li_core as C

ERASE = ("Remove the emblem completely. Continue the surface underneath - paint, panel crease or shut line, or the "
         "wheel's centre plate or cap - straight and unbroken through the spot where it was, with matching colour, "
         "lighting and reflections. Keep the exact shape and outline of that plate or cap (a square plate stays "
         "square). Keep everything else in the image exactly unchanged.")
CHROME = ("Turn the flat logo {where} into a real, physical polished chrome emblem: bevelled raised edges, crisp "
          "metal reflections and shading that match the scene's lighting, a soft contact shadow. Keep the emblem "
          "EXACTLY where it is, with exactly the same shape, size, angle and perspective, and the same design as "
          "the reference logo. Keep it bright and clearly readable. Change nothing else.")
TEMPLATE = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
            "background), then explain how the user's text instruction should alter or modify the image. Generate a "
            "new image that meets the user's requirements while maintaining consistency with the original input "
            "where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def _u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def _t(a):  # uint8 HxWx3 -> IMAGE [1,H,W,3]
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32) / 255.0)[None]


def _n(img, i=0):  # IMAGE -> uint8 HxWx3
    return (img[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def _scale(img, mp, mult=16):
    s = img.movedim(-1, 1)
    k = math.sqrt(mp * 1024 * 1024 / (s.shape[3] * s.shape[2]))
    w = max(mult, round(s.shape[3] * k / mult) * mult); h = max(mult, round(s.shape[2] * k / mult) * mult)
    return comfy.utils.common_upscale(s, w, h, "area", "disabled").movedim(1, -1)


def encode(clip, vae, prompt, images, ref_mps):
    """TextEncodeQwenImageEditPlus with a chosen size per reference image (stock node: always 1 MP each)."""
    vl, refs, pre = [], [], ""
    for i, (im, mp) in enumerate(zip(images, ref_mps)):
        vl.append(_scale(im, 384 * 384 / (1024 * 1024), mult=1))
        refs.append(vae.encode(_scale(im, mp)[:, :, :, :3]))
        pre += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
    tokens = clip.tokenize(pre + prompt, images=vl, llama_template=TEMPLATE)
    cond = clip.encode_from_tokens_scheduled(tokens)
    return node_helpers.conditioning_set_values(cond, {"reference_latents": refs}, append=True)


class LogoInjectDetailPass:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "BASE Qwen-Image-Edit model (no Logo Inject LoRA), sampling patches applied"}),
            "clip": ("CLIP",), "vae": ("VAE",), "clip_vision": ("CLIP_VISION",),
            "original": ("IMAGE",), "stage1": ("IMAGE",), "logo": ("IMAGE",),
            "sam3_boxes": ("STRING", {"forceInput": True, "lazy": True}), "sam3_scores": ("STRING", {"forceInput": True, "lazy": True}),
            "badge_body": (["open", "solid"], {"default": "open",
                           "tooltip": "open = strokes only (Citroen, Opel); solid = filled badge (Peugeot shield, Alfa roundel). "
                                      "An open logo on a plate the car already had: the plate is erased and the logo sized to 80 % of it. "
                                      "Optional badge spots limit where badges may go; painted runs use the per-spot stage 1 at full res. Output = original pixels + redrawn badges (old badge removed) only; empty spots filled"}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta57"}),
            "crop_mp": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                        "tooltip": "chrome-pass render size (the badge only needs ~200-600 px in the final image)"}),
            "erase_mp": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05,
                         "tooltip": "erase render size - at 0.5 MP the erase left a faint light patch (Opel bonnet, node r1)"}),
            "logo_ref_mp": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 1.0, "step": 0.05}),
            "chrome_tries": ("INT", {"default": 3, "min": 1, "max": 6}),
            "chrome_denoise": ("FLOAT", {"default": 0.65, "min": 0.3, "max": 1.0, "step": 0.01}),
            "logo_tolerance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.5, "step": 0.01,
                               "tooltip": "a chrome try may read this much less like the logo than the flat guide and still be picked. "
                                          "CLIP-H scores shiny metal lower than flat white: 0.08 rejected the only metallic Citroen try, 0.16 the chrome Citroen hub (r6-r8, 0.166)"}),
            "min_logo_delta": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01,
                               "tooltip": "keep a detector box only if stage 1 made it this much more logo-like"}),
            "max_sites": ("INT", {"default": 6, "min": 1, "max": 12}),
            "max_tilt": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 45.0, "step": 0.5,
                         "tooltip": "how far the stroke fit may turn a body badge away from its perspective fit (level horizontal diameter, vertical = the stage-1 oval's conjugate diameter). Higher lets the fit chase stage 1's blurry strokes and over-turn the design (Alfa tailgate, node r4)"}),
        }, "optional": {
            "sam3_masks": ("MASK", {"lazy": True, "tooltip": "SAM3Grounding masks (same order as its boxes): each badge's exact outline"}),
            "spots": ("MASK", {"tooltip": "Logo Inject - Badge Spots: detections outside the painted spots are ignored"}),
            "stage1_full": ("IMAGE", {"tooltip": "Logo Inject - Spot Stage 1: stage 1 rendered per painted spot at full "
                                                 "resolution (used instead of enlarging the ~1 MP stage 1)"}),
        }}
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "site_strips", "report")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def check_lazy_status(self, original=None, spots=None, stage1_full=None, sam3_boxes=None, sam3_scores=None,
                          sam3_masks=None, **kw):
        # painted spots + a full-res per-spot stage 1 = the result is returned as is: the logo detector (SAM3, 3.3 GB)
        # is not needed, and keeping it loaded pushed the chrome pass out of VRAM (ws-test14)
        if (spots is not None and float(spots.min()) < 0.99 and stage1_full is not None and original is not None
                and tuple(stage1_full.shape[1:3]) == tuple(original.shape[1:3])):
            return []
        return [k for k, v in (("sam3_boxes", sam3_boxes), ("sam3_scores", sam3_scores), ("sam3_masks", sam3_masks))
                if v is None]

    # --- helpers -------------------------------------------------------------------------------------------------
    def _emb(self, a):
        e = self.cv.encode_image(_t(a)).image_embeds[0].float()
        return (e / e.norm()).cpu().numpy()

    def _sample(self, crop_np, mask_np, cond, neg, denoise, seed, n, side=None):
        side = side or self.side
        px = _t(cv2.resize(crop_np, (side, side), interpolation=cv2.INTER_AREA))
        lat = self.vae.encode(px)
        lat = lat.repeat((n,) + (1,) * (lat.ndim - 1))
        m = torch.from_numpy(cv2.resize(mask_np.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA))
        latent = {"samples": lat, "noise_mask": m.reshape((-1, 1, side, side))}
        out = nodes.common_ksampler(self.model, seed, self.steps, self.cfg, self.sampler, self.scheduler,
                                    cond, neg, latent, denoise=denoise)[0]["samples"]
        img = self.vae.decode(out)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        return [cv2.resize(_n(img, i), (C.CROP, C.CROP), interpolation=cv2.INTER_CUBIC) for i in range(img.shape[0])]

    def _cond(self, prompt, imgs, mps):
        return encode(self.clip, self.vae, prompt, imgs, mps), encode(self.clip, self.vae, "", imgs, mps)

    # --- main ----------------------------------------------------------------------------------------------------
    def run(self, model, clip, vae, clip_vision, original, stage1, logo, sam3_boxes, sam3_scores, badge_body, seed,
            steps, cfg, sampler_name, scheduler, crop_mp, erase_mp, logo_ref_mp, chrome_tries, chrome_denoise,
            logo_tolerance, min_logo_delta, max_sites, max_tilt=2.0, sam3_masks=None, spots=None, stage1_full=None):
        t0 = time.time()
        self.model, self.clip, self.vae, self.cv = model, clip, vae, clip_vision
        self.steps, self.cfg, self.sampler, self.scheduler = steps, cfg, sampler_name, scheduler
        self.side = int(round(math.sqrt(crop_mp) * 1024 / 16)) * 16
        eside = int(round(math.sqrt(erase_mp) * 1024 / 16)) * 16
        o = _n(original); s1n = _n(stage1); lg = _n(logo)
        if (spots is not None and float(spots.min()) < 0.99 and stage1_full is not None
                and tuple(stage1_full.shape[1:3]) == o.shape[:2]):
            # painted spots: the per-spot stage 1 IS the result - one masked edit per spot, stitched into the original,
            # exactly like WS Qwen Edit. The badge redraw below warped the logos to its fit (Alfa hub roundel squashed,
            # DS skewed) and left grey patches where the old hub plates were (ws-test8, Yovan: "I still see distortion")
            return (stage1_full, _t(np.zeros((384, 384 * 9, 3), np.uint8)),
                    json.dumps(dict(sites=[], painted_stage1_only=True, seconds=round(time.time() - t0, 1))))
        H, W = s1n.shape[:2]
        boxes = json.loads(sam3_boxes or "[]"); scores = json.loads(sam3_scores or "[]")
        outside = []
        if spots is not None:  # painted badge spots: a detection whose box is mostly outside them can't become a site
            sp = cv2.resize(spots[0].cpu().numpy().astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            for k, b in enumerate(boxes):
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                if x1 > x0 and y1 > y0 and sp[max(0, y0):y1, max(0, x0):x1].mean() < 0.3:
                    scores = list(scores); scores[k] = 0.0; outside.append(k)
        e_ref = self._emb(lg)
        osm = cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA)
        deltas = []
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; r = 0.65 * max(b[2] - b[0], b[3] - b[1])
            y0, y1, x0, x1 = [int(max(0, v)) for v in (cy - r, cy + r, cx - r, cx + r)]
            deltas.append(float(self._emb(s1n[y0:y1, x0:x1]) @ e_ref - self._emb(osm[y0:y1, x0:x1]) @ e_ref))
        sites, dm, dlog = C.sites_from_detections(o, s1n, boxes, scores, deltas, min_delta=min_logo_delta,
                                                  max_sites=max_sites)
        if stage1_full is not None and tuple(stage1_full.shape[1:3]) == o.shape[:2]:
            s1 = _n(stage1_full)  # per-spot stage 1, already full-res and sharp (WS Qwen Edit crop/stitch)
        else:
            s1 = cv2.resize(s1n, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if sam3_masks is not None:
            mk = sam3_masks.reshape(-1, sam3_masks.shape[-2], sam3_masks.shape[-1])
            for st in sites:
                if st["det_index"] < mk.shape[0]:
                    m = (mk[st["det_index"]].cpu().numpy() * 255).astype(np.uint8)
                    st["det_mask"] = cv2.resize(m, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LINEAR)
        # painted spots stage 1 left empty (Alfa far hub, seed 7): stage 2 makes the badge there itself - whatever is in
        # the spot is erased (the user painted it), the logo is fitted to its middle 70 %
        if spots is not None and float(sp.min()) < 0.99:
            kx, ky = o.shape[1] / W, o.shape[0] / H
            n_sp, lab_sp = cv2.connectedComponents((sp > 0.5).astype(np.uint8))
            g_sp = max(1, int(0.02 * min(H, W)))  # LogoInjectSpots' default grow: undo it to get the painted blob back
            for j in range(1, n_sp):
                comp = (lab_sp == j).astype(np.uint8)
                if any(comp[min(H - 1, int(q["cy"] / ky)), min(W - 1, int(q["cx"] / kx))] for q in sites):
                    continue
                raw = cv2.erode(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g_sp + 1,) * 2))
                if raw.sum() < 20:
                    raw = comp
                raw_full = cv2.resize(raw * 255, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.nonzero(raw_full)
                foot_full = (C.shrink_footprint(raw_full.astype(np.float32) / 255, 0.7) * 255).astype(np.uint8)
                fy, fx = np.nonzero(foot_full)
                bx0, by0, bw, bh = int(fx.min()), int(fy.min()), int(np.ptp(fx)) + 1, int(np.ptp(fy)) + 1
                sites.append(dict(det_index=-1, cx=float(xs.mean()), cy=float(ys.mean()), w=bw, h=bh, area=bw * bh,
                                  det_box=[bx0, by0, bw, bh], bbox=[bx0, by0, bw, bh], det_mask=foot_full,
                                  spot_erase=raw_full, from_spot=True))
        # painted spots = the user chose every badge position: no 'second badge below another' removal then
        painted = spots is not None and float(sp.min()) < 0.99
        dups = set() if painted else C.stacked_duplicates(sites)
        if painted:  # like WS Qwen Edit: nothing outside the painted spots may change, and the passes run at 1 MP
            sp_full = cv2.resize(sp, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LINEAR)
            self.side = 1024

        def paste(img, crop_img, c, mask):
            if painted:
                bx0, by0, bside = c["box"]
                mask = mask * cv2.resize(sp_full[by0:by0 + bside, bx0:bx0 + bside], (C.CROP, C.CROP),
                                         interpolation=cv2.INTER_LINEAR)
            return C.paste_back(img, crop_img, c["box"], mask)
        for k in outside:  # say why, instead of the generic "low score"
            for d in dlog:
                if d.get("det") == k: d["why"] = "outside the painted badge spots"
        # Stitch like WS Qwen Edit (InpaintCrop/InpaintStitch): the output starts from the ORIGINAL full-res pixels and
        # only the badge areas are pasted in. Starting from stage 1 (a 1 MP render scaled back up) softened the whole
        # car (detail 98 -> 15 on the Alfa) and re-toned every painted spot ("drift", Yovan 2026-10-01).
        cur = o.copy(); report = dict(sites=[], detections=dlog, crop_px=self.side, t_detect=round(time.time() - t0, 1),
                                       outside_spots=outside)
        logo_t = _t(lg); strips = []
        pbar = comfy.utils.ProgressBar(max(1, len(sites)))
        for i, st in enumerate(sites):
            ts = time.time(); rep = dict(site=i, center=[int(st["cx"]), int(st["cy"])]); report["sites"].append(rep)
            c = C.crop_site(s1, o, st, dm)
            if not c["valid"]:
                rep["kept_stage1"] = "no badge marks in the box"; pbar.update(1); continue
            if i in dups:  # a second badge stacked below another one on the same face: put the original surface back
                # generous mask: the badge's contact shadow reaches past its outline, and the original is plain panel
                g = int(0.8 * c["foot_r"]) | 1  # 0.5 left an arc of stage-1 shadow (alfa r6)
                rm = cv2.dilate((c["footprint"] > 0.5).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
                rm = cv2.GaussianBlur(rm, (0, 0), g / 3).astype(np.float32) / 255
                rep["removed_duplicate"] = True  # nothing to paste: the output already holds the original surface
                og = c["original_crop"]
                strips.append(np.hstack([cv2.resize(x, (384, 384)) for x in (c["stage1_crop"], og)] + [np.zeros((384, 384, 3), np.uint8)] * 7))
                rep["seconds"] = round(time.time() - ts, 1); pbar.update(1); continue
            if st.get("from_spot"):  # an empty painted spot: erase all of it, fit the logo to the plain outline
                bx0, by0, bside = c["box"]
                se = cv2.resize(st["spot_erase"][by0:by0 + bside, bx0:bx0 + bside], (C.CROP, C.CROP),
                                interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                c["erase_mask"] = np.maximum(c["erase_mask"], cv2.GaussianBlur(se, (0, 0), 6))
                c["marks"] = cv2.Canny((c["footprint"] > 0.5).astype(np.uint8) * 255, 50, 150).astype(np.float32) / 255
                rep["from_spot"] = True
            # the site's area = the new badge (erase mask) + the old badge stage 1 removed around it: both get erased and
            # pasted, otherwise the original's old badge shows through the stitched output
            c["erase_mask"] = np.maximum(c["erase_mask"], C.old_badge_area(c["original_crop"], c["stage1_crop"],
                                                                           c["footprint"], c["foot_r"]))
            # the crop the passes work on: stage 1 only over the badge (erase area + margin), the sharp original around it
            k1 = cv2.dilate((c["erase_mask"] > 0.05).astype(np.uint8) * 255, np.ones((31, 31), np.uint8))
            k1 = cv2.GaussianBlur(k1, (0, 0), 6).astype(np.float32)[..., None] / 255
            crop = _u8(c["stage1_crop"] * k1 + c["original_crop"] * (1 - k1))
            crop_t = _t(cv2.resize(crop, (eside, eside), interpolation=cv2.INTER_AREA))
            foot, marks = c["footprint"], c["marks"]
            # an outline logo on a plate the car already had: erase the plate too (Yovan, 2026-10-01: the disc kept behind
            # the DS looked unnatural) with a wider mask (r1 left a dark remnant of it), then fit the logo to 80 % of it
            plate = badge_body == "open" and C.plate_under_badge(c["original_crop"], foot); rep["plate"] = plate
            # ---- erase (checked; one retry with another seed and a wider mask) ----
            er = None
            pos, neg = self._cond(ERASE, [crop_t], [erase_mp])
            for attempt, (sd, grow) in enumerate(((seed + 35, 41), (seed, 81)) if plate else ((seed + 35, 0), (seed, 41))):
                em = c["erase_mask"]
                if grow:
                    em = cv2.dilate((em > 0.3).astype(np.uint8) * 255, np.ones((grow, grow), np.uint8))
                    em = cv2.GaussianBlur(em, (0, 0), 8).astype(np.float32) / 255
                cand = self._sample(crop, em, pos, neg, 1.0, sd, 1, side=eside)[0]
                res = C.erase_residual(crop, cand, foot); rep[f"erase{attempt}_residual"] = round(res, 3)
                if res <= 0.45:
                    er = cand; break
            if er is None:
                rep["kept_stage1"] = "erase left the old badge twice"
                cur = paste(cur, crop, c, c["erase_mask"]); pbar.update(1); continue
            if plate:
                foot = C.shrink_footprint(foot); marks = marks * foot
            # ---- clean logo fitted to the stage-1 badge (checked + clamped) ----
            guide, cm, meta = C.badge_guide(er, foot, marks, lg, body=badge_body, max_tilt=max_tilt)
            rep.update(hub=bool(meta.get("hub")), outline_fallback=meta.get("outline_fallback"), tilt=meta.get("tilt"), htilt=meta.get("htilt"))
            if guide is None:
                rep["kept_stage1"] = "fit rejected: " + meta["rejected"]
                cur = paste(cur, crop, c, c["erase_mask"]); pbar.update(1); continue
            # ---- chrome: all tries in one batched call, keep the best chrome that still reads as the logo ----
            guide_t = _t(cv2.resize(guide, (self.side, self.side), interpolation=cv2.INTER_AREA))
            prompt = CHROME.format(where="on the wheel's centre cap" if meta["hub"] else "on the car's body")
            pos, neg = self._cond(prompt, [guide_t, logo_t], [crop_mp, logo_ref_mp])
            outs = self._sample(guide, cm, pos, neg, chrome_denoise, seed, chrome_tries)
            e_guide = float(self._emb(_region(guide, foot)) @ e_ref)
            cands = []
            for k, x in enumerate(outs):
                sc, design_ok = C.chrome_score(x, guide, er, badge_body == "solid")
                cands.append(dict(k=k, img=x, metal=sc, design_ok=design_ok,
                                  sim=float(self._emb(_region(x, foot)) @ e_ref)))
            ok = [x for x in cands if x["sim"] >= e_guide - logo_tolerance and x["design_ok"]]                 or [x for x in cands if x["sim"] >= e_guide - logo_tolerance] or cands
            best = max(ok, key=lambda x: x["metal"])
            e1 = float(self._emb(_region(crop, foot)) @ e_ref)
            rep.update(chrome=[dict(k=x["k"], metal=round(x["metal"], 1), design_ok=x["design_ok"], sim=round(x["sim"], 3)) for x in cands],
                       pick=best["k"], sim_guide=round(e_guide, 3), sim_stage1=round(e1, 3), sim_stage2=round(best["sim"], 3))
            tiles = [cv2.resize(x, (384, 384)) for x in [crop, er, guide] + [x["img"] for x in cands]]
            cv2.rectangle(tiles[3 + best["k"]], (2, 2), (381, 381), (0, 255, 0), 6)
            strips.append(np.hstack(tiles + [np.zeros((384, 384, 3), np.uint8)] * (6 - chrome_tries)))
            if best["sim"] < e1:
                rep["kept_stage1"] = "stage 2 not more logo-like than stage 1"
                cur = paste(cur, crop, c, c["erase_mask"])
            else:
                cur = paste(cur, best["img"], c, c["erase_mask"]); rep["pasted"] = True
            rep["seconds"] = round(time.time() - ts, 1); pbar.update(1)
        report["seconds"] = round(time.time() - t0, 1); report["duplicates_removed"] = sorted(dups)
        for st in sites:
            st.pop("det_mask", None)
        strip = np.vstack(strips) if strips else np.zeros((384, 384 * 9, 3), np.uint8)
        return (_t(cur), _t(strip), json.dumps(report, default=str))


def _region(img, foot, pad=0.3):
    ys, xs = np.nonzero(foot > 0.5); r = (1 + pad) * 0.5 * max(np.ptp(xs), np.ptp(ys))
    cx, cy = xs.mean(), ys.mean()
    y0, y1, x0, x1 = [int(np.clip(v, 0, img.shape[0])) for v in (cy - r, cy + r, cx - r, cx + r)]
    return img[y0:y1, x0:x1]
