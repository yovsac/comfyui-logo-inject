# Made by Yovan Vujacic
"""Logo Inject stage 2 - badge detail pass. Nodes wrap li_core (pure numpy/cv2).

Graph per badge site i (i = 0..N-1), after stage 1 (whole-image Logo Inject):
  LogoInjectSiteCrop(i) -> [Qwen erase, masked by erase_mask] -> LogoInjectBadgeGuide
  -> [Qwen chrome pass, image2 = logo, masked by chrome_mask, low denoise] -> LogoInjectPasteBack (chained).
A site index past the number of badges found yields a tiny 64 px dummy crop and PasteBack passes the image through,
so a graph built for N sites works on images with fewer badges.
"""
import json
import numpy as np
import torch
from . import li_core as C
from .detail_pass import LogoInjectDetailPass
from .spots import LogoInjectSpots
from .spot_stage1 import LogoInjectSpotStage1
from .dev_dump import LogoInjectDevDump
from .v2_node import LogoInjectV2

SITES = "LI_SITES"
INFO = "LI_SITE_INFO"


def to_np(img):  # IMAGE [B,H,W,3] 0..1 -> uint8 HxWx3 (first of batch)
    return (img[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def to_img(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0)[None]


def to_mask(m):
    return torch.from_numpy(m.astype(np.float32))[None]


class LogoInjectFindSites:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "stage1": ("IMAGE",),
                             "max_sites": ("INT", {"default": 4, "min": 1, "max": 12}),
                             "threshold": ("INT", {"default": 25, "min": 5, "max": 120})}}
    RETURN_TYPES = ("IMAGE", SITES, "INT", "STRING")
    RETURN_NAMES = ("stage1_full", "sites", "count", "report")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def run(self, original, stage1, max_sites, threshold):
        import cv2
        o = to_np(original); s1 = to_np(stage1)
        sites, dm = C.find_sites(o, s1, max_sites=max_sites, thresh=threshold)  # compared at stage-1 res
        # stage 1 runs at ~1 MP: stage 2 works (and outputs) at the input resolution
        s = cv2.resize(s1, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        payload = {"sites": sites, "original": o, "stage1": s, "change": dm}
        rep = json.dumps([{k: v for k, v in x.items() if k != "bbox"} for x in sites])
        return (to_img(s), payload, len(sites), rep)


class LogoInjectSiteCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sites": (SITES,), "index": ("INT", {"default": 0, "min": 0, "max": 11}),
                             "crop_factor": ("FLOAT", {"default": 3.5, "min": 2.0, "max": 8.0, "step": 0.1}),
                             "size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64})}}
    RETURN_TYPES = ("IMAGE", "MASK", INFO)
    RETURN_NAMES = ("crop", "erase_mask", "info")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def run(self, sites, index, crop_factor, size):
        if index >= len(sites["sites"]):
            return (torch.zeros(1, 64, 64, 3), torch.zeros(1, 64, 64), {"valid": False})
        c = C.crop_site(sites["stage1"], sites["original"], sites["sites"][index], sites["change"], crop_factor, size)
        if not c["valid"]:  # no badge-like markings at this site: pass through, never draw a logo there
            return (torch.zeros(1, 64, 64, 3), torch.zeros(1, 64, 64), {"valid": False})
        info = {"valid": True, "box": c["box"], "footprint": c["footprint"], "marks": c["marks"],
                "erase_mask": c["erase_mask"]}
        return (to_img(c["stage1_crop"]), to_mask(c["erase_mask"]), info)


class LogoInjectBadgeGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"erased": ("IMAGE",), "logo": ("IMAGE",), "info": (INFO,),
                             "badge_body": (["open", "solid"], {"default": "open"}),
                             "hub_centering": (["auto", "off"], {"default": "auto"})}}
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("guide", "chrome_mask", "report")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def run(self, erased, logo, info, badge_body, hub_centering):
        if not info.get("valid"):
            return (erased, torch.zeros(1, erased.shape[1], erased.shape[2]), "no site")
        g, m, meta = C.badge_guide(to_np(erased), info["footprint"], info["marks"], to_np(logo),
                                   body=badge_body, hub_mode=hub_centering)
        if g is None:  # fit rejected -> keep stage 1's badge (empty chrome mask = nothing pasted)
            info["valid"] = False
            return (erased, torch.zeros(1, erased.shape[1], erased.shape[2]), json.dumps({"rejected": meta["rejected"]}))
        info["is_hub"] = bool(meta["hub"])
        rep = json.dumps({"hub": meta["hub"], "stroke_fit": meta["stroke_fit"]})
        return (to_img(g), to_mask(m), rep)


class LogoInjectChromePrompt:
    """Picks the chrome-pass wording (body vs wheel cap) after the guide node has decided which it is."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"info": (INFO,), "report": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Logo Inject"
    BODY = ("Turn the flat logo {where} into a real, physical polished chrome emblem: bevelled raised edges, crisp "
            "metal reflections and shading that match the scene's lighting, a soft contact shadow. Keep the emblem "
            "EXACTLY where it is, with exactly the same shape, size, angle and perspective, and the same design as "
            "the reference logo. Keep it bright and clearly readable. Change nothing else.")

    def run(self, info, report):
        hub = '"hub": {' in report
        return (self.BODY.format(where="on the wheel's centre cap" if hub else "on the car's body"),)


class LogoInjectPasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "crop": ("IMAGE",), "info": (INFO,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def run(self, image, crop, info):
        if not info.get("valid"):
            return (image,)
        return (to_img(C.paste_back(to_np(image), to_np(crop), info["box"], info["erase_mask"])),)


NODE_CLASS_MAPPINGS = {
    "LogoInjectFindSites": LogoInjectFindSites,
    "LogoInjectSiteCrop": LogoInjectSiteCrop,
    "LogoInjectBadgeGuide": LogoInjectBadgeGuide,
    "LogoInjectChromePrompt": LogoInjectChromePrompt,
    "LogoInjectPasteBack": LogoInjectPasteBack,
    "LogoInjectDetailPass": LogoInjectDetailPass,
    "LogoInjectSpots": LogoInjectSpots,
    "LogoInjectSpotStage1": LogoInjectSpotStage1,
    "LogoInjectDevDump": LogoInjectDevDump,
    "LogoInjectV2": LogoInjectV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LogoInjectFindSites": "Logo Inject - Find Badge Sites",
    "LogoInjectSiteCrop": "Logo Inject - Crop Site",
    "LogoInjectBadgeGuide": "Logo Inject - Badge Guide",
    "LogoInjectChromePrompt": "Logo Inject - Chrome Prompt",
    "LogoInjectPasteBack": "Logo Inject - Paste Back",
    "LogoInjectDetailPass": "Logo Inject - Badge Detail Pass (all badges, one node)",
    "LogoInjectSpots": "Logo Inject - Badge Spots (optional painted mask)",
    "LogoInjectSpotStage1": "Logo Inject - Spot Stage 1 (per painted spot, crop + stitch)",
    "LogoInjectV2": "Logo Inject v2 - painted spots: erase, exact logo in perspective, real badge",
}
