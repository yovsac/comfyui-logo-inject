# Made by Yovan Vujacic
"""Optional 'badge spots' mask for Logo Inject.

Stage 1 decides where badges go, and that placement is seed-dependent: extra wordmarks, a blank licence plate, badges
on wheel-arch flares, a garbled grille side badge (seed tests, 2026-10-01). The user can paint where the badges
belong on the car image (Workflow Studio's mask brush: white strokes on black). This node turns that drawing into
the mask that limits stage 1 (noise mask + composite back onto the original) and stage 2 (detections outside are
ignored).

No drawing = Workflow Studio's default empty.png, which reads as all-white -> a full mask -> the tool behaves exactly
as it did without this node.
"""
import cv2
import numpy as np
import torch


class LogoInjectSpots:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "the car at the size stage 1 works at"}),
            "spots": ("IMAGE", {"tooltip": "painted badge spots: white on black, any size (empty / all white = whole car)"}),
            "grow": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005,
                               "tooltip": "grow each painted spot by this share of the image's short side, so a rough "
                                          "stroke still gives stage 1 room for the whole badge"}),
        }}
    RETURN_TYPES = ("MASK", "BOOLEAN")
    RETURN_NAMES = ("mask", "painted")
    FUNCTION = "run"
    CATEGORY = "Logo Inject"

    def run(self, image, spots, grow):
        H, W = image.shape[1], image.shape[2]
        m = spots[0, ..., 0].cpu().numpy()  # red channel
        if m.mean() > 0.98 or m.max() < 0.5:  # nothing painted (empty.png is white; an all-black drawing is no drawing)
            return (torch.ones(1, H, W), False)
        m = cv2.resize((m > 0.5).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        g = max(1, int(grow * min(H, W)))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, g / 2)).astype(np.float32) / 255
        return (torch.from_numpy(m)[None], True)
