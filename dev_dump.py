# Made by Yovan Vujacic
"""Dev only: save a raw IMAGE tensor (e.g. Depth Anything 3 raw depth) + a string to disk as .npy/.txt for offline work."""
import os
import numpy as np


class LogoInjectDevDump:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "text": ("STRING", {"forceInput": True}),
                             "path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Logo Inject/dev"

    def run(self, image, text, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + ".npy", image[0].cpu().numpy().astype(np.float32))
        open(path + ".txt", "w", encoding="utf-8").write(text)
        return {}
