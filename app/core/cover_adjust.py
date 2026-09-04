"""Cover-look adjustments for mockup display/export.

Applied to a *copy* of the cover photo only. Uploaded artwork is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np


@dataclass
class CoverLook:
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    clarity: float = 0.0
    saturation: float = 0.0
    exposure: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    opacity: float = 1.0

    def is_identity(self) -> bool:
        for item in fields(self):
            value = float(getattr(self, item.name))
            default = float(item.default) if item.default is not None else 0.0
            if abs(value - default) > 1e-6:
                return False
        return True


def apply_cover_look(cover_layer_rgba: np.ndarray, look: CoverLook) -> np.ndarray:
    """Return an adjusted copy of the cover/artwork layer. Input array is not modified."""
    if look.is_identity():
        return cover_layer_rgba
    img = cover_layer_rgba.astype(np.float32) / 255.0
    rgb = np.clip(img[..., :3], 0.0, 1.0)
    alpha = np.clip(img[..., 3], 0.0, 1.0)

    rgb = rgb * (2.0 ** float(look.exposure))
    rgb = rgb + float(look.brightness)
    rgb = (rgb - 0.5) * (1.0 + float(look.contrast)) + 0.5

    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    if abs(look.highlights) > 1e-6:
        w = np.clip((luma - 0.45) / 0.55, 0.0, 1.0)[..., None]
        rgb = rgb + w * float(look.highlights) * 0.45
    if abs(look.shadows) > 1e-6:
        w = np.clip((0.55 - luma) / 0.55, 0.0, 1.0)[..., None]
        rgb = rgb + w * float(look.shadows) * 0.45

    if abs(look.saturation) > 1e-6:
        gray = luma[..., None]
        sat = 1.0 + float(look.saturation)
        rgb = gray + (rgb - gray) * sat

    rgb = np.clip(rgb, 0.0, 1.0)
    h, w = rgb.shape[:2]
    min_side = float(max(8, min(h, w)))
    if look.sharpness > 1e-6:
        sigma = max(0.6, 0.0018 * min_side)
        blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
        rgb = np.clip(rgb + float(look.sharpness) * (rgb - blur), 0.0, 1.0)
    if look.clarity > 1e-6:
        sigma = max(2.2, 0.012 * min_side)
        blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
        rgb = np.clip(rgb + float(look.clarity) * 0.85 * (rgb - blur), 0.0, 1.0)

    opacity = float(np.clip(look.opacity, 0.0, 1.0))
    alpha = alpha * opacity

    # Guarantee transparent pixels in the cover/artwork layer remain strictly transparent
    zero_mask = alpha < 1e-4
    if np.any(zero_mask):
        rgb[zero_mask] = 0.0
        alpha[zero_mask] = 0.0

    out = np.concatenate([rgb, alpha[..., None]], axis=2)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
