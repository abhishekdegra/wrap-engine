"""Artwork inside the printable mask — original colors, no cover haze.

The cover is used for geometry. Printable pixels show the uploaded design
unchanged. Cover pixels are kept only where the print mask is zero
(camera, bumper, background).
"""

from __future__ import annotations

import numpy as np

from app.utils.constants import COVER_OVERLAY_STRENGTH


def composite_design_under_cover(
    cover_rgba: np.ndarray,
    design_canvas_rgba: np.ndarray,
    print_mask: np.ndarray,
    cover_overlay: float = COVER_OVERLAY_STRENGTH,
) -> np.ndarray:
    """Lerp cover → design using the antialiased print mask only.

    ``cover_overlay`` is unused unless explicitly > 0 (optional later effect).
    """
    cover = cover_rgba.astype(np.float32) / 255.0
    design = design_canvas_rgba.astype(np.float32) / 255.0
    mask = _as_float_mask(print_mask)

    design_rgb = design[..., :3]
    design_a = np.clip(design[..., 3], 0.0, 1.0)
    cover_rgb = cover[..., :3]
    cover_a = np.clip(cover[..., 3], 0.0, 1.0)

    overlay = float(np.clip(cover_overlay, 0.0, 1.0))
    if overlay > 0.0:
        mix = overlay * mask
        design_rgb = design_rgb * (1.0 - mix[..., None]) + cover_rgb * mix[..., None]

    art_w = np.clip(mask * design_a, 0.0, 1.0)
    keep = 1.0 - art_w

    out_rgb = design_rgb * art_w[..., None] + cover_rgb * keep[..., None]
    out_a = np.clip(art_w + cover_a * keep, 0.0, 1.0)

    out = np.concatenate([out_rgb, out_a[..., None]], axis=2)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_mask_to_alpha(image_rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply image alpha by an antialiased mask (no RGB halo)."""
    out = image_rgba.copy()
    m = _as_float_mask(mask)
    alpha = out[..., 3].astype(np.float32) * m
    out[..., 3] = np.clip(alpha + 0.5, 0, 255).astype(np.uint8)
    return out


def _as_float_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.0 + 1e-3:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)
