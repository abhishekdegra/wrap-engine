"""Artwork inside the printable mask — original colors, no cover haze.

The cover is used for geometry. Printable pixels show the uploaded design
unchanged. Cover pixels are kept only where the print mask is zero
(camera, bumper, background).
"""

from __future__ import annotations

import numpy as np

from app.utils.constants import COVER_OVERLAY_STRENGTH, SURFACE_LIGHTING_STRENGTH


def composite_design_under_cover(
    cover_rgba: np.ndarray,
    design_canvas_rgba: np.ndarray,
    print_mask: np.ndarray,
    cover_overlay: float = COVER_OVERLAY_STRENGTH,
    surface_lighting: float = SURFACE_LIGHTING_STRENGTH,
    camera_exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Lerp cover → design using the antialiased print mask only.

    Applies realistic surface lighting (reflections/highlights from cover material)
    via soft-light blending when ``surface_lighting`` > 0.

    When ``camera_exclusion_mask`` is provided, adds a subtle darkening band
    along the camera rim edge to simulate the physical shadow where the
    printed skin wraps around the rim.
    """
    cover = cover_rgba.astype(np.float32) / 255.0
    design = design_canvas_rgba.astype(np.float32) / 255.0
    mask = _as_float_mask(print_mask)

    design_rgb = design[..., :3]
    design_a = np.clip(design[..., 3], 0.0, 1.0)
    cover_rgb = cover[..., :3]
    cover_a = np.clip(cover[..., 3], 0.0, 1.0)

    # Realistic surface lighting extraction & blend
    lighting_str = float(np.clip(surface_lighting, 0.0, 1.0))
    if lighting_str > 0.0:
        cover_lum = 0.2126 * cover_rgb[..., 0] + 0.7152 * cover_rgb[..., 1] + 0.0722 * cover_rgb[..., 2]
        m_idx = mask > 0.5
        if m_idx.any():
            med_lum = float(np.median(cover_lum[m_idx]))
            if med_lum > 1e-3:
                light_map = np.clip(cover_lum / (2.0 * med_lum), 0.0, 1.0)
            else:
                light_map = cover_lum
        else:
            light_map = np.full_like(cover_lum, 0.5)
        design_rgb = _apply_soft_light(design_rgb, light_map, lighting_str)

        # Subtle rim shadow along the artwork edge wrapping around the camera island.
        if camera_exclusion_mask is not None:
            cam_m = _as_float_mask(camera_exclusion_mask)
            if cam_m.max() > 0.01:
                import cv2
                cam_blur = cv2.GaussianBlur(cam_m, (0, 0), sigmaX=2.5)
                rim_shadow = np.clip(cam_blur * mask, 0.0, 1.0)
                darken = 1.0 - rim_shadow * lighting_str * 0.35
                design_rgb = design_rgb * darken[..., None]

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


def _apply_soft_light(design_rgb: np.ndarray, light_map: np.ndarray, strength: float) -> np.ndarray:
    blend = light_map[..., None]
    low = design_rgb - (1.0 - 2.0 * blend) * design_rgb * (1.0 - design_rgb)
    high = design_rgb + (2.0 * blend - 1.0) * (np.sqrt(np.maximum(design_rgb, 0.0)) - design_rgb)
    sl = np.where(blend <= 0.5, low, high)
    return np.clip(design_rgb * (1.0 - strength) + sl * strength, 0.0, 1.0)


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
