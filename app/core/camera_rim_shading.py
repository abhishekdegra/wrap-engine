"""Physical 3D camera-rim realism and relief shading.

Preserves the physical 3D appearance of the raised protective camera-cutout rim
when artwork is composited onto a phone cover.

Extracts original highlights, specular reflections, bevel curvature, and
groove shadows from the cover photo and reapplies them over the wrapped artwork
strictly within the detected camera rim band.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_3d_camera_rim_shading(
    cover_rgb: np.ndarray,
    composited_rgb: np.ndarray,
    print_mask: np.ndarray,
    camera_exclusion_mask: np.ndarray | None,
    artwork_weight: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply realistic 3D lighting, highlights, and shadows of the physical camera rim.

    Parameters
    ----------
    cover_rgb : np.ndarray
        Original phone cover RGB in float32 [0.0, 1.0].
    composited_rgb : np.ndarray
        Composited artwork + phone RGB in float32 [0.0, 1.0].
    print_mask : np.ndarray
        Antialiased printable area mask in float32 [0.0, 1.0].
    camera_exclusion_mask : np.ndarray | None
        Antialiased camera cutout mask in float32 [0.0, 1.0] (1.0 = camera hardware).
    artwork_weight : np.ndarray | None
        Artwork opacity/weight mask in float32 [0.0, 1.0]. Ensures shading is applied
        only over the artwork layer, leaving bare phone cover 100% untouched.
    strength : float
        Overall strength factor in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Enhanced RGB image with physical 3D camera rim relief restored.
    """
    if camera_exclusion_mask is None or camera_exclusion_mask.size == 0:
        return composited_rgb

    cam_m = np.clip(camera_exclusion_mask.astype(np.float32), 0.0, 1.0)
    if cam_m.max() < 0.05:
        return composited_rgb

    mask = np.clip(print_mask.astype(np.float32), 0.0, 1.0)
    cam_bin = cam_m > 0.35
    if not cam_bin.any():
        return composited_rgb

    height, width = cover_rgb.shape[:2]

    # 1. Derive physical rim width dynamically from camera cutout scale
    ys, xs = np.where(cam_bin)
    cam_w = float(xs.max() - xs.min() + 1)
    cam_h = float(ys.max() - ys.min() + 1)
    cam_scale = min(cam_w, cam_h)

    # Physical camera rims typically span ~6% to 10% of the camera island width
    rim_w = float(np.clip(0.082 * cam_scale, 4.0, 42.0))

    # 2. Euclidean distance outward from the camera cutout into the printable area
    dist_out = cv2.distanceTransform((~cam_bin).astype(np.uint8), cv2.DIST_L2, 5)

    # 3. Smooth, antialiased rim band mask
    # Cosine falloff from 1.0 at cutout edge down to 0.0 at rim_w
    u = np.clip(dist_out / rim_w, 0.0, 1.0)
    rim_falloff = 0.5 * (1.0 + np.cos(np.pi * u)) * (dist_out > 0.4)

    # Strictly restrict to printable area, outside camera cutout, and over artwork
    rim_band = rim_falloff * mask * (1.0 - np.clip(cam_m, 0.0, 1.0))
    if artwork_weight is not None:
        art_w_f = np.clip(artwork_weight.astype(np.float32), 0.0, 1.0)
        rim_band = rim_band * art_w_f
    if rim_band.max() < 1e-4:
        return composited_rgb

    # 4. Extract original cover luminance and local background baseline
    L_cover = (
        0.2126 * cover_rgb[..., 0]
        + 0.7152 * cover_rgb[..., 1]
        + 0.0722 * cover_rgb[..., 2]
    )

    # Local background baseline using a spatial blur proportional to rim scale
    sigma_bg = max(3.5, rim_w * 1.6)
    L_base = cv2.GaussianBlur(L_cover, (0, 0), sigmaX=sigma_bg)

    # High-frequency detail and relief (Difference of Gaussians) for crisp glints and bevels
    sigma_detail = max(1.0, rim_w * 0.16)
    dog = L_cover - cv2.GaussianBlur(L_cover, (0, 0), sigmaX=sigma_detail)

    # Lighting delta relative to baseline back panel
    delta = (L_cover - L_base) * rim_band

    str_val = float(np.clip(strength, 0.0, 1.0))
    out = composited_rgb.copy()

    # 5. Inner groove / bevel shadow (Multiply blend)
    # Reconstructs the physical trough where the raised plastic rim steps down to the camera plate
    shadow_groove = np.clip(-delta, 0.0, 1.0)
    # Subtle contact shadow line right along the camera perimeter edge
    cam_contact = cv2.GaussianBlur(cam_m, (0, 0), sigmaX=max(1.2, rim_w * 0.14)) * mask * (1.0 - cam_m)
    shadow_total = np.clip(shadow_groove * 1.15 + cam_contact * 0.32, 0.0, 1.0) * str_val
    shadow_darken = 1.0 - shadow_total * 0.85
    out = out * shadow_darken[..., None]

    # 6. Specular and ridge highlights (Screen / Additive blend)
    # Highlights the outer bevel and reflective ridge of the raised rim
    highlight = np.clip(delta * 1.25 + dog * rim_band * 0.75, 0.0, 1.0) * str_val
    out = 1.0 - (1.0 - out) * (1.0 - highlight[..., None])

    # 7. Subtle physical rim material / glass translucency blend
    # Allows the physical cover material (clear plastic / black rubber / metal) to subtly overlay
    # the artwork, preserving natural material texture without obscuring the design
    material_blend = (rim_band[..., None] * 0.20 * str_val).astype(np.float32)
    out = out * (1.0 - material_blend) + cover_rgb * material_blend

    # 8. Strict protection: inside camera cutout and outside printable area
    keep_cover = np.clip(cam_m, 0.0, 1.0)[..., None]
    final = out * (1.0 - keep_cover) + cover_rgb * keep_cover

    # Outside print mask, guarantee exact cover pixels
    keep_outside = (1.0 - mask)[..., None]
    final = final * (1.0 - keep_outside) + cover_rgb * keep_outside

    return np.clip(final, 0.0, 1.0)
