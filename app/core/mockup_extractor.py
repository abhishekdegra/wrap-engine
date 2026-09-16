"""Foreground extraction engine for phone mockups.

Isolates the complete phone/case silhouette (including rounded corners, camera area,
buttons, transparent/colored rails, and printed artwork) from its studio background
with sub-pixel edge matting and de-fringing (zero white halo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class ExtractedPhone:
    """Isolated phone foreground ready for commercial tabletop compositing."""

    rgba: np.ndarray  # H x W x 4 (uint8) tightly cropped with clean alpha
    mask: np.ndarray  # H x W (float32 in 0..1)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original composite space
    aspect_ratio: float  # w / h
    original_size: Tuple[int, int]  # (width, height) of source image


def defringe_edge_pixels(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    alpha_low: float = 0.02,
    alpha_high: float = 0.98,
) -> np.ndarray:
    """Remove background color halo from semi-transparent edge pixels.
    
    Uses physical color de-contamination:
        C_true = (C_observed - (1 - alpha) * C_bg) / alpha
    """
    clean_rgb = rgb.astype(np.float32).copy()
    bg = np.array(bg_color, dtype=np.float32)

    # Edge mask where alpha is semi-transparent
    edge_mask = (alpha > alpha_low) & (alpha < alpha_high)
    if not np.any(edge_mask):
        return np.clip(clean_rgb, 0, 255).astype(np.uint8)

    a = alpha[edge_mask, np.newaxis]
    c_obs = clean_rgb[edge_mask]

    # De-contaminate
    c_pure = (c_obs - (1.0 - a) * bg) / np.maximum(a, 0.01)
    clean_rgb[edge_mask] = np.clip(c_pure, 0.0, 255.0)

    # Completely zero out color where alpha <= alpha_low
    zero_mask = alpha <= alpha_low
    clean_rgb[zero_mask] = 0.0

    return np.clip(clean_rgb, 0, 255).astype(np.uint8)


def extract_phone_foreground(
    composite_rgba: np.ndarray,
    outer_mask: np.ndarray | None = None,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    padding_px: int = 4,
) -> ExtractedPhone:
    """Isolates the phone + case from the background and returns an ExtractedPhone.
    
    Args:
        composite_rgba: Input image (H x W x 3 or H x W x 4).
        outer_mask: Precomputed subpixel outer mask of the phone silhouette (float 0..1 or uint8 0..255).
                    If None, automatic segmentation is performed.
        bg_color: Background color to de-fringe against (default white studio: 255, 255, 255).
        padding_px: Extra padding around the tight bounding box.
    """
    orig_h, orig_w = composite_rgba.shape[:2]

    # Separate RGB and existing Alpha
    if composite_rgba.shape[2] == 4:
        rgb = composite_rgba[:, :, :3]
        in_alpha = composite_rgba[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = composite_rgba
        in_alpha = np.ones((orig_h, orig_w), dtype=np.float32)

    # Determine the silhouette mask
    if outer_mask is not None and outer_mask.size > 0:
        if outer_mask.dtype == np.uint8:
            mask = outer_mask.astype(np.float32) / 255.0
        else:
            mask = outer_mask.astype(np.float32).copy()

        # Resize mask if dimensions don't match composite exactly
        if mask.shape[:2] != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # If composite already has transparency, combine both
        if composite_rgba.shape[2] == 4:
            mask = np.minimum(mask, in_alpha)
    elif composite_rgba.shape[2] == 4 and np.any(in_alpha < 0.99):
        # Already has transparency and no separate outer_mask provided
        mask = in_alpha
    else:
        # Fallback: Automatic background isolation
        mask = _segment_phone_from_background(rgb, bg_color)

    # Clean sub-pixel alpha: smooth gentle anti-aliasing on binary threshold if needed
    mask = np.clip(mask, 0.0, 1.0)

    # De-fringe RGB against background color
    clean_rgb = defringe_edge_pixels(rgb, mask, bg_color=bg_color)

    # Compute tight bounding box
    active_y, active_x = np.where(mask > 0.02)
    if len(active_y) == 0 or len(active_x) == 0:
        # Fallback to full frame if empty
        x0, y0, x1, y1 = 0, 0, orig_w, orig_h
    else:
        x0 = max(0, int(np.min(active_x)) - padding_px)
        y0 = max(0, int(np.min(active_y)) - padding_px)
        x1 = min(orig_w, int(np.max(active_x)) + 1 + padding_px)
        y1 = min(orig_h, int(np.max(active_y)) + 1 + padding_px)

    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)

    cropped_rgb = clean_rgb[y0:y1, x0:x1]
    cropped_alpha = mask[y0:y1, x0:x1]

    # Combine into 4-channel RGBA uint8
    cropped_rgba = np.dstack([
        cropped_rgb,
        (cropped_alpha * 255.0 + 0.5).clip(0, 255).astype(np.uint8),
    ])

    aspect = float(crop_w) / float(crop_h)

    return ExtractedPhone(
        rgba=cropped_rgba,
        mask=cropped_alpha,
        bbox=(x0, y0, crop_w, crop_h),
        aspect_ratio=aspect,
        original_size=(orig_w, orig_h),
    )


def _segment_phone_from_background(
    rgb: np.ndarray,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Robust fallback segmentation for unseen phone covers on uniform studio backgrounds."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.shape[2] == 3 else rgb

    # Distance to background color
    bg = np.array(bg_color, dtype=np.float32)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)

    # Otsu or adaptive threshold on distance
    dist_u8 = np.clip(dist * (255.0 / (dist.max() + 1e-5)), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Find largest contour in the center
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    # Subpixel anti-aliasing feathering
    feathered = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 1.2)
    return np.clip(feathered, 0.0, 1.0)
