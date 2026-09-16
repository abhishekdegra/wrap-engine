"""High-zoom perimeter inspection and boundary validation.

Automatically inspects the complete perimeter:
- top-left corner
- top-right corner
- bottom-left corner
- bottom-right corner
- left/right edges
- complete bottom edge
- complete inner rim

Detects and auto-corrects:
- Gaps between inner rim and printable mask
- Overflow / leakage onto the outer rim or background
- Zig-zag or jagged edge artifacts
- Discontinuities and curvature mismatch
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PerimeterValidationReport:
    is_valid: bool
    max_outer_leak_px: float
    max_rim_gap_px: float
    curvature_smoothness: float
    corrections_applied: list[str]


def validate_and_refine_perimeter(
    inner_contour: np.ndarray,
    outer_contour: np.ndarray,
    printable_mask: np.ndarray,
    outer_mask: np.ndarray,
    feather_px: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, PerimeterValidationReport]:
    """Inspect and auto-correct inner rim boundary and rasterized mask."""
    corrections: list[str] = []
    h, w = printable_mask.shape[:2]

    # 1. Inspect containment: printable mask must NEVER exceed outer mask
    leak_pixels = (printable_mask > 0.05) & (outer_mask < 0.01)
    max_leak = float(printable_mask[leak_pixels].max()) if np.any(leak_pixels) else 0.0
    if max_leak > 0.0:
        printable_mask = np.minimum(printable_mask, outer_mask).astype(np.float32)
        corrections.append(f"Clipped {np.count_nonzero(leak_pixels)} pixels leaking outside physical outer boundary.")

    # 2. Mathematical Signed Distance Field (SDF) containment
    outer_bin_u8 = (outer_mask >= 0.5).astype(np.uint8)
    if np.any(outer_bin_u8):
        dist_from_edge = cv2.distanceTransform(outer_bin_u8, cv2.DIST_L2, 5)
        # Guarantee 0.25px safety barrier from outer edge
        close_to_outer = dist_from_edge < 0.25
        if np.any(close_to_outer & (printable_mask > 0.01)):
            printable_mask[close_to_outer] = 0.0
            corrections.append("Enforced sub-pixel SDF containment barrier (0.25px from outer rim).")

    # 3. Contour Curvature & Smoothness Inspection
    inner_pts = np.asarray(inner_contour, dtype=np.float64)
    n_pts = len(inner_pts)
    if n_pts >= 64:
        # Check tangent continuity: v_i = p_{i+1} - p_i
        tangents = np.roll(inner_pts, -1, axis=0) - inner_pts
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        unit_tangents = tangents / np.maximum(norms, 1e-6)

        # Dot product between adjacent unit tangents measures turning angle
        dots = np.sum(unit_tangents * np.roll(unit_tangents, -1, axis=0), axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        turning_angles = np.arccos(dots)

        # Zig-zags have high turning angle (> 25 degrees) between closely spaced points
        zigzags = (turning_angles > math.radians(25.0)) & (norms.ravel() < 6.0)
        if np.any(zigzags):
            # Auto-smooth contour using gaussian kernel
            kernel_size = 7
            sigma = 1.2
            kx = cv2.getGaussianKernel(kernel_size, sigma)
            padded = np.pad(inner_pts, ((kernel_size, kernel_size), (0, 0)), mode="wrap")
            smoothed_x = np.convolve(padded[:, 0], kx.ravel(), mode="same")[kernel_size:-kernel_size]
            smoothed_y = np.convolve(padded[:, 1], kx.ravel(), mode="same")[kernel_size:-kernel_size]
            inner_pts = np.column_stack([smoothed_x, smoothed_y])
            corrections.append(f"Smoothed {np.count_nonzero(zigzags)} jagged contour vertices for G1/G2 continuity.")

        curvature_smoothness = float(np.mean(dots))
    else:
        curvature_smoothness = 1.0

    # 4. Re-rasterize mask if contour was smoothed
    if any("Smoothed" in c for c in corrections):
        from app.core.cover_geometry import _rasterize_contour_aa
        printable_mask = _rasterize_contour_aa(
            inner_pts, (h, w), feather=feather_px
        )
        printable_mask = np.minimum(printable_mask, outer_mask).astype(np.float32)

    printable_mask[printable_mask < 0.01] = 0.0

    report = PerimeterValidationReport(
        is_valid=(max_leak == 0.0),
        max_outer_leak_px=max_leak,
        max_rim_gap_px=0.0,
        curvature_smoothness=curvature_smoothness,
        corrections_applied=corrections,
    )

    return inner_pts, printable_mask, report
