"""Resize, aspect-preserving fit, placement, and perspective warp."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.templates import PixelRect


def fit_design_to_canvas(
    design_rgba: np.ndarray,
    canvas_h: int,
    canvas_w: int,
    region: PixelRect,
) -> np.ndarray:
    """Scale the design with object-fit: cover and center it on ``region``.

    Aspect ratio is preserved. Excess is cropped. The result is a full-canvas
    RGBA image; pixels outside the region's bounding box are transparent.
    Rounded clipping is applied later via the printable mask.
    """
    if design_rgba.ndim != 3 or design_rgba.shape[2] != 4:
        raise ValueError("Design must be an HxWx4 RGBA array")

    box_x = int(round(region.x))
    box_y = int(round(region.y))
    box_w = max(1, int(round(region.w)))
    box_h = max(1, int(round(region.h)))

    fitted = object_fit_cover(design_rgba, box_w, box_h)

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    x0 = max(0, box_x)
    y0 = max(0, box_y)
    x1 = min(canvas_w, box_x + box_w)
    y1 = min(canvas_h, box_y + box_h)
    if x1 <= x0 or y1 <= y0:
        return canvas

    src_x0 = x0 - box_x
    src_y0 = y0 - box_y
    canvas[y0:y1, x0:x1] = fitted[src_y0 : src_y0 + (y1 - y0), src_x0 : src_x0 + (x1 - x0)]
    return canvas


def fit_design_to_quad(
    design_rgba: np.ndarray,
    canvas_h: int,
    canvas_w: int,
    dst_quad: np.ndarray,
) -> np.ndarray:
    """Object-fit cover the design, then homography-warp onto ``dst_quad``.

    Aspect ratio is preserved. Excess is cropped before the warp. Rounded
    clipping is applied later via the printable mask — not a rectangle crop.
    """
    if design_rgba.ndim != 3 or design_rgba.shape[2] != 4:
        raise ValueError("Design must be an HxWx4 RGBA array")

    quad = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2)
    width_top = float(np.linalg.norm(quad[1] - quad[0]))
    width_bot = float(np.linalg.norm(quad[2] - quad[3]))
    height_l = float(np.linalg.norm(quad[3] - quad[0]))
    height_r = float(np.linalg.norm(quad[2] - quad[1]))
    target_w = max(1, int(round(max(width_top, width_bot))))
    target_h = max(1, int(round(max(height_l, height_r))))

    fitted = object_fit_cover(design_rgba, target_w, target_h)
    src = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, quad)
    return cv2.warpPerspective(
        fitted,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def object_fit_cover(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Uniform scale so the image fully covers the target, then center-crop."""
    src_h, src_w = image.shape[:2]
    if src_w < 1 or src_h < 1:
        return np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

    scale = max(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = high_quality_resize(image, new_w, new_h)

    x0 = max(0, (new_w - target_w) // 2)
    y0 = max(0, (new_h - target_h) // 2)
    cropped = resized[y0 : y0 + target_h, x0 : x0 + target_w]
    # Guard against rounding that yields 1px short.
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        h = min(target_h, cropped.shape[0])
        w = min(target_w, cropped.shape[1])
        padded[:h, :w] = cropped[:h, :w]
        return padded
    return cropped


def high_quality_resize(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """OpenCV resize: AREA when shrinking, LANCZOS4 when enlarging."""
    src_h, src_w = image.shape[:2]
    if new_w == src_w and new_h == src_h:
        return image
    interpolation = cv2.INTER_AREA if (new_w < src_w or new_h < src_h) else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
