"""Camera-module exclusion: full island contour, not just lenses.

Low-confidence detections are discarded so a bad mask is never applied.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.mask_generator import dilate_binary, fill_binary_holes, smooth_silhouette
from app.utils.constants import CAMERA_MIN_CONFIDENCE


def detect_camera(
    cover_rgba: np.ndarray,
    outer_filled: np.ndarray,
    back_panel: np.ndarray,
    safety_px: float,
) -> tuple[np.ndarray, bool, float, list[str]]:
    """Return (mask, found, confidence, warnings). Empty mask if uncertain."""
    warnings: list[str] = []
    height, width = cover_rgba.shape[:2]
    alpha = cover_rgba[..., 3]
    rgb = cover_rgba[..., :3]
    outer = outer_filled.astype(bool)
    back = back_panel.astype(bool)
    has_alpha = int(alpha.max()) >= 12 and int(np.percentile(alpha, 5)) < 240

    mask = np.zeros((height, width), dtype=bool)
    confidence = 0.0

    if has_alpha:
        holes = _holes_inside_cover(alpha, outer, height, width)
        island = _opaque_island(alpha, back, height, width)
        if holes.any() or island.any():
            combined = holes | island
            if holes.any():
                combined = combined | _filled_contour_around(holes)
            if _plausible_camera_island(combined, back):
                mask, confidence = combined, 0.82 if (holes.any() and island.any()) else 0.7

    if not mask.any():
        mask, confidence = _rgb_camera_module(rgb, back, height, width)

    if not mask.any() or confidence < CAMERA_MIN_CONFIDENCE:
        return (
            np.zeros((height, width), dtype=bool),
            False,
            float(confidence),
            warnings,
        )

    mask = fill_binary_holes(mask) & back
    mask = _largest_component(mask)
    mask = _smooth_island_contour(mask)
    mask = _largest_component(mask)
    mask = _regularize_camera_island(mask, back)
    mask = _largest_component(mask)
    if not _plausible_camera_island(mask, back):
        return np.zeros((height, width), dtype=bool), False, 0.0, warnings

    mask = dilate_binary(mask, safety_px) & back

    back_area = max(float(back.mean()), 1e-6)
    if float(mask.mean()) / back_area > 0.16:
        return np.zeros((height, width), dtype=bool), False, 0.0, warnings
    if not _plausible_camera_island(mask, back):
        return np.zeros((height, width), dtype=bool), False, 0.0, warnings

    return mask, True, float(confidence), warnings


def _plausible_camera_island(mask: np.ndarray, back: np.ndarray) -> bool:
    """Reject horizontal strips, sparse bites, and blobs that are not a module."""
    if mask is None or not mask.any() or not back.any():
        return False
    island = mask.astype(bool) & back.astype(bool)
    if not island.any():
        return False
    bys, bxs = np.where(back)
    y0, y1 = int(bys.min()), int(bys.max())
    x0, x1 = int(bxs.min()), int(bxs.max())
    bh = max(1, y1 - y0)
    bw = max(1, x1 - x0)
    back_area = float(np.count_nonzero(back))

    ys, xs = np.where(island)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    area = float(np.count_nonzero(island))
    aspect = max(w, h) / float(max(min(w, h), 1))
    extent = area / float(max(w * h, 1))
    cy = float(ys.mean())
    cx = float(xs.mean())
    frac = area / max(back_area, 1.0)
    frac_x = (cx - x0) / float(bw)
    frac_y = (cy - y0) / float(bh)

    if aspect > 2.05:
        return False
    if w > 0.50 * bw or h > 0.42 * bh:
        return False
    if frac < 0.0035 or frac > 0.145:
        return False
    if extent < 0.50:
        return False
    if frac_y > 0.40:
        return False
    if h < 0.06 * bh and w > 0.28 * bw:
        return False
    if w > 0.45 * bw and aspect > 1.9:
        return False
    if 0.42 < frac_x < 0.58:
        return False
    return True


def _island_extent(mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return 0.0
    ys, xs = np.where(mask)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    return float(np.count_nonzero(mask)) / float(max(w * h, 1))


def _regularize_camera_island(mask: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Replace a jagged Canny island with the compact module fitted to those pixels."""
    if mask is None or not mask.any():
        return mask
    island = mask.astype(bool) & back.astype(bool)
    if not island.any():
        return mask
    ys, xs = np.where(island)
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if pts.shape[0] < 12:
        return island
    (_cx, _cy), (rw, rh), ang = cv2.minAreaRect(pts)
    rw = max(float(rw), 4.0) * 1.06
    rh = max(float(rh), 4.0) * 1.06
    box = cv2.boxPoints(((_cx, _cy), (rw, rh), ang)).astype(np.int32)
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, box, 1)
    rad = max(3, int(round(0.22 * min(rw, rh))))
    if rad % 2 == 0:
        rad += 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad)))
    fitted = fill_binary_holes(out.astype(bool)) & back
    if not fitted.any():
        return island
    # Keep the fit only if it still covers the detected optics/island core.
    overlap = float(np.count_nonzero(fitted & island)) / max(float(np.count_nonzero(island)), 1.0)
    if overlap < 0.82:
        return island
    return fitted


def _smooth_island_contour(mask: np.ndarray) -> np.ndarray:
    """Close lens/flash gaps; keep one compact island (no strip expansion)."""
    if not mask.any():
        return mask
    ys, xs = np.where(mask)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    min_side = float(max(8, min(w, h)))
    k = max(5, int(round(0.08 * min_side)))
    k = min(k, max(5, int(0.22 * min_side)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    closed = fill_binary_holes(closed.astype(bool))
    # Convex hull only if it stays a compact module, not a giant bite.
    hull = _filled_contour_around(closed)
    if hull.any():
        if float(np.count_nonzero(hull)) <= 1.45 * max(float(np.count_nonzero(closed)), 1.0):
            closed = hull
    return smooth_silhouette(closed, sigma=max(1.0, 0.012 * min_side))


def _holes_inside_cover(
    alpha: np.ndarray,
    outer: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if int(alpha.max()) < 8:
        return np.zeros((height, width), dtype=bool)
    filled = outer.astype(np.uint8)
    holes = (alpha < 18) & (filled > 0)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    holes = holes & (dist > 8)
    return _keep_blobs(holes, height, width, min_frac=0.0002, max_frac=0.04, top_frac=0.5, region=outer)


def _opaque_island(
    alpha: np.ndarray,
    back: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if not back.any() or int(alpha.max()) < 8:
        return np.zeros((height, width), dtype=bool)
    median = float(np.median(alpha[back]))
    thresh = max(median + 55.0, 120.0)
    if thresh >= 250:
        return np.zeros((height, width), dtype=bool)
    raw = (alpha >= thresh) & back
    island = _keep_blobs(raw, height, width, min_frac=0.004, max_frac=0.12, top_frac=0.42, region=back)
    if island.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        island = cv2.morphologyEx(island.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return island


def _rgb_camera_module(
    rgb: np.ndarray,
    back: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, float]:
    """Find the full camera housing (lenses + flash + sensors) as one island."""
    if not back.any():
        return np.zeros((height, width), dtype=bool), 0.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(back)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh = max(1, y1 - y0)
    bw = max(1, x1 - x0)
    back_area = float(np.count_nonzero(back))
    min_side = float(min(bw, bh))

    top = np.zeros_like(back)
    top_h = int(0.38 * bh)
    top[y0 : y0 + top_h, x0:x1] = back[y0 : y0 + top_h, x0:x1]

    lenses_raw = _find_lenses(gray, top, back_area, min_side)
    hough = _hough_lenses(gray, top, min_side)
    lenses, n_lenses = _cluster_optics(lenses_raw, hough, top, back, min_side)
    if n_lenses >= 1:
        extra_roi = _optics_in_island_roi(gray, lenses, top, back, back_area, min_side)
        if extra_roi.any():
            lenses = fill_binary_holes(lenses | extra_roi)
            n_lenses = max(
                n_lenses,
                int(cv2.connectedComponents(extra_roi.astype(np.uint8), connectivity=8)[0] - 1),
            )

    ranked: list[tuple[float, np.ndarray]] = []

    boxed = _rounded_box_around_lenses(lenses, top, back, min_side)
    if n_lenses >= 1 and _plausible_camera_island(boxed, back):
        ranked.append((0.98 if n_lenses >= 2 else 0.72, boxed))

    if n_lenses >= 2:
        clustered = _island_from_lenses(lenses, top, back, back_area, min_side)
        if _plausible_camera_island(clustered, back):
            ranked.append((0.90, clustered))

    if n_lenses >= 1:
        around = _island_around_lenses(gray, lenses, top, back, back_area, min_side)
        if _plausible_camera_island(around, back):
            ranked.append((0.80 if n_lenses >= 2 else 0.66, around))

    housing = _dark_housing(gray, top, back, back_area, y0, bh, min_side)
    if _plausible_camera_island(housing, back):
        lens_cover = float(np.count_nonzero(housing & lenses)) / max(float(np.count_nonzero(lenses)), 1.0)
        if n_lenses >= 2 and lens_cover < 0.85:
            housing = np.zeros_like(housing)
        elif lenses.any():
            housing = fill_binary_holes(housing | (lenses & back))
        if _plausible_camera_island(housing, back):
            ranked.append((0.70 if n_lenses >= 2 else 0.84, housing))

    contour_mask, contour_score = _module_contour(gray, top, back, lenses, y0, bh, back_area)
    if contour_mask.any() and contour_score >= 0.45 and _plausible_camera_island(contour_mask, back):
        conf = max(0.62, float(contour_score))
        if n_lenses >= 2:
            conf = max(0.72, float(contour_score))
        ranked.append((conf, contour_mask))

    if not ranked:
        fallback = _edge_contrast_module(rgb, gray, top, back, back_area, min_side, y0, bh, x0, bw)
        if _plausible_camera_island(fallback, back):
            ranked.append((0.64, fallback))

    if not ranked:
        return np.zeros((height, width), dtype=bool), 0.0

    def _lens_bonus(m: np.ndarray) -> float:
        if not lenses.any():
            return 0.0
        return 0.08 if np.any(m & lenses) else 0.0

    ranked.sort(key=lambda item: item[0] + _lens_bonus(item[1]), reverse=True)
    return ranked[0][1], ranked[0][0]


def _edge_contrast_module(
    rgb: np.ndarray,
    gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    min_side: float,
    y0: int,
    bh: int,
    x0: int,
    bw: int,
) -> np.ndarray:
    """Light-colored camera island: closed edge loop in a top corner of the back."""
    if not top.any() or not back.any():
        return np.zeros(gray.shape, dtype=bool)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    edges = cv2.Canny(blur, 18, 64)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, grad_bin = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = np.maximum(edges, grad_bin)
    combined[~top] = 0
    combined[~back] = 0
    k = max(5, int(round(0.035 * min_side)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = np.zeros(gray.shape, dtype=bool)
    best_score = -1.0
    min_a = 0.008 * back_area
    max_a = 0.13 * back_area
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_a or area > max_a:
            continue
        peri = float(cv2.arcLength(contour, True))
        if peri < 16:
            continue
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ > 0.92:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / float(max(min(w, h), 1))
        if aspect > 2.05:
            continue
        solidity = area / float(max(w * h, 1))
        if solidity < 0.52:
            continue
        M = cv2.moments(contour)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        fy = (cy - y0) / float(max(bh, 1))
        fx = (cx - x0) / float(max(bw, 1))
        if fy > 0.38:
            continue
        if 0.28 < fx < 0.72:
            continue
        filled = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        blob = filled.astype(bool) & top & back
        if not _plausible_camera_island(blob, back):
            continue
        corner = 0.15 if fx < 0.28 or fx > 0.72 else 0.0
        score = solidity + corner - 0.15 * abs(aspect - 1.15)
        if score > best_score:
            best_score = score
            best = blob
    return fill_binary_holes(best) if best.any() else best


def _cluster_optics(
    dark_lenses: np.ndarray,
    hough: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
) -> np.ndarray:
    """Keep only optics that form one compact corner cluster (ignore glare circles)."""
    seeds = dark_lenses.astype(bool) & top & back
    extra = hough.astype(bool) & top & back
    if seeds.any():
        dist = cv2.distanceTransform((~seeds).astype(np.uint8), cv2.DIST_L2, 5)
        near = extra & (dist <= max(8.0, 0.14 * min_side))
        clustered = seeds | near
    elif extra.any():
        clustered = extra
        if not _compact_optics_bbox(_largest_component(clustered), min_side):
            clustered = np.zeros_like(extra)
    else:
        return np.zeros_like(dark_lenses), 0
    n_optics = max(0, int(cv2.connectedComponents(clustered.astype(np.uint8), connectivity=8)[0] - 1))
    if not clustered.any():
        return clustered, n_optics
    k = max(5, int(round(0.045 * min_side)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        clustered.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    joined = closed.astype(bool) & top & back
    # Prefer the component that covers the most original optics.
    u8 = joined.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n <= 1:
        return joined, n_optics
    best_i = 1
    best_overlap = -1
    for i in range(1, n):
        overlap = int(np.count_nonzero((labels == i) & clustered))
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = i
    return labels == best_i, n_optics


def _optics_in_island_roi(
    gray: np.ndarray,
    seeds: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    min_side: float,
) -> np.ndarray:
    """Search the square around the first lenses for flash / remaining glass."""
    if not seeds.any():
        return np.zeros_like(seeds)
    height, width = gray.shape
    ys, xs = np.where(seeds)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    span = int(max(x1 - x0 + 1, y1 - y0 + 1, 8))
    pad = max(4, int(round(0.38 * span)))
    rx0, rx1 = max(0, x0 - pad), min(width, x1 + pad + 1)
    ry0, ry1 = max(0, y0 - pad), min(height, y1 + pad + 1)
    roi = np.zeros_like(top)
    roi[ry0:ry1, rx0:rx1] = top[ry0:ry1, rx0:rx1] & back[ry0:ry1, rx0:rx1]
    found = _find_lenses(gray, roi, back_area, min_side)
    hough = _hough_lenses(gray, roi, min_side)
    return (found | hough | seeds) & roi


def _compact_optics_bbox(mask: np.ndarray, min_side: float) -> bool:
    if not mask.any():
        return False
    ys, xs = np.where(mask)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    if max(w, h) > 0.28 * min_side:
        return False
    aspect = max(w, h) / float(max(min(w, h), 1))
    return aspect <= 2.2


def _snap_to_module_contour(
    gray: np.ndarray,
    lenses: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
    back_area: float,
) -> np.ndarray:
    """Fill the real camera-housing outline around the lens cluster."""
    if not lenses.any():
        return np.zeros_like(lenses)
    height, width = gray.shape
    ys, xs = np.where(lenses)
    pad = max(8, int(round(0.045 * min_side)))
    x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1)
    roi = np.zeros_like(gray)
    roi[y0:y1, x0:x1] = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(blur, 22, 70)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, grad_bin = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = np.maximum(edges, grad_bin)
    combined[~top] = 0
    combined[~back] = 0
    k = max(3, int(round(0.012 * min_side)))
    if k % 2 == 0:
        k += 1
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cents = _blob_centroids(lenses)
    best = np.zeros(gray.shape, dtype=bool)
    best_area = 1e18
    min_a = max(float(np.count_nonzero(lenses)) * 1.15, 0.004 * back_area)
    max_a = 0.14 * back_area
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_a or area > max_a:
            continue
        peri = cv2.arcLength(contour, True)
        if peri < 12:
            continue
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ > 0.92:
            continue
        _x, _y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / float(max(min(bw, bh), 1))
        if aspect > 2.0:
            continue
        solidity = area / float(max(bw * bh, 1))
        if solidity < 0.62:
            continue
        filled = np.zeros(gray.shape, dtype=np.uint8)
        hull = cv2.convexHull(contour)
        cv2.drawContours(filled, [hull], -1, 1, thickness=cv2.FILLED)
        fb = filled.astype(bool) & back & top
        contained = 0
        for px, py in cents:
            iy, ix = int(round(py)), int(round(px))
            if 0 <= iy < height and 0 <= ix < width and fb[iy, ix]:
                contained += 1
        if contained < len(cents):
            continue
        if area < best_area:
            best_area = area
            best = fb
    if best.any():
        return fill_binary_holes(best)
    return _rounded_box_around_lenses(lenses, top, back, min_side)


def _island_from_lenses(
    lenses: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    min_side: float,
) -> np.ndarray:
    """One filled island covering every lens plus flash/sensors between them."""
    if not lenses.any():
        return np.zeros_like(lenses)
    boxed = _rounded_box_around_lenses(lenses, top, back, min_side)
    if boxed.any():
        return boxed
    hull = _filled_contour_around(lenses)
    cents = _blob_centroids(lenses)
    gap = 12.0
    if len(cents) >= 2:
        dists = [
            float(np.hypot(cents[i][0] - cents[j][0], cents[i][1] - cents[j][1]))
            for i in range(len(cents))
            for j in range(i + 1, len(cents))
        ]
        gap = float(np.median(dists))
    k = max(9, int(round(gap * 0.85)))
    k = min(k, max(9, int(0.16 * min_side)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        (hull | lenses).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    closed = fill_binary_holes(closed.astype(bool))
    pad = max(3.0, 0.022 * min_side)
    closed = dilate_binary(closed, pad) & top & back
    closed = fill_binary_holes(closed)
    if float(np.count_nonzero(closed)) > 0.12 * back_area:
        closed = fill_binary_holes(hull) & top & back
    return closed


def _rounded_box_around_lenses(
    lenses: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
) -> np.ndarray:
    """Camera housing as one rounded rectangle covering every optic."""
    if not lenses.any():
        return np.zeros_like(lenses)
    ys, xs = np.where(lenses)
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if pts.shape[0] < 4:
        hull = _filled_contour_around(lenses)
        return fill_binary_holes(dilate_binary(hull, max(4.0, 0.028 * min_side)) & top & back)
    (_cx, _cy), (rw, rh), ang = cv2.minAreaRect(pts)
    rw = max(float(rw), 4.0) * 1.58
    rh = max(float(rh), 4.0) * 1.58
    side = max(rw, rh)
    rw = max(rw, 0.80 * side)
    rh = max(rh, 0.80 * side)
    box = cv2.boxPoints(((_cx, _cy), (rw, rh), ang)).astype(np.int32)
    out = np.zeros(lenses.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, box, 1)
    rad = max(5, int(round(0.20 * min(rw, rh))))
    if rad % 2 == 0:
        rad += 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad)))
    island = fill_binary_holes(out.astype(bool)) & top & back
    return fill_binary_holes(island | (lenses & back))


def _hough_lenses(gray: np.ndarray, top: np.ndarray, min_side: float) -> np.ndarray:
    """Circular optics on light-colored backs where a global dark threshold misses."""
    if not top.any():
        return np.zeros(gray.shape, dtype=bool)
    work = gray.copy()
    work[~top] = int(np.median(gray[top]))
    blur = cv2.GaussianBlur(work, (5, 5), 1.1)
    min_r = max(3, int(round(0.012 * min_side)))
    max_r = max(min_r + 2, int(round(0.075 * min_side)))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(max(6, min_r * 2)),
        param1=70,
        param2=14,
        minRadius=min_r,
        maxRadius=max_r,
    )
    keep = np.zeros(gray.shape, dtype=bool)
    if circles is None:
        return keep
    med = float(np.median(gray[top]))
    height, width = gray.shape
    for x, y, r in np.round(circles[0]).astype(int):
        if not (0 <= x < width and 0 <= y < height and top[y, x]):
            continue
        yy, xx = np.ogrid[0:height, 0:width]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= (r * r)
        disk &= top
        if float(np.count_nonzero(disk)) < 8:
            continue
        inside = float(np.mean(gray[disk]))
        ring = ((xx - x) ** 2 + (yy - y) ** 2 <= ((r + 4) ** 2)) & ~disk & top
        outside = float(np.mean(gray[ring])) if ring.any() else med
        if inside > med - 12.0:
            continue
        if outside - inside < 10.0:
            continue
        keep |= disk
    return keep


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if mask is None or not mask.any():
        return mask
    u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n <= 2:
        return mask.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _dark_housing(
    gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    y0: int,
    bh: int,
    min_side: float,
) -> np.ndarray:
    """Raised dark camera bump (common on mockup PNGs and black modules)."""
    med = float(np.median(gray[top])) if top.any() else 128.0
    dark = (gray < (med - 18.0)) & top
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    best = np.zeros(gray.shape, dtype=bool)
    best_area = 0.0
    min_a = 0.005 * back_area
    max_a = 0.14 * back_area
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_a or area > max_a:
            continue
        w = float(stats[i, cv2.CC_STAT_WIDTH])
        h = float(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / float(max(min(w, h), 1))
        if aspect > 1.85:
            continue
        if cents[i][1] > y0 + 0.36 * bh:
            continue
        if h > 0.24 * bh and w / max(h, 1.0) < 0.85:
            continue
        if w > 0.42 * min_side:
            continue
        blob = labels == i
        if not ((gray < med - 12.0) & blob).any():
            continue
        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        peri = cv2.arcLength(cnts[0], True)
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ > 0.93:
            continue
        if area > best_area:
            best_area = area
            best = blob
    if not best.any():
        return best
    return fill_binary_holes(best) & back


def _island_around_lenses(
    gray: np.ndarray,
    lenses: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    min_side: float,
) -> np.ndarray:
    """Expand from detected lenses to the housing rim using local edges."""
    height, width = gray.shape
    ys, xs = np.where(lenses)
    if ys.size == 0:
        return np.zeros_like(lenses)
    pad = max(8, int(0.08 * min_side))
    x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad)
    roi = np.zeros_like(gray)
    roi[y0:y1, x0:x1] = gray[y0:y1, x0:x1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 8))
    eq = clahe.apply(roi)
    edges = cv2.Canny(cv2.GaussianBlur(eq, (3, 3), 0), 18, 80)
    edges[~top] = 0
    k = max(3, pad // 4)
    if k % 2 == 0:
        k += 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lens_pts = _blob_centroids(lenses)
    best = np.zeros(gray.shape, dtype=bool)
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.008 * back_area or area > 0.16 * back_area:
            continue
        peri = cv2.arcLength(contour, True)
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ > 0.93:
            continue
        filled = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        fb = filled.astype(bool) & back
        contained = sum(
            1
            for px, py in lens_pts
            if 0 <= int(py) < height and 0 <= int(px) < width and fb[int(py), int(px)]
        )
        if lens_pts and contained == 0:
            continue
        score = 0.4 + 0.3 * (contained / max(len(lens_pts), 1)) + (0.15 if circ < 0.75 else 0)
        if score > best_score:
            best_score = score
            best = fb
    if best_score >= 0.5:
        return fill_binary_holes(best)

    # Fallback: close the lens cluster so flash/sensors between them are inside.
    k = max(11, int(0.07 * min_side))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        lenses.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    return fill_binary_holes(closed.astype(bool)) & top


def _contrast_module(
    gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    y0: int,
    bh: int,
) -> np.ndarray:
    """Module that is darker than the (often white) back glass, including the island."""
    if not back.any():
        return np.zeros(gray.shape, dtype=bool)
    med = float(np.median(gray[back]))
    darker = (gray < med - 16.0) & top
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    darker = cv2.morphologyEx(darker.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(darker, 8)
    best = np.zeros(gray.shape, dtype=bool)
    best_area = 0.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 0.015 * back_area or area > 0.12 * back_area:
            continue
        w = float(stats[i, cv2.CC_STAT_WIDTH])
        h = float(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / float(max(min(w, h), 1))
        if aspect > 1.85:
            continue
        if cents[i][1] > y0 + 0.38 * bh:
            continue
        if h > 0.25 * bh and w / max(h, 1.0) < 0.85:
            continue
        blob = labels == i
        if not ((gray < 110) & blob).any():
            continue
        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        peri = cv2.arcLength(cnts[0], True)
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ > 0.93:
            continue
        if area > best_area:
            best_area = area
            best = blob.astype(bool)
    if not best.any():
        return best
    return fill_binary_holes(best) & back


def _find_lenses(gray: np.ndarray, top: np.ndarray, back_area: float, min_side: float) -> np.ndarray:
    """Near-black circular optics in the upper back panel."""
    if not top.any():
        return np.zeros(gray.shape, dtype=bool)
    local_med = float(np.median(gray[top]))
    thresh = float(max(28.0, min(local_med - 28.0, 140.0)))
    dark = (gray < thresh) & top
    u8 = dark.astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(u8, 8)
    keep = np.zeros(gray.shape, dtype=bool)
    min_area = max(6.0, 0.00018 * back_area)
    max_area = max(min_area * 4.0, 0.035 * back_area)
    min_d = max(2.0, 0.008 * min_side)
    max_d = max(min_d + 1.0, 0.18 * min_side)

    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        w = float(stats[i, cv2.CC_STAT_WIDTH])
        h = float(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if min(w, h) < min_d or max(w, h) > max_d:
            continue
        blob = labels == i
        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        peri = cv2.arcLength(cnts[0], True)
        if peri < 1:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.22:
            continue
        keep |= blob
    return keep


def _module_contour(
    gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    lenses: np.ndarray,
    y0: int,
    bh: int,
    back_area: float,
) -> tuple[np.ndarray, float]:
    """Largest non-circular housing contour in the upper panel that wraps the lenses."""
    height, width = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    edges = cv2.Canny(blur, 28, 95)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _, grad_bin = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = np.maximum(edges, grad_bin)
    combined[~top] = 0

    k = max(3, int(round(min(height, width) * 0.006)))
    if k % 2 == 0:
        k += 1
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lens_pts = _blob_centroids(lenses)
    best_mask = np.zeros((height, width), dtype=bool)
    best_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.010 * back_area or area > 0.10 * back_area:
            continue
        peri = cv2.arcLength(contour, True)
        if peri < 8:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        M = cv2.moments(contour)
        if M["m00"] < 1:
            continue
        cy = M["m01"] / M["m00"]
        cx = M["m10"] / M["m00"]
        if (cy - y0) / float(bh) > 0.40:
            continue
        if circ > 0.91:
            continue  # MagSafe  # MagSafe
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / float(max(min(w, h), 1))
        if aspect > 2.1:
            continue

        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        filled_b = filled.astype(bool) & back
        if not filled.any():
            continue

        contained = 0
        for px, py in lens_pts:
            if 0 <= int(py) < height and 0 <= int(px) < width and filled_b[int(py), int(px)]:
                contained += 1
        if lens_pts:
            if contained == 0:
                continue
        else:
            solidity = area / float(max(w * h, 1))
            if solidity < 0.70 or aspect > 1.5:
                continue

        score = 0.35
        if 0.7 <= aspect <= 1.55:
            score += 0.25
        if 0.018 <= area / back_area <= 0.11:
            score += 0.2
        if contained >= 2:
            score += 0.3
        elif contained == 1:
            score += 0.1
        if circ < 0.72:
            score += 0.1
        if score > best_score:
            best_score = score
            best_mask = filled_b

    if best_score >= 0.5:
        return fill_binary_holes(best_mask), best_score
    return np.zeros((height, width), dtype=bool), 0.0


def _grow_to_bezel(
    gray: np.ndarray,
    seed: np.ndarray,
    top: np.ndarray,
    back_area: float,
) -> np.ndarray:
    """Expand a lens cluster until the housing rim, using a closed contour if possible."""
    filled = _filled_contour_around(seed)
    if not filled.any():
        filled = seed
    # Close gaps between dual/triple lenses so flash/sensors sit inside.
    k = max(9, int(round(np.sqrt(max(np.count_nonzero(seed), 1)) * 0.9)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        filled.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    closed = fill_binary_holes(closed.astype(bool))
    if float(np.count_nonzero(closed)) > 0.16 * back_area:
        return filled
    return closed & top


def _filled_contour_around(binary: np.ndarray) -> np.ndarray:
    u8 = binary.astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary.astype(bool)
    pts = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    out = np.zeros(binary.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, approx if len(approx) >= 3 else hull, 1)
    return fill_binary_holes(out.astype(bool))


def _blob_centroids(binary: np.ndarray) -> list[tuple[float, float]]:
    u8 = binary.astype(np.uint8)
    n, _labels, _stats, cents = cv2.connectedComponentsWithStats(u8, 8)
    return [(float(cents[i][0]), float(cents[i][1])) for i in range(1, n)]


def _keep_blobs(
    binary: np.ndarray,
    height: int,
    width: int,
    *,
    min_frac: float,
    max_frac: float,
    top_frac: float,
    region: np.ndarray,
) -> np.ndarray:
    area_img = float(height * width)
    ys = np.where(region)[0]
    if ys.size == 0:
        return np.zeros(binary.shape, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    top_lim = y0 + int(top_frac * max(1, y1 - y0))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = np.zeros(binary.shape, dtype=bool)
    for i in range(1, n):
        frac = float(stats[i, cv2.CC_STAT_AREA]) / max(area_img, 1.0)
        if frac < min_frac or frac > max_frac:
            continue
        if stats[i, cv2.CC_STAT_WIDTH] < 4 or stats[i, cv2.CC_STAT_HEIGHT] < 4:
            continue
        if cents[i][1] > top_lim:
            continue
        keep |= labels == i
    return keep
