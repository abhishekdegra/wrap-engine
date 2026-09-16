"""Robust phone case localization and pose estimation for arbitrary orientations.

Locates the physical phone/case object in ANY uploaded image (regardless of
rotation, tilt, perspective, scale, position, aspect ratio, transparent or colored
case, background, or lighting) and normalizes it to a canonical upright geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PhonePose:
    contour: np.ndarray  # Nx2 float64: contour points in original image coordinates
    quad: np.ndarray  # 4x2 float32: [TL, TR, BR, BL] in original image coordinates
    angle: float  # Rotation angle relative to vertical (degrees, 0 = upright)
    canonical_size: tuple[int, int]  # (width, height) of canonical normalized frame
    H_to_canon: np.ndarray  # 3x3 float32: perspective transform from original to canonical
    H_from_canon: np.ndarray  # 3x3 float32: perspective transform from canonical to original
    confidence: float  # Pose detection confidence [0.0 to 1.0]
    is_upright: bool  # True if phone is already approximately axis-aligned upright


def detect_phone_pose(cover_rgba: np.ndarray) -> PhonePose:
    """Locate phone/case and compute canonical normalization transform."""
    if cover_rgba.ndim != 3 or cover_rgba.shape[2] != 4:
        raise ValueError("cover_rgba must be an HxWx4 array")

    height, width = cover_rgba.shape[:2]
    rgb = cover_rgba[..., :3]
    alpha = cover_rgba[..., 3]
    has_alpha = bool(int(alpha.min()) < 240 and int(np.percentile(alpha, 10)) < 230)

    # For fast candidate generation on high-resolution images, compute on scaled copy
    scale = 1.0
    longest = max(height, width)
    if longest > 1200:
        scale = 1200.0 / float(longest)
        nw = max(1, int(round(width * scale)))
        nh = max(1, int(round(height * scale)))
        rgb_work = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        alpha_work = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        work_h, work_w = nh, nw
    else:
        rgb_work = rgb
        alpha_work = alpha
        work_h, work_w = height, width

    # Collect phone candidates from multiple independent strategies
    candidates: list[dict] = []

    # Strategy 1: Alpha channel (transparent PNG mockup)
    if has_alpha:
        _collect_alpha_candidates(alpha_work, work_h, work_w, candidates)

    # Strategy 2: Multi-signal Canny & morphological edges (hierarchical tree)
    _collect_edge_tree_candidates(rgb_work, work_h, work_w, candidates)

    # Strategy 3: Background contrast (corners & border color modeling)
    _collect_bg_contrast_candidates(rgb_work, work_h, work_w, candidates)

    # Strategy 4: Dark bumper / rim segmentation (clear case with black/dark rim)
    _collect_dark_rim_candidates(rgb_work, work_h, work_w, candidates)

    # Strategy 5: Adaptive / Otsu multi-thresholding
    _collect_adaptive_candidates(rgb_work, work_h, work_w, candidates)

    if not candidates:
        # Fallback: full image canvas
        return _fallback_pose(width, height)

    # Rank candidates by phone score
    for cand in candidates:
        cand["score"] = _score_phone_candidate(cand, rgb_work, work_h, work_w)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    if scale != 1.0:
        contour = (best["contour"] / scale).astype(np.float64)
        (cx, cy), (rw, rh), angle = best["rect"]
        rect = ((cx / scale, cy / scale), (rw / scale, rh / scale), angle)
        box = (cv2.boxPoints(best["rect"]) / scale).astype(np.float32)
    else:
        contour = best["contour"].astype(np.float64)
        rect = best["rect"]
        box = cv2.boxPoints(rect).astype(np.float32)

    # Refine quad using robust Huber line intersections along 4 rails
    quad_refined = _fit_perspective_rails(contour, rect, box)

    # Order corners canonical [TL, TR, BR, BL] with camera at top
    quad_canon, canon_size, angle_deg = _order_quad_canonical(quad_refined, contour, rgb)

    target_w, target_h = canon_size
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )

    H_to_canon = cv2.getPerspectiveTransform(quad_canon, dst)
    H_from_canon = np.linalg.inv(H_to_canon).astype(np.float32)

    is_upright = (
        abs(angle_deg) < 3.5
        and abs(float(quad_canon[0, 1] - quad_canon[1, 1])) < max(8.0, 0.02 * target_w)
        and abs(float(quad_canon[0, 0] - quad_canon[3, 0])) < max(8.0, 0.02 * target_h)
    )

    conf = float(np.clip(best["score"], 0.2, 0.99))

    return PhonePose(
        contour=contour.astype(np.float64),
        quad=quad_canon,
        angle=angle_deg,
        canonical_size=canon_size,
        H_to_canon=H_to_canon,
        H_from_canon=H_from_canon,
        confidence=conf,
        is_upright=is_upright,
    )


# ==============================================================================
# CANDIDATE GENERATORS
# ==============================================================================

def _collect_alpha_candidates(
    alpha: np.ndarray, height: int, width: int, out: list[dict]
) -> None:
    total_area = float(height * width)
    # Exclude soft drop shadows
    a_bin = (alpha > 40).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a_clean = cv2.morphologyEx(a_bin, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(a_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 0.05 * total_area or area > 0.98 * total_area:
            continue
        rect = cv2.minAreaRect(c)
        pw, pl = sorted(rect[1])
        if pw < 30.0:
            continue
        aspect = pl / max(pw, 1.0)
        if 1.40 <= aspect <= 2.85:
            out.append({"src": "alpha", "contour": c.reshape(-1, 2), "rect": rect, "area": area})


def _collect_edge_tree_candidates(
    rgb: np.ndarray, height: int, width: int, out: list[dict]
) -> None:
    total_area = float(height * width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    blurred = cv2.bilateralFilter(gray, 7, 45, 45) if min(width, height) > 200 else cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 90)

    # Morphological gradient to catch subtle edges
    kernel_m = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_m)
    edges_comb = cv2.bitwise_or(edges, (morph_grad > 28).astype(np.uint8) * 255)

    # Close small gaps in perimeter
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_closed = cv2.morphologyEx(edges_comb, cv2.MORPH_CLOSE, kernel_close)

    # RETR_TREE allows locating phone inside nested screenshot boxes or product cards
    cnts, hier = cv2.findContours(edges_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 0.03 * total_area or area > 0.98 * total_area:
            continue
        rect = cv2.minAreaRect(c)
        pw, pl = sorted(rect[1])
        if pw < 30.0:
            continue
        aspect = pl / max(pw, 1.0)
        if 1.40 <= aspect <= 2.85:
            out.append({"src": "edge_tree", "contour": c.reshape(-1, 2), "rect": rect, "area": area})


def _collect_bg_contrast_candidates(
    rgb: np.ndarray, height: int, width: int, out: list[dict]
) -> None:
    total_area = float(height * width)
    cw = max(2, int(0.04 * width))
    ch = max(2, int(0.04 * height))
    corners = np.vstack([
        rgb[:ch, :cw].reshape(-1, 3),
        rgb[:ch, -cw:].reshape(-1, 3),
        rgb[-ch:, :cw].reshape(-1, 3),
        rgb[-ch:, -cw:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)
    diff = np.linalg.norm(rgb.astype(np.float32) - bg_color, axis=2)

    for thresh in (20.0, 35.0, 50.0):
        fg = (diff > thresh).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < 0.04 * total_area or area > 0.98 * total_area:
                continue
            rect = cv2.minAreaRect(c)
            pw, pl = sorted(rect[1])
            if pw < 30.0:
                continue
            aspect = pl / max(pw, 1.0)
            if 1.40 <= aspect <= 2.85:
                out.append({"src": "bg_contrast", "contour": c.reshape(-1, 2), "rect": rect, "area": area})


def _collect_dark_rim_candidates(
    rgb: np.ndarray, height: int, width: int, out: list[dict]
) -> None:
    """Segment dark bumpers / shockproof rims enclosing a phone case."""
    total_area = float(height * width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 65).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    # Find loops enclosing an area
    cnts, hier = cv2.findContours(dark_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None and len(hier[0]) > 0:
        for i, c in enumerate(cnts):
            area = float(cv2.contourArea(c))
            if area < 0.03 * total_area or area > 0.98 * total_area:
                continue
            rect = cv2.minAreaRect(c)
            pw, pl = sorted(rect[1])
            if pw < 30.0:
                continue
            aspect = pl / max(pw, 1.0)
            if 1.40 <= aspect <= 2.85:
                out.append({"src": "dark_rim", "contour": c.reshape(-1, 2), "rect": rect, "area": area})


def _collect_adaptive_candidates(
    rgb: np.ndarray, height: int, width: int, out: list[dict]
) -> None:
    total_area = float(height * width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    block_size = max(31, (min(height, width) // 16) | 1)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    adapt_clean = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(adapt_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 0.04 * total_area or area > 0.98 * total_area:
            continue
        rect = cv2.minAreaRect(c)
        pw, pl = sorted(rect[1])
        if pw < 30.0:
            continue
        aspect = pl / max(pw, 1.0)
        if 1.40 <= aspect <= 2.85:
            out.append({"src": "adaptive", "contour": c.reshape(-1, 2), "rect": rect, "area": area})


# ==============================================================================
# SCORING & SELECTION
# ==============================================================================

def _score_phone_candidate(cand: dict, rgb: np.ndarray, height: int, width: int) -> float:
    """Score candidate by aspect ratio, rectangle fill, convexity, and edge gradients."""
    rect = cand["rect"]
    pw, pl = sorted(rect[1])
    aspect = pl / max(pw, 1.0)
    # Typical phone aspect ratio is 1.95 - 2.25. Score bell-curves around 2.1
    aspect_score = math.exp(-0.5 * ((aspect - 2.12) / 0.35) ** 2)

    area = cand["area"]
    rect_area = pw * pl
    fill_ratio = area / max(1.0, rect_area)
    # Real phone with rounded corners has fill ratio ~0.88 - 0.97
    fill_score = math.exp(-0.5 * ((fill_ratio - 0.92) / 0.12) ** 2)

    c = cand["contour"]
    hull = cv2.convexHull(c.reshape(-1, 1, 2).astype(np.int32))
    hull_area = float(cv2.contourArea(hull))
    convexity = area / max(1.0, hull_area)

    # Source bonus
    src_weight = {
        "alpha": 1.25,
        "edge_tree": 1.15,
        "dark_rim": 1.10,
        "bg_contrast": 1.05,
        "adaptive": 0.95,
    }.get(cand.get("src", ""), 1.0)

    # Size penalty if too small (<5% canvas) or almost canvas (>96%)
    total_area = float(height * width)
    area_frac = area / total_area
    size_weight = 1.0
    if area_frac < 0.06:
        size_weight = 0.5
    elif area_frac > 0.95:
        size_weight = 0.7

    score = 0.40 * aspect_score + 0.30 * fill_score + 0.30 * convexity
    score = score * src_weight * size_weight
    return float(score)


# ==============================================================================
# PERSPECTIVE QUAD & ORIENTATION NORMALIZATION
# ==============================================================================

def _fit_perspective_rails(contour: np.ndarray, rect: tuple, box: np.ndarray) -> np.ndarray:
    """Fit 4 Huber lines along contour rails to handle perspective tapering."""
    (cx0, cy0), (rw, rh), angle = rect
    if rw > rh:
        rw, rh = rh, rw
        angle = angle + 90.0

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    pts = contour.astype(np.float64)
    dx = pts[:, 0] - cx0
    dy = pts[:, 1] - cy0
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    mid_h = 0.32 * rh
    mid_w = 0.32 * rw

    left_mask = (u < 0) & (np.abs(v) < mid_h)
    right_mask = (u > 0) & (np.abs(v) < mid_h)
    top_mask = (v < 0) & (np.abs(u) < mid_w)
    bot_mask = (v > 0) & (np.abs(u) < mid_w)

    if not (left_mask.any() and right_mask.any() and top_mask.any() and bot_mask.any()):
        return box

    def _fit_line(subset_pts):
        if len(subset_pts) < 6:
            return None
        vx, vy, x0, y0 = cv2.fitLine(subset_pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        d = np.array([float(vx.flat[0]), float(vy.flat[0])], dtype=np.float64)
        p = np.array([float(x0.flat[0]), float(y0.flat[0])], dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return None
        return p, d / norm

    l_res = _fit_line(pts[left_mask])
    r_res = _fit_line(pts[right_mask])
    t_res = _fit_line(pts[top_mask])
    b_res = _fit_line(pts[bot_mask])

    if l_res is None or r_res is None or t_res is None or b_res is None:
        return box

    p_l, d_l = l_res
    p_r, d_r = r_res
    p_t, d_t = t_res
    p_b, d_b = b_res

    def _intersect(p1, d1, p2, d2):
        mat = np.column_stack([d1, -d2])
        if abs(np.linalg.det(mat)) < 1e-5:
            return (p1 + p2) / 2.0
        t = np.linalg.solve(mat, p2 - p1)[0]
        return p1 + t * d1

    v_tl = _intersect(p_t, d_t, p_l, d_l)
    v_tr = _intersect(p_t, d_t, p_r, d_r)
    v_br = _intersect(p_b, d_b, p_r, d_r)
    v_bl = _intersect(p_b, d_b, p_l, d_l)

    quad = np.array([v_tl, v_tr, v_br, v_bl], dtype=np.float32)

    # Sanity check: quad points must be close to box points
    diff = np.linalg.norm(quad - box, axis=1).max()
    if diff > 0.20 * max(rw, rh):
        return box

    return quad


def _order_quad_canonical(
    box: np.ndarray, pts: np.ndarray, rgb: np.ndarray
) -> tuple[np.ndarray, tuple[int, int], float]:
    """Order quad corners [TL, TR, BR, BL] ensuring upright orientation and camera at top."""
    d01 = float(np.linalg.norm(box[1] - box[0]))
    d12 = float(np.linalg.norm(box[2] - box[1]))

    if d01 <= d12:
        short_len, long_len = d01, d12
        p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
    else:
        short_len, long_len = d12, d01
        p0, p1, p2, p3 = box[1], box[2], box[3], box[0]

    # Ensure right-handed orientation (positive cross product)
    v_short = p1 - p0
    v_long = p3 - p0
    cross = v_short[0] * v_long[1] - v_short[1] * v_long[0]
    if cross < 0:
        p0, p1, p2, p3 = p1, p0, p3, p2

    quad_A = np.array([p0, p1, p2, p3], dtype=np.float32)
    quad_B = np.array([p2, p3, p0, p1], dtype=np.float32)

    # Determine if phone in image coordinates is vertical or horizontal
    y_top_A = float(quad_A[0, 1] + quad_A[1, 1]) / 2.0
    y_bot_A = float(quad_A[2, 1] + quad_A[3, 1]) / 2.0
    x_left_A = float(quad_A[0, 0] + quad_A[3, 0]) / 2.0
    x_right_A = float(quad_A[1, 0] + quad_A[2, 0]) / 2.0

    is_vertical_in_photo = abs(y_bot_A - y_top_A) > abs(x_right_A - x_left_A)

    if is_vertical_in_photo:
        # If quad_A's top rail is physically lower in image than its bottom rail, swap quad_A and quad_B
        # so quad_A is GUARANTEED to be upright in image space
        if y_top_A > y_bot_A:
            quad_A, quad_B = quad_B, quad_A

    canon_w = max(100, int(round(short_len)))
    canon_h = max(100, int(round(long_len)))

    # Supersample canonical target for high precision
    target_h = max(1400, canon_h)
    target_w = int(round(target_h * (canon_w / float(canon_h))))

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )

    H_A = cv2.getPerspectiveTransform(quad_A, dst)
    warp_A = cv2.warpPerspective(rgb, H_A, (target_w, target_h))

    # Score camera presence in top half
    score_A, n_c_t, n_c_b, d_t, d_b = _camera_optical_analysis(warp_A)

    if is_vertical_in_photo:
        # quad_A is physically upright in the user's photo.
        # NEVER invert 180° unless there is decisive optical proof of camera at the bottom:
        strong_inverted_proof = (n_c_b >= 1 and n_c_t == 0 and score_A < -800.0) or (score_A < -3500.0 and d_b > d_t + 15000)
        if strong_inverted_proof:
            chosen_quad = quad_B
            v_up = -(quad_B[3] - quad_B[0])
        else:
            chosen_quad = quad_A
            v_up = -(quad_A[3] - quad_A[0])
    else:
        # Horizontal / landscape orientation: pick side with higher optical score
        if score_A >= 0:
            chosen_quad = quad_A
            v_up = -(quad_A[3] - quad_A[0])
        else:
            chosen_quad = quad_B
            v_up = -(quad_B[3] - quad_B[0])

    angle_rad = math.atan2(float(v_up[0]), -float(v_up[1]))
    angle_deg = float(math.degrees(angle_rad))

    return chosen_quad, (target_w, target_h), angle_deg


def _camera_optical_analysis(img_canonical: np.ndarray) -> tuple[float, int, int, float, float]:
    """Analyze top 38% vs bottom 38% for optical features (camera lenses, dark islands, circles)."""
    h, w = img_canonical.shape[:2]
    top_roi = img_canonical[: int(0.38 * h), :]
    bot_roi = img_canonical[int(0.62 * h) :, :]

    gray_t = cv2.cvtColor(top_roi, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(bot_roi, cv2.COLOR_RGB2GRAY)

    grad_t = float(cv2.Sobel(gray_t, cv2.CV_32F, 1, 1).var())
    grad_b = float(cv2.Sobel(gray_b, cv2.CV_32F, 1, 1).var())

    dark_t = float(np.count_nonzero(gray_t < 60))
    dark_b = float(np.count_nonzero(gray_b < 60))

    # Detect circles (camera optics)
    circles_t = cv2.HoughCircles(
        cv2.GaussianBlur(gray_t, (5, 5), 0),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w * 0.08,
        param1=80,
        param2=22,
        minRadius=int(w * 0.03),
        maxRadius=int(w * 0.22),
    )
    n_circles_t = len(circles_t[0]) if circles_t is not None else 0

    circles_b = cv2.HoughCircles(
        cv2.GaussianBlur(gray_b, (5, 5), 0),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w * 0.08,
        param1=80,
        param2=22,
        minRadius=int(w * 0.03),
        maxRadius=int(w * 0.22),
    )
    n_circles_b = len(circles_b[0]) if circles_b is not None else 0

    score = (grad_t - grad_b) + 0.005 * (dark_t - dark_b) + 60.0 * (n_circles_t - n_circles_b)
    return float(score), n_circles_t, n_circles_b, dark_t, dark_b


def _camera_top_score(img_canonical: np.ndarray) -> float:
    return _camera_optical_analysis(img_canonical)[0]


def _fallback_pose(width: int, height: int) -> PhonePose:
    """Axis-aligned canvas fallback pose."""
    w, h = float(width), float(height)
    quad = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32)
    cnt = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    H_eye = np.eye(3, dtype=np.float32)
    return PhonePose(
        contour=cnt,
        quad=quad,
        angle=0.0,
        canonical_size=(width, height),
        H_to_canon=H_eye,
        H_from_canon=H_eye,
        confidence=0.30,
        is_upright=True,
    )
