"""Antialiased printable / camera / edge masks.

Template-based generation remains available for fixtures. Runtime covers
use ``masks_from_binaries`` after OpenCV detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.templates import CoverTemplate, PixelCircle, PixelRect


@dataclass(frozen=True)
class MaskSet:
    printable_back: np.ndarray
    camera_exclusion: np.ndarray
    edge_exclusion: np.ndarray
    final_print: np.ndarray
    outer: np.ndarray | None = None


def generate_masks(template: CoverTemplate, height: int, width: int) -> MaskSet:
    """Build masks from CoverTemplate geometry (tests / sample PNG builder)."""
    feather = float(template.mask_feather_px)

    outer = template.outer_case.to_pixels(width, height)
    printable = template.printable_area.to_pixels(width, height).inset(template.edge_margin_px)
    camera = template.camera_exclusion.to_pixels(width, height).expand(
        template.camera_safety_margin_px
    )
    holes = tuple(
        hole.to_pixels(width, height).expand(template.camera_safety_margin_px * 0.35)
        for hole in template.camera_holes
    )

    outer_mask = rounded_rect_mask(height, width, outer, feather)
    printable_back = rounded_rect_mask(height, width, printable, feather)
    printable_back = np.minimum(printable_back, outer_mask)

    camera_exclusion = rounded_rect_mask(height, width, camera, feather)
    for hole in holes:
        camera_exclusion = np.maximum(camera_exclusion, circle_mask(height, width, hole, feather))

    edge_exclusion = np.clip(outer_mask - printable_back, 0.0, 1.0)
    final_print = np.clip(printable_back * (1.0 - camera_exclusion), 0.0, 1.0).astype(np.float32)

    return MaskSet(
        printable_back=printable_back.astype(np.float32),
        camera_exclusion=camera_exclusion.astype(np.float32),
        edge_exclusion=edge_exclusion.astype(np.float32),
        final_print=final_print,
        outer=outer_mask.astype(np.float32),
    )


def masks_from_binaries(
    outer_bin: np.ndarray,
    back_bin: np.ndarray,
    camera_bin: np.ndarray,
    feather_px: float,
) -> MaskSet:
    """Antialiased masks from detected silhouettes — perspective-aware and uniform rim."""
    feather = float(feather_px)
    outer_src = fill_binary_holes(np.asarray(outer_bin).astype(bool))
    back_src = fill_binary_holes(np.asarray(back_bin).astype(bool)) & outer_src

    quad = _fit_perspective_phone_quad(_contour_points(outer_src))
    if quad is not None:
        w_top = np.linalg.norm(quad[1] - quad[0])
        w_bot = np.linalg.norm(quad[2] - quad[3])
        avg_w = float((w_top + w_bot) / 2.0)
        outer_pts = rounded_quad_polyline(quad, radius_ratio=0.126, num_arc_pts=36)
        inset_px = _median_boundary_inset(outer_src, back_src)
        bumper_px = float(max(inset_px, 0.020 * avg_w))
        back_pts = _offset_closed_polyline(outer_pts, -bumper_px)
        outer_m = _rasterize_wrap_polyline(outer_pts, outer_src.shape, feather, cap_bin=None)
        back_m = _rasterize_wrap_polyline(back_pts, outer_src.shape, feather, cap_bin=None)
        back_m = np.minimum(back_m, outer_m)
    else:
        outer_pts = _finish_wrap_polyline(_contour_points(outer_src))
        inset_px = _median_boundary_inset(outer_src, back_src)
        if outer_pts is not None and inset_px > 0.35:
            back_pts = _offset_closed_polyline(outer_pts, -float(inset_px))
            outer_m = _rasterize_wrap_polyline(outer_pts, outer_src.shape, feather, cap_bin=None)
            back_m = _rasterize_wrap_polyline(back_pts, outer_src.shape, feather, cap_bin=None)
            back_m = np.minimum(back_m, outer_m)
        else:
            outer_m = silhouette_to_aa_mask(outer_src, feather)
            back_m = silhouette_to_aa_mask(back_src, feather)
            back_m = np.minimum(back_m, outer_m)

    camera_m = _camera_aa_mask(camera_bin, feather)
    back_m = np.minimum(back_m, outer_m)
    camera_m = np.minimum(camera_m, outer_m)
    edge_m = np.clip(outer_m - back_m, 0.0, 1.0)
    final_print = np.clip(back_m * (1.0 - camera_m), 0.0, 1.0).astype(np.float32)
    final_print[final_print < 0.02] = 0.0
    return MaskSet(
        printable_back=back_m.astype(np.float32),
        camera_exclusion=camera_m.astype(np.float32),
        edge_exclusion=edge_m.astype(np.float32),
        final_print=final_print,
        outer=outer_m.astype(np.float32),
    )


def override_camera_exclusion(masks: MaskSet, camera_aa: np.ndarray) -> MaskSet:
    """Replace camera cutout with a user-drawn antialiased mask (exact shape)."""
    cam = np.clip(np.asarray(camera_aa, dtype=np.float32), 0.0, 1.0)
    if cam.ndim == 3:
        cam = cam[..., 0]
    back = masks.printable_back.astype(np.float32)
    outer = masks.outer if masks.outer is not None else np.ones_like(back)
    if cam.shape != back.shape:
        cam = cv2.resize(cam, (back.shape[1], back.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam = np.minimum(cam, outer.astype(np.float32))
    final_print = np.clip(back * (1.0 - cam), 0.0, 1.0).astype(np.float32)
    final_print[final_print < 0.02] = 0.0
    return MaskSet(
        printable_back=back,
        camera_exclusion=cam.astype(np.float32),
        edge_exclusion=masks.edge_exclusion.astype(np.float32),
        final_print=final_print,
        outer=outer.astype(np.float32) if outer is not None else None,
    )


def _camera_aa_mask(camera_bin: np.ndarray, feather: float) -> np.ndarray:
    """Hi-res AA for a compact island; SDF-only if the blob is a smear."""
    if camera_bin is None or not np.any(camera_bin):
        shape = camera_bin.shape[:2] if camera_bin is not None else (0, 0)
        return np.zeros(shape, dtype=np.float32)
    ys, xs = np.where(camera_bin)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    area = float(np.count_nonzero(camera_bin))
    aspect = max(w, h) / float(max(min(w, h), 1))
    extent = area / float(max(w * h, 1))
    if aspect <= 2.05 and extent >= 0.52:
        return _island_aa_mask(camera_bin, feather)
    return binary_to_antialiased(camera_bin, feather)


def refine_outer_binary(binary: np.ndarray) -> np.ndarray:
    """Replace a jagged silhouette with a filled, uniformly smoothed contour."""
    if binary is None or not np.any(binary):
        return binary
    finished = fill_binary_holes(np.asarray(binary).astype(bool))
    pts = _finish_wrap_polyline(_contour_points(finished))
    if pts is None:
        return finished
    canvas = np.zeros(finished.shape, dtype=np.uint8)
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [poly], 255)
    filled = fill_binary_holes(canvas > 0)
    if float(np.count_nonzero(filled)) < 0.92 * max(float(np.count_nonzero(finished)), 1.0):
        return finished
    return filled


def silhouette_to_aa_mask(binary: np.ndarray, feather: float) -> np.ndarray:
    """Rasterize the real cover contour after spline/arc finishing, at full-res AA."""
    if binary is None or not np.any(binary):
        shape = binary.shape[:2] if binary is not None else (0, 0)
        return np.zeros(shape, dtype=np.float32)
    finished = fill_binary_holes(binary.astype(bool))
    pts = _finish_wrap_polyline(_contour_points(finished))
    if pts is None:
        return _island_aa_mask(finished, feather)
    cap = dilate_binary(finished, 2.4)
    return _rasterize_wrap_polyline(pts, finished.shape, feather, cap_bin=cap, cap_slack=1.6)


def _island_aa_mask(binary: np.ndarray, feather: float) -> np.ndarray:
    """Previous silhouette AA path — used for camera islands only."""
    from PIL import Image

    if binary is None or not np.any(binary):
        shape = binary.shape[:2] if binary is not None else (0, 0)
        return np.zeros(shape, dtype=np.float32)

    finished = fill_binary_holes(binary.astype(bool))
    height, width = finished.shape
    scale = _aa_scale(height, width)
    hi_w, hi_h = width * scale, height * scale

    orig_hi = cv2.resize(
        finished.astype(np.float32),
        (hi_w, hi_h),
        interpolation=cv2.INTER_LINEAR,
    )
    orig_bin = (orig_hi >= 0.45).astype(np.uint8)
    cap = cv2.dilate(orig_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    floor = cv2.erode(orig_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    canvas = orig_bin.copy()
    contour = _main_contour(finished)
    if contour is not None and cv2.contourArea(contour) > 32:
        pts = contour.reshape(-1, 2).astype(np.float64)
        pts = _smooth_closed_polyline(pts, sigma_frac=0.0028)
        peri = float(cv2.arcLength(contour, True))
        npts = max(len(pts), int(peri * scale * 1.25), 64)
        pts = _resample_closed_polyline(pts, npts)
        poly = np.round(pts * scale).astype(np.int32).reshape(-1, 1, 2)
        smooth_hi = np.zeros((hi_h, hi_w), dtype=np.uint8)
        cv2.fillPoly(smooth_hi, [poly], 1, lineType=cv2.LINE_AA)
        canvas = np.maximum(floor, np.minimum(smooth_hi, cap))

    dist_in = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, 5)
    sdf = np.where(canvas > 0, -dist_in, dist_out).astype(np.float32)
    coverage = coverage_from_sdf(sdf, max(float(feather), 0.75) * scale)

    cov_u8 = np.clip(coverage * 255.0 + 0.5, 0, 255).astype(np.uint8)
    down = Image.fromarray(cov_u8, mode="L").resize((width, height), Image.Resampling.BOX)
    return (np.asarray(down).astype(np.float32) / 255.0)


def _aa_scale(height: int, width: int) -> int:
    from app.utils.constants import MASK_SUPERSAMPLE

    area = float(max(height * width, 1))
    wanted = max(2, int(MASK_SUPERSAMPLE))
    for scale in range(wanted, 1, -1):
        if area * scale * scale <= 24_000_000:
            return scale
    return 2


def _wrap_aa_scale(height: int, width: int) -> int:
    """Supersample the cover crop as high as memory allows (not the full frame)."""
    area = float(max(height * width, 1))
    for scale in (8, 6, 5, 4, 3, 2):
        if area * scale * scale <= 96_000_000:
            return int(scale)
    return 2


def _contour_points(binary: np.ndarray) -> np.ndarray | None:
    contour = _main_contour(binary)
    if contour is None or cv2.contourArea(contour) < 32:
        return None
    return contour.reshape(-1, 2).astype(np.float64)


def _median_boundary_inset(outer: np.ndarray, inner: np.ndarray) -> float:
    """Typical gap from the outer silhouette to the inner printable lip."""
    if inner is None or not np.any(inner) or not np.any(outer):
        return 0.0
    dist = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 5)
    u8 = inner.astype(np.uint8)
    ring = cv2.dilate(u8, np.ones((3, 3), np.uint8)) & (1 - u8)
    if not np.any(ring):
        return 0.0
    vals = dist[ring.astype(bool)]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _clamp_polyline_inside(pts: np.ndarray, binary: np.ndarray, slack: float = 0.35) -> np.ndarray:
    """Keep a finished contour from ballooning into the bumper / background."""
    if pts is None or pts.shape[0] < 8 or binary is None or not np.any(binary):
        return pts
    height, width = binary.shape[:2]
    outside = (~binary.astype(bool)).astype(np.uint8)
    dist_out = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    xs = np.clip(pts[:, 0].astype(np.float32), 0.0, float(width - 1))
    ys = np.clip(pts[:, 1].astype(np.float32), 0.0, float(height - 1))
    sampled = cv2.remap(
        dist_out.astype(np.float32),
        xs.reshape(1, -1),
        ys.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()
    over = sampled > float(slack)
    if not np.any(over):
        return pts
    centroid = pts.mean(axis=0)
    inward = centroid - pts
    lengths = np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-6)
    inward = inward / lengths
    out = pts.copy()
    extra = (sampled[over] - float(slack) + 0.15)[:, None]
    out[over] = pts[over] + inward[over] * extra
    return out


def _finish_wrap_polyline(pts: np.ndarray | None) -> np.ndarray | None:
    """Sub-pixel regularized phone contour — eliminates side button bumps and bottom shadow leaks."""
    if pts is None or pts.shape[0] < 16:
        return pts
    quad = _fit_perspective_phone_quad(pts)
    if quad is not None:
        radius_ratio = _estimate_quad_corner_radius(pts, quad)
        return rounded_quad_polyline(quad, radius_ratio=radius_ratio, num_arc_pts=28)

    peri = float(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum())
    dense_n = int(np.clip(peri * 1.25, 256, 4096))
    work = _resample_closed_polyline(pts.astype(np.float64), dense_n)
    work = _rolling_median_closed(work, 5)
    work = _chaikin_closed(work, iterations=2)
    if work.shape[0] > 2048:
        work = _resample_closed_polyline(work, 2048)
    work = _smooth_closed_polyline(work, sigma_frac=0.0035)
    return work


def _estimate_quad_corner_radii(pts: np.ndarray | None, quad: np.ndarray) -> list[float]:
    """Measure the physical corner radius for all 4 corners (TL, TR, BR, BL) individually."""
    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bot = np.linalg.norm(quad[2] - quad[3])
    avg_w = float((w_top + w_bot) / 2.0)
    default_r = 0.126 * avg_w

    if pts is None or pts.shape[0] < 32 or avg_w < 10.0:
        return [default_r, default_r, default_r, default_r]

    radii = []
    for corner in quad:
        dist = np.linalg.norm(pts - corner, axis=1)
        close_pts = pts[dist < 0.26 * avg_w]
        r_val = default_r
        if close_pts.shape[0] >= 8:
            fit = _fit_circle(close_pts)
            if fit is not None:
                _cx, _cy, r = fit
                if 0.110 * avg_w <= r <= 0.145 * avg_w:
                    r_val = float(r)
        radii.append(r_val)

    med_r = float(np.median(radii)) if radii else default_r
    return [r if 0.110 * avg_w <= r <= 0.145 * avg_w else med_r for r in radii]


def _estimate_quad_corner_radius(pts: np.ndarray | None, quad: np.ndarray) -> float:
    radii = _estimate_quad_corner_radii(pts, quad)
    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bot = np.linalg.norm(quad[2] - quad[3])
    avg_w = float((w_top + w_bot) / 2.0)
    return float(np.median(radii) / max(avg_w, 1.0))


def _fit_perspective_phone_quad(pts: np.ndarray | None) -> np.ndarray | None:
    """Fit 4 robust edge lines to phone contour in perspective space to handle trapezoid taper."""
    if pts is None or pts.shape[0] < 32:
        return None

    pts_f = pts.astype(np.float32).reshape(-1, 1, 2)
    (cx0, cy0), (rw, rh), angle = cv2.minAreaRect(pts_f)
    if rw > rh:
        rw, rh = rh, rw
        angle = angle + 90.0

    aspect = rh / float(max(rw, 1.0))
    if aspect < 1.15 or aspect > 3.2:
        return None

    rad = np.deg2rad(angle)
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))

    pts_2d = pts.astype(np.float64)
    dx = pts_2d[:, 0] - cx0
    dy = pts_2d[:, 1] - cy0
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    mid_h = 0.35 * rh
    mid_w = 0.35 * rw
    left_mask = (u < 0) & (np.abs(v) < mid_h)
    right_mask = (u > 0) & (np.abs(v) < mid_h)
    top_mask = (v < 0) & (np.abs(u) < mid_w)
    bot_mask = (v > 0) & (np.abs(u) < mid_w) & (np.abs(u) > 0.06 * rw)
    if not bot_mask.any():
        bot_mask = (v > 0) & (np.abs(u) < mid_w)

    if not (left_mask.any() and right_mask.any() and top_mask.any() and bot_mask.any()):
        return None

    def robust_line_fit(subset_pts, is_vertical=True):
        if subset_pts.shape[0] < 6:
            return None
        vx, vy, x0, y0 = cv2.fitLine(subset_pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
        d = np.array([float(vx.flat[0]), float(vy.flat[0])], dtype=np.float64)
        p = np.array([float(x0.flat[0]), float(y0.flat[0])], dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return None
        d = d / norm
        n = np.array([-d[1], d[0]])
        dist = np.abs(np.dot(subset_pts - p, n))
        inliers = subset_pts[dist < max(3.0, np.median(dist) * 2.5)]
        if inliers.shape[0] >= 6:
            vx, vy, x0, y0 = cv2.fitLine(inliers.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
            d = np.array([float(vx.flat[0]), float(vy.flat[0])], dtype=np.float64)
            p = np.array([float(x0.flat[0]), float(y0.flat[0])], dtype=np.float64)
            d = d / max(np.linalg.norm(d), 1e-6)

        if is_vertical and d[1] < 0:
            d = -d
        elif not is_vertical and d[0] < 0:
            d = -d
        return p, d

    res_l = robust_line_fit(pts_2d[left_mask], is_vertical=True)
    res_r = robust_line_fit(pts_2d[right_mask], is_vertical=True)
    res_t = robust_line_fit(pts_2d[top_mask], is_vertical=False)
    res_b = robust_line_fit(pts_2d[bot_mask], is_vertical=False)

    if res_l is None or res_r is None or res_t is None or res_b is None:
        return None

    p_left, d_left = res_l
    p_right, d_right = res_r
    p_top, d_top = res_t
    p_bot, d_bot = res_b

    def line_intersection(p1, d1, p2, d2):
        mat = np.column_stack([d1, -d2])
        det = np.linalg.det(mat)
        if abs(det) < 1e-6:
            return (p1 + p2) / 2.0
        t = np.linalg.solve(mat, p2 - p1)[0]
        return p1 + t * d1

    v_tl = line_intersection(p_top, d_top, p_left, d_left)
    v_tr = line_intersection(p_top, d_top, p_right, d_right)
    v_br = line_intersection(p_bot, d_bot, p_right, d_right)
    v_bl = line_intersection(p_bot, d_bot, p_left, d_left)

    return np.array([v_tl, v_tr, v_br, v_bl], dtype=np.float64)


def inset_quad(quad: np.ndarray, inset_px: float) -> np.ndarray:
    """Offset each quadrilateral edge inward by exactly inset_px perpendicular distance."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centroid = quad.mean(axis=0)
    shifted_lines = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        tangent = p2 - p1
        length = np.linalg.norm(tangent)
        u_tangent = tangent / max(length, 1e-6)
        n = np.array([-u_tangent[1], u_tangent[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, centroid - mid) < 0:
            n = -n
        p_shifted = p1 + n * float(inset_px)
        shifted_lines.append((p_shifted, u_tangent))

    def line_intersection(p1, d1, p2, d2):
        mat = np.column_stack([d1, -d2])
        det = np.linalg.det(mat)
        if abs(det) < 1e-6:
            return (p1 + p2) / 2.0
        t = np.linalg.solve(mat, p2 - p1)[0]
        return p1 + t * d1

    v0_new = line_intersection(shifted_lines[3][0], shifted_lines[3][1], shifted_lines[0][0], shifted_lines[0][1])
    v1_new = line_intersection(shifted_lines[0][0], shifted_lines[0][1], shifted_lines[1][0], shifted_lines[1][1])
    v2_new = line_intersection(shifted_lines[1][0], shifted_lines[1][1], shifted_lines[2][0], shifted_lines[2][1])
    v3_new = line_intersection(shifted_lines[2][0], shifted_lines[2][1], shifted_lines[3][0], shifted_lines[3][1])
    return np.array([v0_new, v1_new, v2_new, v3_new], dtype=np.float64)


def rounded_quad_polyline(
    quad: np.ndarray,
    radius_ratio: float | list[float] = 0.126,
    num_arc_pts: int = 36,
) -> np.ndarray:
    """Generate dense, smooth closed polyline with true inward circular corner arcs."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bot = np.linalg.norm(quad[2] - quad[3])
    avg_w = (w_top + w_bot) / 2.0

    if isinstance(radius_ratio, (list, tuple, np.ndarray)):
        r_list = [float(r) if float(r) > 1.0 else float(r) * avg_w for r in radius_ratio]
    else:
        r_val = float(radius_ratio) if float(radius_ratio) > 1.0 else float(radius_ratio) * avg_w
        r_list = [r_val] * 4

    segs = []
    centroid = quad.mean(axis=0)

    for i in range(4):
        r = r_list[i]
        v_prev = quad[(i - 1) % 4]
        v_curr = quad[i]
        v_next = quad[(i + 1) % 4]

        d_in = v_curr - v_prev
        len_in = np.linalg.norm(d_in)
        u_in = d_in / max(len_in, 1e-6)

        d_out = v_next - v_curr
        len_out = np.linalg.norm(d_out)
        u_out = d_out / max(len_out, 1e-6)

        n_in = np.array([-u_in[1], u_in[0]])
        if np.dot(n_in, centroid - v_curr) < 0:
            n_in = -n_in

        cos_corner = np.clip(-np.dot(u_in, u_out), -1.0, 1.0)
        half_angle = np.arccos(cos_corner) / 2.0
        t_dist = min(r / max(np.tan(half_angle), 1e-4), 0.35 * min(len_in, len_out))
        actual_r = t_dist * np.tan(half_angle)

        p_in = v_curr - t_dist * u_in
        p_out = v_curr + t_dist * u_out

        center = p_in + n_in * actual_r

        v_start = p_in - center
        v_end = p_out - center
        ang_start = np.arctan2(v_start[1], v_start[0])
        ang_end = np.arctan2(v_end[1], v_end[0])

        diff = (ang_end - ang_start + np.pi) % (2 * np.pi) - np.pi
        angles = ang_start + diff * np.linspace(0.0, 1.0, num_arc_pts, endpoint=False)
        arc = center + actual_r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        segs.append(arc)

        next_v = v_next
        next_d_in = next_v - v_curr
        next_len_in = np.linalg.norm(next_d_in)
        next_u_in = next_d_in / max(next_len_in, 1e-6)
        next_d_out = quad[(i + 2) % 4] - next_v
        next_u_out = next_d_out / max(np.linalg.norm(next_d_out), 1e-6)
        next_cos = np.clip(-np.dot(next_u_in, next_u_out), -1.0, 1.0)
        next_t = min(r / max(np.tan(np.arccos(next_cos) / 2.0), 1e-4), 0.35 * min(next_len_in, np.linalg.norm(next_d_out)))
        next_p_in = next_v - next_t * next_u_in

        if np.linalg.norm(next_p_in - p_out) > 1.0:
            straight = np.linspace(p_out, next_p_in, 16, endpoint=False)
            segs.append(straight)

    return np.concatenate(segs, axis=0)


def _corner_proximity_mask(pts: np.ndarray, frac: float = 0.18) -> np.ndarray:
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    reach = frac * min(max(x1 - x0, 8.0), max(y1 - y0, 8.0))
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)
    d = np.linalg.norm(pts[:, None, :] - corners[None, :, :], axis=2).min(axis=1)
    return d <= reach


def _despike_side_bumps(pts: np.ndarray) -> np.ndarray:
    """Clamp leftover notches/protrusions on the long edges; leave corner fans alone."""
    n = pts.shape[0]
    if n < 48:
        return pts
    win = int(np.clip(0.018 * n, 11, 41)) | 1
    med = _rolling_median_closed(pts, win)
    near_c = _corner_proximity_mask(pts, 0.17)
    prev = np.roll(med, 1, axis=0)
    nxt = np.roll(med, -1, axis=0)
    tang = nxt - prev
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    lengths = np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    normal = normal / lengths
    signed = np.sum((pts - med) * normal, axis=1)
    cap = 1.35
    clamped = np.clip(signed, -cap, cap)
    out = med + normal * clamped[:, None]
    out[near_c] = pts[near_c]
    return out


def _smooth_sides_keep_corners(pts: np.ndarray) -> np.ndarray:
    """Heavier Gaussian on straight rims; corners stay for circular reconstruction."""
    if pts.shape[0] < 32:
        return pts
    near_c = _corner_proximity_mask(pts, 0.16)
    smoothed = _smooth_closed_polyline(pts, sigma_frac=0.0042)
    out = pts.copy()
    out[~near_c] = smoothed[~near_c]
    return out


def _harmonize_four_corners(pts: np.ndarray) -> np.ndarray:
    """Same radius and tangent construction on every rounded corner."""
    if pts.shape[0] < 64:
        return pts
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = max(x1 - x0, 8.0), max(y1 - y0, 8.0)
    min_side = min(bw, bh)
    walk = 0.18 * min_side
    corners = (
        np.array([x0, y0], dtype=np.float64),
        np.array([x1, y0], dtype=np.float64),
        np.array([x0, y1], dtype=np.float64),
        np.array([x1, y1], dtype=np.float64),
    )
    radii: list[float] = []
    for corner in corners:
        fitted = _measure_corner_radius(pts, corner, walk, min_side)
        if fitted is not None:
            radii.append(fitted)
    if radii:
        radius = float(np.median(radii))
    else:
        radius = 0.12 * min_side
    radius = float(np.clip(radius, 0.07 * min_side, 0.18 * min_side))
    lines = _side_lines(pts, x0, y0, x1, y1)
    out = pts
    pairs = (
        (corners[0], lines["top"], lines["left"]),
        (corners[1], lines["top"], lines["right"]),
        (corners[2], lines["bottom"], lines["left"]),
        (corners[3], lines["bottom"], lines["right"]),
    )
    centroid = pts.mean(axis=0)
    for corner, line_a, line_b in pairs:
        if line_a is None or line_b is None:
            out = _replace_corner_with_arc(out, corner, walk, min_side, radius_override=radius)
            continue
        rebuilt = _replace_corner_from_tangents(out, corner, line_a, line_b, radius, centroid)
        out = rebuilt
    return out


def _side_lines(
    pts: np.ndarray, x0: float, y0: float, x1: float, y1: float
) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    w, h = max(x1 - x0, 8.0), max(y1 - y0, 8.0)
    mx, my = 0.16 * w, 0.16 * h
    bands = {
        "left": pts[(pts[:, 0] <= x0 + 0.10 * w) & (pts[:, 1] > y0 + my) & (pts[:, 1] < y1 - my)],
        "right": pts[(pts[:, 0] >= x1 - 0.10 * w) & (pts[:, 1] > y0 + my) & (pts[:, 1] < y1 - my)],
        "top": pts[(pts[:, 1] <= y0 + 0.10 * h) & (pts[:, 0] > x0 + mx) & (pts[:, 0] < x1 - mx)],
        "bottom": pts[(pts[:, 1] >= y1 - 0.10 * h) & (pts[:, 0] > x0 + mx) & (pts[:, 0] < x1 - mx)],
    }
    out: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for name, sample in bands.items():
        out[name] = _fit_line_segment(sample) if sample.shape[0] >= 12 else None
    return out


def _fit_line_segment(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if xy.shape[0] < 8:
        return None
    vx, vy, px, py = cv2.fitLine(xy.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vx.flat[0]), float(vy.flat[0])], dtype=np.float64)
    origin = np.array([float(px.flat[0]), float(py.flat[0])], dtype=np.float64)
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-8:
        return None
    return origin, direction / nrm


def _measure_corner_radius(
    pts: np.ndarray, corner: np.ndarray, walk: float, min_side: float
) -> float | None:
    n = pts.shape[0]
    dist = np.linalg.norm(pts - corner, axis=1)
    idxs = np.where(dist <= walk)[0]
    if idxs.size < 12:
        return None
    # Keep the contiguous fan around the nearest point.
    i0 = int(np.argmin(dist))
    span = _wrapped_index_range(*_expand_to_walk(pts, i0, corner, walk), n)
    xy = pts[span]
    fit = _fit_circle(xy)
    if fit is None:
        return None
    _cx, _cy, radius = fit
    if radius < 0.06 * min_side or radius > 0.20 * min_side:
        return None
    residual = np.abs(np.linalg.norm(xy - np.array([_cx, _cy]), axis=1) - radius)
    if float(np.median(residual)) > 2.6:
        return None
    return float(radius)


def _expand_to_walk(pts: np.ndarray, i0: int, corner: np.ndarray, walk: float) -> tuple[int, int]:
    n = pts.shape[0]
    left = i0
    for _ in range(n):
        nxt = (left - 1) % n
        if float(np.linalg.norm(pts[nxt] - corner)) > walk:
            break
        left = nxt
        if left == i0:
            break
    right = i0
    for _ in range(n):
        nxt = (right + 1) % n
        if float(np.linalg.norm(pts[nxt] - corner)) > walk:
            break
        right = nxt
        if right == i0:
            break
    return left, right


def _replace_corner_from_tangents(
    pts: np.ndarray,
    corner: np.ndarray,
    line_a: tuple[np.ndarray, np.ndarray],
    line_b: tuple[np.ndarray, np.ndarray],
    radius: float,
    centroid: np.ndarray,
) -> np.ndarray:
    pa, da = line_a
    pb, db = line_b
    na = _inward_normal(da, pa, centroid)
    nb = _inward_normal(db, pb, centroid)
    oa = pa + na * radius
    ob = pb + nb * radius
    center = _intersect_lines(oa, da, ob, db)
    if center is None:
        return pts
    # Tangent points on the original side lines.
    ta = center - na * radius
    tb = center - nb * radius
    i_a = int(np.argmin(np.linalg.norm(pts - ta, axis=1)))
    i_b = int(np.argmin(np.linalg.norm(pts - tb, axis=1)))
    i_c = int(np.argmin(np.linalg.norm(pts - corner, axis=1)))
    n = pts.shape[0]
    range_ab = _wrapped_index_range(i_a, i_b, n)
    range_ba = _wrapped_index_range(i_b, i_a, n)
    idxs = range_ab if i_c in set(range_ab.tolist()) else range_ba
    if len(idxs) < 8:
        return pts
    a0 = np.arctan2(ta[1] - center[1], ta[0] - center[0])
    a1 = np.arctan2(tb[1] - center[1], tb[0] - center[0])
    da_ang = a1 - a0
    while da_ang > np.pi:
        da_ang -= 2 * np.pi
    while da_ang < -np.pi:
        da_ang += 2 * np.pi
    # The short 90° turn should point toward the bbox corner.
    mid_short = np.array(
        [center[0] + radius * np.cos(a0 + da_ang * 0.5), center[1] + radius * np.sin(a0 + da_ang * 0.5)]
    )
    mid_long = np.array(
        [
            center[0] + radius * np.cos(a0 + (da_ang - np.sign(da_ang) * 2 * np.pi) * 0.5),
            center[1] + radius * np.sin(a0 + (da_ang - np.sign(da_ang) * 2 * np.pi) * 0.5),
        ]
    )
    if float(np.linalg.norm(mid_long - corner)) < float(np.linalg.norm(mid_short - corner)):
        da_ang = da_ang - np.sign(da_ang) * 2 * np.pi if da_ang != 0 else np.pi
    if abs(da_ang) < 0.40 or abs(da_ang) > 2.15:
        return pts
    angles = np.linspace(a0, a0 + da_ang, len(idxs))
    arc = np.stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)],
        axis=1,
    )
    out = pts.copy()
    out[idxs] = arc
    return out


def _inward_normal(direction: np.ndarray, origin: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    nrm = np.array([-direction[1], direction[0]], dtype=np.float64)
    length = float(np.linalg.norm(nrm))
    if length < 1e-8:
        nrm = centroid - origin
        length = float(np.linalg.norm(nrm)) or 1.0
    nrm = nrm / length
    if float(np.dot(nrm, centroid - origin)) < 0:
        nrm = -nrm
    return nrm


def _intersect_lines(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    mat = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-8:
        return None
    delta = p2 - p1
    t = float(np.linalg.solve(mat, delta)[0])
    return p1 + t * d1


def _rasterize_wrap_polyline(
    pts: np.ndarray,
    shape: tuple[int, int],
    feather: float,
    cap_bin: np.ndarray | None = None,
    cap_slack: float = 1.25,
) -> np.ndarray:
    """Fill a finished contour on a supersampled crop, then BOX-downsample."""
    from PIL import Image

    height, width = int(shape[0]), int(shape[1])
    if pts is None or pts.shape[0] < 8:
        return np.zeros((height, width), dtype=np.float32)

    pad = 14
    x0 = int(max(0, np.floor(pts[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - pad))
    x1 = int(min(width, np.ceil(pts[:, 0].max()) + pad + 1))
    y1 = int(min(height, np.ceil(pts[:, 1].max()) + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((height, width), dtype=np.float32)

    cw, ch = x1 - x0, y1 - y0
    scale = _wrap_aa_scale(ch, cw)
    local = pts - np.array([x0, y0], dtype=np.float64)
    peri = float(np.linalg.norm(np.diff(np.vstack([local, local[0]]), axis=0), axis=1).sum())
    npts = int(min(6144, max(512, peri * scale * 1.35)))
    dense = _catmull_rom_closed(local, npts)

    hi_w, hi_h = cw * scale, ch * scale
    shift = 3
    mul = 1 << shift
    poly = np.round(dense * scale * mul).astype(np.int32).reshape(-1, 1, 2)
    hi = np.zeros((hi_h, hi_w), dtype=np.uint8)
    cv2.fillPoly(hi, [poly], 255, lineType=cv2.LINE_AA, shift=shift)

    if cap_bin is not None and np.any(cap_bin):
        outside = (~cap_bin[y0:y1, x0:x1].astype(bool)).astype(np.uint8)
        dist_cap = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        dist_hi = cv2.resize(
            dist_cap.astype(np.float32),
            (hi_w, hi_h),
            interpolation=cv2.INTER_LINEAR,
        )
        hi[dist_hi > (float(cap_slack) * scale)] = 0

    coverage = hi.astype(np.float32) / 255.0
    # Blur in hi-res pixels so BOX downsample yields ~1px native AA, not a binary edge.
    sigma = float(np.clip(0.42 * scale, 0.85, 2.15))
    coverage = cv2.GaussianBlur(coverage, (0, 0), sigmaX=sigma)

    cov_u8 = np.clip(coverage * 255.0 + 0.5, 0, 255).astype(np.uint8)
    down = Image.fromarray(cov_u8, mode="L").resize((cw, ch), Image.Resampling.BOX)
    crop = np.asarray(down).astype(np.float32) / 255.0
    crop[crop < 0.02] = 0.0
    out = np.zeros((height, width), dtype=np.float32)
    out[y0:y1, x0:x1] = crop
    return out


def _rolling_median_closed(pts: np.ndarray, k: int = 5) -> np.ndarray:
    k = int(k) | 1
    if pts.shape[0] < k:
        return pts
    pad = k // 2
    ext = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0)
    out = np.empty_like(pts, dtype=np.float64)
    for i in range(pts.shape[0]):
        window = ext[i : i + k]
        out[i] = np.median(window, axis=0)
    return out


def _chaikin_closed(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    work = pts.astype(np.float64)
    for _ in range(max(1, int(iterations))):
        nxt = np.roll(work, -1, axis=0)
        q = 0.75 * work + 0.25 * nxt
        r = 0.25 * work + 0.75 * nxt
        stacked = np.empty((work.shape[0] * 2, 2), dtype=np.float64)
        stacked[0::2] = q
        stacked[1::2] = r
        work = stacked
    return work


def _catmull_rom_closed(pts: np.ndarray, count: int) -> np.ndarray:
    """C1 cubic interpolation of a closed polyline, even arc-length samples."""
    n = int(pts.shape[0])
    count = int(max(count, n, 32))
    if n < 4:
        return _resample_closed_polyline(pts, count)
    closed = np.vstack([pts, pts[0]])
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    length = float(seg.sum())
    if length < 1.0:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, length, count, endpoint=False)
    i = np.searchsorted(cum, samples, side="right") - 1
    i = np.clip(i, 0, n - 1)
    span = np.maximum(seg[i], 1e-9)
    t = ((samples - cum[i]) / span)[:, None]
    p0 = pts[(i - 1) % n]
    p1 = pts[i % n]
    p2 = pts[(i + 1) % n]
    p3 = pts[(i + 2) % n]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _offset_closed_polyline(pts: np.ndarray, delta: float) -> np.ndarray:
    """Move a closed contour along its outward normal (positive = grow)."""
    if abs(float(delta)) < 1e-6 or pts.shape[0] < 8:
        return pts
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tangent = nxt - prev
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-6)
    normal = normal / lengths
    centroid = pts.mean(axis=0)
    # findContours outer is CCW: (ty, -tx) points outward. Flip if not.
    probe = pts[0] + normal[0]
    inward = centroid - pts[0]
    if float(np.dot(probe - pts[0], inward)) > 0:
        normal = -normal
    return pts + normal * float(delta)


def _blend_measured_corner_arcs(pts: np.ndarray) -> np.ndarray:
    """Replace only the four rounded-corner spans with fitted circular arcs."""
    if pts.shape[0] < 64:
        return pts
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = max(x1 - x0, 8.0), max(y1 - y0, 8.0)
    min_side = min(bw, bh)
    walk = 0.165 * min_side
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    out = pts.copy()
    for corner in corners:
        out = _replace_corner_with_arc(out, np.asarray(corner, dtype=np.float64), walk, min_side)
    return out


def _replace_corner_with_arc(
    pts: np.ndarray,
    corner: np.ndarray,
    walk: float,
    min_side: float,
    radius_override: float | None = None,
) -> np.ndarray:
    n = pts.shape[0]
    dist = np.linalg.norm(pts - corner, axis=1)
    i0 = int(np.argmin(dist))
    left, right = _expand_to_walk(pts, i0, corner, walk)
    idxs = _wrapped_index_range(left, right, n)
    if len(idxs) < 10:
        return pts
    xy = pts[idxs]
    fit = _fit_circle(xy)
    if fit is None and radius_override is None:
        return pts
    if fit is not None:
        cx, cy, radius = fit
    else:
        radius = float(radius_override)
        # Axis-aligned fallback center.
        sign = np.sign(np.array([pts[:, 0].mean(), pts[:, 1].mean()]) - corner)
        sign[sign == 0] = 1
        cx = float(corner[0] + sign[0] * radius)
        cy = float(corner[1] + sign[1] * radius)
    if radius_override is not None:
        radius = float(radius_override)
        sign = np.sign(np.array([pts[:, 0].mean(), pts[:, 1].mean()]) - corner)
        sign[sign == 0] = 1.0
        cx = float(corner[0] + sign[0] * radius)
        cy = float(corner[1] + sign[1] * radius)
    if radius < 0.055 * min_side or radius > 0.22 * min_side:
        return pts
    if radius_override is None:
        residual = np.abs(np.linalg.norm(xy - np.array([cx, cy]), axis=1) - radius)
        if float(np.median(residual)) > 2.15:
            return pts
    mid = np.array([(pts[:, 0].min() + pts[:, 0].max()) * 0.5, (pts[:, 1].min() + pts[:, 1].max()) * 0.5])
    if float(np.dot(np.array([cx, cy]) - corner, mid - corner)) < 0:
        return pts
    a0 = np.arctan2(xy[0, 1] - cy, xy[0, 0] - cx)
    a1 = np.arctan2(xy[-1, 1] - cy, xy[-1, 0] - cx)
    da = a1 - a0
    while da > np.pi:
        da -= 2 * np.pi
    while da < -np.pi:
        da += 2 * np.pi
    if abs(da) < 0.35 or abs(da) > 2.0:
        return pts
    angles = np.linspace(a0, a0 + da, len(idxs))
    arc = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    out = pts.copy()
    out[idxs] = arc
    return out


def _wrapped_index_range(start: int, end: int, n: int) -> np.ndarray:
    if start <= end:
        return np.arange(start, end + 1)
    return np.concatenate([np.arange(start, n), np.arange(0, end + 1)])


def _fit_circle(xy: np.ndarray) -> tuple[float, float, float] | None:
    if xy.shape[0] < 6:
        return None
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones(xy.shape[0])])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if r2 <= 1.0:
        return None
    return cx, cy, float(np.sqrt(r2))


def finish_silhouette(binary: np.ndarray) -> np.ndarray:
    """1-pixel morph cleanup only — large kernels bulge corners."""
    if binary is None or not binary.any():
        return np.zeros(binary.shape[:2], dtype=bool) if binary is not None else binary
    u8 = (binary.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    filled = fill_binary_holes(closed > 0)
    return _largest_blob(filled)


def refine_panel_shape(binary: np.ndarray) -> np.ndarray:
    """Clean the detected outline while keeping its real curvature."""
    return finish_silhouette(binary)


def _trim_aa_tail(coverage: np.ndarray, tail: float) -> np.ndarray:
    """Zero the faintest AA fringe; keep a linear ramp so the rim stays flush."""
    t = float(np.clip(tail, 0.0, 0.4))
    return np.clip((coverage.astype(np.float32) - t) / max(1.0 - t, 1e-4), 0.0, 1.0)


def _main_contour(binary: np.ndarray) -> np.ndarray | None:
    u8 = (binary.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _largest_blob(binary: np.ndarray) -> np.ndarray:
    u8 = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return binary.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _smooth_closed_polyline(pts: np.ndarray, sigma_frac: float = 0.004) -> np.ndarray:
    """Periodic Gaussian smoothing of a closed contour (pixel coordinates)."""
    if pts.shape[0] < 12:
        return pts.astype(np.float64)
    n = int(pts.shape[0])
    sigma = float(max(1.2, sigma_frac * n))
    pad = int(max(3, round(sigma * 3)))
    ext = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0).astype(np.float32)
    xs = cv2.GaussianBlur(ext[:, 0].reshape(-1, 1), (0, 0), sigmaX=sigma).ravel()
    ys = cv2.GaussianBlur(ext[:, 1].reshape(-1, 1), (0, 0), sigmaX=sigma).ravel()
    return np.stack([xs[pad : pad + n], ys[pad : pad + n]], axis=1).astype(np.float64)


def _resample_closed_polyline(pts: np.ndarray, count: int) -> np.ndarray:
    """Even arc-length resampling so fillPoly edges stay dense after smoothing."""
    if pts.shape[0] < 3:
        return pts
    closed = np.vstack([pts, pts[0]])
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    length = float(seg.sum())
    if length < 1.0:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, length, int(count), endpoint=False)
    out = np.empty((int(count), 2), dtype=np.float64)
    out[:, 0] = np.interp(samples, cum, closed[:, 0])
    out[:, 1] = np.interp(samples, cum, closed[:, 1])
    return out


def refined_outer_sdf(binary: np.ndarray) -> np.ndarray | None:
    """Signed distance of the printable outer: rounded-rect SDF union smoothed contour.

    Negative inside. Corner arcs come from the analytic rounded rectangle fitted to
    the detected bbox, not from a polygonal approximation of Canny pixels.
    """
    if binary is None or not binary.any():
        return None
    height, width = binary.shape
    box = _mask_bbox(binary)
    if box is None:
        return None
    x, y, bw, bh = box
    radius = _estimate_corner_radius(binary, x, y, bw, bh)
    x, y, bw, bh = _expand_bbox_for_chords(binary, x, y, bw, bh, radius)
    rect = PixelRect(float(x), float(y), float(bw), float(bh), float(radius))
    sdf_round = _rounded_rect_sdf(height, width, rect)
    smoothed = smooth_silhouette(binary, sigma=max(1.4, 0.0035 * min(bw, bh)))
    sdf_smooth = _binary_sdf(smoothed)
    sdf_union = np.minimum(sdf_round, sdf_smooth)
    pad = max(6.0, min(radius * 0.38, 0.028 * min(bw, bh)))
    sdf_allow = _binary_sdf(dilate_binary(binary, pad))
    return np.maximum(sdf_union, sdf_allow).astype(np.float32)


def _binary_sdf(binary: np.ndarray) -> np.ndarray:
    inside = (binary > 0).astype(np.uint8)
    if not inside.any():
        return np.full(binary.shape, 1.0e6, dtype=np.float32)
    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    return np.where(inside > 0, -dist_in, dist_out).astype(np.float32)


def smooth_silhouette(binary: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """Gaussian round-off of stair-steps without converting to a polygon."""
    if not binary.any() or sigma <= 0:
        return binary.astype(bool)
    k = int(max(3, round(sigma * 6))) | 1
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (k, k), sigmaX=float(sigma))
    return blurred >= 0.5


def inset_smooth(binary: np.ndarray, pixels: float, sigma: float = 1.2) -> np.ndarray:
    """Distance-transform inset, then a light blur so the inner lip stays smooth."""
    inset = inset_binary(binary, pixels)
    if sigma > 0:
        inset = smooth_silhouette(inset, sigma=sigma)
    return inset & binary


def binary_to_antialiased(binary: np.ndarray, feather: float) -> np.ndarray:
    """Signed-distance coverage, computed at 2× then downsampled."""
    from app.utils.constants import MASK_SUPERSAMPLE

    inside = (binary > 0).astype(np.uint8)
    if not inside.any():
        return np.zeros(binary.shape[:2], dtype=np.float32)
    scale = max(1, int(MASK_SUPERSAMPLE))
    height, width = inside.shape
    if scale > 1:
        hi = cv2.resize(
            inside.astype(np.float32),
            (width * scale, height * scale),
            interpolation=cv2.INTER_LINEAR,
        )
        hi_bin = (hi >= 0.5).astype(np.uint8)
        dist_in = cv2.distanceTransform(hi_bin, cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform(1 - hi_bin, cv2.DIST_L2, 5)
        sdf = np.where(hi_bin > 0, -dist_in, dist_out).astype(np.float32)
        coverage = coverage_from_sdf(sdf, float(feather) * scale)
        return cv2.resize(coverage, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    sdf = np.where(inside > 0, -dist_in, dist_out).astype(np.float32)
    return coverage_from_sdf(sdf, feather)


def _expand_bbox_for_chords(
    filled: np.ndarray,
    x: int,
    y: int,
    bw: int,
    bh: int,
    radius: float,
) -> tuple[int, int, int, int]:
    """If a bbox edge is a filled chord (missing round), grow that side slightly."""
    h, w = filled.shape
    extra = int(round(max(4.0, min(radius * 0.42, 0.03 * min(bw, bh)))))
    dt, db, dl, dr = 0, 0, 0, 0
    if float(filled[y, x : x + bw].mean()) > 0.82:
        dt = extra
    if float(filled[y + bh - 1, x : x + bw].mean()) > 0.82:
        db = extra
    if float(filled[y : y + bh, x].mean()) > 0.82:
        dl = extra
    if float(filled[y : y + bh, x + bw - 1].mean()) > 0.82:
        dr = extra
    x0 = max(0, x - dl)
    y0 = max(0, y - dt)
    x1 = min(w, x + bw + dr)
    y1 = min(h, y + bh + db)
    return x0, y0, x1 - x0, y1 - y0


def _mask_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _estimate_corner_radius(filled: np.ndarray, x: int, y: int, bw: int, bh: int) -> float:
    """Radius from how soon the silhouette meets the bbox edges (phone-like)."""
    min_side = float(min(bw, bh))
    default = 0.12 * min_side
    samples: list[float] = []
    h, w = filled.shape
    for off in (3, 5, 8):
        row = y + off
        if 0 <= row < h:
            cols = np.where(filled[row, x : x + bw])[0]
            if cols.size:
                samples.append(float(cols[0]))
        col = x + off
        if 0 <= col < w:
            rows = np.where(filled[y : y + bh, col])[0]
            if rows.size:
                samples.append(float(rows[0]))
        row_b = y + bh - 1 - off
        if 0 <= row_b < h:
            cols = np.where(filled[row_b, x : x + bw])[0]
            if cols.size:
                samples.append(float(cols[0]))
    if samples:
        r = float(np.median(samples))
        if r < 0.045 * min_side:
            r = default
        return float(np.clip(r, 0.07 * min_side, 0.20 * min_side))
    return default


def fill_binary_holes(binary: np.ndarray) -> np.ndarray:
    u8 = (binary.astype(np.uint8)) * 255
    h, w = u8.shape
    flood = u8.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    if u8[0, 0] == 0:
        cv2.floodFill(flood, mask, (0, 0), 255)
    else:
        ys, xs = np.where(u8 == 0)
        if ys.size == 0:
            return binary.astype(bool)
        cv2.floodFill(flood, mask, (int(xs[0]), int(ys[0])), 255)
    holes = cv2.bitwise_not(flood)
    return (u8 | holes) > 0


def inset_binary(binary: np.ndarray, pixels: float) -> np.ndarray:
    """Erode a binary mask by ``pixels`` using the distance transform."""
    inside = (binary > 0).astype(np.uint8)
    if not inside.any() or pixels <= 0:
        return inside.astype(bool)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    return dist > float(pixels)


def dilate_binary(binary: np.ndarray, pixels: float) -> np.ndarray:
    inside = (binary > 0).astype(np.uint8)
    if pixels <= 0:
        return inside.astype(bool)
    dist_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    return (inside > 0) | (dist_out <= float(pixels))


def coverage_from_sdf(sdf: np.ndarray, feather: float) -> np.ndarray:
    width = max(float(feather), 1e-4)
    return np.clip(0.5 - sdf / width, 0.0, 1.0).astype(np.float32)


def rounded_rect_mask(
    height: int,
    width: int,
    rect: PixelRect,
    feather: float,
) -> np.ndarray:
    sdf = _rounded_rect_sdf(height, width, rect)
    return coverage_from_sdf(sdf, feather)


def circle_mask(
    height: int,
    width: int,
    circle: PixelCircle,
    feather: float,
) -> np.ndarray:
    sdf = _circle_sdf(height, width, circle)
    return coverage_from_sdf(sdf, feather)


def _mesh(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[0:height, 0:width]
    px = xx.astype(np.float32) + 0.5
    py = yy.astype(np.float32) + 0.5
    return px, py


def _rounded_rect_sdf(height: int, width: int, rect: PixelRect) -> np.ndarray:
    px, py = _mesh(height, width)
    cx = rect.x + rect.w * 0.5
    cy = rect.y + rect.h * 0.5
    hx = rect.w * 0.5
    hy = rect.h * 0.5
    radius = float(min(max(rect.radius, 0.0), hx, hy))

    dx = np.abs(px - cx) - hx + radius
    dy = np.abs(py - cy) - hy + radius
    outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return (outside + inside - radius).astype(np.float32)


def _circle_sdf(height: int, width: int, circle: PixelCircle) -> np.ndarray:
    px, py = _mesh(height, width)
    return (np.sqrt((px - circle.cx) ** 2 + (py - circle.cy) ** 2) - circle.r).astype(np.float32)
