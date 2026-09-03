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
    camera_contour: np.ndarray | None = None,
    bumper_px: float | None = None,
    has_alpha: bool = False,
    cover_rgba: np.ndarray | None = None,
) -> MaskSet:
    """Antialiased masks from detected silhouettes — artwork terminates at the inner edge of the physical rim."""
    feather = float(feather_px)
    outer_src = fill_binary_holes(np.asarray(outer_bin).astype(bool))
    back_src = fill_binary_holes(np.asarray(back_bin).astype(bool)) & outer_src

    outer_pts = _detected_rim_polyline(outer_src)
    if outer_pts is not None:
        outer_m = _rasterize_wrap_polyline(outer_pts, outer_src.shape, feather, cap_bin=None)
    else:
        outer_m = silhouette_to_aa_mask(outer_src, feather)

    inner_pts = _finish_inner_lip_polyline(_contour_points(back_src))
    if inner_pts is not None and inner_pts.shape[0] >= 16:
        if cover_rgba is not None and outer_pts is not None:
            inner_pts = _trace_continuous_inner_rim(inner_pts, cover_rgba, outer_src, back_src)
        else:
            inner_pts = _snap_polyline_to_boundary(inner_pts, back_src)
        back_m = _rasterize_wrap_polyline(inner_pts, outer_src.shape, feather, cap_bin=outer_src)
    else:
        inset_px = _median_boundary_inset(outer_src, back_src)
        if bumper_px is not None and bumper_px > 0:
            inset_px = float(bumper_px)
            
        if outer_pts is not None and inset_px > 0.4:
            back_pts = _offset_closed_polyline(outer_pts, -float(inset_px))
            if cover_rgba is not None:
                back_pts = _trace_continuous_inner_rim(back_pts, cover_rgba, outer_src, back_src)
            else:
                back_pts = _snap_polyline_to_boundary(back_pts, back_src)
            back_m = _rasterize_wrap_polyline(back_pts, outer_src.shape, feather, cap_bin=outer_src)
        else:
            back_m = silhouette_to_aa_mask(back_src, feather)

    back_m = np.minimum(back_m, outer_m)
    back_m[back_m < 0.04] = 0.0

    if camera_contour is not None and camera_contour.shape[0] >= 12:
        camera_m = _camera_contour_aa_mask(camera_contour, camera_bin, feather)
    else:
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


def _camera_contour_aa_mask(
    contour: np.ndarray,
    camera_bin: np.ndarray,
    feather: float,
) -> np.ndarray:
    """High-quality AA camera mask from a refined contour polyline.

    Rasterizes the refined contour at supersampled resolution with exact
    SDF antialiasing and BOX downsampling, giving the camera rim crisp,
    sub-pixel antialiasing with zero bleed into the camera area.
    """
    from PIL import Image

    if camera_bin is None or not np.any(camera_bin):
        shape = camera_bin.shape[:2] if camera_bin is not None else (0, 0)
        return np.zeros(shape, dtype=np.float32)

    finished = fill_binary_holes(camera_bin.astype(bool))
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

    if contour is not None and contour.shape[0] >= 12:
        pts = _smooth_closed_polyline(contour, sigma_frac=0.0025)
        peri = float(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum())
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


def _quad_from_binary(binary: np.ndarray) -> np.ndarray:
    """Axis-aligned fallback quad (TL, TR, BR, BL) from a silhouette."""
    ys, xs = np.where(binary)
    if ys.size == 0:
        h, w = binary.shape[:2]
        return np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _outer_min_side(outer_src: np.ndarray) -> float:
    ys, xs = np.where(outer_src)
    if ys.size == 0:
        return float(min(outer_src.shape[:2]))
    return float(min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))


def _finish_inner_lip_polyline(pts: np.ndarray | None) -> np.ndarray | None:
    """Smooth the detected inner lip while keeping real corner fans (no rounded-rect replace)."""
    if pts is None or pts.shape[0] < 16:
        return pts
    peri = float(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum())
    dense_n = int(np.clip(peri * 1.35, 256, 4096))
    work = _resample_closed_polyline(pts.astype(np.float64), dense_n)
    work = _rolling_median_closed(work, 5)
    work = _smooth_sides_keep_corners(work)
    work = _smooth_closed_polyline(work, sigma_frac=0.0018)
    if work.shape[0] > 2048:
        work = _resample_closed_polyline(work, 2048)
    return work


def _polyline_outward_normals(pts: np.ndarray) -> np.ndarray:
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tangent = nxt - prev
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    lengths = np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    normal = normal / lengths
    centroid = pts.mean(axis=0)
    probe = pts[0] + normal[0]
    inward = centroid - pts[0]
    if float(np.dot(probe - pts[0], inward)) > 0:
        normal = -normal
    return normal


def _nudge_print_lip_local(
    pts: np.ndarray,
    quad: np.ndarray,
    right_px: float,
    corner_px: float,
) -> np.ndarray:
    """Move only the right rim and four corner fans slightly toward the inner phone lip."""
    if pts is None or pts.shape[0] < 8 or quad is None:
        return pts
    normal = _polyline_outward_normals(pts)
    x0, x1 = float(quad[:, 0].min()), float(quad[:, 0].max())
    y0, y1 = float(quad[:, 1].min()), float(quad[:, 1].max())
    uw = max(x1 - x0, 1.0)
    vh = max(y1 - y0, 1.0)
    u = (pts[:, 0] - x0) / uw
    v = (pts[:, 1] - y0) / vh
    corner = np.zeros(pts.shape[0], dtype=np.float64)
    for uc, vc in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        dist = np.sqrt(((u - uc) / 0.20) ** 2 + ((v - vc) / 0.20) ** 2)
        corner = np.maximum(corner, np.clip(1.0 - dist, 0.0, 1.0))
    right = np.clip((u - 0.76) / 0.16, 0.0, 1.0)
    delta = right_px * right + corner_px * corner
    return pts + normal * delta[:, None]


def _detected_rim_polyline(binary: np.ndarray) -> np.ndarray | None:
    """OpenCV outer-rim contour: real corner curvature, button bumps despiked, sides smoothed."""
    pts = _contour_points(binary)
    if pts is None or pts.shape[0] < 16:
        return pts
    peri = float(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum())
    dense_n = int(np.clip(peri * 1.6, 384, 4096))
    work = _resample_closed_polyline(pts.astype(np.float64), dense_n)
    work = _rolling_median_closed(work, 5)
    work = _despike_side_bumps(work)
    work = _smooth_sides_keep_corners(work)
    work = _smooth_closed_polyline(work, sigma_frac=0.00145)
    if work.shape[0] > 2560:
        work = _resample_closed_polyline(work, 2560)
    return work


def _ss3_print_inset(
    outer_src: np.ndarray,
    back_src: np.ndarray,
    avg_w: float,
    bumper_px: float | None,
    has_alpha: bool,
) -> float:
    """Inset for the ss-3 flush wrap: circular 0.126 lip sitting on the inner rim."""
    detected = _median_boundary_inset(outer_src, back_src)
    floor = float(max(0.45, 0.0028 * _outer_min_side(outer_src)))
    wall = float(bumper_px) if bumper_px is not None and bumper_px > 0 else detected
    if has_alpha:
        return float(max(floor, detected, wall, 0.020 * avg_w))
    cap = float(np.clip(0.010 * avg_w, 2.0, 9.5))
    return float(max(floor, min(wall, cap)))


def _print_lip_inset(outer_src: np.ndarray, back_src: np.ndarray) -> float:
    detected = _median_boundary_inset(outer_src, back_src)
    floor = float(max(0.45, 0.0028 * _outer_min_side(outer_src)))
    return float(max(floor, detected)) if detected > 0.5 else floor


def _inner_print_lip_polyline(
    back_src: np.ndarray,
    outer_src: np.ndarray,
    outer_pts: np.ndarray,
) -> np.ndarray:
    """Parallel offset of the detected outer rim — same curvature, flush to the inner lip."""
    lip_inset = _print_lip_inset(outer_src, back_src)
    pts = _offset_closed_polyline(outer_pts, -float(max(0.35, lip_inset)))
    return _clamp_polyline_inside(pts, outer_src, slack=0.32)


def _pixel_right_corner_weights(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    yy, xx = np.indices((height, width), dtype=np.float32)
    x0, x1 = float(quad[:, 0].min()), float(quad[:, 0].max())
    y0, y1 = float(quad[:, 1].min()), float(quad[:, 1].max())
    u = (xx - x0) / max(x1 - x0, 1.0)
    v = (yy - y0) / max(y1 - y0, 1.0)
    corner = np.zeros((height, width), dtype=np.float32)
    for uc, vc in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        dist = np.sqrt(((u - uc) / 0.20) ** 2 + ((v - vc) / 0.20) ** 2)
        corner = np.maximum(corner, np.clip(1.0 - dist, 0.0, 1.0))
    right = np.clip((u - 0.82) / 0.12, 0.0, 1.0)
    left = np.clip((0.18 - u) / 0.12, 0.0, 1.0)
    top = np.clip((0.18 - v) / 0.12, 0.0, 1.0)
    bot = np.clip((v - 0.82) / 0.12, 0.0, 1.0)
    # Straight edges only — do not harden the restored circular corner fans.
    rim = np.maximum.reduce([right, left, top, bot])
    rim = rim * (1.0 - corner)
    return np.clip(rim, 0.0, 1.0)


def _solidify_print_microgaps(
    back_m: np.ndarray,
    outer_m: np.ndarray,
    camera_m: np.ndarray,
    quad: np.ndarray | None,
) -> np.ndarray:
    """Opaque-up the right-edge / corner AA fringe so phone color cannot show through.

    Camera coverage is used only as a keep-out; the camera mask itself is not modified.
    """
    if quad is None or back_m.size == 0:
        return back_m
    weight = _pixel_right_corner_weights(back_m.shape, quad)
    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=1.15)
    cam = np.clip(np.asarray(camera_m, dtype=np.float32), 0.0, 1.0)
    if cam.ndim == 3:
        cam = cam[..., 0]
    # Do not alter print near the camera island.
    weight = weight * np.clip(1.0 - cam * 3.0, 0.0, 1.0)
    if float(weight.max()) < 1e-4:
        return back_m
    # Coverage at/above ~0.52 becomes solid in the weighted region (no iso-shift).
    solid = np.where(back_m >= 0.52, 1.0, back_m)
    mixed = back_m * (1.0 - weight) + solid * weight
    return np.minimum(mixed, outer_m).astype(np.float32)


def _side_boundary_inset(outer_src: np.ndarray, back_src: np.ndarray, quad: np.ndarray) -> float:
    """Median outer→back gap on the straight rims only (ignores tight corner fans)."""
    dist = cv2.distanceTransform(outer_src.astype(np.uint8), cv2.DIST_L2, 5)
    u8 = back_src.astype(np.uint8)
    ring = cv2.dilate(u8, np.ones((3, 3), np.uint8)) & (1 - u8)
    if not np.any(ring):
        return _median_boundary_inset(outer_src, back_src)
    ys, xs = np.where(ring)
    x0, x1 = float(quad[:, 0].min()), float(quad[:, 0].max())
    y0, y1 = float(quad[:, 1].min()), float(quad[:, 1].max())
    uw = max(x1 - x0, 1.0)
    vh = max(y1 - y0, 1.0)
    u = (xs.astype(np.float64) - x0) / uw
    v = (ys.astype(np.float64) - y0) / vh
    near_c = np.zeros(xs.shape[0], dtype=bool)
    for uc, vc in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        near_c |= np.sqrt(((u - uc) / 0.22) ** 2 + ((v - vc) / 0.22) ** 2) < 1.0
    side = ~near_c
    if int(np.count_nonzero(side)) < 24:
        return _median_boundary_inset(outer_src, back_src)
    return float(np.median(dist[ys[side], xs[side]]))


def _finish_print_corners(
    back_m: np.ndarray,
    outer_m: np.ndarray,
    outer_src: np.ndarray,
    back_src: np.ndarray,
    camera_m: np.ndarray,
    quad: np.ndarray | None,
) -> np.ndarray:
    """Restore circular wrap corners to the same bumper gap as the already-correct sides.

    Straight edges are left unchanged. Camera coverage is a keep-out only.
    """
    if quad is None or back_m.size == 0:
        return back_m
    side_in = _side_boundary_inset(outer_src, back_src, quad)
    dist_o = cv2.distanceTransform(outer_src.astype(np.uint8), cv2.DIST_L2, 5)
    # Coverage 1 inside the side-matching lip, AA across ~1px at the rim.
    target = np.clip(0.5 + (dist_o - float(side_in)) / max(float(0.85), 1e-3), 0.0, 1.0)
    target = np.minimum(target.astype(np.float32), outer_m.astype(np.float32))

    height, width = back_m.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    x0, x1 = float(quad[:, 0].min()), float(quad[:, 0].max())
    y0, y1 = float(quad[:, 1].min()), float(quad[:, 1].max())
    u = (xx - x0) / max(x1 - x0, 1.0)
    v = (yy - y0) / max(y1 - y0, 1.0)
    corner = np.zeros((height, width), dtype=np.float32)
    for uc, vc in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        d = np.sqrt(((u - uc) / 0.20) ** 2 + ((v - vc) / 0.20) ** 2)
        corner = np.maximum(corner, np.clip(1.0 - d, 0.0, 1.0))
    corner = cv2.GaussianBlur(corner, (0, 0), sigmaX=1.35)
    cam = np.clip(np.asarray(camera_m, dtype=np.float32), 0.0, 1.0)
    if cam.ndim == 3:
        cam = cam[..., 0]
    corner = corner * np.clip(1.0 - cam * 3.0, 0.0, 1.0)
    mixed = back_m * (1.0 - corner) + np.maximum(back_m, target) * corner
    return np.minimum(mixed, outer_m).astype(np.float32)


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


def _refine_bottom_corners_and_gap(
    inner_pts: np.ndarray,
    outer_pts: np.ndarray,
    cover_rgba: np.ndarray | None,
    outer_src: np.ndarray,
    back_src: np.ndarray,
) -> np.ndarray:
    """Reconstruct the bottom corners by tracking the true inner rim edge from the physical outer rim."""
    if inner_pts is None or inner_pts.shape[0] < 16 or outer_pts is None or cover_rgba is None:
        return inner_pts
        
    ys_out = outer_pts[:, 1]
    y_min, y_max = ys_out.min(), ys_out.max()
    y_mid = y_min + (y_max - y_min) * 0.5
    
    # 1. Calculate inward normals of the perfectly detected outer physical rim
    prev = np.roll(outer_pts, 1, axis=0)
    nxt = np.roll(outer_pts, -1, axis=0)
    tangent = nxt - prev
    normal_out = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    lengths = np.maximum(np.linalg.norm(normal_out, axis=1, keepdims=True), 1e-6)
    normal_out = normal_out / lengths
    
    centroid = outer_pts.mean(axis=0)
    probe = outer_pts[0] + normal_out[0]
    if float(np.dot(probe - outer_pts[0], centroid - outer_pts[0])) < 0:
        normal_out = -normal_out # Ensure normals point INWARD towards centroid
        
    # 2. Ray cast INWARD from outer_pts to detect the actual physical inner rim
    height, width = cover_rgba.shape[:2]
    gray = cv2.cvtColor(cover_rgba[..., :3], cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    
    true_inner = outer_pts.copy()
    max_search = int(max(15, 0.05 * min(width, height)))
    default_step = max(3.0, _median_boundary_inset(outer_src, back_src))
    
    for i in range(outer_pts.shape[0]):
        x, y = outer_pts[i]
        
        # Only analyze the bottom half of the phone
        if y < y_mid:
            nx, ny = normal_out[i]
            true_inner[i, 0] = x + nx * default_step
            true_inner[i, 1] = y + ny * default_step
            continue
            
        nx, ny = normal_out[i]
        best_step = default_step
        max_grad = 15.0 # Noise immunity threshold
        
        # The true inner edge MUST be physically near the median rim width.
        # Constrain the search to a tight local band to prevent snapping to deep interior shadows.
        search_start = max(2, int(default_step * 0.3))
        search_end = max(search_start + 4, int(default_step * 1.8) + 2)
        
        # Search inward for the strongest physical rim edge
        for step in range(search_start, search_end):
            sx = int(round(x + nx * step))
            sy = int(round(y + ny * step))
            
            if 0 <= sx < width and 0 <= sy < height:
                g = mag[sy, sx]
                if g > max_grad:
                    max_grad = g
                    best_step = step
                    
        true_inner[i, 0] = x + nx * best_step
        true_inner[i, 1] = y + ny * best_step
        
    # Fit a smooth, continuous contour through the detected edge points
    true_inner = _smooth_closed_polyline(true_inner, sigma_frac=0.003)
    
    # 3. Connect it continuously to the existing correct side/bottom boundaries
    out = inner_pts.copy()
    c = out.mean(axis=0)
    
    ang1 = np.arctan2(out[:, 1] - c[1], out[:, 0] - c[0])
    ang2 = np.arctan2(true_inner[:, 1] - c[1], true_inner[:, 0] - c[0])
    
    sort_idx = np.argsort(ang2)
    ang2_sorted = ang2[sort_idx] + np.arange(ang2.size) * 1e-7
    pts2_sorted = true_inner[sort_idx]
    
    ang2_ext = np.concatenate([ang2_sorted - 2*np.pi, ang2_sorted, ang2_sorted + 2*np.pi])
    pts2_ext = np.vstack([pts2_sorted, pts2_sorted, pts2_sorted])
    
    x2_interp = np.interp(ang1, ang2_ext, pts2_ext[:, 0])
    y2_interp = np.interp(ang1, ang2_ext, pts2_ext[:, 1])
    ideal_mapped = np.stack([x2_interp, y2_interp], axis=1)
    xs_out = outer_pts[:, 0]
    x_mid = xs_out.min() + (xs_out.max() - xs_out.min()) * 0.5
    y_thresh = y_min + (y_max - y_min) * 0.65
    
    weight = np.zeros(out.shape[0], dtype=np.float64)
    for i in range(out.shape[0]):
        x, y = out[i]
        
        # EXPLICITLY ISOLATE BOTTOM-LEFT CORNER ONLY
        # Do not touch the already-perfect bottom-right corner or any other edges.
        if y > y_thresh and x < x_mid:
            # Smooth weight transition for a seamless connection to the correct side/bottom edges
            weight[i] = np.clip((y - y_thresh) / ((y_max - y_min) * 0.1), 0.0, 1.0)
            
    out = out * (1.0 - weight[:, None]) + ideal_mapped * weight[:, None]
    
    return _smooth_closed_polyline(out, sigma_frac=0.001)


def _trace_continuous_inner_rim(
    pts: np.ndarray,
    cover_rgba: np.ndarray,
    outer_src: np.ndarray,
    back_src: np.ndarray,
) -> np.ndarray:
    """Continuously track the true physical inner rim from the calibrated top edge around the entire perimeter."""
    if pts is None or pts.shape[0] < 16 or cover_rgba is None or outer_src is None:
        return pts

    height, width = cover_rgba.shape[:2]
    gray = cv2.cvtColor(cover_rgba[..., :3], cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # Correct distance transform: distance from INSIDE the phone to the outer edge/background
    dist_inside = cv2.distanceTransform(outer_src.astype(np.uint8), cv2.DIST_L2, 5)

    n = pts.shape[0]
    ys = pts[:, 1]
    y_min, y_max = float(ys.min()), float(ys.max())
    h = max(y_max - y_min, 10.0)

    xs = pts[:, 0]
    x_min, x_max = float(xs.min()), float(xs.max())
    w = max(x_max - x_min, 10.0)

    # 1. Calibrate at the known-correct TOP INNER RIM
    # Points near top-center
    top_mask = (ys <= y_min + 0.06 * h) & (np.abs(xs - (x_min + 0.5 * w)) <= 0.35 * w)
    if np.any(top_mask):
        sample_y = np.clip(np.round(ys[top_mask]).astype(int), 0, height - 1)
        sample_x = np.clip(np.round(xs[top_mask]).astype(int), 0, width - 1)
        calibrated_rim_w = float(np.median(dist_inside[sample_y, sample_x]))
        ref_grad = float(np.median(mag[sample_y, sample_x]))
    else:
        calibrated_rim_w = float(_median_boundary_inset(outer_src, back_src))
        ref_grad = 25.0

    if calibrated_rim_w < 3.0:
        calibrated_rim_w = float(max(3.0, _median_boundary_inset(outer_src, back_src)))
    if ref_grad < 10.0:
        ref_grad = 20.0

    # 2. Compute smooth outward normals
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tangent = nxt - prev
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    lengths = np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    normal = normal / lengths

    centroid = pts.mean(axis=0)
    probe = pts[0] + normal[0]
    inward_vec = centroid - pts[0]
    if float(np.dot(probe - pts[0], inward_vec)) > 0:
        normal = -normal  # Ensure normal points OUTWARD toward physical rim

    # Find the top-center index as the anchor point
    i_top = int(np.argmin(ys))
    i_bottom = int(np.argmax(ys))

    # Identify the locked top zone (top edge where y <= y_min + 0.05 * h)
    is_locked_top = ys <= (y_min + 0.05 * h)

    def find_best_edge_step(idx: int, prev_step: float) -> float:
        x, y = pts[idx]
        nx, ny = normal[idx]

        # Search locally along the boundary normal around prev_step
        search_radius = 8.0
        steps = np.arange(prev_step - search_radius, prev_step + search_radius + 0.5, 0.5)

        sampled_g = []
        sampled_d = []
        valid_steps = []

        for s in steps:
            sx = int(round(x + nx * s))
            sy = int(round(y + ny * s))
            if 0 <= sx < width and 0 <= sy < height:
                d = dist_inside[sy, sx]
                # Reject if outside phone, hitting outer rim, or sitting deep inside printable surface
                if d < max(2.0, calibrated_rim_w * 0.35) or d > calibrated_rim_w * 2.3:
                    continue
                g = float(mag[sy, sx])
                sampled_g.append(g)
                sampled_d.append(d)
                valid_steps.append(s)

        if len(sampled_g) < 3:
            return prev_step

        arr_g = np.array(sampled_g)
        arr_d = np.array(sampled_d)
        arr_s = np.array(valid_steps)

        max_g = float(arr_g.max())
        if max_g < max(6.0, ref_grad * 0.20):
            return prev_step

        best_s = prev_step
        best_cost = 9999.0

        for p_idx in range(1, len(arr_g) - 1):
            if arr_g[p_idx] >= arr_g[p_idx - 1] and arr_g[p_idx] >= arr_g[p_idx + 1]:
                g_val = arr_g[p_idx]
                if g_val < max(6.0, ref_grad * 0.22):
                    continue
                s_val = arr_s[p_idx]
                d_val = arr_d[p_idx]

                # Sub-pixel quadratic peak refinement
                denom = 2.0 * (arr_g[p_idx - 1] - 2.0 * g_val + arr_g[p_idx + 1])
                if abs(denom) > 1e-5:
                    delta_s = (arr_g[p_idx - 1] - arr_g[p_idx + 1]) / denom
                    s_sub = s_val + np.clip(delta_s * 0.5, -0.5, 0.5)
                else:
                    s_sub = s_val

                # Cost function favoring CONTINUITY of the same physical rim
                continuity_penalty = abs(s_sub - prev_step)
                thickness_penalty = abs(d_val - calibrated_rim_w) / max(calibrated_rim_w, 1.0)
                gradient_bonus = g_val / max(max_g, 1.0)

                cost = continuity_penalty + 1.2 * thickness_penalty - 0.7 * gradient_bonus

                if cost < best_cost:
                    best_cost = cost
                    best_s = float(s_sub)

        # If candidate jumped too far (e.g. button, reflection, gap), enforce continuity
        if abs(best_s - prev_step) > 2.8:
            return prev_step

        return 0.70 * prev_step + 0.30 * best_s

    # Build cyclic traversal order starting from i_top
    cw_indices = [(i_top + k) % n for k in range(n)]
    k_bot = cw_indices.index(i_bottom)

    offsets_cw = np.zeros(n, dtype=np.float64)
    curr_s = 0.0
    for k in range(0, k_bot + 1):
        idx = cw_indices[k]
        if is_locked_top[idx]:
            curr_s = 0.0
            offsets_cw[idx] = 0.0
        else:
            curr_s = find_best_edge_step(idx, curr_s)
            offsets_cw[idx] = curr_s

    offsets_ccw = np.zeros(n, dtype=np.float64)
    curr_s = 0.0
    for k in range(n - 1, k_bot - 1, -1):
        idx = cw_indices[k]
        if is_locked_top[idx]:
            curr_s = 0.0
            offsets_ccw[idx] = 0.0
        else:
            curr_s = find_best_edge_step(idx, curr_s)
            offsets_ccw[idx] = curr_s

    # Blend CW and CCW around the bottom meeting region for seamless continuity
    final_offsets = np.zeros(n, dtype=np.float64)
    blend_w = 24
    for k in range(n):
        idx = cw_indices[k]
        if is_locked_top[idx]:
            final_offsets[idx] = 0.0
            continue

        if k_bot - blend_w <= k <= k_bot + blend_w:
            alpha = (k - (k_bot - blend_w)) / float(2 * blend_w)
            final_offsets[idx] = (1.0 - alpha) * offsets_cw[idx] + alpha * offsets_ccw[idx]
        elif k < k_bot - blend_w:
            final_offsets[idx] = offsets_cw[idx]
        else:
            final_offsets[idx] = offsets_ccw[idx]

    # Median smooth offsets along perimeter to eliminate any discrete stepping
    win = 15
    padded = np.concatenate([final_offsets[-win:], final_offsets, final_offsets[:win]])
    med_offsets = np.array([float(np.median(padded[j : j + win])) for j in range(win, win + n)])

    out = pts + normal * med_offsets[:, None]
    return _smooth_closed_polyline(out, sigma_frac=0.0008)


def _snap_polyline_to_boundary(pts: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Snap a smoothed polyline outward exactly to the contact edge of the binary mask."""
    if pts is None or pts.shape[0] < 8 or binary is None or not np.any(binary):
        return pts
    height, width = binary.shape[:2]
    
    # Distance from inside the mask to the nearest boundary edge
    dist_in = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    
    xs = np.clip(pts[:, 0].astype(np.float32), 0.0, float(width - 1))
    ys = np.clip(pts[:, 1].astype(np.float32), 0.0, float(height - 1))
    
    sampled_dist = cv2.remap(
        dist_in.astype(np.float32),
        xs.reshape(1, -1),
        ys.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()
    
    inside = sampled_dist > 0.1
    if not np.any(inside):
        return pts
        
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tangent = nxt - prev
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    lengths = np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    normal = normal / lengths
    
    centroid = pts.mean(axis=0)
    probe = pts[0] + normal[0]
    inward_vec = centroid - pts[0]
    if float(np.dot(probe - pts[0], inward_vec)) > 0:
        normal = -normal
        
    out = pts.copy()
    push = sampled_dist[inside][:, None]
    out[inside] = pts[inside] + normal[inside] * push
    
    # Smooth slightly to absorb the snap without altering shape
    return _smooth_closed_polyline(out, sigma_frac=0.0005)


def _finish_wrap_polyline(pts: np.ndarray | None) -> np.ndarray | None:
    """Sub-pixel regularized phone contour — eliminates side button bumps and bottom shadow leaks."""
    if pts is None or pts.shape[0] < 16:
        return pts

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
    """Measure the exact physical corner radius for all 4 corners (TL, TR, BR, BL) individually.
    Uses the geometric apex distance from the quad corner vertex to the closest contour point.
    For a corner with interior angle theta, distance from vertex to curve apex is:
        d = R * (1/sin(theta/2) - 1)
    Therefore:
        R = d * sin(theta/2) / (1 - sin(theta/2))
    This provides exact, sub-pixel accurate corner radii matching the phone's true physical curvature.
    """
    if quad is None or len(quad) != 4:
        return [0.126 * 100.0] * 4
    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bot = np.linalg.norm(quad[2] - quad[3])
    avg_w = float((w_top + w_bot) / 2.0)
    default_r = 0.126 * avg_w

    if pts is None or pts.shape[0] < 32 or avg_w < 10.0:
        return [default_r, default_r, default_r, default_r]

    raw_radii: list[float] = []
    centroid = quad.mean(axis=0)

    for i, corner in enumerate(quad):
        v_prev = quad[(i - 1) % 4]
        v_curr = quad[i]
        v_next = quad[(i + 1) % 4]
        len_in = np.linalg.norm(v_curr - v_prev)
        len_out = np.linalg.norm(v_next - v_curr)
        if len_in < 1e-4 or len_out < 1e-4:
            raw_radii.append(default_r)
            continue
        u_in = (v_curr - v_prev) / len_in
        u_out = (v_next - v_curr) / len_out
        cos_t = np.clip(-np.dot(u_in, u_out), -1.0, 1.0)
        theta = np.arccos(cos_t)
        half_theta = max(theta / 2.0, 1e-4)
        sin_half = np.sin(half_theta)

        u_bisect = -u_in + u_out
        len_b = np.linalg.norm(u_bisect)
        if len_b > 1e-6:
            u_bisect = u_bisect / len_b
        if np.dot(u_bisect, centroid - v_curr) < 0:
            u_bisect = -u_bisect

        dists = np.linalg.norm(pts - corner, axis=1)
        corner_mask = dists < 0.35 * min(len_in, len_out, avg_w)
        if not np.any(corner_mask):
            raw_radii.append(default_r)
            continue

        corner_pts = pts[corner_mask]
        vecs = corner_pts - corner
        proj = np.dot(vecs, u_bisect)
        inward_pts = corner_pts[proj > 0]
        if inward_pts.shape[0] >= 3:
            inward_dists = np.linalg.norm(inward_pts - corner, axis=1)
            min_d = float(np.min(inward_dists))
        else:
            min_d = float(np.min(dists[corner_mask]))

        if sin_half < 0.999:
            r_val = min_d * sin_half / max(1.0 - sin_half, 1e-4)
        else:
            r_val = min_d

        if 0.04 * avg_w <= r_val <= 0.30 * avg_w:
            raw_radii.append(float(r_val))
        else:
            raw_radii.append(default_r)

    valid = [r for r in raw_radii if 0.05 * avg_w <= r <= 0.28 * avg_w]
    med_r = float(np.median(valid)) if valid else default_r
    final_radii = [r if 0.65 * med_r <= r <= 1.5 * med_r else med_r for r in raw_radii]
    return final_radii


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

    raw_quad = np.array([v_tl, v_tr, v_br, v_bl], dtype=np.float64)
    ysort = raw_quad[np.argsort(raw_quad[:, 1])]
    top_two = ysort[:2]
    bot_two = ysort[2:]
    tl, tr = top_two[np.argsort(top_two[:, 0])]
    bl, br = bot_two[np.argsort(bot_two[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float64)


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
    num_arc_pts: int = 48,
) -> np.ndarray:
    """Generate dense, smooth closed polyline with true inward circular corner arcs and exact tangent continuity."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bot = np.linalg.norm(quad[2] - quad[3])
    avg_w = (w_top + w_bot) / 2.0

    if isinstance(radius_ratio, (list, tuple, np.ndarray)):
        r_list = [float(r) if float(r) > 1.0 else float(r) * avg_w for r in radius_ratio]
    else:
        r_val = float(radius_ratio) if float(radius_ratio) > 1.0 else float(radius_ratio) * avg_w
        r_list = [r_val] * 4

    centroid = quad.mean(axis=0)
    corners_info = []

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

        cos_theta = np.clip(-np.dot(u_in, u_out), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        half_theta = max(theta / 2.0, 1e-4)

        t = min(r / max(np.tan(half_theta), 1e-4), 0.45 * min(len_in, len_out))
        actual_r = t * np.tan(half_theta)

        p_in = v_curr - t * u_in
        p_out = v_curr + t * u_out

        u_bisect = -u_in + u_out
        len_b = np.linalg.norm(u_bisect)
        if len_b > 1e-6:
            u_bisect = u_bisect / len_b
        if np.dot(u_bisect, centroid - v_curr) < 0:
            u_bisect = -u_bisect

        center = v_curr + (actual_r / max(np.sin(half_theta), 1e-4)) * u_bisect

        v_start = p_in - center
        v_end = p_out - center
        ang_start = np.arctan2(v_start[1], v_start[0])
        ang_end = np.arctan2(v_end[1], v_end[0])

        diff = (ang_end - ang_start + np.pi) % (2 * np.pi) - np.pi
        angles = ang_start + diff * np.linspace(0.0, 1.0, num_arc_pts, endpoint=True)
        arc = center + actual_r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        corners_info.append({"p_in": p_in, "p_out": p_out, "arc": arc})

    segs = []
    for i in range(4):
        segs.append(corners_info[i]["arc"][:-1])
        next_i = (i + 1) % 4
        p_out_curr = corners_info[i]["p_out"]
        p_in_next = corners_info[next_i]["p_in"]
        if np.linalg.norm(p_in_next - p_out_curr) > 0.5:
            straight = np.linspace(p_out_curr, p_in_next, 24, endpoint=False)
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


def _offset_closed_polyline(
    pts: np.ndarray,
    delta: float | tuple[float, float, float, float] = 0.0,
) -> np.ndarray:
    """Move a closed contour along its outward normal (positive = grow).
    delta can be a float or (top, right, bottom, left) directional values.
    """
    if pts.shape[0] < 8:
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

    if isinstance(delta, (list, tuple, np.ndarray)) and len(delta) == 4:
        d_top, d_right, d_bot, d_left = [float(x) for x in delta]
        nx = normal[:, 0]
        ny = normal[:, 1]
        w_right = np.clip(nx, 0.0, 1.0)
        w_left = np.clip(-nx, 0.0, 1.0)
        w_bot = np.clip(ny, 0.0, 1.0)
        w_top = np.clip(-ny, 0.0, 1.0)
        w_sum = np.maximum(w_right + w_left + w_bot + w_top, 1e-6)
        local_delta = (w_top * d_top + w_right * d_right + w_bot * d_bot + w_left * d_left) / w_sum
        return pts + normal * local_delta[:, None]

    if abs(float(delta)) < 1e-6:
        return pts
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
