"""Analyze an uploaded cover image: outer contour, back panel, perspective quad.

JPG/PNG product photos (no alpha) use edge/contour geometry — not flood-fill
of the MagSafe ring, which is the common failure on white-background shots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.camera_detector import detect_camera
from app.core.mask_generator import (
    MaskSet,
    inset_binary,
    inset_smooth,
    masks_from_binaries,
    override_camera_exclusion,
    refine_outer_binary,
)
from app.utils.constants import (
    CAMERA_SAFETY_FRACTION,
    CAMERA_SAFETY_MIN_PX,
    DETECTION_CONFIDENCE_WARN,
    MASK_FEATHER_PX,
    PRINT_MARGIN_FRACTION,
    PRINT_MARGIN_MIN_PX,
    SIDE_WALL_MAX_FRACTION,
)
from app.utils.image_utils import CoverError, contour_overlay, mask_to_preview


@dataclass
class CoverDetection:
    masks: MaskSet
    outer_bin: np.ndarray
    back_bin: np.ndarray
    back_full: np.ndarray
    camera_bin: np.ndarray
    back_quad: np.ndarray
    confidence: float
    warnings: list[str] = field(default_factory=list)
    has_alpha: bool = True
    camera_found: bool = False
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    raw_outer: np.ndarray | None = None


def detect_cover(cover_rgba: np.ndarray) -> CoverDetection:
    """Build printable geometry from a user-uploaded cover (PNG or JPG)."""
    if cover_rgba.ndim != 3 or cover_rgba.shape[2] != 4:
        raise CoverError("The cover image could not be read as an RGBA image.")

    height, width = cover_rgba.shape[:2]
    alpha = cover_rgba[..., 3]
    has_alpha = bool(int(alpha.min()) < 250 or int(np.percentile(alpha, 5)) < 240)

    if has_alpha and int(alpha.max()) >= 12:
        outer = _outer_from_alpha(alpha)
        # If alpha silhouette is tiny/circular, treat as opaque photo.
        if _phone_score(outer, height, width) < 0.35:
            outer = _outer_from_rgb(cover_rgba[..., :3])
            has_alpha = False
    else:
        outer = _outer_from_rgb(cover_rgba[..., :3])
        has_alpha = False

    if not outer.any():
        raise CoverError(
            "Could not find a phone cover in this image.\n"
            "Try a front-facing photo of the case on a plain background."
        )

    raw_outer = _fill_holes(_largest_component(outer))
    outer_filled = raw_outer

    if _phone_score(outer_filled, height, width) < 0.28:
        raise CoverError(
            "Could not find a large phone-shaped outline in this image.\n"
            "Use a clearer front-facing photo of the whole case."
        )

    min_side = float(min(width, height))
    bbox = _bbox(outer_filled)
    if bbox is not None:
        min_side = float(min(bbox[2], bbox[3]))

    outer_filled = _polish_cover_contour(_clean_cover_silhouette(outer_filled))
    outer_filled = _fill_shallow_notches(outer_filled, min_side)
    outer_filled = refine_outer_binary(outer_filled)
    bbox = _bbox(outer_filled)
    if bbox is not None:
        min_side = float(min(bbox[2], bbox[3]))

    safety_px = max(CAMERA_SAFETY_MIN_PX, CAMERA_SAFETY_FRACTION * min_side)
    back_full = _inner_printable_panel(cover_rgba, outer_filled, has_alpha, min_side)
    if not back_full.any():
        raise CoverError(
            "The cover outline was found, but the inner back panel is too small.\n"
            "Try a flatter, front-facing photo of the case."
        )

    camera, camera_found, cam_conf, cam_warnings = detect_camera(
        cover_rgba, outer_filled, back_full, safety_px
    )
    if camera_found and cam_conf < 0.48:
        camera = np.zeros_like(back_full)
        camera_found = False
        cam_warnings = list(cam_warnings) + [
            "Camera detection was uncertain, so it was not applied. "
            "Use “Mark camera area” if the cutout is missing."
        ]
    quad = back_panel_quad(back_full)
    print_margin = max(PRINT_MARGIN_MIN_PX, PRINT_MARGIN_FRACTION * min_side)
    print_body = back_full
    if print_margin > 0.5:
        inset_body = inset_smooth(back_full, print_margin, sigma=0.6)
        if inset_body.any():
            print_body = inset_body
    back_printable = print_body & ~camera

    if not back_printable.any() or float(back_printable.mean()) < 0.04:
        # Camera exclusion must never wipe the panel — drop auto camera instead of failing.
        camera = np.zeros_like(back_full)
        camera_found = False
        cam_warnings = [
            "Automatic camera exclusion removed too much of the back panel, so it was skipped. "
            "Mark the camera area if needed."
        ]
        back_printable = print_body

    return _pack_detection(
        cover_rgba,
        outer_filled,
        back_full,
        back_printable,
        camera,
        quad,
        camera_found,
        has_alpha,
        cam_warnings,
        raw_outer=raw_outer,
    )


def apply_manual_camera(
    cover_rgba: np.ndarray,
    detection: CoverDetection,
    rect_xywh: tuple[float, float, float, float],
) -> CoverDetection:
    """Replace camera exclusion with a user-drawn rectangle (image pixels)."""
    x, y, w, h = rect_xywh
    height, width = cover_rgba.shape[:2]
    camera = np.zeros((height, width), dtype=bool)
    x0 = int(max(0, round(x)))
    y0 = int(max(0, round(y)))
    x1 = int(min(width, round(x + w)))
    y1 = int(min(height, round(y + h)))
    if x1 <= x0 or y1 <= y0:
        raise CoverError("Draw a box around the camera area on the cover preview.")

    # Rounded-rect-ish: filled ellipse covering the drag box (typical camera island).
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    yy, xx = np.ogrid[0:height, 0:width]
    camera = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    camera = camera & detection.outer_bin

    print_body = detection.back_full
    back_printable = print_body & ~camera
    if not back_printable.any():
        raise CoverError("That camera box covers the whole back panel. Draw a smaller box.")

    quad = back_panel_quad(detection.back_full)
    return _pack_detection(
        cover_rgba,
        detection.outer_bin,
        detection.back_full,
        back_printable,
        camera,
        quad,
        True,
        detection.has_alpha,
        [],
        raw_outer=detection.raw_outer if detection.raw_outer is not None else detection.outer_bin,
    )


def apply_manual_camera_mask(
    cover_rgba: np.ndarray,
    detection: CoverDetection,
    camera_aa: np.ndarray,
) -> CoverDetection:
    """Replace camera exclusion with a user-drawn antialiased mask (image pixels)."""
    cam = np.clip(np.asarray(camera_aa, dtype=np.float32), 0.0, 1.0)
    if cam.ndim == 3:
        cam = cam[..., 0]
    height, width = cover_rgba.shape[:2]
    if cam.shape != (height, width):
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    outer = detection.outer_bin.astype(np.float32)
    cam = cam * outer
    camera_bin = cam > 0.12
    print_body = detection.back_full
    back_printable = print_body & ~(cam > 0.45)
    if print_body.any() and float(np.count_nonzero(back_printable)) < 0.04 * float(np.count_nonzero(print_body)):
        raise CoverError("That camera mask covers the whole back panel. Draw a smaller area.")
    quad = back_panel_quad(detection.back_full)
    packed = _pack_detection(
        cover_rgba,
        detection.outer_bin,
        detection.back_full,
        back_printable,
        camera_bin,
        quad,
        bool(camera_bin.any()),
        detection.has_alpha,
        [],
        raw_outer=detection.raw_outer if detection.raw_outer is not None else detection.outer_bin,
    )
    packed.masks = override_camera_exclusion(packed.masks, cam)
    packed.camera_bin = camera_bin
    packed.debug_images["camera_exclusion"] = mask_to_preview(packed.masks.camera_exclusion)
    packed.debug_images["final_print_mask"] = mask_to_preview(packed.masks.final_print)
    return packed


def _pack_detection(
    cover_rgba: np.ndarray,
    outer_filled: np.ndarray,
    back_full: np.ndarray,
    back_printable: np.ndarray,
    camera: np.ndarray,
    quad: np.ndarray,
    camera_found: bool,
    has_alpha: bool,
    extra_warnings: list[str],
    raw_outer: np.ndarray | None = None,
) -> CoverDetection:
    masks = masks_from_binaries(outer_filled, back_full, camera, MASK_FEATHER_PX)
    confidence, warnings = _score(
        cover_rgba, outer_filled, back_full, camera_found, has_alpha
    )
    warnings.extend(extra_warnings)
    raw = raw_outer if raw_outer is not None else outer_filled

    debug = {
        "uploaded_cover": cover_rgba,
        "raw_contour": contour_overlay(cover_rgba, raw, (255, 90, 70)),
        "cleaned_contour": contour_overlay(cover_rgba, outer_filled, (40, 200, 90)),
        "printable_boundary": contour_overlay(cover_rgba, back_full, (80, 180, 255)),
        "final_print_mask": mask_to_preview(masks.final_print),
        "outer_contour": contour_overlay(cover_rgba, outer_filled, (40, 200, 90)),
        "back_contour": contour_overlay(cover_rgba, back_full, (80, 180, 255)),
        "back_panel": contour_overlay(cover_rgba, back_full, (80, 180, 255)),
        "camera_exclusion": mask_to_preview(masks.camera_exclusion),
        "edge_exclusion": mask_to_preview(masks.edge_exclusion),
    }

    return CoverDetection(
        masks=masks,
        outer_bin=outer_filled,
        back_bin=back_printable,
        back_full=back_full,
        camera_bin=camera,
        back_quad=quad,
        confidence=confidence,
        warnings=warnings,
        has_alpha=has_alpha,
        camera_found=camera_found,
        debug_images=debug,
        raw_outer=raw,
    )


def back_panel_quad(back_bin: np.ndarray) -> np.ndarray:
    """Four corners (TL, TR, BR, BL) of the inner back panel."""
    ys, xs = np.where(back_bin)
    if ys.size == 0:
        h, w = back_bin.shape
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    aabb = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    u8 = (back_bin.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return aabb

    contour = max(contours, key=cv2.contourArea)
    (_cx, _cy), (_rw, _rh), angle = cv2.minAreaRect(contour)
    tilt = abs(angle) % 90.0
    tilt = min(tilt, 90.0 - tilt)
    if tilt < 6.0:
        return aabb
    pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    return order_corners(pts)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[int(np.argmin(s))]
    ordered[2] = pts[int(np.argmax(s))]
    d = pts[:, 0] - pts[:, 1]
    ordered[1] = pts[int(np.argmax(d))]
    ordered[3] = pts[int(np.argmin(d))]
    return ordered


def _outer_from_alpha(alpha: np.ndarray) -> np.ndarray:
    raw = alpha > 12
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def _outer_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Full device silhouette vs background — never an inner glass/chroma sticker."""
    height, width = rgb.shape[:2]
    best: np.ndarray | None = None
    best_key = -1.0
    for raw in (
        _lab_border_fg(rgb),
        _grabcut_border_fg(rgb),
        _flood_fill_fg(rgb),
        _edge_outer_only(rgb),
        _dark_object_fg(rgb),
    ):
        if raw is None or not raw.any():
            continue
        filled = _clean_cover_silhouette(raw)
        key = _cover_choice_score(filled, height, width)
        if key > best_key:
            best, best_key = filled, key
    if best is None:
        return np.zeros((height, width), dtype=bool)
    return _polish_cover_contour(best)


def _cover_choice_score(mask: np.ndarray, height: int, width: int) -> float:
    """Prefer the complete cover. Penalize jagged inner blobs and tiny slivers."""
    ps = _phone_score(mask, height, width)
    if ps < 0.22:
        return -1.0
    area = float(mask.mean())
    rough = _contour_roughness(mask)
    return float(ps + 0.55 * min(area / 0.50, 1.0) - 0.22 * rough)


def _contour_roughness(mask: np.ndarray) -> float:
    """0 = smooth silhouette, 1 = torn/jagged perimeter."""
    if mask is None or not mask.any():
        return 1.0
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if area < 8 or peri < 8:
        return 1.0
    compact = (peri * peri) / area
    # Smooth rounded-rect ≈ 16–22; torn sticker ≫ 35.
    return float(np.clip((compact - 20.0) / 35.0, 0.0, 1.0))


def _polish_cover_contour(mask: np.ndarray) -> np.ndarray:
    """Kill 1px teeth on the real silhouette. Never replace it with a hull or polygon."""
    from app.core.mask_generator import smooth_silhouette

    if mask is None or not mask.any():
        return mask
    filled = _fill_holes(_largest_component(mask.astype(bool)))
    h, w = filled.shape
    k = max(5, int(round(0.0055 * min(h, w))))
    if k % 2 == 0:
        k += 1
    k = min(k, 13)
    closed = cv2.morphologyEx(
        filled.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    filled = _fill_holes(closed.astype(bool))
    # 3×3 open removes 1px rim specks without shrinking the real corners.
    opened = cv2.morphologyEx(
        filled.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    filled = _fill_holes(_largest_component(opened.astype(bool)))
    sigma = float(np.clip(0.0024 * min(h, w), 1.05, 2.15))
    return _fill_holes(smooth_silhouette(filled, sigma=sigma))


def _roundness_bonus(mask: np.ndarray) -> float:
    """1 if corners are cut (rounded phone), 0 if the bbox corners are filled (a box)."""
    box = _bbox(mask)
    if box is None:
        return 0.0
    x, y, bw, bh = box
    r = max(3, int(0.06 * min(bw, bh)))
    corners = (
        mask[y : y + r, x : x + r],
        mask[y : y + r, x + bw - r : x + bw],
        mask[y + bh - r : y + bh, x : x + r],
        mask[y + bh - r : y + bh, x + bw - r : x + bw],
    )
    fills = [float(c.mean()) for c in corners if c.size]
    if not fills:
        return 0.0
    mean_fill = float(np.mean(fills))
    # Rounded corners leave bbox corners mostly empty.
    if mean_fill < 0.35:
        return 1.0
    if mean_fill < 0.55:
        return 0.45
    return 0.0


def _photograph_region(rgb: np.ndarray) -> np.ndarray:
    """Light product photo versus dark UI chrome, or the full frame if already a photo."""
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    light = _clean_cover_silhouette(L > 48)
    frac = float(light.mean())
    border_l = float(
        np.mean(
            np.concatenate(
                [L[:6, :].ravel(), L[-6:, :].ravel(), L[:, :6].ravel(), L[:, -6:].ravel()]
            )
        )
    )
    if border_l < 70 and 0.10 < frac < 0.90:
        return light
    return np.ones((height, width), dtype=bool)


def _phone_from_scene(rgb: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Phone vs table: sample the scene border as backdrop, keep the interior object."""
    if scene is None or not scene.any():
        scene = np.ones(rgb.shape[:2], dtype=bool)
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    dist_s = cv2.distanceTransform(scene.astype(np.uint8), cv2.DIST_L2, 5)
    max_d = float(dist_s.max()) if scene.any() else 1.0
    band = max(4.0, 0.055 * min(height, width), 0.08 * max_d)
    backdrop = scene & (dist_s <= band) & (dist_s > 0)
    if float(np.count_nonzero(backdrop)) < 64:
        bw = max(4, int(0.05 * min(height, width)))
        backdrop = np.zeros((height, width), dtype=bool)
        backdrop[:bw, :] = True
        backdrop[-bw:, :] = True
        backdrop[:, :bw] = True
        backdrop[:, -bw:] = True
        backdrop &= scene

    if np.any(backdrop):
        med_lab = np.median(lab[backdrop], axis=0)
        med_s = float(np.median(hsv[backdrop, 1]))
        dlab = np.linalg.norm(lab - med_lab[None, None, :], axis=2)
        sat = hsv[:, :, 1]
        color_fg = dlab > (float(np.percentile(dlab[backdrop], 88)) + 4.0)
        sat_fg = sat > (med_s + 8.0)
        fg = scene & (color_fg | sat_fg)
        fg &= dist_s >= 2.0
        seed = _clean_cover_silhouette(fg)
    else:
        seed = np.zeros((height, width), dtype=bool)
    flooded = _flood_from_phone_interior(rgb, scene)
    contoured = _cover_contour_around_seed(rgb, scene, seed)
    grown = _clean_cover_silhouette(_grow_phone_to_rim(rgb, seed, scene, backdrop))
    gc = _clean_cover_silhouette(_grabcut_from_prior(rgb, flooded if flooded.any() else seed, scene=scene))
    chroma = _chroma_object(lab, scene, backdrop)
    profile = _phone_from_chroma_profile(lab, scene)
    ranked = []
    scene_area = max(float(np.count_nonzero(scene)), 1.0)
    sb = _bbox(scene)
    for cand in (profile, chroma, flooded, contoured, gc, grown, seed):
        if cand is None or not cand.any():
            continue
        score = _phone_score(cand, height, width) + 0.22 * _roundness_bonus(cand)
        ratio = float(np.count_nonzero(cand)) / scene_area
        if 0.60 <= ratio <= 0.90:
            score += 0.28
        elif ratio < 0.52:
            score -= 0.45
        elif ratio > 0.92:
            score -= 0.40
        cb = _bbox(cand)
        if sb is not None and cb is not None:
            width_frac = cb[2] / float(max(sb[2], 1))
            if width_frac < 0.58:
                score -= 0.50
            elif 0.62 <= width_frac <= 0.92:
                score += 0.18
        ranked.append((score, cand))
    ranked.sort(key=lambda t: t[0], reverse=True)
    chosen = ranked[0][1] if ranked else seed
    return _expand_to_case_rim(rgb, chosen, scene, lab, backdrop)


def _expand_to_case_rim(
    rgb: np.ndarray,
    phone: np.ndarray,
    scene: np.ndarray,
    lab: np.ndarray,
    backdrop: np.ndarray,
) -> np.ndarray:
    """Grow the glass seed to the visible case outline (clear bumper), then stop."""
    if not phone.any():
        return phone
    height, width = rgb.shape[:2]
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.hypot(a, b)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if backdrop.any():
        table_c = float(np.median(chroma[backdrop]))
    elif np.any(scene & ~phone):
        table_c = float(np.median(chroma[scene & ~phone]))
    else:
        table_c = 0.0
    phone_c = float(np.median(chroma[phone]))
    mid_c = 0.45 * phone_c + 0.55 * table_c
    max_px = max(3.0, 0.038 * min(height, width))
    dist_out = cv2.distanceTransform((~phone).astype(np.uint8), cv2.DIST_L2, 5)
    rim = (
        scene
        & (~phone)
        & (dist_out <= max_px)
        & ((chroma >= mid_c) | (grad >= max(8, int(np.percentile(grad[scene], 70)))))
    )
    grown = _clean_cover_silhouette(phone | rim)
    if _phone_score(grown, height, width) >= 0.4:
        return grown
    return phone


def _chroma_object(lab: np.ndarray, scene: np.ndarray, backdrop: np.ndarray) -> np.ndarray:
    """Colored cover versus gray table using Lab chroma."""
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.hypot(a, b)
    if backdrop.any():
        t = float(np.percentile(chroma[backdrop], 75) + 4.0)
    else:
        t = float(np.median(chroma[scene])) + 3.0
    fg = scene & (chroma > t)
    return _clean_cover_silhouette(fg)


def _phone_from_chroma_profile(lab: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Find phone extents by chroma jumping up from the table at each side."""
    if not scene.any():
        return np.zeros(lab.shape[:2], dtype=bool)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.hypot(a, b)
    chroma[~scene] = 0
    box = _bbox(scene)
    if box is None:
        return np.zeros(lab.shape[:2], dtype=bool)
    x0, y0, bw, bh = box
    table = float(np.median(chroma[scene & (cv2.distanceTransform(scene.astype(np.uint8), cv2.DIST_L2, 5) <= 6)]))
    thresh = table + 5.0
    col_m = np.array([float(np.mean(chroma[y0 : y0 + bh, x])) for x in range(x0, x0 + bw)])
    row_m = np.array([float(np.mean(chroma[y, x0 : x0 + bw])) for y in range(y0, y0 + bh)])
    xs = np.where(col_m > thresh)[0]
    ys = np.where(row_m > thresh)[0]
    if xs.size < 8 or ys.size < 8:
        return np.zeros(lab.shape[:2], dtype=bool)
    px0, px1 = x0 + int(xs.min()), x0 + int(xs.max()) + 1
    py0, py1 = y0 + int(ys.min()), y0 + int(ys.max()) + 1
    # Keep a solid panel, then round corners via the true chroma blob inside this box.
    roi = np.zeros(lab.shape[:2], dtype=bool)
    roi[py0:py1, px0:px1] = scene[py0:py1, px0:px1]
    inside = roi & (chroma > thresh)
    filled = _clean_cover_silhouette(inside)
    if float(filled.mean()) < 0.08:
        return roi
    # If chroma blob is a sliver, use the profile rectangle clipped to scene.
    fb = _bbox(filled)
    if fb is not None and fb[2] < 0.62 * (px1 - px0):
        return _clean_cover_silhouette(roi)
    return filled


def _flood_from_phone_interior(rgb: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Flood from the most saturated interior pixel so the full back glass is one region."""
    if not scene.any():
        return np.zeros(rgb.shape[:2], dtype=bool)
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dist = cv2.distanceTransform(scene.astype(np.uint8), cv2.DIST_L2, 5)
    interior = scene & (dist >= max(6.0, 0.06 * min(height, width)))
    if not interior.any():
        interior = scene
    sat = hsv[:, :, 1].astype(np.float32)
    sat[~interior] = -1
    idx = int(np.argmax(sat))
    cy, cx = divmod(idx, width)
    best = np.zeros((height, width), dtype=bool)
    best_score = -1.0
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    for tol in (10, 16, 22, 30, 40, 52):
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        try:
            cv2.floodFill(rgb.copy(), mask, (int(cx), int(cy)), (0, 0, 0), (tol, tol, tol), (tol, tol, tol), flags)
        except cv2.error:
            continue
        fg = (mask[1:-1, 1:-1] > 0) & scene
        fg = _clean_cover_silhouette(fg)
        ratio = float(np.count_nonzero(fg)) / max(float(np.count_nonzero(scene)), 1.0)
        score = _phone_score(fg, height, width) + 0.22 * _roundness_bonus(fg)
        if 0.60 <= ratio <= 0.90:
            score += 0.30
        if ratio > 0.93:
            score -= 0.45
        if ratio < 0.50:
            score -= 0.35
        if score > best_score:
            best, best_score = fg, score
    return best


def _cover_contour_around_seed(rgb: np.ndarray, scene: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Largest scene contour that wraps the phone seed — the real cover outline."""
    if not scene.any() or not seed.any():
        return seed
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fill_val = int(np.median(gray[scene])) if scene.any() else 0
    work = gray.copy()
    work[~scene] = fill_val
    blur = cv2.GaussianBlur(work, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 70)
    edges[~scene] = 0
    k = max(5, int(round(0.012 * min(height, width))))
    if k % 2 == 0:
        k += 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys, xs = np.where(seed)
    cy, cx = float(ys.mean()), float(xs.mean())
    seed_area = float(np.count_nonzero(seed))
    best = np.zeros((height, width), dtype=bool)
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(seed_area * 1.35, 0.10 * height * width):
            continue
        if area > 0.92 * float(np.count_nonzero(scene)):
            continue
        if cv2.pointPolygonTest(contour, (cx, cy), False) < 0:
            continue
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        fb = filled.astype(bool) & scene
        if float(np.count_nonzero(fb & seed)) < 0.85 * seed_area:
            continue
        score = _phone_score(fb, height, width) + 0.3 * _roundness_bonus(fb)
        if score > best_score:
            best_score = score
            best = fb
    return _fill_holes(best) if best.any() else best


def _grow_phone_to_rim(
    rgb: np.ndarray,
    seed: np.ndarray,
    scene: np.ndarray,
    backdrop: np.ndarray,
) -> np.ndarray:
    """Expand an interior seed until pixels look more like the table than the phone."""
    if not seed.any() or not scene.any():
        return seed
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    seed_m = np.median(lab[seed], axis=0)
    back_src = backdrop if backdrop.any() else ~scene
    if not np.any(back_src):
        return seed
    back_m = np.median(lab[back_src], axis=0)
    fg = seed.astype(bool) & scene
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(120):
        dil = cv2.dilate(fg.astype(np.uint8), kernel).astype(bool) & scene
        ring = dil & ~fg
        if not ring.any():
            break
        samples = lab[ring]
        d_s = np.linalg.norm(samples - seed_m, axis=1)
        d_b = np.linalg.norm(samples - back_m, axis=1)
        accept = d_s <= (d_b * 1.35 + 3.0)
        if int(np.count_nonzero(accept)) < 6:
            break
        ys, xs = np.where(ring)
        fg[ys[accept], xs[accept]] = True
    return fg


def _grabcut_from_prior(
    rgb: np.ndarray,
    prior: np.ndarray,
    scene: np.ndarray | None = None,
) -> np.ndarray:
    """GrabCut guided by a prior — never paint a rectangle as sure-foreground."""
    height, width = rgb.shape[:2]
    if prior is None or not prior.any():
        return np.zeros((height, width), dtype=bool)
    scale = min(1.0, 560.0 / float(max(height, width)))
    if scale < 0.99:
        small = cv2.resize(
            rgb,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        prior_s = cv2.resize(prior.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        scene_s = (
            cv2.resize(scene.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if scene is not None
            else np.ones(small.shape[:2], dtype=bool)
        )
    else:
        small, prior_s, scene_s = rgb, prior.astype(bool), (scene.astype(bool) if scene is not None else np.ones((height, width), dtype=bool))
    sh, sw = small.shape[:2]
    mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    mask[scene_s] = cv2.GC_PR_BGD
    mask[prior_s] = cv2.GC_PR_FGD
    k = max(3, min(9, int(round(0.025 * min(sh, sw)))))
    if k % 2 == 0:
        k += 1
    sure = cv2.erode(prior_s.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if sure.any():
        mask[sure.astype(bool)] = cv2.GC_FGD
    mx, my = max(2, int(0.04 * sw)), max(2, int(0.04 * sh))
    mask[:my, :] = cv2.GC_BGD
    mask[-my:, :] = cv2.GC_BGD
    mask[:, :mx] = cv2.GC_BGD
    mask[:, -mx:] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, (0, 0, sw - 1, sh - 1), bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return prior.astype(bool)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if scale < 0.99:
        fg = cv2.resize(fg.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return fg


def _clean_cover_silhouette(mask: np.ndarray) -> np.ndarray:
    """Fill holes and join tiny rim gaps without shrinking the panel."""
    if mask is None or not mask.any():
        return np.zeros(mask.shape[:2], dtype=bool) if mask is not None else mask
    filled = _fill_holes(_largest_component(mask.astype(bool)))
    h, w = filled.shape
    k = max(3, int(round(0.004 * min(h, w))))
    if k % 2 == 0:
        k += 1
    k = min(k, 9)
    closed = cv2.morphologyEx(
        filled.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    return _fill_holes(_largest_component(closed.astype(bool)))


def _lab_border_fg(rgb: np.ndarray) -> np.ndarray:
    """Pixels whose color is unlike the image-border background."""
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    band = max(6, int(0.055 * min(height, width)))
    border = np.zeros((height, width), dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    med = np.median(lab[border], axis=0)
    dist = np.linalg.norm(lab - med[None, None, :], axis=2)
    border_d = dist[border]
    thresh = float(np.percentile(border_d, 92) + max(6.0, 0.55 * float(np.std(border_d))))
    fg = dist > thresh
    # If almost everything is foreground, the phone matches the backdrop — abort.
    if float(fg.mean()) > 0.88:
        return np.zeros((height, width), dtype=bool)
    return fg


def _grabcut_border_fg(rgb: np.ndarray) -> np.ndarray:
    """GrabCut with a sure-background frame so the full phone is foreground."""
    height, width = rgb.shape[:2]
    scale = min(1.0, 560.0 / float(max(height, width)))
    if scale < 0.99:
        small = cv2.resize(
            rgb,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb
    sh, sw = small.shape[:2]
    mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
    mx = max(2, int(0.07 * sw))
    my = max(2, int(0.07 * sh))
    mask[:my, :] = cv2.GC_BGD
    mask[-my:, :] = cv2.GC_BGD
    mask[:, :mx] = cv2.GC_BGD
    mask[:, -mx:] = cv2.GC_BGD
    y0, y1 = int(0.10 * sh), int(0.90 * sh)
    x0, x1 = int(0.16 * sw), int(0.84 * sw)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, (0, 0, sw - 1, sh - 1), bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((height, width), dtype=bool)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if scale < 0.99:
        fg = cv2.resize(fg.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return fg


def _edge_outer_only(rgb: np.ndarray) -> np.ndarray:
    """Largest external closed outline — ignore internal camera/logo edges."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    k = max(5, int(round(min(height, width) * 0.012)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = np.zeros((height, width), dtype=np.uint8)
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.10 * height * width:
            continue
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        score = _phone_score(filled.astype(bool), height, width)
        if score > best_score:
            best_score = score
            best = filled
    if best_score < 0:
        return np.zeros((height, width), dtype=bool)
    return best.astype(bool)


def _inner_printable_panel(
    cover_rgba: np.ndarray,
    outer: np.ndarray,
    has_alpha: bool,
    min_side: float,
) -> np.ndarray:
    """Inner printable lip from this cover's geometry (nested contour, then measured inset)."""
    rgb = cover_rgba[..., :3]
    outer_area = max(float(np.count_nonzero(outer)), 1.0)

    for candidate in (
        _inner_from_visible_lip(rgb, outer, min_side),
        _inner_from_nested_edges(rgb, outer, min_side),
    ):
        if candidate is None or not np.any(candidate):
            continue
        panel = _fill_holes(candidate.astype(bool) & outer)
        frac = float(np.count_nonzero(panel)) / outer_area
        if 0.84 <= frac <= 0.985 and _phone_score(panel, outer.shape[0], outer.shape[1]) >= 0.30:
            return panel

    bumper = _side_wall_width(cover_rgba, outer, has_alpha, min_side)
    ceiling = float(SIDE_WALL_MAX_FRACTION) * min_side
    floor = _bumper_floor_px(min_side)
    bumper = float(np.clip(bumper, floor, ceiling))
    back, _px = _inset_keeping_large_panel(outer, bumper, min_side)
    if float(np.count_nonzero(back)) < 0.84 * outer_area:
        back, _px = _inset_keeping_large_panel(outer, max(floor, 0.004 * min_side), min_side)
    return back & outer


def _inner_from_nested_edges(rgb: np.ndarray, outer: np.ndarray, min_side: float) -> np.ndarray:
    """Largest closed inner edge loop inside the cover (contour hierarchy)."""
    if not outer.any():
        return np.zeros(outer.shape, dtype=bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 22, 76)
    dist = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 5)
    lo = max(1.0, 0.004 * min_side)
    hi = max(lo + 2.0, 0.055 * min_side)
    band = outer & (dist >= lo) & (dist <= hi)
    work = np.zeros_like(edges)
    work[band] = edges[band]
    k = max(3, int(round(0.009 * min_side)))
    if k % 2 == 0:
        k += 1
    closed = cv2.morphologyEx(
        work, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return np.zeros(outer.shape, dtype=bool)
    outer_area = float(np.count_nonzero(outer))
    best = np.zeros(outer.shape, dtype=bool)
    best_score = -1.0
    hier = hierarchy[0]
    for i, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < 0.82 * outer_area or area > 0.98 * outer_area:
            continue
        filled = np.zeros(outer.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        panel = filled.astype(bool) & outer
        frac = float(np.count_nonzero(panel)) / max(outer_area, 1.0)
        if frac < 0.82 or frac > 0.98:
            continue
        parent = int(hier[i][3])
        nest = 0.12 if parent >= 0 else 0.0
        score = frac + nest + 0.12 * _phone_score(panel, outer.shape[0], outer.shape[1])
        if score > best_score:
            best_score = score
            best = panel
    return _fill_holes(best) if best.any() else best


def _fill_shallow_notches(mask: np.ndarray, min_side: float) -> np.ndarray:
    """Fill tiny convexity nicks; leave real rounds and port cutouts alone."""
    if mask is None or not mask.any():
        return mask
    u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask.astype(bool)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 16:
        return mask.astype(bool)
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return mask.astype(bool)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return mask.astype(bool)
    if defects is None:
        return mask.astype(bool)
    limit = 0.028 * float(min_side)
    out = u8.copy()
    for row in defects:
        start_i, end_i, far_i, depth_raw = (int(v) for v in row[0])
        depth = float(depth_raw) / 256.0
        if depth < 1.25 or depth > limit:
            continue
        tri = np.array(
            [contour[start_i][0], contour[end_i][0], contour[far_i][0]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(out, tri, 255)
    return _fill_holes(out > 0)


def _inner_from_visible_lip(rgb: np.ndarray, outer: np.ndarray, min_side: float) -> np.ndarray:
    """Find the inner case lip as a nested contour similar to the outer cover."""
    if not outer.any():
        return np.zeros(outer.shape, dtype=bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dist = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 5)
    lo = max(2.0, 0.008 * min_side)
    hi = max(lo + 2.0, 0.042 * min_side)
    band = outer & (dist >= lo) & (dist <= hi)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 24, 80)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, grad_bin = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = np.maximum(edges, grad_bin)
    combined[~band] = 0
    k = max(3, int(round(0.008 * min_side)))
    if k % 2 == 0:
        k += 1
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    outer_area = float(np.count_nonzero(outer))
    best = np.zeros(outer.shape, dtype=bool)
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.80 * outer_area or area > 0.97 * outer_area:
            continue
        peri = cv2.arcLength(contour, True)
        if peri < 16:
            continue
        filled = np.zeros(outer.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        fb = filled.astype(bool) & outer
        if float(np.count_nonzero(fb)) < 0.80 * outer_area:
            continue
        # Boundary should sit in the bumper band, not on the outer rim or the center.
        u8 = fb.astype(np.uint8)
        boundary = cv2.dilate(u8, np.ones((3, 3), np.uint8)) & (1 - u8)
        if not boundary.any():
            continue
        mean_d = float(np.mean(dist[boundary.astype(bool)]))
        if mean_d < lo * 0.7 or mean_d > hi * 1.15:
            continue
        score = area / outer_area + 0.15 * _phone_score(fb, outer.shape[0], outer.shape[1])
        if score > best_score:
            best_score = score
            best = fb
    return _fill_holes(best) if best.any() else best


def _edge_silhouette(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 25, 90)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = np.clip(mag / max(float(mag.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    _, mag_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = np.maximum(edges, mag_bin)

    k = max(7, int(round(min(height, width) * 0.018)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros((height, width), dtype=np.uint8)
    best_score = -1.0
    for contour in contours:
        if cv2.contourArea(contour) < height * width * 0.04:
            continue
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        score = _phone_score(filled.astype(bool), height, width)
        if score > best_score:
            best_score = score
            canvas = filled
    if best_score < 0:
        return closed.astype(bool)
    return canvas.astype(bool)


def _flood_fill_fg(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    best = np.zeros((height, width), dtype=bool)
    best_score = -1.0
    seeds = ((1, 1), (width - 2, 1), (1, height - 2), (width - 2, height - 2))
    for tol in (16, 24, 34, 48, 64):
        flood = rgb.copy()
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        lo = hi = (tol, tol, tol)
        for sx, sy in seeds:
            try:
                cv2.floodFill(flood, mask, (sx, sy), (0, 0, 0), lo, hi, flags)
            except cv2.error:
                continue
        bg = mask[1:-1, 1:-1] > 0
        fg = ~bg
        score = _phone_score(fg, height, width)
        if score > best_score:
            best, best_score = fg, score
    if best_score < 0.3:
        return np.zeros((height, width), dtype=bool)
    return best


def _grabcut_fg(rgb: np.ndarray) -> np.ndarray:
    """Coarse foreground on a downscaled copy — helps plain studio photos."""
    height, width = rgb.shape[:2]
    scale = min(1.0, 480.0 / float(max(height, width)))
    if scale < 0.99:
        small = cv2.resize(rgb, (max(8, int(width * scale)), max(8, int(height * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    margin_x = max(2, int(0.04 * sw))
    margin_y = max(2, int(0.04 * sh))
    rect = (margin_x, margin_y, max(1, sw - 2 * margin_x), max(1, sh - 2 * margin_y))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((height, width), dtype=bool)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if scale < 0.99:
        fg = cv2.resize(fg.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if _phone_score(fg, height, width) < 0.3:
        return np.zeros((height, width), dtype=bool)
    return fg


def _dark_object_fg(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if float(np.mean(gray)) < 80:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = otsu > 0
    else:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = otsu == 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if _phone_score(fg, height, width) < 0.3:
        return np.zeros((height, width), dtype=bool)
    return fg


def _phone_score(mask: np.ndarray, height: int, width: int) -> float:
    """Score how much a blob looks like a whole phone cover, not a strip or ring."""
    if mask is None or not mask.any():
        return 0.0
    area_frac = float(mask.mean())
    box = _bbox(mask)
    if box is None:
        return 0.0
    _x, _y, bw, bh = box
    if bw < 8 or bh < 8:
        return 0.0
    # Internal highlights / MagSafe leftovers are narrow relative to the frame.
    if bw < 0.14 * width or bh < 0.20 * height:
        return 0.04
    if area_frac < 0.08 or area_frac > 0.94:
        return 0.06

    aspect = bh / float(bw)
    if aspect < 1.0:
        aspect = 1.0 / max(aspect, 1e-6)

    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 1.0
    solidity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        area_px = float(cv2.contourArea(contour))
        if peri > 1:
            circularity = float(4.0 * np.pi * area_px / (peri * peri))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 1:
            solidity = area_px / hull_area
    rectness = float(mask.sum()) / float(max(bw * bh, 1))

    score = 0.0
    if 1.50 <= aspect <= 2.65:
        score += 0.38
    elif 1.15 <= aspect <= 3.05:
        score += 0.20
    if 0.22 <= area_frac <= 0.72:
        score += 0.18
    if 0.16 <= area_frac <= 0.82:
        score += 0.22
    elif 0.10 <= area_frac <= 0.90:
        score += 0.10
    # MagSafe/lens leftovers are small and round; a full cover is large.
    if area_frac < 0.20 and circularity > 0.72:
        score -= 0.45
    elif circularity < 0.75:
        score += 0.12
    if rectness > 0.78:
        score += 0.14
    elif rectness < 0.55:
        score -= 0.20
    if solidity > 0.90:
        score += 0.10
    return float(np.clip(score, 0.0, 1.0))


def _largest_component(binary: np.ndarray) -> np.ndarray:
    u8 = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return binary.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    u8 = (binary.astype(np.uint8)) * 255
    h, w = u8.shape
    flood = u8.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    if u8[0, 0] != 0:
        # Background may not include origin; flood from a dark border pixel.
        ys, xs = np.where(u8 == 0)
        if ys.size == 0:
            return binary.astype(bool)
        cv2.floodFill(flood, mask, (int(xs[0]), int(ys[0])), 255)
    else:
        cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = u8 | holes
    return filled > 0


def _bumper_floor_px(min_side: float) -> float:
    """Minimum bumper in pixels, scaled to this photo — not a fixed 2px on tiny shots."""
    return float(max(0.45, 0.0028 * float(min_side)))


def _side_wall_width(
    cover_rgba: np.ndarray,
    outer: np.ndarray,
    has_alpha: bool,
    min_side: float,
) -> float:
    """Width of the visible side-wall / bumper ring, from a radial profile.

    Printable back starts just inside this ring so artwork reaches the back
    rim without covering case thickness. Falls back to a thin inset when the
    photo has no clear bumper lip (white-on-white).
    """
    floor = _bumper_floor_px(min_side)
    ceiling = float(min(SIDE_WALL_MAX_FRACTION, 0.025) * min_side)
    fallback = max(floor, 0.0045 * min_side)
    if not outer.any():
        return fallback

    dist = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 5)
    search = min(float(0.048 * min_side), float(dist.max()) * 0.40)
    max_t = int(max(5, round(search)))
    if has_alpha:
        signal = cover_rgba[..., 3].astype(np.float32)
    else:
        signal = cv2.cvtColor(cover_rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)

    means = np.full(max_t + 1, np.nan, dtype=np.float32)
    min_band = max(16, int(0.0015 * float(np.count_nonzero(outer))))
    for t in range(1, max_t + 1):
        band = outer & (dist >= (t - 0.6)) & (dist < (t + 0.6))
        if int(np.count_nonzero(band)) < min_band:
            continue
        means[t] = float(np.mean(signal[band]))

    valid = np.isfinite(means)
    if int(valid.sum()) < 5:
        return fallback

    # Fill tiny gaps in the profile.
    idx = np.arange(max_t + 1)
    good = valid & (idx > 0)
    if int(good.sum()) < 5:
        return fallback
    interp = np.interp(idx.astype(np.float32), idx[good].astype(np.float32), means[good])
    smooth = cv2.GaussianBlur(interp.reshape(-1, 1), (0, 0), sigmaX=1.1).ravel()
    deriv = np.abs(np.diff(smooth))
    deriv[0] = 0.0
    if deriv.size < 4:
        return fallback
    peak_i = int(np.argmax(deriv[1:]) + 1)
    peak = float(deriv[peak_i])
    noise = float(np.median(deriv[deriv > 0])) if (deriv > 0).any() else 0.0
    # Real bumper lip: a sharp change a few pixels in from the outer edge.
    if peak > max(4.0, 1.65 * noise) and 2 <= peak_i <= max_t - 1:
        return float(np.clip(float(peak_i), floor, ceiling))
    return float(np.clip(fallback, floor, ceiling))


def _bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _inset_keeping_large_panel(
    outer_filled: np.ndarray, bumper_px: float, min_side: float
) -> tuple[np.ndarray, float]:
    """Erode bumper but never shrink the back panel below ~78% of the outline."""
    outer_mean = float(outer_filled.mean())
    px = float(bumper_px)
    min_px = min(px, _bumper_floor_px(float(min_side)))
    for _ in range(12):
        back = inset_binary(outer_filled, px)
        ratio = float(back.mean()) / max(outer_mean, 1e-6)
        if ratio >= 0.92 and back.any():
            return back, px
        if px <= min_px:
            break
        px = max(min_px, px * 0.65)
    back = inset_binary(outer_filled, min_px)
    return back, min_px


def _refine_bumper_from_alpha(
    alpha: np.ndarray,
    outer_filled: np.ndarray,
    bumper_px: float,
    has_alpha: bool,
) -> float:
    if not has_alpha or not outer_filled.any():
        return bumper_px
    dist = cv2.distanceTransform(outer_filled.astype(np.uint8), cv2.DIST_L2, 5)
    vals = alpha[outer_filled]
    median = float(np.median(vals))
    ring = outer_filled & (dist > 1) & (dist < bumper_px * 1.6)
    if not ring.any():
        return bumper_px
    ring_a = float(np.mean(alpha[ring]))
    inner = outer_filled & (dist > bumper_px * 1.2)
    if not inner.any():
        return bumper_px
    inner_a = float(np.mean(alpha[inner]))
    if ring_a > inner_a + 25:
        for t in np.linspace(bumper_px * 0.5, bumper_px * 1.8, 12):
            band = outer_filled & (dist >= t - 2) & (dist <= t + 2)
            if not band.any():
                continue
            if float(np.mean(alpha[band])) <= median + 18:
                return max(bumper_px, float(t))
    return bumper_px


def _score(
    cover: np.ndarray,
    outer: np.ndarray,
    back: np.ndarray,
    camera_found: bool,
    has_alpha: bool,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    outer_frac = float(outer.mean())
    back_frac = float(back.mean())
    ratio = back_frac / max(outer_frac, 1e-6)

    score = 0.4
    if 0.12 <= outer_frac <= 0.92:
        score += 0.25
    if 0.70 <= ratio <= 0.995:
        score += 0.25
    elif 0.55 <= ratio < 0.70:
        score += 0.1
    if camera_found:
        score += 0.1
    score = float(np.clip(score, 0.0, 1.0))

    if ratio < 0.55:
        warnings.append("Inner back panel looks unusually small — inspect DEBUG masks before exporting.")
        score = min(score, 0.5)
    if not camera_found:
        warnings.append(
            "Camera area was not detected automatically. Processing continues. "
            "Use “Mark camera area” and drag a box on the cover preview if needed."
        )
    if score < DETECTION_CONFIDENCE_WARN:
        warnings.append(
            f"Automatic detection confidence is {score:.2f}. Inspect DEBUG masks before exporting."
        )
    return score, warnings
