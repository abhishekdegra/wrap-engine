"""Camera-module exclusion: robust hierarchical detection of the physical outer camera rim.

Detects the camera island / housing outer rim for any shape (rounded square,
rounded rectangle, circle, oval, pill, multi-lens custom contour, etc.) while
treating individual camera lenses, rings, flash, and sensors as internal landmarks.
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
) -> tuple[np.ndarray, bool, float, list[str], np.ndarray | None, list[np.ndarray], np.ndarray | None]:
    """Return (mask, found, confidence, warnings, contour, openings, outer_rim_contour).

    ``contour`` is the inner camera opening contour (final artwork cutout boundary).
    ``outer_rim_contour`` is the physical outer boundary of the raised camera rim.
    ``openings`` is a list of binary masks for individual lens/flash cutouts.
    """
    warnings: list[str] = []
    height, width = cover_rgba.shape[:2]
    alpha = cover_rgba[..., 3]
    rgb = cover_rgba[..., :3]
    outer = outer_filled.astype(bool)
    back = back_panel.astype(bool)
    has_alpha = int(alpha.max()) >= 12 and int(np.percentile(alpha, 5)) < 240

    _empty = (
        np.zeros((height, width), dtype=bool),
        False,
        0.0,
        warnings,
        None,
        [],
        None,
    )

    if not back.any():
        return _empty

    # Collect camera island candidates across all strategies: (conf, mask, in_contour, source, out_contour)
    candidates: list[tuple[float, np.ndarray, np.ndarray | None, str, np.ndarray | None]] = []

    # ------------------------------------------------------------------
    # Strategy 1: Alpha-channel islands & holes (transparent mockup PNGs)
    # ------------------------------------------------------------------
    if has_alpha:
        alpha_cands = _alpha_camera_candidates(alpha, outer, back, height, width)
        for conf, c_mask, c_pts in alpha_cands:
            if _plausible_camera_island(c_mask, back):
                candidates.append((conf, c_mask, c_pts, "alpha", None))

    # ------------------------------------------------------------------
    # Strategy 2: Hierarchical RGB Camera-Module Detection
    # ------------------------------------------------------------------
    rgb_cands = _rgb_camera_candidates(rgb, back, height, width)
    for conf, c_mask, c_pts, src, out_pts in rgb_cands:
        if _plausible_camera_island(c_mask, back):
            candidates.append((conf, c_mask, c_pts, src, out_pts))

    if not candidates:
        return _empty

    # Detect internal optics/lenses for enclosure scoring and clustering.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bys, bxs = np.where(back)
    y0, y1 = int(bys.min()), int(bys.max())
    x0, x1 = int(bxs.min()), int(bxs.max())
    bh, bw = max(1, y1 - y0), max(1, x1 - x0)
    min_side = float(min(bw, bh))
    back_area = float(np.count_nonzero(back))
    top = np.zeros_like(back)
    top_h = int(0.55 * bh) if (bh / max(bw, 1) >= 1.4) else bh
    top[y0 : y0 + top_h, x0:x1] = back[y0 : y0 + top_h, x0:x1]

    lenses, n_lenses, optics_list = _detect_and_cluster_lenses(gray, top, back, min_side, back_area)

    # Score and rank candidates.
    ranked = _rank_camera_candidates(candidates, rgb, back, height, width, lenses=lenses, n_lenses=n_lenses)
    if not ranked:
        return _empty

    best_score, best_mask, best_contour, best_source, best_outer_contour = ranked[0]
    if best_score < CAMERA_MIN_CONFIDENCE:
        return _empty

    # Clean up mask.
    mask = fill_binary_holes(best_mask.astype(bool)) & back
    mask = _largest_component(mask)
    if not _plausible_camera_island(mask, back):
        return _empty

    # Extract or refine the inner camera opening contour directly from the final geometry.
    if best_contour is not None and best_contour.shape[0] >= 8:
        contour = _refine_existing_contour(best_contour, mask, rgb)
    else:
        contour = _extract_refined_contour(mask, back)
        if contour is not None:
            snapped = _snap_contour_to_rim(rgb, contour, mask, back)
            if snapped is not None:
                contour = snapped

    # Ensure mask strictly matches the refined inner camera opening polyline.
    if contour is not None and contour.shape[0] >= 6:
        m_poly = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(m_poly, [np.round(contour).astype(np.int32).reshape(-1, 1, 2)], 1)
        mask = fill_binary_holes((m_poly > 0) & back)

    # Detect internal lens / flash openings inside the camera island.
    openings = _detect_camera_openings(rgb, alpha, mask, back, has_alpha, optics_list=optics_list)

    back_area = max(float(back.mean()), 1e-6)
    if float(mask.mean()) / back_area > 0.35:
        return _empty
    if not _plausible_camera_island(mask, back):
        return _empty

    return mask, True, float(best_score), warnings, contour, openings, best_outer_contour



# ---------------------------------------------------------------------------
# Candidate Generation: Alpha Channel (transparent PNG cases)
# ---------------------------------------------------------------------------


def _alpha_camera_candidates(
    alpha: np.ndarray,
    outer: np.ndarray,
    back: np.ndarray,
    height: int,
    width: int,
) -> list[tuple[float, np.ndarray, np.ndarray | None]]:
    """Find camera cutouts and opaque camera islands in transparent PNG covers."""
    cands: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    min_side = float(min(height, width))

    # Opaque camera island (alpha is noticeably higher than transparent back).
    if back.any():
        med_a = float(np.median(alpha[back]))
        thresh = max(med_a + 45.0, 110.0)
        if thresh < 250:
            raw_island = fill_binary_holes((alpha >= thresh) & back)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(raw_island.astype(np.uint8), 8)
            back_cnt = float(np.count_nonzero(back))
            for i in range(1, n):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if 0.005 * back_cnt <= area <= 0.25 * back_cnt:
                    blob = labels == i
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    closed = cv2.morphologyEx(blob.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
                    closed = fill_binary_holes(closed) & back
                    if _plausible_camera_island(closed, back):
                        cands.append((0.96, closed, None))

    # Transparent holes inside the cover (lens cutouts).
    holes = (alpha < 18) & outer
    dist_outer = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 5)
    holes = holes & (dist_outer > 8)

    if holes.any():
        n_h, labels_h, stats_h, cents_h = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
        valid_holes: list[tuple[int, float, float]] = []
        for i in range(1, n_h):
            area = float(stats_h[i, cv2.CC_STAT_AREA])
            if area > 16:
                valid_holes.append((i, float(cents_h[i][0]), float(cents_h[i][1])))

        if valid_holes:
            bys, _ = np.where(back)
            y0 = int(bys.min()) if bys.size else 0
            bh = int(bys.max() - y0) if bys.size else height
            top_holes = [h for h in valid_holes if h[2] < y0 + 0.55 * bh]
            if top_holes:
                pts_arr = np.array([[h[1], h[2]] for h in top_holes])
                if len(top_holes) == 1:
                    clusters = [[top_holes[0][0]]]
                else:
                    max_gap = max(24.0, 0.25 * min_side)
                    visited = set()
                    clusters = []
                    for idx in range(len(top_holes)):
                        if idx in visited:
                            continue
                        cluster = [top_holes[idx][0]]
                        visited.add(idx)
                        queue = [idx]
                        while queue:
                            curr = queue.pop(0)
                            for other in range(len(top_holes)):
                                if other not in visited:
                                    d = np.hypot(pts_arr[curr, 0] - pts_arr[other, 0], pts_arr[curr, 1] - pts_arr[other, 1])
                                    if d < max_gap:
                                        visited.add(other)
                                        cluster.append(top_holes[other][0])
                                        queue.append(other)
                        clusters.append(cluster)

                for cluster in clusters:
                    cl_mask = np.zeros((height, width), dtype=bool)
                    for lbl in cluster:
                        cl_mask |= (labels_h == lbl)
                    pad = max(8.0, 0.045 * min_side)
                    dil = dilate_binary(cl_mask, pad) & back
                    closed = _smooth_island_contour(dil) & back
                    if _plausible_camera_island(closed, back):
                        cands.append((0.95 if len(cluster) >= 2 else 0.85, closed, None))

    return cands


# ---------------------------------------------------------------------------
# Candidate Generation: RGB Image (JPGs, photos, opaque phone covers)
# ---------------------------------------------------------------------------


def _rgb_camera_candidates(
    rgb: np.ndarray,
    back: np.ndarray,
    height: int,
    width: int,
) -> list[tuple[float, np.ndarray, np.ndarray | None]]:
    """Multi-strategy camera module detection from RGB image."""
    cands: list[tuple[float, np.ndarray, np.ndarray | None, str]] = []

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(back)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh = max(1, y1 - y0)
    bw = max(1, x1 - x0)
    back_area = float(np.count_nonzero(back))
    min_side = float(min(bw, bh))

    if bh / max(bw, 1) < 1.4:
        top = back.copy()
    else:
        top = np.zeros_like(back)
        top_h = int(0.58 * bh)
        top[y0 : y0 + top_h, x0:x1] = back[y0 : y0 + top_h, x0:x1]

    # Pre-process top region with edge-preserving bilateral filter.
    bi = cv2.bilateralFilter(gray, 7, 50, 50)

    # 1. Detect candidate optics (internal landmarks: lenses, flash, rings).
    lenses, n_lenses, optics_list = _detect_and_cluster_lenses(gray, top, back, min_side, back_area)

    # 2. Outer bevel rim search around the optics cluster (PRIMARY strategy).
    if n_lenses >= 1 and lenses.any():
        r_mask, r_in_pts, r_out_pts = _trace_outer_bevel_rim(rgb, lenses, top, back, min_side, back_area, optics_list)
        if r_mask is not None and r_mask.any() and _plausible_camera_island(r_mask, back):
            conf = 1.0 if n_lenses >= 2 else 0.95
            cands.append((conf, r_mask, r_in_pts, "rim_trace", r_out_pts))
        else:
            # Fallback: Convex cluster hull with rounded padding as candidate.
            ys_l, xs_l = np.where(lenses)
            hull_pts = cv2.convexHull(np.column_stack([xs_l, ys_l]))
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(hull_mask, [hull_pts], 1)
            pad = max(10.0, 0.045 * min_side)
            dil_lenses = dilate_binary(hull_mask > 0, pad) & top & back
            smooth_hull = _smooth_island_contour(dil_lenses) & back
            if _plausible_camera_island(smooth_hull, back):
                cnt = _extract_refined_contour(smooth_hull, back)
                if cnt is not None:
                    snapped = _snap_contour_to_rim(rgb, cnt, smooth_hull, back)
                    if snapped is not None:
                        cnt = snapped
                cands.append((0.88 if n_lenses >= 2 else 0.78, smooth_hull, cnt, "convex_hull", None))

    # 3. Gradient energy plateau (detects full glass/opaque camera module bump).
    plateau_cands = _gradient_energy_plateau(rgb, top, back, min_side, back_area, y0, bh)
    for conf, p_mask, p_pts in plateau_cands:
        if _plausible_camera_island(p_mask, back):
            if n_lenses >= 1 and lenses.any():
                cover_ratio = float(np.count_nonzero(p_mask & lenses)) / max(float(np.count_nonzero(lenses)), 1.0)
                if cover_ratio >= 0.70:
                    conf = 0.97
            cands.append((conf, p_mask, p_pts, "energy_plateau", None))

    # 4. Contour detection on multi-threshold edge maps (closed loops in upper panel).
    edge_loops = _find_closed_edge_loops(bi, top, back, min_side, back_area, y0, bh, x0, bw, lenses, n_lenses)
    for score, loop_mask, loop_contour in edge_loops:
        cands.append((score, loop_mask, loop_contour, "edge_loop", None))

    # 5. Color / Tone contrast housing (dark bump on light phone, or vice-versa).
    tone_cands = _tone_contrast_candidates(rgb, gray, top, back, back_area, min_side, y0, bh)
    for conf, t_mask in tone_cands:
        if _plausible_camera_island(t_mask, back):
            cands.append((conf, t_mask, None, "tone_contrast", None))

    # 6. Multi-channel color gradient loops.
    grad_cands = _color_gradient_candidates(rgb, top, back, back_area, min_side, y0, bh, x0, bw)
    for conf, g_mask, g_pts in grad_cands:
        if _plausible_camera_island(g_mask, back):
            cands.append((conf, g_mask, g_pts, "color_grad", None))

    return cands



# ---------------------------------------------------------------------------
# Strategy 3: Internal Optics & Landmark Detection
# ---------------------------------------------------------------------------


def _detect_and_cluster_lenses(
    gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
    back_area: float,
) -> tuple[np.ndarray, int, list[tuple[float, float, float, float]]]:
    """Detect circular camera lenses and bright flash elements, then cluster them.

    Returns:
        (clustered_mask, count_in_cluster, optics_list)
        where optics_list is a list of (cx, cy, radius, contrast_diff).
    """
    height, width = gray.shape
    if not top.any():
        return np.zeros((height, width), dtype=bool), 0, []

    # Downscale for fast circle search if image is very large
    scale = 1.0
    longest = max(height, width)
    if longest > 1200:
        scale = 1000.0 / float(longest)
        nw, nh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        gray_work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        top_work = cv2.resize(top.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    else:
        gray_work = gray
        top_work = top

    bi_work = cv2.bilateralFilter(gray_work, 7, 45, 45)
    local_med = float(np.median(gray_work[top_work])) if top_work.any() else 128.0

    min_r_w = max(3, int(round(0.012 * min_side * scale)))
    max_r_w = max(min_r_w + 3, int(round(0.12 * min_side * scale)))

    detected_optics: list[tuple[float, float, float, float]] = []

    # 1. Hough circles with strict radial edge contrast validation.
    circles = cv2.HoughCircles(
        bi_work,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=float(min_r_w * 1.6),
        param1=45,
        param2=18,
        minRadius=min_r_w,
        maxRadius=max_r_w,
    )
    if circles is not None:
        for cx_w, cy_w, cr_w in np.round(circles[0]).astype(int):
            if 0 <= cx_w < gray_work.shape[1] and 0 <= cy_w < gray_work.shape[0] and top_work[cy_w, cx_w]:
                yy, xx = np.ogrid[0:gray_work.shape[0], 0:gray_work.shape[1]]
                dist_sq = (xx - cx_w) ** 2 + (yy - cy_w) ** 2
                inner = (dist_sq <= (cr_w * 0.70) ** 2) & top_work
                outer = (dist_sq > (cr_w * 0.90) ** 2) & (dist_sq <= (cr_w * 1.35) ** 2) & top_work
                if inner.any() and outer.any():
                    in_val = float(np.mean(gray_work[inner]))
                    out_val = float(np.mean(gray_work[outer]))
                    diff = out_val - in_val
                    if diff >= 8.0 or diff <= -15.0:
                        detected_optics.append((
                            float(cx_w) / scale,
                            float(cy_w) / scale,
                            float(cr_w) / scale,
                            diff,
                        ))

    # 2. High-contrast circular / compact blobs (optics blobs).
    dark_bin = (gray_work < max(20.0, min(local_med - 16.0, 160.0))) & top_work
    bright_bin = (gray_work > max(195.0, local_med + 45.0)) & top_work
    opt_bin = dark_bin | bright_bin

    n, labels, stats, cents = cv2.connectedComponentsWithStats(opt_bin.astype(np.uint8), 8)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        w_b = float(stats[i, cv2.CC_STAT_WIDTH])
        h_b = float(stats[i, cv2.CC_STAT_HEIGHT])
        if area < np.pi * (min_r_w * 0.7) ** 2 or area > np.pi * (max_r_w * 1.2) ** 2:
            continue
        if min(w_b, h_b) < min_r_w * 0.8 or max(w_b, h_b) > max_r_w * 2.2:
            continue
        aspect = max(w_b, h_b) / max(min(w_b, h_b), 1.0)
        if aspect > 1.45:
            continue
        cx_w, cy_w = float(cents[i][0]), float(cents[i][1])
        cr_w = float(max(w_b, h_b) / 2.0)
        already = False
        for ocx, ocy, ocr, _ in detected_optics:
            if np.hypot(cx_w / scale - ocx, cy_w / scale - ocy) < max(cr_w / scale, ocr) * 0.8:
                already = True
                break
        if not already:
            in_val = float(np.mean(gray_work[labels == i]))
            diff = local_med - in_val
            detected_optics.append((
                cx_w / scale,
                cy_w / scale,
                cr_w / scale,
                diff,
            ))

    if not detected_optics:
        return np.zeros((height, width), dtype=bool), 0, []

    # 3. Spatial clustering of optics into a cohesive camera module cluster.
    pts = np.array([[o[0], o[1]] for o in detected_optics])
    max_d = max(24.0, 0.25 * min_side)

    clusters: list[list[int]] = []
    visited: set[int] = set()
    for i in range(len(detected_optics)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        q = [i]
        while q:
            curr = q.pop(0)
            for j in range(len(detected_optics)):
                if j not in visited:
                    d = float(np.hypot(pts[curr, 0] - pts[j, 0], pts[curr, 1] - pts[j, 1]))
                    if d <= max_d:
                        visited.add(j)
                        cluster.append(j)
                        q.append(j)
        clusters.append(cluster)

    best_cl = max(clusters, key=len)
    cl_optics = [detected_optics[idx] for idx in best_cl]

    clustered_mask = np.zeros((height, width), dtype=bool)
    for cx, cy, cr, _ in cl_optics:
        cv2.circle(
            clustered_mask.view(np.uint8),
            (int(round(cx)), int(round(cy))),
            int(round(cr)),
            1,
            thickness=cv2.FILLED,
        )

    return clustered_mask & top & back, len(cl_optics), cl_optics


# ---------------------------------------------------------------------------
# Strategy 4: Direct Directional Outer Rim Ridge Search
# ---------------------------------------------------------------------------


def _build_smooth_rounded_poly(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    radius: float | None = None,
    n_pts: int = 256,
) -> np.ndarray:
    """Construct a dense, smooth 2D rounded polygon (squircle/capsule/circle) with n_pts vertices."""
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    if radius is None:
        radius = max(6.0, min(0.18 * min(w, h), 45.0))
    radius = min(radius, 0.45 * min(w, h))

    n_c = n_pts // 4
    poly = []
    # Top-Right arc
    for a in np.linspace(-np.pi / 2, 0, n_c, endpoint=False):
        poly.append([x_max - radius + radius * np.cos(a), y_min + radius + radius * np.sin(a)])
    # Bottom-Right arc
    for a in np.linspace(0, np.pi / 2, n_c, endpoint=False):
        poly.append([x_max - radius + radius * np.cos(a), y_max - radius + radius * np.sin(a)])
    # Bottom-Left arc
    for a in np.linspace(np.pi / 2, np.pi, n_c, endpoint=False):
        poly.append([x_min + radius + radius * np.cos(a), y_max - radius + radius * np.sin(a)])
    # Top-Left arc
    for a in np.linspace(np.pi, 3 * np.pi / 2, n_c, endpoint=False):
        poly.append([x_min + radius + radius * np.cos(a), y_min + radius + radius * np.sin(a)])

    return np.array(poly, dtype=np.float64)


def _trace_outer_bevel_rim(
    rgb: np.ndarray,
    lenses: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
    back_area: float,
    optics_list: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Trace the actual physical camera rim contour geometrically from image edge gradients.

    Determines the exact physical bounds along Left, Right, Top, Bottom and
    constructs a smooth, perspective-regularized squircle / rounded polygon
    snapped to the outer contact ridge of the raised camera rim.
    """
    height, width = rgb.shape[:2]
    if not optics_list or len(optics_list) == 0:
        ys, xs = np.where(lenses)
        if ys.size == 0:
            return np.zeros((height, width), dtype=bool), None
        c_xmin, c_xmax = float(xs.min()), float(xs.max())
        c_ymin, c_ymax = float(ys.min()), float(ys.max())
    else:
        xs_all = []
        ys_all = []
        for cx, cy, cr, _ in optics_list:
            xs_all.extend([cx - cr, cx + cr])
            ys_all.extend([cy - cr, cy + cr])
        c_xmin, c_xmax = float(min(xs_all)), float(max(xs_all))
        c_ymin, c_ymax = float(min(ys_all)), float(max(ys_all))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bi = cv2.bilateralFilter(gray, 7, 50, 50)
    gx = np.abs(cv2.Sobel(bi, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(bi, cv2.CV_32F, 0, 1, ksize=3))
    gx[~back] = 0
    gy[~back] = 0

    min_pad = max(2.0, 0.008 * min_side)
    max_pad = max(min_pad + 4.0, 0.045 * min_side)

    bys, bxs = np.where(back)
    b_xmin, b_xmax = int(bxs.min()), int(bxs.max())
    b_ymin, b_ymax = int(bys.min()), int(bys.max())

    # 1. Left rim peak
    l_min = max(b_xmin + 2, int(round(c_xmin - max_pad)))
    l_max = max(b_xmin + 2, int(round(c_xmin - min_pad)))
    l_range = range(l_min, min(width - 1, l_max + 1))
    gl = [float(np.mean(gx[int(max(0, c_ymin)):int(min(height, c_ymax + 1)), x])) for x in l_range]
    left_x = float(l_range[int(np.argmax(gl))]) if gl else c_xmin - min_pad

    # 2. Right rim peak
    r_min = min(b_xmax - 2, int(round(c_xmax + min_pad)))
    r_max = min(b_xmax - 2, int(round(c_xmax + max_pad)))
    r_range = range(r_min, min(width - 1, r_max + 1))
    gr = [float(np.mean(gx[int(max(0, c_ymin)):int(min(height, c_ymax + 1)), x])) for x in r_range]
    right_x = float(r_range[int(np.argmax(gr))]) if gr else c_xmax + min_pad

    # 3. Top rim peak
    t_min = max(b_ymin + 2, int(round(c_ymin - max_pad)))
    t_max = max(b_ymin + 2, int(round(c_ymin - min_pad)))
    t_range = range(t_min, min(height - 1, t_max + 1))
    gt = [float(np.mean(gy[y, int(max(0, c_xmin)):int(min(width, c_xmax + 1))])) for y in t_range]
    top_y = float(t_range[int(np.argmax(gt))]) if gt else c_ymin - min_pad

    # 4. Bottom rim peak
    b_min = min(b_ymax - 2, int(round(c_ymax + min_pad)))
    b_max = min(b_ymax - 2, int(round(c_ymax + max_pad)))
    b_range = range(b_min, min(height - 1, b_max + 1))
    gb = [float(np.mean(gy[y, int(max(0, c_xmin)):int(min(width, c_xmax + 1))])) for y in b_range]
    bot_y = float(b_range[int(np.argmax(gb))]) if gb else c_ymax + min_pad

    w_box = max(1.0, right_x - left_x)
    h_box = max(1.0, bot_y - top_y)
    aspect = max(w_box, h_box) / min(w_box, h_box)

    # 1. Outer rim radius
    if len(optics_list) == 1 and aspect < 1.15:
        radius_out = 0.50 * min(w_box, h_box)
    elif aspect > 1.65:
        radius_out = 0.50 * min(w_box, h_box)
    else:
        radius_out = max(6.0, min(0.18 * min(w_box, h_box), 45.0))
        radius_out = min(radius_out, 0.45 * min(w_box, h_box))

    # 2. Inner opening bounds (tightly enclosing the optics plateau)
    in_pad = max(1.5, min(0.006 * min_side, 4.0))
    in_left = max(left_x, c_xmin - in_pad)
    in_right = min(right_x, c_xmax + in_pad)
    in_top = max(top_y, c_ymin - in_pad)
    in_bot = min(bot_y, c_ymax + in_pad)

    w_in = max(1.0, in_right - in_left)
    h_in = max(1.0, in_bot - in_top)
    aspect_in = max(w_in, h_in) / min(w_in, h_in)

    if len(optics_list) == 1 and aspect_in < 1.15:
        radius_in = 0.50 * min(w_in, h_in)
    elif aspect_in > 1.65:
        radius_in = 0.50 * min(w_in, h_in)
    else:
        radius_in = max(5.0, min(0.18 * min(w_in, h_in), 38.0))
        radius_in = min(radius_in, 0.45 * min(w_in, h_in))

    # Construct both smooth rounded polygons
    in_poly = _build_smooth_rounded_poly(in_left, in_top, in_right, in_bot, radius=radius_in, n_pts=256)
    out_poly = _build_smooth_rounded_poly(left_x, top_y, right_x, bot_y, radius=radius_out, n_pts=256)

    # Sub-pixel snap to physical rim edges
    snapped_in = _snap_contour_to_rim(rgb, in_poly, lenses, back)
    if snapped_in is not None:
        in_poly = snapped_in

    snapped_out = _snap_contour_to_rim(rgb, out_poly, lenses, back)
    if snapped_out is not None:
        out_poly = snapped_out

    final_in_contour = _smooth_closed(_resample_closed(in_poly, 256), sigma_frac=0.0025)
    final_out_contour = _smooth_closed(_resample_closed(out_poly, 256), sigma_frac=0.0025)

    # The exclusion mask is rasterized from the INNER camera opening,
    # so the artwork WRAPS/PRINTS on top of the raised rim and terminates
    # right at the actual camera hardware plateau opening!
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(final_in_contour).astype(np.int32).reshape(-1, 1, 2)], 1)
    b_mask = (mask > 0) & back
    b_mask = fill_binary_holes(b_mask)

    if _plausible_camera_island(b_mask, back):
        return b_mask, final_in_contour, final_out_contour

    return np.zeros((height, width), dtype=bool), None, None





# ---------------------------------------------------------------------------
# Strategy 5: Closed Edge Loops & Gradient Contours
# ---------------------------------------------------------------------------


def _find_closed_edge_loops(
    blur_gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
    back_area: float,
    y0: int,
    bh: int,
    x0: int,
    bw: int,
    lenses: np.ndarray | None = None,
    n_lenses: int = 0,
) -> list[tuple[float, np.ndarray, np.ndarray | None]]:
    """Extract closed contour loops from multi-threshold Canny & Morphological gradients."""
    results: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    height, width = blur_gray.shape

    bi = cv2.bilateralFilter(blur_gray, 9, 75, 75)

    edge_maps = []
    for low, high in [(8, 30), (12, 45), (18, 60), (25, 80)]:
        canny = cv2.Canny(bi, low, high)
        canny[~top] = 0
        edge_maps.append(canny)

    grad = cv2.morphologyEx(bi, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    grad[~top] = 0
    _, grad_bin = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_maps.append(grad_bin)

    min_a = 0.012 * back_area
    max_a = 0.40 * back_area

    for emap in edge_maps:
        for k_val in [5, 9, 13, max(5, int(round(0.018 * min_side)) | 1)]:
            closed = cv2.morphologyEx(emap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_val, k_val)))
            contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_a or area > max_a:
                    continue
                peri = float(cv2.arcLength(contour, True))
                if peri < 20:
                    continue
                circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
                if circ > 0.65 and area < 0.025 * back_area:
                    continue
                if circ > 0.95:
                    continue

                x, y, cw, ch = cv2.boundingRect(contour)
                aspect = max(cw, ch) / float(max(min(cw, ch), 1))
                if aspect > 2.5:
                    continue
                solidity = area / float(max(cw * ch, 1))
                if solidity < 0.45:
                    continue

                M = cv2.moments(contour)
                if M["m00"] < 1:
                    continue
                cy = M["m01"] / M["m00"]
                fy = (cy - y0) / float(max(bh, 1))
                if fy > 0.58:
                    continue

                filled = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
                blob = (filled > 0) & top & back
                blob = fill_binary_holes(blob)

                if not _plausible_camera_island(blob, back):
                    continue

                pts = contour.reshape(-1, 2).astype(np.float64)
                rect_bonus = 0.15 if (aspect <= 1.4 and solidity >= 0.80) else 0.0
                score = 0.75 + 0.12 * solidity + rect_bonus

                if n_lenses >= 1 and lenses is not None and lenses.any():
                    cover_frac = float(np.count_nonzero(blob & lenses)) / max(float(np.count_nonzero(lenses)), 1.0)
                    if cover_frac >= 0.70:
                        score = min(0.98, score + 0.25)
                    elif cover_frac < 0.25:
                        score = max(0.15, score - 0.35)

                results.append((score, blob, pts))

    return results


# ---------------------------------------------------------------------------
# Strategy 6: Gradient Energy Plateau
# ---------------------------------------------------------------------------


def _gradient_energy_plateau(
    rgb: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    min_side: float,
    back_area: float,
    y0: int,
    bh: int,
) -> list[tuple[float, np.ndarray, np.ndarray | None]]:
    """Detect the full camera bump / plateau (glass or opaque) via localized gradient energy."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag[~top] = 0

    if not top.any():
        return []

    kw = max(7, int(round(0.06 * min_side))) | 1
    energy = cv2.boxFilter(grad_mag, -1, (kw, kw), normalize=True)
    energy[~top] = 0

    top_energy = energy[top]
    med_e = float(np.median(top_energy))
    p75_e = float(np.percentile(top_energy, 75))
    thresh_e = max(3.0, med_e + (p75_e - med_e) * 0.40)

    bin_energy = (energy >= thresh_e) & top

    k_close = max(11, int(round(0.08 * min_side))) | 1
    closed_energy = cv2.morphologyEx(
        bin_energy.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)),
    )
    closed_energy = fill_binary_holes(closed_energy.astype(bool)) & back

    n, labels, stats, cents = cv2.connectedComponentsWithStats(closed_energy.astype(np.uint8), 8)
    results = []

    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if 0.015 * back_area <= area <= 0.25 * back_area:
            cw = float(stats[i, cv2.CC_STAT_WIDTH])
            ch = float(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = max(cw, ch) / max(min(cw, ch), 1.0)
            if aspect <= 2.4 and cents[i][1] < y0 + 0.48 * bh:
                blob = fill_binary_holes(labels == i) & back
                cnt = _extract_refined_contour(blob, back)
                if cnt is not None:
                    snapped = _snap_contour_to_rim(rgb, cnt, blob, back)
                    if snapped is not None:
                        cnt = snapped
                results.append((0.95, blob, cnt))

    return results


# ---------------------------------------------------------------------------
# Strategy 7 & 8: Tone Contrast & Multi-Channel Gradient Loops
# ---------------------------------------------------------------------------


def _tone_contrast_candidates(
    rgb: np.ndarray,
    gray: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    min_side: float,
    y0: int,
    bh: int,
) -> list[tuple[float, np.ndarray]]:
    """Regions with different lightness/chroma from the back panel (camera bump)."""
    results: list[tuple[float, np.ndarray]] = []
    if not back.any():
        return results

    med_l = float(np.median(gray[back]))
    dark_bump = (gray < med_l - 18.0) & top
    light_bump = (gray > med_l + 25.0) & top

    for bump_raw, base_conf in [(dark_bump, 0.75), (light_bump, 0.65)]:
        k = max(7, int(round(0.03 * min_side))) | 1
        closed = cv2.morphologyEx(bump_raw.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(closed, 8)
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if 0.008 * back_area <= area <= 0.20 * back_area:
                w = float(stats[i, cv2.CC_STAT_WIDTH])
                h = float(stats[i, cv2.CC_STAT_HEIGHT])
                aspect = max(w, h) / float(max(min(w, h), 1))
                if aspect <= 2.3 and cents[i][1] < y0 + 0.48 * bh:
                    blob = fill_binary_holes(labels == i) & back
                    results.append((base_conf, blob))
    return results


def _color_gradient_candidates(
    rgb: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    back_area: float,
    min_side: float,
    y0: int,
    bh: int,
    x0: int,
    bw: int,
) -> list[tuple[float, np.ndarray, np.ndarray | None]]:
    """Multi-channel gradient edges to catch glass / transparent camera bevels."""
    results: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    height, width = rgb.shape[:2]

    grads = []
    for c in range(3):
        ch = rgb[..., c]
        blur = cv2.GaussianBlur(ch, (3, 3), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(np.sqrt(gx * gx + gy * gy))

    max_grad = np.maximum.reduce(grads)
    max_grad[~top] = 0

    if not top.any():
        return results

    med_g = float(np.median(max_grad[top]))
    thresh = max(18.0, med_g * 2.2)
    edge_bin = (max_grad > thresh).astype(np.uint8)

    k = max(5, int(round(0.02 * min_side))) | 1
    closed = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_a = 0.008 * back_area
    max_a = 0.20 * back_area

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_a or area > max_a:
            continue
        peri = float(cv2.arcLength(contour, True))
        if peri < 20:
            continue
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ > 0.72 and area < 0.025 * back_area:
            continue
        if circ > 0.93:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect = max(cw, ch) / float(max(min(cw, ch), 1))
        if aspect > 2.2:
            continue
        solidity = area / float(max(cw * ch, 1))
        if solidity < 0.50:
            continue

        M = cv2.moments(contour)
        if M["m00"] < 1:
            continue
        cy = M["m01"] / M["m00"]
        if (cy - y0) / float(max(bh, 1)) > 0.48:
            continue

        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        blob = fill_binary_holes((filled > 0) & back & top)
        if _plausible_camera_island(blob, back):
            pts = contour.reshape(-1, 2).astype(np.float64)
            results.append((0.72 + 0.15 * solidity, blob, pts))

    return results


# ---------------------------------------------------------------------------
# Candidate Ranking & Selection
# ---------------------------------------------------------------------------


def _rank_camera_candidates(
    candidates: list[tuple[float, np.ndarray, np.ndarray | None, str, np.ndarray | None]],
    rgb: np.ndarray,
    back: np.ndarray,
    height: int,
    width: int,
    lenses: np.ndarray | None = None,
    n_lenses: int = 0,
) -> list[tuple[float, np.ndarray, np.ndarray | None, str, np.ndarray | None]]:
    """Score and rank all camera island candidates, strictly enforcing whole-module enclosure."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

    back_area = float(max(np.count_nonzero(back), 1))
    scored: list[tuple[float, np.ndarray, np.ndarray | None, str, np.ndarray | None]] = []

    for item in candidates:
        if len(item) == 5:
            base_conf, mask, contour, source, out_contour = item
        else:
            base_conf, mask, contour, source = item[:4]
            out_contour = None

        if not mask.any():
            continue

        u8 = mask.astype(np.uint8)
        boundary = cv2.dilate(u8, np.ones((3, 3), np.uint8)) - u8
        if boundary.any():
            mean_boundary_grad = float(np.mean(grad_mag[boundary > 0]))
            norm_grad = min(1.0, mean_boundary_grad / 40.0)
        else:
            norm_grad = 0.5

        ys, xs = np.where(mask)
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area = float(np.count_nonzero(mask))
        area_frac = area / back_area
        aspect = max(w, h) / float(max(min(w, h), 1))
        solidity = area / float(max(w * h, 1))

        bys, bxs = np.where(back)
        bw = max(1, int(bxs.max() - bxs.min() + 1))
        bh = max(1, int(bys.max() - bys.min() + 1))
        fx = (float(xs.mean()) - float(bxs.min())) / float(bw)
        fy = (float(ys.mean()) - float(bys.min())) / float(bh)

        corner_bonus = 0.08 if (fx < 0.35 or fx > 0.65 or 0.42 < fx < 0.58) else 0.0
        top_bonus = max(0.0, 0.10 * (1.0 - fy / 0.50))

        if 0.020 <= area_frac <= 0.30:
            area_bonus = 0.25
        elif area_frac < 0.012:
            area_bonus = -0.50
        else:
            area_bonus = 0.0

        cnts_u8, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        peri = float(cv2.arcLength(cnts_u8[0], True)) if cnts_u8 else 0.0
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0) if peri > 0 else 0.0

        cover_ratio = 1.0
        lens_bonus = 0.0
        single_lens_penalty = 0.0
        if lenses is not None and lenses.any():
            cover_ratio = float(np.count_nonzero(mask & lenses)) / max(float(np.count_nonzero(lenses)), 1.0)
            if cover_ratio >= 0.70:
                lens_bonus = 0.50
            elif cover_ratio < 0.35:
                lens_bonus = -0.50

            if (circ > 0.60 or cover_ratio < 0.40) and area_frac < 0.12 and n_lenses >= 2:
                single_lens_penalty = -0.90  # Strictly reject individual lens circular cutouts

        squircle_bonus = 0.22 if (aspect <= 1.50 and solidity >= 0.75 and area_frac >= 0.020) else 0.0
        rim_bonus = 0.25 if source == "rim_trace" else 0.0

        total_score = (
            0.35 * base_conf
            + 0.18 * norm_grad
            + 0.10 * solidity
            + corner_bonus
            + top_bonus
            + area_bonus
            + squircle_bonus
            + rim_bonus
            + lens_bonus
            + single_lens_penalty
            - 0.06 * max(0.0, aspect - 1.5)
        )
        scored.append((total_score, mask, contour, source, out_contour))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored



# ---------------------------------------------------------------------------
# Contour Extraction & Shape-Preserving Refinement
# ---------------------------------------------------------------------------


def _extract_refined_contour(
    mask: np.ndarray,
    back: np.ndarray,
) -> np.ndarray | None:
    """Extract a smooth, shape-preserving polyline from the camera mask."""
    if mask is None or not mask.any():
        return None

    u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 32:
        return None

    peri = cv2.arcLength(contour, True)
    epsilon = max(0.003 * peri, 0.5)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 6:
        pts = contour.reshape(-1, 2).astype(np.float64)

    n_pts = int(np.clip(peri * 0.6, 64, 2048))
    pts = _resample_closed(pts, n_pts)
    pts = _smooth_closed(pts, sigma_frac=0.003)
    return pts


def _refine_existing_contour(
    contour: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
) -> np.ndarray:
    """Refine a detected contour polyline with arc-length resampling and gentle smoothing."""
    peri = float(np.linalg.norm(np.diff(np.vstack([contour, contour[0]]), axis=0), axis=1).sum())
    n_pts = int(np.clip(peri * 0.6, 64, 2048))
    resampled = _resample_closed(contour, n_pts)
    smoothed = _smooth_closed(resampled, sigma_frac=0.0025)
    return smoothed


def _snap_contour_to_rim(
    rgb: np.ndarray,
    contour: np.ndarray | None,
    mask: np.ndarray,
    back: np.ndarray,
) -> np.ndarray | None:
    """Snap contour vertices to the physical camera rim's outer contact edge.

    Searches in a tight local band (+- 4.5px) along the vertex normals to locate
    the sub-pixel gradient ridge of the raised rim without jumping into distant
    artwork/graphics.
    """
    if contour is None or contour.shape[0] < 12:
        return None

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bi = cv2.bilateralFilter(gray, 7, 50, 50)

    gx = cv2.Sobel(bi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bi, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad[~back] = 0

    n = contour.shape[0]
    prev = np.roll(contour, 1, axis=0)
    nxt = np.roll(contour, -1, axis=0)
    tangent = nxt - prev
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    lengths = np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    normal = normal / lengths

    centroid = contour.mean(axis=0)
    for i in range(n):
        if np.dot(normal[i], contour[i] - centroid) < 0:
            normal[i] = -normal[i]

    search_px = 4.5
    n_samples = 17

    snapped = contour.copy()
    moved_count = 0

    for i in range(n):
        offsets = np.linspace(-search_px, search_px, n_samples)
        samples_x = contour[i, 0] + offsets * normal[i, 0]
        samples_y = contour[i, 1] + offsets * normal[i, 1]
        sx = np.clip(samples_x, 0, width - 1).astype(np.float32)
        sy = np.clip(samples_y, 0, height - 1).astype(np.float32)
        vals = cv2.remap(
            grad,
            sx.reshape(1, -1),
            sy.reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).ravel()

        peak = int(np.argmax(vals))
        if vals[peak] > 8.0:
            if 1 <= peak <= n_samples - 2:
                v_m = float(vals[peak - 1])
                v_0 = float(vals[peak])
                v_p = float(vals[peak + 1])
                denom = v_m - 2.0 * v_0 + v_p
                sub = 0.5 * (v_m - v_p) / denom if abs(denom) > 1e-6 else 0.0
                sub = max(-0.5, min(0.5, sub))
                t = (peak + sub) / max(n_samples - 1, 1)
            else:
                t = peak / max(n_samples - 1, 1)
            offset_val = -search_px + t * (2.0 * search_px)
            snapped[i, 0] = contour[i, 0] + offset_val * normal[i, 0]
            snapped[i, 1] = contour[i, 1] + offset_val * normal[i, 1]
            moved_count += 1

    if moved_count < n * 0.10:
        return None

    snapped = _resample_closed(snapped, contour.shape[0])
    snapped = _smooth_closed(snapped, sigma_frac=0.0025)
    return snapped



# ---------------------------------------------------------------------------
# Internal Openings Detection
# ---------------------------------------------------------------------------


def _detect_camera_openings(
    rgb: np.ndarray,
    alpha: np.ndarray,
    camera_mask: np.ndarray,
    back: np.ndarray,
    has_alpha: bool,
    optics_list: list[tuple[float, float, float, float]] | None = None,
) -> list[np.ndarray]:
    """Find individual lens/flash openings inside the camera rim."""
    if camera_mask is None or not camera_mask.any():
        return []

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    island = camera_mask.astype(bool) & back.astype(bool)
    if not island.any():
        return []

    island_area = float(np.count_nonzero(island))
    min_side = max(8, int(np.sqrt(island_area)))
    openings: list[np.ndarray] = []

    # 1. From detected optics landmarks list
    if optics_list:
        for cx, cy, cr, _ in optics_list:
            if 0 <= cx < width and 0 <= cy < height and island[int(round(cy)), int(round(cx))]:
                circ = np.zeros((height, width), dtype=bool)
                cv2.circle(
                    circ.view(np.uint8),
                    (int(round(cx)), int(round(cy))),
                    int(round(cr)),
                    1,
                    thickness=cv2.FILLED,
                )
                circ = circ & island
                if circ.any():
                    openings.append(circ)

    # 2. Alpha holes (transparent covers).
    if has_alpha and int(alpha.max()) >= 12 and not openings:
        holes = (alpha < 18) & island
        if holes.any():
            n_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
            for i in range(1, n_h):
                area = float(stats_h[i, cv2.CC_STAT_AREA])
                if 0.002 * island_area <= area <= 0.45 * island_area:
                    openings.append((labels_h == i).astype(bool))

    # 3. Dark circular optics inside the island.
    if not openings:
        med_inside = float(np.median(gray[island]))
        thresh = max(24.0, min(med_inside - 16.0, 160.0))
        dark = (gray < thresh) & island
        if dark.any():
            n_d, labels_d, stats_d, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
            for i in range(1, n_d):
                area = float(stats_d[i, cv2.CC_STAT_AREA])
                if 0.003 * island_area <= area <= 0.45 * island_area:
                    w = float(stats_d[i, cv2.CC_STAT_WIDTH])
                    h = float(stats_d[i, cv2.CC_STAT_HEIGHT])
                    if max(w, h) / max(min(w, h), 1.0) <= 2.2:
                        openings.append((labels_d == i).astype(bool) & island)

    return openings


# ---------------------------------------------------------------------------
# Plausibility & Helpers
# ---------------------------------------------------------------------------


def _plausible_camera_island(mask: np.ndarray, back: np.ndarray) -> bool:
    """Reject horizontal strips, full panel wipes, and out-of-bounds regions."""
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
    frac = area / max(back_area, 1.0)
    frac_y = (cy - y0) / float(bh)
    is_tall = (bh / float(max(bw, 1)) >= 1.4)
    max_w = 0.85 * bw if is_tall else 0.96 * bw
    max_h = 0.55 * bh if is_tall else 0.96 * bh
    max_frac = 0.35 if is_tall else 0.88
    max_frac_y = 0.58 if is_tall else 0.88

    if aspect > 2.80:
        return False
    if w > max_w or h > max_h:
        return False
    if frac < 0.0030 or frac > max_frac:
        return False
    if extent < 0.38:
        return False
    if frac_y > max_frac_y:
        return False
    if h < 0.025 * bh and w > 0.40 * bw:
        return False
    return True


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if mask is None or not mask.any():
        return mask
    u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n <= 2:
        return mask.astype(bool)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def _smooth_island_contour(mask: np.ndarray) -> np.ndarray:
    """Close gaps and lightly smooth the binary island."""
    if not mask.any():
        return mask
    ys, xs = np.where(mask)
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    min_side = float(max(8, min(w, h)))
    k = max(5, int(round(0.08 * min_side)))
    k = min(k, max(5, int(0.20 * min_side))) | 1
    closed = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    closed = fill_binary_holes(closed.astype(bool))
    return smooth_silhouette(closed, sigma=max(1.0, 0.012 * min_side))


def _resample_closed(pts: np.ndarray, count: int) -> np.ndarray:
    """Even arc-length resampling of a closed polyline."""
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


def _smooth_closed(pts: np.ndarray, sigma_frac: float = 0.003) -> np.ndarray:
    """Periodic Gaussian smoothing of a closed polyline."""
    if pts.shape[0] < 12:
        return pts.astype(np.float64)
    n = int(pts.shape[0])
    sigma = float(max(1.2, sigma_frac * n))
    pad = int(max(3, round(sigma * 3)))
    ext = np.concatenate([pts[-pad:], pts, pts[:pad]], axis=0).astype(np.float32)
    xs = cv2.GaussianBlur(ext[:, 0].reshape(-1, 1), (0, 0), sigmaX=sigma).ravel()
    ys = cv2.GaussianBlur(ext[:, 1].reshape(-1, 1), (0, 0), sigmaX=sigma).ravel()
    return np.stack([xs[pad : pad + n], ys[pad : pad + n]], axis=1).astype(np.float64)
