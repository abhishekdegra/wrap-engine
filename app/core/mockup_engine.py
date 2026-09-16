"""Photorealistic commercial mockup compositing engine.

Handles dynamic tabletop surface analysis, perspective homography,
multi-layer physical shadow simulation (contact, cast penumbra, ambient occlusion),
glossy surface reflections, and ambient light harmonization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.mockup_extractor import ExtractedPhone, extract_phone_foreground
from app.utils.constants import MOCKUP_BG_DIR


@dataclass
class SceneDescriptor:
    """Parameters describing the lighting, geometry, and placement for a background."""
    scene_id: str
    scene_name: str
    horizon_y: int  # Y pixel where table surface begins
    placement_center: Tuple[int, int]  # (cx, cy) on desk
    phone_target_height: int  # Desired phone height in pixels
    light_dir: Tuple[float, float]  # Normalized (dx, dy) direction of shadow cast
    shadow_softness: float  # Multiplier for cast shadow blur
    shadow_opacity: float  # Overall shadow strength (0.0 to 1.0)
    is_glossy: bool  # True for polished marble or shiny surfaces
    reflection_strength: float  # 0.0 to 0.35
    ambient_tint_strength: float  # 0.0 to 0.05
    recommended_rotation: float  # Recommended angle in degrees (-12.0 etc.)
    perspective_strength: float  # Subtle tabletop foreshortening (0.0 to 0.08)


# Curated presets tuned for the 10 scenes in assets/mockup_backgrounds
SCENE_PRESETS: Dict[str, SceneDescriptor] = {
    "01": SceneDescriptor(
        scene_id="01",
        scene_name="Morning Sunlit Oak Desk",
        horizon_y=550,
        placement_center=(560, 960),
        phone_target_height=540,
        light_dir=(0.75, 0.66),
        shadow_softness=1.0,
        shadow_opacity=0.75,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.03,
        recommended_rotation=-11.0,
        perspective_strength=0.045,
    ),
    "02": SceneDescriptor(
        scene_id="02",
        scene_name="Artisan Wooden Workspace",
        horizon_y=530,
        placement_center=(560, 970),
        phone_target_height=535,
        light_dir=(0.80, 0.60),
        shadow_softness=1.1,
        shadow_opacity=0.72,
        is_glossy=True,
        reflection_strength=0.10,
        ambient_tint_strength=0.035,
        recommended_rotation=-10.0,
        perspective_strength=0.045,
    ),
    "03": SceneDescriptor(
        scene_id="03",
        scene_name="Minimalist Slate Studio",
        horizon_y=560,
        placement_center=(560, 965),
        phone_target_height=540,
        light_dir=(0.70, 0.71),
        shadow_softness=1.15,
        shadow_opacity=0.70,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.02,
        recommended_rotation=-12.0,
        perspective_strength=0.05,
    ),
    "04": SceneDescriptor(
        scene_id="04",
        scene_name="Dark Moody Executive Desk",
        horizon_y=540,
        placement_center=(560, 960),
        phone_target_height=535,
        light_dir=(0.60, 0.80),
        shadow_softness=1.2,
        shadow_opacity=0.85,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.02,
        recommended_rotation=-10.5,
        perspective_strength=0.045,
    ),
    "05": SceneDescriptor(
        scene_id="05",
        scene_name="Scandinavian Bright Birch",
        horizon_y=560,
        placement_center=(560, 975),
        phone_target_height=540,
        light_dir=(0.72, 0.69),
        shadow_softness=1.0,
        shadow_opacity=0.68,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.03,
        recommended_rotation=-11.0,
        perspective_strength=0.045,
    ),
    "06": SceneDescriptor(
        scene_id="06",
        scene_name="Polished Carrara Marble",
        horizon_y=540,
        placement_center=(560, 960),
        phone_target_height=535,
        light_dir=(0.65, 0.76),
        shadow_softness=0.95,
        shadow_opacity=0.72,
        is_glossy=True,
        reflection_strength=0.18,
        ambient_tint_strength=0.02,
        recommended_rotation=-9.0,
        perspective_strength=0.04,
    ),
    "07": SceneDescriptor(
        scene_id="07",
        scene_name="Warm Window Cafe Table",
        horizon_y=570,
        placement_center=(560, 975),
        phone_target_height=545,
        light_dir=(0.82, 0.57),
        shadow_softness=1.2,
        shadow_opacity=0.70,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.04,
        recommended_rotation=-12.0,
        perspective_strength=0.05,
    ),
    "08": SceneDescriptor(
        scene_id="08",
        scene_name="Modern Travertine Stone",
        horizon_y=530,
        placement_center=(560, 955),
        phone_target_height=530,
        light_dir=(0.70, 0.71),
        shadow_softness=1.0,
        shadow_opacity=0.74,
        is_glossy=True,
        reflection_strength=0.14,
        ambient_tint_strength=0.025,
        recommended_rotation=-10.0,
        perspective_strength=0.045,
    ),
    "09": SceneDescriptor(
        scene_id="09",
        scene_name="Clean Neutral Staging Desk",
        horizon_y=550,
        placement_center=(560, 965),
        phone_target_height=540,
        light_dir=(0.68, 0.73),
        shadow_softness=1.05,
        shadow_opacity=0.70,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.025,
        recommended_rotation=-11.0,
        perspective_strength=0.045,
    ),
    "10": SceneDescriptor(
        scene_id="10",
        scene_name="Terrace Natural Timber",
        horizon_y=540,
        placement_center=(560, 970),
        phone_target_height=540,
        light_dir=(0.75, 0.66),
        shadow_softness=1.1,
        shadow_opacity=0.75,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.03,
        recommended_rotation=-11.5,
        perspective_strength=0.05,
    ),
}


def analyze_background_scene(bg_bgr: np.ndarray, scene_hint: Optional[str] = None) -> SceneDescriptor:
    """Analyze a background scene to detect horizon, lighting vector, and placement zone.
    
    If scene_hint matches a known preset (e.g. '01', 'background_01'), returns the tuned
    profile with any fine-grained dynamic adjustments.
    """
    h, w = bg_bgr.shape[:2]

    # Extract ID if matching filename pattern
    if scene_hint:
        for key in SCENE_PRESETS:
            if f"background_{key}" in scene_hint or f"bg_{key}" in scene_hint or scene_hint == key:
                preset = SCENE_PRESETS[key]
                # Scale preset if image resolution differs from standard 1122x1402
                scale_y = h / 1402.0
                scale_x = w / 1122.0
                if abs(scale_y - 1.0) > 0.05 or abs(scale_x - 1.0) > 0.05:
                    return SceneDescriptor(
                        scene_id=preset.scene_id,
                        scene_name=preset.scene_name,
                        horizon_y=int(preset.horizon_y * scale_y),
                        placement_center=(int(preset.placement_center[0] * scale_x), int(preset.placement_center[1] * scale_y)),
                        phone_target_height=int(preset.phone_target_height * scale_y),
                        light_dir=preset.light_dir,
                        shadow_softness=preset.shadow_softness,
                        shadow_opacity=preset.shadow_opacity,
                        is_glossy=preset.is_glossy,
                        reflection_strength=preset.reflection_strength,
                        ambient_tint_strength=preset.ambient_tint_strength,
                        recommended_rotation=preset.recommended_rotation,
                        perspective_strength=preset.perspective_strength,
                    )
                return preset

    # Automatic analysis for unseen custom background images
    gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Horizon detection: gradient along Y
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    row_gradient = np.mean(np.abs(sobel_y), axis=1)
    
    # Look for horizon between 30% and 65% of image height
    min_row, max_row = int(h * 0.30), int(h * 0.65)
    search_gradient = row_gradient[min_row:max_row]
    horizon_y = min_row + int(np.argmax(search_gradient)) if len(search_gradient) > 0 else int(h * 0.45)
    
    # 2. Table placement center: center-x, mid-way between horizon and bottom margin
    table_top = horizon_y
    table_bottom = int(h * 0.95)
    cy = int((table_top + table_bottom) / 2)
    cx = int(w / 2)
    
    # 3. Light direction analysis: left vs right brightness differential
    left_side = np.mean(gray[:, :w // 3])
    right_side = np.mean(gray[:, 2 * w // 3:])
    dx = 0.75 if left_side >= right_side else -0.75
    dy = 0.66
    norm = np.hypot(dx, dy)
    light_dir = (dx / norm, dy / norm)

    # 4. Target height is ~38% of background height
    phone_target_height = int(h * 0.385)

    return SceneDescriptor(
        scene_id="custom",
        scene_name="Custom Workspace Table",
        horizon_y=horizon_y,
        placement_center=(cx, cy),
        phone_target_height=phone_target_height,
        light_dir=light_dir,
        shadow_softness=1.0,
        shadow_opacity=0.72,
        is_glossy=False,
        reflection_strength=0.0,
        ambient_tint_strength=0.03,
        recommended_rotation=-11.0,
        perspective_strength=0.045,
    )


def transform_phone_for_scene(
    phone_rgba: np.ndarray,
    target_height: int,
    rotation_deg: float = -11.0,
    perspective_strength: float = 0.045,
    user_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scales, foreshortens with perspective homography, and rotates the phone.
    
    Returns:
        (transformed_rgba, transformed_mask)
    """
    src_h, src_w = phone_rgba.shape[:2]
    
    # Base scale to target height
    scale = (target_height * user_scale) / float(src_h)
    scaled_w = max(4, int(round(src_w * scale)))
    scaled_h = max(4, int(round(src_h * scale)))
    
    # Smooth resizing with sub-pixel interpolation
    scaled_rgba = cv2.resize(phone_rgba, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4)
    
    # Apply subtle perspective foreshortening (trapezoid where top is slightly narrower)
    if perspective_strength > 0.001:
        pw_top_factor = 1.0 - (perspective_strength * 0.7)
        pw_bot_factor = 1.0 + (perspective_strength * 0.3)
        
        src_pts = np.float32([
            [0, 0],
            [scaled_w - 1, 0],
            [scaled_w - 1, scaled_h - 1],
            [0, scaled_h - 1]
        ])
        
        top_inset = (scaled_w * (1.0 - pw_top_factor)) / 2.0
        bot_outset = (scaled_w * (pw_bot_factor - 1.0)) / 2.0
        
        # Warp in a slightly padded canvas to prevent clipping
        pad_x = int(bot_outset + 10)
        canvas_w = scaled_w + 2 * pad_x
        
        dst_pts = np.float32([
            [pad_x + top_inset, 0],
            [pad_x + scaled_w - top_inset, 0],
            [pad_x + scaled_w + bot_outset, scaled_h - 1],
            [pad_x - bot_outset, scaled_h - 1]
        ])
        
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        scaled_rgba = cv2.warpPerspective(
            scaled_rgba, M_persp, (canvas_w, scaled_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        scaled_w = canvas_w

    # Rotate with safe padding
    if abs(rotation_deg) > 0.05:
        # Calculate bounding size after rotation
        rad = np.radians(abs(rotation_deg))
        cos_r = np.cos(rad)
        sin_r = np.sin(rad)
        rot_w = int(scaled_h * sin_r + scaled_w * cos_r) + 8
        rot_h = int(scaled_h * cos_r + scaled_w * sin_r) + 8
        
        # Build rotation matrix around center
        center_x = scaled_w / 2.0
        center_y = scaled_h / 2.0
        M_rot = cv2.getRotationMatrix2D((center_x, center_y), -rotation_deg, 1.0)
        
        # Shift translation to center in expanded canvas
        M_rot[0, 2] += (rot_w / 2.0) - center_x
        M_rot[1, 2] += (rot_h / 2.0) - center_y
        
        rotated_rgba = cv2.warpAffine(
            scaled_rgba, M_rot, (rot_w, rot_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
    else:
        rotated_rgba = scaled_rgba

    alpha_channel = (rotated_rgba[:, :, 3].astype(np.float32) / 255.0).clip(0.0, 1.0)
    return rotated_rgba, alpha_channel


def generate_multi_layer_shadow(
    mask: np.ndarray,
    light_dir: Tuple[float, float],
    shadow_softness: float = 1.0,
    shadow_opacity: float = 0.72,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates 3 distinct physical shadow layers:
    
    1. Contact Shadow (Umbra): Sharp, dark rim directly under casing.
    2. Cast Shadow (Penumbra): Diffuse directional shadow reaching away from light.
    3. Ambient Occlusion (AO): Soft omnidirectional grounding contact halo.
    
    Returns:
        (umbra_canvas, penumbra_canvas, ao_canvas) matching mask shape with float 0..1 values.
    """
    h, w = mask.shape[:2]
    dx, dy = light_dir

    # 1. Contact Umbra: very tight Gaussian blur, tiny offset
    u_off_x = int(round(dx * 4.0))
    u_off_y = int(round(dy * 5.0))
    u_blur_size = max(3, int(round(5 * shadow_softness)) | 1)
    
    # Translate mask for umbra
    M_u = np.float32([[1, 0, u_off_x], [0, 1, u_off_y]])
    u_mask = cv2.warpAffine(mask, M_u, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    u_shadow = cv2.GaussianBlur(u_mask, (u_blur_size, u_blur_size), 1.8 * shadow_softness)
    # Erode slight bleed outside contact zone
    u_shadow = np.clip(u_shadow * 1.25, 0.0, 1.0) * (shadow_opacity * 0.75)

    # 2. Soft Cast Penumbra: larger offset, directional stretch & high diffusion
    p_off_x = int(round(dx * 26.0))
    p_off_y = int(round(dy * 32.0))
    p_blur_size = max(15, int(round(45 * shadow_softness)) | 1)
    
    M_p = np.float32([[1, 0, p_off_x], [0, 1, p_off_y]])
    p_mask = cv2.warpAffine(mask, M_p, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p_shadow = cv2.GaussianBlur(p_mask, (p_blur_size, p_blur_size), 16.0 * shadow_softness)
    p_shadow = np.clip(p_shadow, 0.0, 1.0) * (shadow_opacity * 0.42)

    # 3. Ambient Occlusion (AO): omnidirectional contact glow
    ao_blur_size = max(9, int(round(25 * shadow_softness)) | 1)
    ao_shadow = cv2.GaussianBlur(mask, (ao_blur_size, ao_blur_size), 7.5 * shadow_softness)
    ao_shadow = np.clip(ao_shadow, 0.0, 1.0) * (shadow_opacity * 0.28)

    return u_shadow, p_shadow, ao_shadow


def apply_glossy_table_reflection(
    bg_rgb: np.ndarray,
    phone_rgba: np.ndarray,
    phone_pos: Tuple[int, int],  # (x, y) top-left placement in bg
    reflection_strength: float = 0.15,
) -> np.ndarray:
    """Creates a subtle, realistic tabletop reflection for glossy surfaces (marble/polished wood).
    
    Operates strictly in RGB color space.
    """
    if reflection_strength <= 0.005:
        return bg_rgb

    bg_h, bg_w = bg_rgb.shape[:2]
    fg_h, fg_w = phone_rgba.shape[:2]
    px, py = phone_pos

    # Vertical flip of phone
    flipped = cv2.flip(phone_rgba, 0)
    
    # Anisotropic horizontal blur (simulating table surface roughness)
    f_rgb = flipped[:, :, :3]
    f_alpha = flipped[:, :, 3].astype(np.float32) / 255.0
    
    blurred_rgb = cv2.GaussianBlur(f_rgb, (25, 5), 4.0)
    
    # Vertical gradient fade: reflection quickly fades as distance from contact point increases
    y_indices = np.linspace(1.0, 0.0, fg_h, dtype=np.float32).reshape(-1, 1)
    fade_mask = np.power(y_indices, 1.8) * reflection_strength
    final_alpha = np.clip(f_alpha * fade_mask, 0.0, 1.0)

    # Position reflection immediately beneath the phone
    refl_y = py + fg_h - int(fg_h * 0.08)  # slight overlap at contact edge
    refl_h = min(fg_h, bg_h - refl_y)
    
    if refl_h <= 0 or px >= bg_w or px + fg_w <= 0:
        return bg_rgb

    rx0 = max(0, px)
    rx1 = min(bg_w, px + fg_w)
    sx0 = rx0 - px
    sx1 = sx0 + (rx1 - rx0)

    output = bg_rgb.copy()
    bg_crop = output[refl_y:refl_y + refl_h, rx0:rx1].astype(np.float32)
    fg_crop = blurred_rgb[:refl_h, sx0:sx1].astype(np.float32)
    a_crop = final_alpha[:refl_h, sx0:sx1, np.newaxis]

    # Screen / Alpha blend into background
    blended = (fg_crop * a_crop) + (bg_crop * (1.0 - a_crop))
    output[refl_y:refl_y + refl_h, rx0:rx1] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


def composite_mockup_on_background(
    bg_image: np.ndarray,
    extracted_phone: ExtractedPhone,
    scene: SceneDescriptor,
    user_scale: float = 1.0,
    user_rotation_offset: float = 0.0,
    user_nudge_x: int = 0,
    user_nudge_y: int = 0,
    shadow_intensity_mult: float = 1.0,
    *,
    bg_bgr: Optional[np.ndarray] = None,
    bg_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Composites an extracted phone onto a background with full physics simulation.
    
    Operates strictly in RGB color space to preserve 100% exact phone case colors.
    Returns:
        H x W x 3 uint8 image in RGB format.
    """
    if bg_rgb is not None:
        bg_image = bg_rgb
    elif bg_bgr is not None:
        bg_image = bg_bgr

    bg_h, bg_w = bg_image.shape[:2]

    # 1. Transform phone (scale, perspective, rotation)
    total_rot = scene.recommended_rotation + user_rotation_offset
    phone_rgba, phone_mask = transform_phone_for_scene(
        extracted_phone.rgba,
        target_height=scene.phone_target_height,
        rotation_deg=total_rot,
        perspective_strength=scene.perspective_strength,
        user_scale=user_scale,
    )
    fg_h, fg_w = phone_rgba.shape[:2]

    # 2. Determine placement coordinates
    cx, cy = scene.placement_center
    top_x = cx - (fg_w // 2) + user_nudge_x
    top_y = cy - (fg_h // 2) + user_nudge_y

    canvas_bg = bg_image.copy()

    # 3. Apply glossy tabletop reflection (if applicable) in RGB space
    if scene.is_glossy and scene.reflection_strength > 0.01:
        canvas_bg = apply_glossy_table_reflection(
            canvas_bg, phone_rgba, (top_x, top_y),
            reflection_strength=scene.reflection_strength * shadow_intensity_mult
        )

    # 4. Generate multi-layer shadows in phone bounding box
    eff_shadow_opacity = np.clip(scene.shadow_opacity * shadow_intensity_mult, 0.0, 1.0)
    u_shadow, p_shadow, ao_shadow = generate_multi_layer_shadow(
        phone_mask,
        light_dir=scene.light_dir,
        shadow_softness=scene.shadow_softness,
        shadow_opacity=eff_shadow_opacity,
    )

    # Combine shadows via multiply blend onto background
    pad = 80
    box_x0 = max(0, top_x - pad)
    box_y0 = max(0, top_y - pad)
    box_x1 = min(bg_w, top_x + fg_w + pad)
    box_y1 = min(bg_h, top_y + fg_h + pad)

    if box_x1 > box_x0 and box_y1 > box_y0:
        roi = canvas_bg[box_y0:box_y1, box_x0:box_x1].astype(np.float32)
        roi_h, roi_w = roi.shape[:2]

        px_in_roi = top_x - box_x0
        py_in_roi = top_y - box_y0

        roi_u = np.zeros((roi_h, roi_w), dtype=np.float32)
        roi_p = np.zeros((roi_h, roi_w), dtype=np.float32)
        roi_ao = np.zeros((roi_h, roi_w), dtype=np.float32)

        src_x0 = max(0, -px_in_roi)
        src_y0 = max(0, -py_in_roi)
        src_x1 = min(fg_w, roi_w - px_in_roi)
        src_y1 = min(fg_h, roi_h - py_in_roi)

        dst_x0 = max(0, px_in_roi)
        dst_y0 = max(0, py_in_roi)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 > src_x0 and src_y1 > src_y0:
            roi_u[dst_y0:dst_y1, dst_x0:dst_x1] = u_shadow[src_y0:src_y1, src_x0:src_x1]
            roi_p[dst_y0:dst_y1, dst_x0:dst_x1] = p_shadow[src_y0:src_y1, src_x0:src_x1]
            roi_ao[dst_y0:dst_y1, dst_x0:dst_x1] = ao_shadow[src_y0:src_y1, src_x0:src_x1]

        combined_shadow = 1.0 - (roi_u * 0.85 + roi_p * 0.65 + roi_ao * 0.45).clip(0.0, 0.90)
        roi = roi * combined_shadow[:, :, np.newaxis]
        canvas_bg[box_y0:box_y1, box_x0:box_x1] = np.clip(roi, 0, 255).astype(np.uint8)

    # 5. Harmonize phone with ambient room lighting (in RGB space)
    phone_rgb = phone_rgba[:, :, :3].astype(np.float32)
    phone_alpha = phone_rgba[:, :, 3].astype(np.float32) / 255.0

    if scene.ambient_tint_strength > 0.005:
        sample_y = min(bg_h - 20, max(0, cy + int(fg_h * 0.35)))
        sample_x = min(bg_w - 20, max(0, cx))
        ambient_sample = np.mean(bg_image[sample_y:sample_y + 15, sample_x:sample_x + 15], axis=(0, 1))
        
        tint_factor = scene.ambient_tint_strength
        phone_rgb = (phone_rgb * (1.0 - tint_factor)) + (ambient_sample * tint_factor)
        phone_rgb = np.clip(phone_rgb, 0.0, 255.0)

    # 6. Composite phone onto background using sub-pixel Alpha Over in pure RGB
    px0 = max(0, top_x)
    py0 = max(0, top_y)
    px1 = min(bg_w, top_x + fg_w)
    py1 = min(bg_h, top_y + fg_h)

    if px1 > px0 and py1 > py0:
        sx0 = px0 - top_x
        sy0 = py0 - top_y
        sx1 = sx0 + (px1 - px0)
        sy1 = sy0 + (py1 - py0)

        bg_slice = canvas_bg[py0:py1, px0:px1].astype(np.float32)
        fg_slice = phone_rgb[sy0:sy1, sx0:sx1]
        a_slice = phone_alpha[sy0:sy1, sx0:sx1, np.newaxis]

        # Porter-Duff Over in RGB
        blended = (fg_slice * a_slice) + (bg_slice * (1.0 - a_slice))
        canvas_bg[py0:py1, px0:px1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return canvas_bg


def load_all_background_paths(bg_dir: Path | str = MOCKUP_BG_DIR) -> List[Path]:
    """Finds all 10 background images sorted by scene number."""
    dir_path = Path(bg_dir)
    if not dir_path.exists():
        return []

    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in dir_path.iterdir() if p.is_file() and any(p.name.lower().endswith(ext) for ext in valid_exts)]
    
    # Sort naturally by filename
    paths.sort(key=lambda p: p.name.lower())
    return paths


def generate_all_mockups(
    extracted_phone: ExtractedPhone,
    bg_dir: Path | str = MOCKUP_BG_DIR,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    user_scale: float = 1.0,
    user_rotation_offset: float = 0.0,
    shadow_intensity_mult: float = 1.0,
) -> List[Tuple[str, np.ndarray]]:
    """Generates mockups for all backgrounds in the directory.
    
    Returns:
        List of (scene_title, composite_rgb_image)
    """
    bg_paths = load_all_background_paths(bg_dir)
    results: List[Tuple[str, np.ndarray]] = []
    total = len(bg_paths)

    for idx, path in enumerate(bg_paths, 1):
        scene_name = path.stem.replace(".png", "").replace(".jpg", "")
        if progress_callback:
            progress_callback(idx, total, f"Generating {scene_name}...")

        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        bg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        scene = analyze_background_scene(bg_rgb, scene_hint=path.name)
        comp_rgb = composite_mockup_on_background(
            bg_image=bg_rgb,
            extracted_phone=extracted_phone,
            scene=scene,
            user_scale=user_scale,
            user_rotation_offset=user_rotation_offset,
            shadow_intensity_mult=shadow_intensity_mult,
        )
        display_title = f"{scene.scene_id}. {scene.scene_name}"
        results.append((display_title, comp_rgb))

    return results
