"""Procedural sample cover + example design.

Geometry comes from CoverTemplate so the PNG and the runtime masks match.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.mask_generator import circle_mask, rounded_rect_mask
from app.core.templates import CoverTemplate, get_default_template
from app.utils.constants import COVERS_DIR, EXAMPLES_DIR, ICONS_DIR, SAMPLE_COVER_FILENAME, SAMPLE_DESIGN_FILENAME


def ensure_sample_assets(template: CoverTemplate | None = None) -> None:
    template = template or get_default_template()
    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    cover_path = template.image_path
    if not cover_path.exists():
        render_sample_cover(template).save(cover_path, format="PNG")

    design_path = EXAMPLES_DIR / SAMPLE_DESIGN_FILENAME
    if not design_path.exists():
        render_sample_design(template.canvas_w, template.canvas_h).save(design_path, format="PNG")

    icon_path = ICONS_DIR / "app_icon.png"
    if not icon_path.exists():
        render_app_icon().save(icon_path, format="PNG")

    jpg_path = COVERS_DIR / "studio_cover.jpg"
    if not jpg_path.exists():
        render_studio_jpg_cover().save(jpg_path, format="JPEG", quality=95)


def render_sample_cover(template: CoverTemplate | None = None) -> Image.Image:
    """Top-down transparent phone case matching ``template`` geometry."""
    template = template or get_default_template()
    width, height = template.canvas_size
    feather = template.mask_feather_px

    outer = template.outer_case.to_pixels(width, height)
    inner = template.printable_area.to_pixels(width, height)
    camera = template.camera_exclusion.to_pixels(width, height)
    holes = [h.to_pixels(width, height) for h in template.camera_holes]

    outer_m = rounded_rect_mask(height, width, outer, feather)
    inner_m = rounded_rect_mask(height, width, inner, feather)
    camera_m = rounded_rect_mask(height, width, camera, feather)

    hole_m = np.zeros((height, width), dtype=np.float32)
    ring_m = np.zeros((height, width), dtype=np.float32)
    for hole in holes:
        hole_m = np.maximum(hole_m, circle_mask(height, width, hole, feather))
        outer_ring = hole.expand(11.0)
        inner_ring = hole.expand(-3.5)
        ring = circle_mask(height, width, outer_ring, feather) * (
            1.0 - circle_mask(height, width, inner_ring, feather)
        )
        ring_m = np.maximum(ring_m, ring)

    bumper_m = np.clip(outer_m * (1.0 - inner_m), 0.0, 1.0)
    lip_m = _sdf_band(inner_m, 0.35, 0.92)

    rgba = np.zeros((height, width, 4), dtype=np.float32)

    # Drop shadow under the case.
    shadow = cv2.GaussianBlur(outer_m, (0, 0), sigmaX=18.0)
    shadow = np.roll(shadow, 16, axis=0)
    _stamp(rgba, shadow, (0, 0, 0), 0.38)

    # Outer frosted body (visible mainly as bumper + faint back).
    _stamp(rgba, outer_m, (214, 224, 232), 0.22)

    # Bumper / rim — thicker, more opaque plastic.
    bumper_grad = _vertical_gradient(height, width, 0.88, 1.08)
    _stamp(rgba, bumper_m * bumper_grad, (186, 198, 210), 0.90)

    # Outer bevel highlight along the rim.
    outer_highlight = _sdf_band(outer_m, 0.55, 0.97) * bumper_m
    _stamp(rgba, outer_highlight, (255, 255, 255), 0.35)

    # Inner lip where the bumper meets the back panel.
    _stamp(rgba, lip_m, (150, 164, 178), 0.55)

    # Flat back panel — mostly transparent so artwork shows through.
    back_grad = _vertical_gradient(height, width, 1.05, 0.82)
    _stamp(rgba, inner_m * (1.0 - camera_m), (228, 236, 244), 0.28 * back_grad)

    # Diagonal specular highlight on the back (glass / plastic).
    spec = _diagonal_highlight(height, width) * inner_m * (1.0 - camera_m)
    _stamp(rgba, spec, (255, 255, 255), 0.18)

    # Camera island (raised protection area — not printable).
    _stamp(rgba, camera_m, (176, 186, 196), 0.82)
    island_edge = _sdf_band(camera_m, 0.45, 0.95)
    _stamp(rgba, island_edge, (255, 255, 255), 0.28)
    _stamp(rgba, camera_m * 0.25, (40, 48, 56), 0.12)

    # Lens rings.
    _stamp(rgba, ring_m, (52, 58, 66), 0.95)
    _stamp(rgba, ring_m * 0.45, (210, 218, 226), 0.25)

    # Flash (between lenses, slightly below).
    if len(holes) >= 2:
        flash_cx = (holes[0].cx + holes[1].cx) * 0.5
        flash_cy = holes[0].cy + max(holes[0].r, holes[1].r) * 0.72
        from app.core.templates import PixelCircle

        flash = PixelCircle(flash_cx, flash_cy, min(holes[0].r, holes[1].r) * 0.28)
        flash_m = circle_mask(height, width, flash, feather)
        _stamp(rgba, flash_m, (248, 244, 220), 0.92)

    # Punch camera holes — fully transparent cutouts.
    keep = 1.0 - hole_m
    rgba[..., 3] *= keep
    rgba[..., :3] *= keep[..., None]

    # Soft inner shadow just inside the bumper for depth.
    inner_shadow = cv2.GaussianBlur(lip_m, (0, 0), sigmaX=6.0) * inner_m
    _stamp(rgba, inner_shadow * (1.0 - camera_m), (90, 110, 130), 0.12)

    out = np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def render_sample_design(width: int, height: int) -> Image.Image:
    """Bold test artwork so mask leaks are obvious during debugging."""
    img = Image.new("RGB", (width, height), (18, 18, 28))
    draw = ImageDraw.Draw(img)

    # Colour blocks — if any colour appears on the bumper, the mask is wrong.
    bands = [
        (0.00, (230, 57, 70)),
        (0.20, (29, 53, 87)),
        (0.40, (69, 123, 157)),
        (0.60, (168, 218, 220)),
        (0.80, (241, 196, 15)),
    ]
    for i, (y0, color) in enumerate(bands):
        y1 = bands[i + 1][0] if i + 1 < len(bands) else 1.0
        draw.rectangle([0, int(y0 * height), width, int(y1 * height)], fill=color)

    # Grid so cropping / rounding is easy to judge.
    step = max(40, width // 16)
    for x in range(0, width, step):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=2)
    for y in range(0, height, step):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=2)

    font = _load_font(int(width * 0.09))
    text = "SAMPLE\nDESIGN"
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=12)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = (height - th) // 2
    # Shadow then fill.
    draw.multiline_text((tx + 4, ty + 4), text, font=font, fill=(0, 0, 0), align="center", spacing=12)
    draw.multiline_text((tx, ty), text, font=font, fill=(255, 255, 255), align="center", spacing=12)

    return img.convert("RGBA")


def render_app_icon(size: int = 256) -> Image.Image:
    template = get_default_template()
    cover = render_sample_cover(template)
    cover.thumbnail((size - 24, size - 24), Image.Resampling.LANCZOS)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - cover.width) // 2
    y = (size - cover.height) // 2
    icon.paste(cover, (x, y), cover)
    return icon


def _stamp(dst: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int], alpha: float | np.ndarray) -> None:
    """Porter-Duff over of a solid colour into ``dst`` (float 0–1 RGBA)."""
    m = mask.astype(np.float32)
    if np.isscalar(alpha):
        src_a = m * float(alpha)
    else:
        src_a = m * alpha.astype(np.float32)
    src_a = np.clip(src_a, 0.0, 1.0)
    color = np.array(rgb, dtype=np.float32) / 255.0
    dst_a = dst[..., 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    src_pre = color * src_a[..., None]
    dst_pre = dst[..., :3] * dst_a[..., None]
    out_pre = src_pre + dst_pre * (1.0 - src_a[..., None])
    rgb_out = np.divide(out_pre, np.maximum(out_a[..., None], 1e-6))
    dst[..., :3] = np.where(out_a[..., None] > 1e-6, rgb_out, 0.0)
    dst[..., 3] = out_a


def _vertical_gradient(height: int, width: int, top: float, bottom: float) -> np.ndarray:
    col = np.linspace(top, bottom, height, dtype=np.float32)[:, None]
    return np.repeat(col, width, axis=1)


def _diagonal_highlight(height: int, width: int) -> np.ndarray:
    yy, xx = np.ogrid[0:height, 0:width]
    t = (xx / max(width, 1) * 0.65 + yy / max(height, 1) * 0.35).astype(np.float32)
    band = np.exp(-((t - 0.28) ** 2) / (2 * 0.045**2))
    return (band * 0.85).astype(np.float32)


def _sdf_band(mask: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """Soft edge band of a 0–1 mask (for bevels / lips)."""
    band = np.clip(mask - inner, 0.0, None) * np.clip(outer - mask, 0.0, None) * 4.0
    return np.clip(band, 0.0, 1.0).astype(np.float32)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_studio_jpg_cover(width: int = 1100, height: int = 2200) -> Image.Image:
    """Opaque JPG: white backdrop, faint bumper, MagSafe ring, dark camera island."""
    from app.core.templates import PixelCircle, PixelRect

    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    outer = PixelRect(x=width * 0.10, y=height * 0.05, w=width * 0.80, h=height * 0.90, radius=width * 0.12)
    inner = PixelRect(x=width * 0.145, y=height * 0.085, w=width * 0.71, h=height * 0.83, radius=width * 0.08)
    outer_m = rounded_rect_mask(height, width, outer, 1.5)
    inner_m = rounded_rect_mask(height, width, inner, 1.5)
    bumper = np.clip(outer_m - inner_m, 0, 1)

    rgb = np.where(outer_m[..., None] > 0.4, np.array([248, 250, 252], dtype=np.uint8), rgb)
    rgb = np.where(bumper[..., None] > 0.4, np.array([168, 176, 186], dtype=np.uint8), rgb)

    # MagSafe — large center ring (must NOT become the printable mask).
    mag = PixelCircle(cx=width * 0.50, cy=height * 0.52, r=width * 0.16)
    ring = circle_mask(height, width, mag, 1.4)
    hole = circle_mask(height, width, PixelCircle(mag.cx, mag.cy, mag.r * 0.78), 1.4)
    magsafe = np.clip(ring - hole, 0, 1)
    rgb = np.where(magsafe[..., None] > 0.5, np.array([210, 214, 218], dtype=np.uint8), rgb)

    cam = PixelRect(x=width * 0.18, y=height * 0.11, w=width * 0.28, h=height * 0.13, radius=width * 0.05)
    cam_m = rounded_rect_mask(height, width, cam, 1.4)
    rgb = np.where(cam_m[..., None] > 0.5, np.array([52, 56, 62], dtype=np.uint8), rgb)
    for cx in (width * 0.26, width * 0.38):
        lens = circle_mask(height, width, PixelCircle(cx, height * 0.175, width * 0.045), 1.2)
        rgb = np.where(lens[..., None] > 0.5, np.array([12, 52, 58], dtype=np.uint8), rgb)

    return Image.fromarray(rgb, mode="RGB")
