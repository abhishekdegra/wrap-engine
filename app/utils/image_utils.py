"""Image loading, conversion, and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.utils.constants import MAX_IMAGE_DIMENSION, SUPPORTED_INPUT_EXTENSIONS


class CoverError(Exception):
    """User-facing processing error."""


def load_image_rgba(path: str | Path, *, cap_dimension: int | None = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Load an image as RGBA uint8, applying EXIF orientation.

    Transparent PNGs keep their alpha. JPEG/WebP without alpha get a
    fully opaque alpha channel. Oversized images are downscaled so the
    longest side does not exceed ``cap_dimension``.
    """
    path = Path(path)
    if not path.exists():
        raise CoverError(f"The file could not be found:\n{path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise CoverError(
            f"Unsupported image format '{suffix}'.\n"
            "Please use PNG, JPG, JPEG, or WebP."
        )

    try:
        with Image.open(path) as src:
            src.load()
            src = ImageOps.exif_transpose(src) or src
            image = src.convert("RGBA")
    except UnidentifiedImageError as exc:
        raise CoverError(
            "This file is not a valid image, or the file is corrupted.\n"
            "Please choose a PNG, JPG, or WebP file."
        ) from exc
    except OSError as exc:
        raise CoverError(
            "The image could not be read. It may be corrupted or incomplete."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise CoverError(
            "This image is extremely large and was rejected for safety.\n"
            "Please use a smaller file."
        ) from exc

    if cap_dimension is not None:
        longest = max(image.size)
        if longest > cap_dimension:
            scale = cap_dimension / float(longest)
            new_size = (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    arr = np.array(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise CoverError("The image could not be converted to RGBA.")
    return arr


def pil_from_rgba(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    return Image.fromarray(array, mode="RGBA")


def mask_to_preview(mask: np.ndarray) -> np.ndarray:
    """Convert a float 0–1 mask to an opaque grayscale RGBA preview."""
    if mask.dtype != np.float32 and mask.dtype != np.float64:
        gray = mask.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
    else:
        gray = mask.astype(np.float32)
    gray = np.clip(gray, 0.0, 1.0)
    u8 = np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.dstack([u8, u8, u8, np.full_like(u8, 255)])


def contour_overlay(
    cover_rgba: np.ndarray,
    binary: np.ndarray,
    rgb: tuple[int, int, int] = (40, 200, 90),
) -> np.ndarray:
    """Draw the contour of ``binary`` on a copy of the cover (debug)."""
    import cv2

    out = cover_rgba.copy()
    u8 = (binary.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return out
    overlay = out[..., :3].copy()
    cv2.drawContours(overlay, contours, -1, rgb, thickness=max(2, min(out.shape[:2]) // 400))
    out[..., :3] = overlay
    return out


def numpy_rgba_to_qimage(array: np.ndarray):
    """Convert HxWx4 uint8 RGBA to a detached QImage."""
    from PySide6.QtGui import QImage

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    height, width, channels = array.shape
    if channels != 4:
        raise ValueError("Expected RGBA array")
    qimage = QImage(array.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
    return qimage.copy()
