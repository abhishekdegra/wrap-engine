"""High-resolution PNG / JPG export."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from app.utils.constants import JPG_QUALITY, OUTPUT_DIR
from app.utils.image_utils import CoverError, pil_from_rgba
import numpy as np


def export_png(image_rgba: np.ndarray, destination: str | Path | None = None) -> Path:
    path = _resolve_destination(destination, ".png")
    try:
        pil_from_rgba(image_rgba).save(path, format="PNG", optimize=True)
    except OSError as exc:
        raise CoverError(f"Could not save PNG:\n{path}") from exc
    return path


def export_jpg(image_rgba: np.ndarray, destination: str | Path | None = None) -> Path:
    """Flatten onto white (JPEG has no alpha) and save at high quality."""
    path = _resolve_destination(destination, ".jpg")
    rgba = pil_from_rgba(image_rgba)
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[3])
    try:
        background.save(path, format="JPEG", quality=JPG_QUALITY, subsampling=0, optimize=True)
    except OSError as exc:
        raise CoverError(f"Could not save JPG:\n{path}") from exc
    return path


def _resolve_destination(destination: str | Path | None, suffix: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if destination is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return OUTPUT_DIR / f"cover_mockup_{stamp}{suffix}"
    path = Path(destination)
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
