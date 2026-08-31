"""Orchestrates cover detection, design fit, mask clip, and export.

GUI code should call CoverProcessor only — not OpenCV internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.compositor import composite_design_under_cover
from app.core.cover_adjust import CoverLook, apply_cover_look
from app.core.cover_detector import CoverDetection, apply_manual_camera, apply_manual_camera_mask, detect_cover
from app.core.export_manager import export_jpg, export_png
from app.core.image_transform import fit_design_to_quad
from app.core.mask_generator import MaskSet
from app.utils.constants import PREVIEW_MAX_SIDE
from app.utils.image_utils import CoverError, load_image_rgba, mask_to_preview


@dataclass
class ProcessingResult:
    cover: np.ndarray
    design_original: np.ndarray | None
    design_fitted: np.ndarray
    composite: np.ndarray
    masks: MaskSet
    confidence: float
    warnings: list[str] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


class CoverProcessor:
    def __init__(self) -> None:
        self.cover: np.ndarray | None = None
        self.design: np.ndarray | None = None
        self.detection: CoverDetection | None = None
        self.masks: MaskSet | None = None
        self.last_result: ProcessingResult | None = None
        self.look = CoverLook()
        self._fitted: np.ndarray | None = None

    def reset(self) -> None:
        self.cover = None
        self.design = None
        self.detection = None
        self.masks = None
        self.last_result = None
        self.look = CoverLook()
        self._fitted = None

    def load_cover(self, path: str | Path) -> CoverDetection:
        cover_path = Path(path)
        if not cover_path.exists():
            raise CoverError("Please choose a phone-cover image first.")

        cover = load_image_rgba(cover_path, cap_dimension=None)
        detection = detect_cover(cover)
        self.cover = cover
        self.detection = detection
        self.masks = detection.masks
        self.last_result = None
        self._fitted = None
        return detection

    def set_manual_camera_rect(self, x: float, y: float, w: float, h: float) -> CoverDetection:
        if self.cover is None or self.detection is None:
            raise CoverError("Upload a phone cover first.")
        detection = apply_manual_camera(self.cover, self.detection, (x, y, w, h))
        self.detection = detection
        self.masks = detection.masks
        self._refresh_composite()
        return detection

    def set_manual_camera_mask(self, camera_aa: np.ndarray) -> CoverDetection:
        if self.cover is None or self.detection is None:
            raise CoverError("Upload a phone cover first.")
        detection = apply_manual_camera_mask(self.cover, self.detection, camera_aa)
        self.detection = detection
        self.masks = detection.masks
        self._refresh_composite()
        return detection

    def set_look(self, look: CoverLook) -> None:
        self.look = look

    def process_design(self, design_path: str | Path) -> ProcessingResult:
        if self.cover is None or self.masks is None or self.detection is None:
            raise CoverError("Upload a phone cover first.")

        path = Path(design_path)
        if not path.exists():
            raise CoverError("Please choose a design image first.")

        design = load_image_rgba(path)
        self.design = design
        return self.process_design_array(design)

    def process_design_array(self, design_rgba: np.ndarray) -> ProcessingResult:
        if self.cover is None or self.masks is None or self.detection is None:
            raise CoverError("Upload a phone cover first.")

        if design_rgba.size == 0:
            raise CoverError("The design image is empty.")

        self.design = design_rgba
        canvas_h, canvas_w = self.cover.shape[:2]
        self._fitted = fit_design_to_quad(
            design_rgba, canvas_h, canvas_w, self.detection.back_quad
        )
        return self._build_result(preview=False)

    def cover_preview_result(self) -> ProcessingResult | None:
        if self.cover is None or self.masks is None or self.detection is None:
            return None
        composite = self._composite(preview=True)
        debug = dict(self.detection.debug_images)
        debug["artwork_before_mask"] = np.zeros_like(self.cover)
        debug["final_composite"] = composite
        return ProcessingResult(
            cover=self.cover,
            design_original=self.design,
            design_fitted=np.zeros_like(self.cover),
            composite=composite,
            masks=self.masks,
            confidence=self.detection.confidence,
            warnings=list(self.detection.warnings),
            debug_images=debug,
        )

    def preview_composite(self) -> np.ndarray | None:
        if self.cover is None or self.masks is None:
            return None
        return self._composite(preview=True)

    def preview_bare_cover(self) -> np.ndarray | None:
        """Cover photo only (no wrap) for aligning the camera mask."""
        if self.cover is None:
            return None
        cover = self.cover
        height, width = cover.shape[:2]
        longest = max(height, width)
        if longest > PREVIEW_MAX_SIDE:
            scale = PREVIEW_MAX_SIDE / float(longest)
            nw = max(1, int(round(width * scale)))
            nh = max(1, int(round(height * scale)))
            cover = cv2.resize(cover, (nw, nh), interpolation=cv2.INTER_AREA)
        return apply_cover_look(cover, self.look)

    def export(self, fmt: str, destination: str | Path | None = None) -> Path:
        if self.design is None or self._fitted is None or self.last_result is None:
            raise CoverError("Upload a cover and a design before exporting.")
        image = self._composite(preview=False)
        fmt = fmt.lower().lstrip(".")
        if fmt == "png":
            return export_png(image, destination)
        if fmt in {"jpg", "jpeg"}:
            return export_jpg(image, destination)
        raise CoverError(f"Unsupported export format '{fmt}'. Use PNG or JPG.")

    def _refresh_composite(self) -> None:
        if self.cover is None or self.masks is None or self.detection is None:
            return
        if self.design is not None and self._fitted is not None:
            self._build_result(preview=False)
        else:
            self.last_result = None

    def _build_result(self, *, preview: bool) -> ProcessingResult:
        assert self.cover is not None and self.masks is not None and self.detection is not None
        composite = self._composite(preview=preview)
        fitted = self._fitted if self._fitted is not None else np.zeros_like(self.cover)
        debug = dict(self.detection.debug_images)
        debug.update(
            {
                "artwork_before_mask": fitted,
                "final_composite": composite,
                "final_print_mask": mask_to_preview(self.masks.final_print),
                "camera_exclusion": mask_to_preview(self.masks.camera_exclusion),
                "edge_exclusion": mask_to_preview(self.masks.edge_exclusion),
            }
        )
        result = ProcessingResult(
            cover=self.cover,
            design_original=self.design,
            design_fitted=fitted,
            composite=composite,
            masks=self.masks,
            confidence=self.detection.confidence,
            warnings=list(self.detection.warnings),
            debug_images=debug,
        )
        self.last_result = result
        return result

    def _composite(self, *, preview: bool) -> np.ndarray:
        assert self.cover is not None and self.masks is not None
        cover, fitted, print_mask = self.cover, self._fitted, self.masks.final_print
        if fitted is None:
            fitted = np.zeros_like(cover)
        if preview:
            cover, fitted, print_mask = _downscale_stack(cover, fitted, print_mask)
        looked = apply_cover_look(cover, self.look)
        return composite_design_under_cover(looked, fitted, print_mask, cover_overlay=0.0)


def _downscale_stack(
    cover: np.ndarray,
    fitted: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = cover.shape[:2]
    longest = max(height, width)
    if longest <= PREVIEW_MAX_SIDE:
        return cover, fitted, mask
    scale = PREVIEW_MAX_SIDE / float(longest)
    nw = max(1, int(round(width * scale)))
    nh = max(1, int(round(height * scale)))
    cover_s = cv2.resize(cover, (nw, nh), interpolation=cv2.INTER_AREA)
    fitted_s = cv2.resize(fitted, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    return cover_s, fitted_s, mask_s
