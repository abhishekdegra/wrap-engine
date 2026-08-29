"""Orchestrates cover detection, design fit, mask clip, and export.

GUI code should call CoverProcessor only — not OpenCV internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.compositor import composite_design_under_cover
from app.core.cover_detector import CoverDetection, apply_manual_camera, detect_cover
from app.core.export_manager import export_jpg, export_png
from app.core.image_transform import fit_design_to_quad
from app.core.mask_generator import MaskSet
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

    def reset(self) -> None:
        self.cover = None
        self.design = None
        self.detection = None
        self.masks = None
        self.last_result = None

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
        return detection

    def set_manual_camera_rect(self, x: float, y: float, w: float, h: float) -> CoverDetection:
        if self.cover is None or self.detection is None:
            raise CoverError("Upload a phone cover first.")
        detection = apply_manual_camera(self.cover, self.detection, (x, y, w, h))
        self.detection = detection
        self.masks = detection.masks
        if self.design is not None:
            self.process_design_array(self.design)
        else:
            self.last_result = None
        return detection

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
        fitted = fit_design_to_quad(
            design_rgba, canvas_h, canvas_w, self.detection.back_quad
        )
        composite = composite_design_under_cover(
            self.cover, fitted, self.masks.final_print, cover_overlay=0.0
        )

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
            design_original=design_rgba,
            design_fitted=fitted,
            composite=composite,
            masks=self.masks,
            confidence=self.detection.confidence,
            warnings=list(self.detection.warnings),
            debug_images=debug,
        )
        self.last_result = result
        return result

    def cover_preview_result(self) -> ProcessingResult | None:
        if self.cover is None or self.masks is None or self.detection is None:
            return None
        debug = dict(self.detection.debug_images)
        debug["artwork_before_mask"] = np.zeros_like(self.cover)
        debug["final_composite"] = self.cover
        return ProcessingResult(
            cover=self.cover,
            design_original=self.design,
            design_fitted=np.zeros_like(self.cover),
            composite=self.cover,
            masks=self.masks,
            confidence=self.detection.confidence,
            warnings=list(self.detection.warnings),
            debug_images=debug,
        )

    def export(self, fmt: str, destination: str | Path | None = None) -> Path:
        if self.last_result is None:
            raise CoverError("Upload a cover and a design before exporting.")
        fmt = fmt.lower().lstrip(".")
        if fmt == "png":
            return export_png(self.last_result.composite, destination)
        if fmt in {"jpg", "jpeg"}:
            return export_jpg(self.last_result.composite, destination)
        raise CoverError(f"Unsupported export format '{fmt}'. Use PNG or JPG.")
