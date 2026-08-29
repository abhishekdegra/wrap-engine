"""Reusable cover-template geometry.

Version 1 ships one sample template. Add another CoverTemplate later
without changing the processing engine — only this module and a PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.utils.constants import COVERS_DIR, SAMPLE_COVER_FILENAME


@dataclass(frozen=True)
class NormalizedRect:
    """Axis-aligned rectangle in fractions of canvas width / height (0–1).

    ``radius`` is a corner radius expressed as a fraction of canvas width.
    """

    x: float
    y: float
    w: float
    h: float
    radius: float = 0.0

    def to_pixels(self, canvas_w: int, canvas_h: int) -> PixelRect:
        return PixelRect(
            x=self.x * canvas_w,
            y=self.y * canvas_h,
            w=self.w * canvas_w,
            h=self.h * canvas_h,
            radius=self.radius * canvas_w,
        )


@dataclass(frozen=True)
class NormalizedCircle:
    """Circle in fractions of canvas size. ``r`` is a fraction of canvas width."""

    cx: float
    cy: float
    r: float

    def to_pixels(self, canvas_w: int, canvas_h: int) -> PixelCircle:
        return PixelCircle(
            cx=self.cx * canvas_w,
            cy=self.cy * canvas_h,
            r=self.r * canvas_w,
        )


@dataclass(frozen=True)
class PixelRect:
    x: float
    y: float
    w: float
    h: float
    radius: float = 0.0

    def inset(self, margin: float) -> PixelRect:
        m = max(0.0, margin)
        new_w = max(1.0, self.w - 2.0 * m)
        new_h = max(1.0, self.h - 2.0 * m)
        new_r = max(0.0, self.radius - m)
        return PixelRect(self.x + m, self.y + m, new_w, new_h, new_r)

    def expand(self, margin: float) -> PixelRect:
        m = max(0.0, margin)
        return PixelRect(
            self.x - m,
            self.y - m,
            self.w + 2.0 * m,
            self.h + 2.0 * m,
            self.radius + m,
        )

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h


@dataclass(frozen=True)
class PixelCircle:
    cx: float
    cy: float
    r: float

    def expand(self, margin: float) -> PixelCircle:
        return PixelCircle(self.cx, self.cy, max(0.0, self.r + margin))


@dataclass(frozen=True)
class CoverTemplate:
    """Geometry for one physical cover model.

    All processing (masks, fit, composite) reads this config instead of
    hard-coded pixel boxes scattered through the engine.
    """

    id: str
    name: str
    image_filename: str
    canvas_size: tuple[int, int]
    outer_case: NormalizedRect
    printable_area: NormalizedRect
    camera_exclusion: NormalizedRect
    camera_holes: tuple[NormalizedCircle, ...] = field(default_factory=tuple)
    # Extra inward safety margin from the printable-area rectangle (pixels).
    edge_margin_px: float = 12.0
    # Extra outward expansion of the camera island so print never kisses it.
    camera_safety_margin_px: float = 8.0
    # Antialias / feather width in pixels for SDF masks.
    mask_feather_px: float = 1.4

    @property
    def image_path(self) -> Path:
        return COVERS_DIR / self.image_filename

    @property
    def canvas_w(self) -> int:
        return self.canvas_size[0]

    @property
    def canvas_h(self) -> int:
        return self.canvas_size[1]


# ---------------------------------------------------------------------------
# Sample cover — coordinates are the single source of truth for both
# sample_cover.png generation and runtime masks.
# Canvas: 1600 × 3200. Outer case, inner back panel, camera island.
# ---------------------------------------------------------------------------

SAMPLE_COVER_TEMPLATE = CoverTemplate(
    id="sample_generic_transparent",
    name="Sample Transparent Cover",
    image_filename=SAMPLE_COVER_FILENAME,
    canvas_size=(1600, 3200),
    # Outer silhouette including bumper / rim.
    outer_case=NormalizedRect(x=0.078, y=0.038, w=0.844, h=0.924, radius=0.122),
    # Flat back panel only — inset from the bumper on every side.
    printable_area=NormalizedRect(x=0.128, y=0.086, w=0.744, h=0.828, radius=0.078),
    # Camera island + protection area (entire module, not just the holes).
    camera_exclusion=NormalizedRect(x=0.148, y=0.104, w=0.292, h=0.128, radius=0.058),
    camera_holes=(
        NormalizedCircle(cx=0.230, cy=0.168, r=0.046),
        NormalizedCircle(cx=0.358, cy=0.168, r=0.038),
    ),
    edge_margin_px=14.0,
    camera_safety_margin_px=10.0,
    mask_feather_px=1.5,
)

_TEMPLATES: dict[str, CoverTemplate] = {
    SAMPLE_COVER_TEMPLATE.id: SAMPLE_COVER_TEMPLATE,
}


def get_default_template() -> CoverTemplate:
    return SAMPLE_COVER_TEMPLATE


def get_template(template_id: str) -> CoverTemplate:
    try:
        return _TEMPLATES[template_id]
    except KeyError as exc:
        known = ", ".join(_TEMPLATES)
        raise KeyError(f"Unknown cover template '{template_id}'. Known: {known}") from exc


def list_templates() -> list[CoverTemplate]:
    return list(_TEMPLATES.values())
