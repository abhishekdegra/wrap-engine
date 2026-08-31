"""Application-wide constants.

Toggle DEBUG_MODE to inspect detection masks in the UI.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Transparent Cover Mockup"
APP_VERSION = "1.0.0"

# Set True to show mask debug previews in the main window.
DEBUG_MODE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
COVERS_DIR = ASSETS_DIR / "covers"
EXAMPLES_DIR = ASSETS_DIR / "examples"
ICONS_DIR = ASSETS_DIR / "icons"
OUTPUT_DIR = PROJECT_ROOT / "output"

SAMPLE_COVER_FILENAME = "sample_cover.png"
SAMPLE_DESIGN_FILENAME = "sample_design.png"

SUPPORTED_INPUT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
SUPPORTED_INPUT_FILTER = (
    "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*.*)"
)

# Guardrail against pathological inputs (pixels on the longest side).
MAX_IMAGE_DIMENSION = 8192

# JPG export quality (1–95+). PNG is lossless.
JPG_QUALITY = 95

# Optional frost over artwork. Default MUST stay 0 — any >0 washes colors out.
COVER_OVERLAY_STRENGTH = 0.0

# Side-wall / thickness exclusion (measured from the photo when possible).
SIDE_WALL_MAX_FRACTION = 0.055

# Tiny extra exclusion around the detected camera island.
CAMERA_SAFETY_FRACTION = 0.005
CAMERA_SAFETY_MIN_PX = 2.0
CAMERA_MIN_CONFIDENCE = 0.48
MASK_FEATHER_PX = 0.72
MASK_SUPERSAMPLE = 6

# Extra print inset after the back surface is found (keep ~0 — gaps come from this).
PRINT_MARGIN_FRACTION = 0.0
PRINT_MARGIN_MIN_PX = 0.0

# Below this, show a warning (still return masks if a silhouette exists).
DETECTION_CONFIDENCE_WARN = 0.55

# Interactive preview longest side (export always uses original resolution).
PREVIEW_MAX_SIDE = 1280
