"""Transparent Cover Mockup — desktop entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transparent Cover Mockup")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a headless pipeline check and exit (no window).",
    )
    parser.add_argument(
        "--gui-check",
        action="store_true",
        help="Create the main window, verify it loads, then exit.",
    )
    return parser.parse_args(argv)


def run_smoke_test() -> int:
    """Verify detection, printable-area masking, color fidelity, and export."""
    import cv2
    import numpy as np

    from app.core.cover_processor import CoverProcessor
    from app.utils.constants import COVERS_DIR, EXAMPLES_DIR, OUTPUT_DIR, SAMPLE_COVER_FILENAME, SAMPLE_DESIGN_FILENAME
    from app.utils.sample_cover_builder import ensure_sample_assets

    print("Generating sample assets…")
    ensure_sample_assets()

    cover_path = COVERS_DIR / SAMPLE_COVER_FILENAME
    design_path = EXAMPLES_DIR / SAMPLE_DESIGN_FILENAME
    processor = CoverProcessor()

    print(f"Detecting cover: {cover_path.name}")
    detection = processor.load_cover(cover_path)
    cover = processor.cover
    assert cover is not None
    print(f"  cover shape: {cover.shape}")
    print(f"  detection confidence: {detection.confidence:.2f}")
    if detection.warnings:
        for warning in detection.warnings:
            print(f"  warning: {warning}")

    print(f"Processing design: {design_path.name}")
    result = processor.process_design(design_path)

    mask = result.masks.final_print
    camera = result.masks.camera_exclusion
    edge = result.masks.edge_exclusion

    untouched = mask < 0.02
    printed = mask > 0.95
    camera_core = camera > 0.92
    if processor.detection is not None and processor.detection.camera_bin.any():
        cam_bin = processor.detection.camera_bin.astype(np.uint8)
        dist_c = cv2.distanceTransform(cam_bin, cv2.DIST_L2, 5)
        camera_core = dist_c > 2.0
    # Solid bumper only — not the 1px inner-rim AA band (that is the print lip).
    back = processor.detection.back_full if processor.detection is not None else (mask > 0.5)
    outer = processor.detection.outer_bin if processor.detection is not None else (edge > 0.5)
    bumper_bin = outer.astype(bool) & ~back.astype(bool)
    if bumper_bin.any():
        dist_b = cv2.distanceTransform(bumper_bin.astype(np.uint8), cv2.DIST_L2, 5)
        bumper_core = dist_b > 2.5
    else:
        bumper_core = edge > 0.95

    if not printed.any():
        print("FAIL: printable mask is empty")
        return 1

    cover_i = cover.astype(np.int16)
    comp_i = result.composite.astype(np.int16)
    fitted = result.design_fitted
    diff_cover = np.abs(comp_i - cover_i).max(axis=2)

    leak_untouched = int(diff_cover[untouched].max()) if untouched.any() else 0
    leak_camera = int(diff_cover[camera_core].max()) if camera_core.any() else 0
    leak_edge = int(diff_cover[bumper_core].max()) if bumper_core.any() else 0
    printed_change = int(diff_cover[printed].max()) if printed.any() else 0

    print(f"  max delta outside print mask: {leak_untouched}")
    print(f"  max delta in camera exclusion: {leak_camera}")
    print(f"  max delta on bumper/edge: {leak_edge}")
    print(f"  max delta inside printable area: {printed_change}")

    failed = False
    if leak_untouched > 2:
        print("FAIL: artwork leaked outside the printable mask")
        failed = True
    if leak_camera > 2:
        print("FAIL: artwork entered the camera exclusion area")
        failed = True
    if leak_edge > 2:
        print("FAIL: artwork leaked onto the bumper / edge")
        failed = True
    if printed_change < 8:
        print("FAIL: design was not applied inside the printable area")
        failed = True

    # Color fidelity: interior print pixels must match the fitted artwork, not a gray haze.
    opaque = printed & (fitted[..., 3] > 250)
    if opaque.any():
        mae = float(np.mean(np.abs(comp_i[opaque, :3].astype(np.float32) - fitted[opaque, :3].astype(np.int16))))
        print(f"  mean RGB error vs artwork in print core: {mae:.2f}")
        if mae > 3.0:
            print("FAIL: artwork colors were washed out / altered in the printable area")
            failed = True
    else:
        print("FAIL: no opaque printable pixels to check color fidelity")
        failed = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = processor.export("png", OUTPUT_DIR / "smoke_test.png")
    jpg_path = processor.export("jpg", OUTPUT_DIR / "smoke_test.jpg")
    print(f"  exported PNG: {png_path}")
    print(f"  exported JPG: {jpg_path}")

    if failed:
        return 1

    print("Testing opaque JPG studio cover (no alpha)…")
    jpg_cover = COVERS_DIR / "studio_cover.jpg"
    from app.utils.sample_cover_builder import render_studio_jpg_cover

    render_studio_jpg_cover().save(jpg_cover, format="JPEG", quality=95)
    jpg_proc = CoverProcessor()
    jpg_det = jpg_proc.load_cover(jpg_cover)
    print(f"  JPG confidence: {jpg_det.confidence:.2f}")
    print(f"  JPG printable fraction: {float(jpg_det.masks.final_print.mean()):.3f}")
    print(f"  JPG outer fraction: {float(jpg_det.outer_bin.mean()):.3f}")
    print(f"  JPG camera found: {jpg_det.camera_found}")
    if float(jpg_det.masks.final_print.mean()) < 0.25:
        print("FAIL: JPG printable area is too small (MagSafe-sized mask)")
        return 1
    if float(jpg_det.outer_bin.mean()) < 0.30:
        print("FAIL: JPG outer silhouette is too small")
        return 1
    jpg_result = jpg_proc.process_design(design_path)
    printed_j = jpg_result.masks.final_print > 0.95
    if not printed_j.any():
        print("FAIL: JPG printable core empty")
        return 1
    mae_j = float(
        np.mean(
            np.abs(
                jpg_result.composite[printed_j, :3].astype(np.float32)
                - jpg_result.design_fitted[printed_j, :3].astype(np.float32)
            )
        )
    )
    print(f"  JPG mean RGB error vs artwork: {mae_j:.2f}")
    if mae_j > 3.0:
        print("FAIL: JPG artwork colors were washed out")
        return 1
    jpg_result_path = OUTPUT_DIR / "smoke_jpg_cover.png"
    jpg_proc.export("png", jpg_result_path)
    print(f"  exported JPG-cover mockup: {jpg_result_path}")

    print("SMOKE TEST PASSED")
    return 0


def run_gui_check() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow
    from app.utils.sample_cover_builder import ensure_sample_assets

    ensure_sample_assets()
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    QTimer.singleShot(1200, app.quit)
    app.exec()
    print("GUI CHECK PASSED")
    return 0


def run_app() -> int:
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow
    from app.utils.constants import APP_NAME
    from app.utils.sample_cover_builder import ensure_sample_assets

    ensure_sample_assets()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return run_smoke_test()
    if args.gui_check:
        return run_gui_check()
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
