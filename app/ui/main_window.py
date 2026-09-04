"""Main application window — UI and user interaction only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


from app.core.cover_adjust import CoverLook
from app.core.cover_processor import CoverProcessor, ProcessingResult
from app.ui.image_preview import DebugThumb, ImagePreview
from app.utils.constants import APP_NAME, APP_VERSION, DEBUG_MODE, ICONS_DIR, OUTPUT_DIR, SUPPORTED_INPUT_FILTER
from app.utils.image_utils import CoverError


DEBUG_VIEWS = (
    ("rim_validation", "1. Rim & Boundary Validation Multi-Overlay"),
    ("outer_cover_contour", "2. Outer Cover Contour (Red)"),
    ("detected_rim", "3. Detected Physical Rim (Amber)"),
    ("printable_boundary", "4. Final Printable Boundary (Green)"),
    ("camera_mask", "5. Camera Mask (Cyan/Yellow)"),
    ("final_artwork_mask", "6. Final Artwork Mask"),
    ("raw_contour", "7. Raw detected contour"),
    ("cleaned_contour", "8. Cleaned contour"),
    ("camera_rim_debug", "9. Camera Rim Debug"),
    ("camera_rim_contour", "10. Camera Rim Contour Overlay"),
    ("camera_openings", "11. Internal Lenses / Flash"),
    ("final_composite", "12. Final composite"),
    ("edge_exclusion", "13. Side wall / edge"),
    ("artwork_before_mask", "14. Artwork before mask"),
)



class CoverThread(QThread):
    finished_detection = Signal(object)
    failed = Signal(str)

    def __init__(self, processor: CoverProcessor, cover_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._processor = processor
        self._cover_path = cover_path

    def run(self) -> None:
        try:
            detection = self._processor.load_cover(self._cover_path)
            self.finished_detection.emit(detection)
        except CoverError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Something went wrong while reading the cover.\n{exc}")


class DesignThread(QThread):
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, processor: CoverProcessor, design_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._processor = processor
        self._design_path = design_path

    def run(self) -> None:
        try:
            result = self._processor.process_design(self._design_path)
            self.finished_result.emit(result)
        except CoverError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Something went wrong while processing the design.\n{exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  ·  v{APP_VERSION}")
        self.resize(1500, 920)
        self.setMinimumSize(1100, 700)

        icon_path = ICONS_DIR / "app_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.processor = CoverProcessor()
        self._result: ProcessingResult | None = None
        self._pending_design: str | None = None
        self._auto_process_design = False
        self._worker_thread: QThread | None = None
        self._worker: QObject | None = None
        self._busy = False
        self._final_key = "final_composite"
        self._look_timer = QTimer(self)
        self._look_timer.setSingleShot(True)
        self._look_timer.setInterval(30)
        self._look_timer.timeout.connect(self._apply_look)
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        self._tool_buttons: dict[str, QPushButton] = {}
        self._job_id = 0
        self._mask_edit = False

        self._build_ui()
        self._apply_style()
        self._set_status("Upload a transparent phone cover.")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 8)
        root_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.final_preview = ImagePreview("Final mockup", zoomable=True)
        self.final_preview.shape_selected.connect(self._on_shape_selected)
        self.final_preview.shape_updated.connect(self._on_shape_updated)
        final = self._mockup_panel()
        right = self._build_controls()

        splitter.addWidget(final)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1100, 320])

        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self._build_thumbs())

        header = QLabel("DEBUG MODE (Press F12 to toggle) — click a view to inspect it in the final preview")
        header.setObjectName("debugHeader")
        self._debug_wrap = QWidget()
        wrap_layout = QVBoxLayout(self._debug_wrap)
        wrap_layout.setContentsMargins(0, 0, 0, 0)
        wrap_layout.setSpacing(6)
        wrap_layout.addWidget(header)

        thumbs_host = QWidget()
        thumbs = QHBoxLayout(thumbs_host)
        thumbs.setContentsMargins(0, 0, 0, 0)
        thumbs.setSpacing(8)
        self._debug_thumbs: dict[str, DebugThumb] = {}
        for key, caption in DEBUG_VIEWS:
            thumb = DebugThumb(key, caption)
            thumb.clicked.connect(self._on_debug_clicked)
            self._debug_thumbs[key] = thumb
            thumbs.addWidget(thumb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(thumbs_host)
        scroll.setMinimumHeight(168)
        wrap_layout.addWidget(scroll)
        root_layout.addWidget(self._debug_wrap)

        self._debug_active = bool(DEBUG_MODE)
        self._debug_wrap.setVisible(self._debug_active)
        self._f12_shortcut = QShortcut(QKeySequence("F12"), self)
        self._f12_shortcut.activated.connect(self._toggle_debug_mode)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _mockup_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Final mockup")
        title.setObjectName("panelTitle")
        header.addWidget(title, 1)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setObjectName("zoomPct")
        header.addWidget(self._zoom_label)
        layout.addLayout(header)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(6)
        btn_in = QPushButton("Zoom in")
        btn_out = QPushButton("Zoom out")
        btn_fit = QPushButton("Fit to screen")
        btn_11 = QPushButton("1:1")
        for btn in (btn_in, btn_out, btn_fit, btn_11):
            btn.setObjectName("zoomBtn")
            zoom_row.addWidget(btn)
        zoom_row.addStretch(1)
        btn_in.clicked.connect(self.final_preview.zoom_in)
        btn_out.clicked.connect(self.final_preview.zoom_out)
        btn_fit.clicked.connect(self.final_preview.fit_to_screen)
        btn_11.clicked.connect(self.final_preview.reset_zoom)
        self.final_preview.zoom_changed.connect(self._on_final_zoom)
        layout.addLayout(zoom_row)
        layout.addWidget(self.final_preview, 1)
        return frame

    def _build_thumbs(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)
        self.cover_preview = ImagePreview("Cover")
        self.cover_preview.setMinimumSize(72, 96)
        self.cover_preview.setMaximumHeight(128)
        self.design_preview = ImagePreview("Design")
        self.design_preview.setMinimumSize(72, 96)
        self.design_preview.setMaximumHeight(128)
        layout.addWidget(self._thumb_block("Cover", self.cover_preview), 1)
        layout.addWidget(self._thumb_block("Design", self.design_preview), 1)
        return frame

    def _thumb_block(self, title: str, widget: QWidget) -> QWidget:
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("muted")
        col.addWidget(label)
        col.addWidget(widget, 1)
        return box

    def _build_controls(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Tools")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.btn_cover = QPushButton("Upload Cover")
        self.btn_cover.setObjectName("primary")
        self.btn_cover.clicked.connect(self._on_upload_cover)
        layout.addWidget(self.btn_cover)

        self.btn_design = QPushButton("Upload Design")
        self.btn_design.clicked.connect(self._on_upload_design)
        layout.addWidget(self.btn_design)

        fit_row = QHBoxLayout()
        fit_row.setSpacing(6)
        fit_lbl = QLabel("Fit mode:")
        fit_lbl.setObjectName("muted")
        self.combo_fit = QComboBox()
        self.combo_fit.addItems(["Cover (Fill)", "Contain (Fit)", "Center"])
        self.combo_fit.currentIndexChanged.connect(self._on_fit_mode_changed)
        fit_row.addWidget(fit_lbl)
        fit_row.addWidget(self.combo_fit, 1)
        layout.addLayout(fit_row)

        layout.addWidget(self._camera_group())
        layout.addWidget(self._adjust_group())

        export = QGroupBox("Export")
        export.setObjectName("group")
        ex = QVBoxLayout(export)
        ex.setSpacing(6)
        self.btn_png = QPushButton("Export PNG")
        self.btn_png.clicked.connect(lambda: self._on_export("png"))
        self.btn_jpg = QPushButton("Export JPG")
        self.btn_jpg.clicked.connect(lambda: self._on_export("jpg"))
        ex.addWidget(self.btn_png)
        ex.addWidget(self.btn_jpg)
        layout.addWidget(export)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        layout.addWidget(self.btn_reset)
        layout.addStretch(1)

        footer = QLabel("Offline  ·  local processing only")
        footer.setObjectName("muted")
        layout.addWidget(footer)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self._set_export_enabled(False)
        frame.setMinimumWidth(280)
        frame.setMaximumWidth(360)
        return frame

    def _camera_group(self) -> QGroupBox:
        box = QGroupBox("Camera mask")
        box.setObjectName("group")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        hint = QLabel(
            "Wrap hides while you edit. Select/move shapes, drag handles, or draw a new one. "
            "Wheel zooms · space-drag pans · delete key removes. Apply when ready."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        layout.addWidget(hint)

        grid = QGridLayout()
        grid.setSpacing(4)
        tools = (
            ("move", "Move / Select"),
            ("rect", "Rectangle"),
            ("roundrect", "Rounded"),
            ("circle", "Circle"),
            ("ellipse", "Ellipse"),
            ("pill", "Capsule / Pill"),
            ("polygon", "Polygon"),
            ("freeform", "Freeform"),
            ("brush", "Brush"),
            ("eraser", "Eraser"),
        )
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for i, (key, label) in enumerate(tools):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setObjectName("toolBtn")
            btn.clicked.connect(lambda _=False, k=key: self._on_tool(k))
            self._tool_group.addButton(btn)
            self._tool_buttons[key] = btn
            grid.addWidget(btn, i // 2, i % 2)
        layout.addLayout(grid)

        # Selected Shape Transform & Properties Panel
        self._shape_prop_group = QGroupBox("Selected shape controls")
        self._shape_prop_group.setObjectName("subGroup")
        sp_layout = QVBoxLayout(self._shape_prop_group)
        sp_layout.setContentsMargins(6, 6, 6, 6)
        sp_layout.setSpacing(4)

        grid_sp = QGridLayout()
        grid_sp.setSpacing(4)

        grid_sp.addWidget(QLabel("X:"), 0, 0)
        self.spin_x = QSpinBox()
        self.spin_x.setRange(-9999, 9999)
        grid_sp.addWidget(self.spin_x, 0, 1)

        grid_sp.addWidget(QLabel("Y:"), 0, 2)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(-9999, 9999)
        grid_sp.addWidget(self.spin_y, 0, 3)

        grid_sp.addWidget(QLabel("W:"), 1, 0)
        self.spin_w = QSpinBox()
        self.spin_w.setRange(2, 9999)
        grid_sp.addWidget(self.spin_w, 1, 1)

        grid_sp.addWidget(QLabel("H:"), 1, 2)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(2, 9999)
        grid_sp.addWidget(self.spin_h, 1, 3)

        grid_sp.addWidget(QLabel("Rot:"), 2, 0)
        self.spin_rot = QDoubleSpinBox()
        self.spin_rot.setRange(-180.0, 180.0)
        self.spin_rot.setSuffix("°")
        grid_sp.addWidget(self.spin_rot, 2, 1)

        grid_sp.addWidget(QLabel("Radius:"), 2, 2)
        self.spin_rad = QSpinBox()
        self.spin_rad.setRange(0, 999)
        self.spin_rad.setSuffix("px")
        grid_sp.addWidget(self.spin_rad, 2, 3)

        sp_layout.addLayout(grid_sp)

        self.chk_aspect = QCheckBox("Lock Aspect Ratio")
        sp_layout.addWidget(self.chk_aspect)

        btn_row = QHBoxLayout()
        self.btn_reset_transform = QPushButton("Reset Transform")
        self.btn_reset_transform.setObjectName("zoomBtn")
        self.btn_delete_shape = QPushButton("Delete Shape")
        self.btn_delete_shape.setObjectName("zoomBtn")
        btn_row.addWidget(self.btn_reset_transform)
        btn_row.addWidget(self.btn_delete_shape)
        sp_layout.addLayout(btn_row)

        layout.addWidget(self._shape_prop_group)
        self._shape_prop_group.setEnabled(False)

        # Wire spinbox events
        self.spin_x.valueChanged.connect(lambda v: self.final_preview.update_selected_shape(x=v))
        self.spin_y.valueChanged.connect(lambda v: self.final_preview.update_selected_shape(y=v))
        self.spin_w.valueChanged.connect(lambda v: self.final_preview.update_selected_shape(width=v))
        self.spin_h.valueChanged.connect(lambda v: self.final_preview.update_selected_shape(height=v))
        self.spin_rot.valueChanged.connect(lambda v: self.final_preview.update_selected_shape(rotation=v))
        self.spin_rad.valueChanged.connect(lambda v: self.final_preview.update_selected_shape(corner_radius=v))
        self.chk_aspect.toggled.connect(lambda v: self.final_preview.update_selected_shape(lock_aspect=v))
        self.btn_reset_transform.clicked.connect(self.final_preview.reset_selected_transform)
        self.btn_delete_shape.clicked.connect(self.final_preview.delete_selected_shape)

        brush_row = QHBoxLayout()
        brush_lab = QLabel("Brush")
        brush_lab.setObjectName("muted")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 80)
        self.brush_slider.setValue(14)
        self.brush_slider.valueChanged.connect(self._on_brush)
        brush_row.addWidget(brush_lab)
        brush_row.addWidget(self.brush_slider, 1)
        layout.addLayout(brush_row)

        row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_clear_mask = QPushButton("Clear")
        self.btn_undo.clicked.connect(self.final_preview.undo_mask)
        self.btn_redo.clicked.connect(self.final_preview.redo_mask)
        self.btn_clear_mask.clicked.connect(self.final_preview.clear_mask)
        for b in (self.btn_undo, self.btn_redo, self.btn_clear_mask):
            b.setObjectName("zoomBtn")
            row.addWidget(b)
        layout.addLayout(row)

        self.btn_camera = QPushButton("Apply mask")
        self.btn_camera.setEnabled(False)
        self.btn_camera.clicked.connect(self._on_apply_camera_mask)
        layout.addWidget(self.btn_camera)
        return box


    def _adjust_group(self) -> QGroupBox:
        box = QGroupBox("Cover adjustments")
        box.setObjectName("group")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        note = QLabel("Looks apply to the cover mockup only — original artwork is unchanged.")
        note.setWordWrap(True)
        note.setObjectName("hint")
        layout.addWidget(note)
        specs = (
            ("brightness", "Brightness", -100, 100, 0),
            ("contrast", "Contrast", -100, 100, 0),
            ("sharpness", "Sharpness", 0, 100, 0),
            ("clarity", "Clarity", 0, 100, 0),
            ("saturation", "Saturation", -100, 100, 0),
            ("exposure", "Exposure", -200, 200, 0),
            ("highlights", "Highlights", -100, 100, 0),
            ("shadows", "Shadows", -100, 100, 0),
            ("opacity", "Opacity", 0, 100, 100),
        )
        for key, label, lo, hi, default in specs:
            layout.addLayout(self._slider_row(key, label, lo, hi, default))
        reset = QPushButton("Reset adjustments")
        reset.setObjectName("zoomBtn")
        reset.clicked.connect(self._reset_look)
        layout.addWidget(reset)
        return box

    def _slider_row(self, key: str, label: str, lo: int, hi: int, default: int) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        name = QLabel(label)
        name.setMinimumWidth(78)
        name.setObjectName("muted")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        slider.setProperty("default", default)
        value = QLabel(str(default))
        value.setObjectName("zoomPct")
        value.setMinimumWidth(36)
        slider.valueChanged.connect(lambda v, lab=value: lab.setText(str(v)))
        slider.valueChanged.connect(lambda _v: self._look_timer.start())
        self._sliders[key] = slider
        self._slider_labels[key] = value
        row.addWidget(name)
        row.addWidget(slider, 1)
        row.addWidget(value)
        return row

    def _on_tool(self, key: str) -> None:
        if self.processor.cover is None:
            self._show_error("Upload a phone cover first.")
            btn = self._tool_buttons.get(key)
            if btn:
                btn.setChecked(False)
            return
        self.final_preview.setFocus()
        self._enter_mask_edit()
        self.final_preview.set_tool(key)
        if key == "move":
            self._set_status("Wrap hidden. Drag the camera mask, or use arrow keys. Apply when it lines up.")
        else:
            self._set_status("Wrap hidden. Draw the camera cutout, then Apply mask.")

    def _on_brush(self, value: int) -> None:
        self.final_preview.brush_radius = float(value)

    def _look_from_sliders(self) -> CoverLook:
        s = self._sliders
        return CoverLook(
            brightness=s["brightness"].value() / 100.0,
            contrast=s["contrast"].value() / 100.0,
            sharpness=s["sharpness"].value() / 100.0,
            clarity=s["clarity"].value() / 100.0,
            saturation=s["saturation"].value() / 100.0,
            exposure=s["exposure"].value() / 100.0,
            highlights=s["highlights"].value() / 100.0,
            shadows=s["shadows"].value() / 100.0,
            opacity=s["opacity"].value() / 100.0,
        )

    def _apply_look(self) -> None:
        if self.processor.cover is None:
            return
        self.processor.set_look(self._look_from_sliders())
        if self.processor.last_result is not None:
            self._result = self.processor.last_result
        if self._mask_edit:
            self._show_bare_cover()
        else:
            self._refresh_previews(keep_mask=True)

    def _reset_look(self) -> None:
        self._look_timer.stop()
        for key, slider in self._sliders.items():
            default = int(slider.property("default"))
            slider.blockSignals(True)
            slider.setValue(default)
            slider.blockSignals(False)
            label = self._slider_labels.get(key)
            if label is not None:
                label.setText(str(default))
        self.processor.set_look(CoverLook())
        if self.processor.last_result is not None:
            self._result = self.processor.last_result
        if self.processor.cover is not None:
            self._refresh_previews(keep_mask=True)

    def _full_reset(self) -> None:
        """Clear session, tools, adjustments, and previews so a new job starts clean."""
        self._job_id += 1
        self._look_timer.stop()
        self._busy = False
        self.processor.reset()
        self._result = None
        self._pending_design = None
        self._auto_process_design = False
        self._final_key = "final_composite"
        self._mask_edit = False

        self.cover_preview.reset_session()
        self.design_preview.reset_session()
        self.final_preview.reset_session()
        self.cover_preview.set_placeholder("Cover")
        self.design_preview.set_placeholder("Design")
        self.final_preview.set_placeholder("Final mockup")

        self._tool_group.setExclusive(False)
        for btn in self._tool_buttons.values():
            btn.setChecked(False)
        self._tool_group.setExclusive(True)

        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(14)
        self.brush_slider.blockSignals(False)
        self.final_preview.brush_radius = 14.0

        if hasattr(self, "combo_fit"):
            self.combo_fit.blockSignals(True)
            self.combo_fit.setCurrentIndex(0)
            self.combo_fit.blockSignals(False)

        self._reset_look()
        if hasattr(self, "_zoom_label"):
            self._zoom_label.setText("100%")
        self.btn_camera.setEnabled(False)
        self._set_buttons_enabled(True)
        self._set_export_enabled(False)
        if DEBUG_MODE:
            for thumb in self._debug_thumbs.values():
                thumb.preview.reset_session()
        self._set_status("Upload a transparent phone cover.")

    def _on_upload_cover(self) -> None:
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a phone cover image", "", SUPPORTED_INPUT_FILTER)
        if not path:
            return
        self._start_cover_load(path)

    def _on_fit_mode_changed(self, index: int) -> None:
        modes = ["cover", "contain", "center"]
        if 0 <= index < len(modes):
            mode = modes[index]
            res = self.processor.set_fit_mode(mode)
            if res is not None:
                self._result = res
                self._refresh_previews(keep_mask=True)

    def _on_upload_design(self) -> None:
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a design image", "", SUPPORTED_INPUT_FILTER)
        if not path:
            return
        self._pending_design = path
        self._auto_process_design = self.processor.cover is not None
        design = self._try_preview_design(path)
        if design is not None:
            self.design_preview.set_image(design)
        if self.processor.cover is None:
            self._set_status("Upload your design. Upload a phone cover next.")
            return
        self._start_design_process(path)

    def _try_preview_design(self, path: str) -> np.ndarray | None:
        try:
            from app.utils.image_utils import load_image_rgba

            return load_image_rgba(path)
        except CoverError:
            return None

    def _start_cover_load(self, path: str) -> None:
        self._job_id += 1
        job = self._job_id
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status("Processing…")

        thread = CoverThread(self.processor, path, self)
        thread.finished_detection.connect(lambda detection, j=job: self._on_cover_loaded(detection, j))
        thread.failed.connect(lambda message, j=job: self._on_process_failed(message, j))
        thread.finished.connect(lambda j=job: self._clear_worker(j))
        self._worker_thread = thread
        self._worker = thread
        thread.start()

    def _start_design_process(self, path: str) -> None:
        self._job_id += 1
        job = self._job_id
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status("Processing…")

        thread = DesignThread(self.processor, path, self)
        thread.finished_result.connect(lambda result, j=job: self._on_process_finished(result, j))
        thread.failed.connect(lambda message, j=job: self._on_process_failed(message, j))
        thread.finished.connect(lambda j=job: self._clear_worker(j))
        self._worker_thread = thread
        self._worker = thread
        thread.start()

    def _clear_worker(self, job: int | None = None) -> None:
        if job is not None and job != self._job_id:
            return
        self._worker_thread = None
        self._worker = None
        self._busy = False
        self._set_buttons_enabled(True)
        if (
            self._auto_process_design
            and self._pending_design
            and self.processor.cover is not None
            and self.processor.last_result is None
        ):
            self._auto_process_design = False
            self._start_design_process(self._pending_design)

    def _on_apply_camera_mask(self) -> None:
        if self.processor.cover is None:
            self._show_error("Upload a phone cover first.")
            return
        mask = self.final_preview.camera_mask()
        if mask is None:
            self._show_error("Draw a camera shape on the mockup first.")
            return
        try:
            detection = self.processor.set_manual_camera_mask(mask)
        except CoverError as exc:
            self._show_error(str(exc))
            return
        if self.processor.last_result is not None:
            self._result = self.processor.last_result
            self._set_export_enabled(self.processor.design is not None)
        else:
            self._result = self.processor.cover_preview_result()
        self._final_key = "final_composite"
        self._exit_mask_edit()
        self._refresh_previews(keep_mask=True)
        self._set_status("Camera mask applied." if detection.camera_found else "Camera mask cleared.")

    def _on_cover_loaded(self, detection, job: int | None = None) -> None:  # noqa: ANN001
        if job is not None and job != self._job_id:
            return
        self._exit_mask_edit()
        self.cover_preview.set_image(self.processor.cover)
        self._result = self.processor.cover_preview_result()
        self._final_key = "final_composite"
        if self.processor.masks is not None:
            self.final_preview.set_camera_mask(self.processor.masks.camera_exclusion)
        self._refresh_previews(keep_mask=True)
        self._set_export_enabled(False)
        self.btn_camera.setEnabled(True)

        serious = [w for w in detection.warnings if "unusually small" in w or "confidence is" in w]
        if serious:
            self._show_warning("\n".join(serious))

        if self._pending_design:
            self._auto_process_design = True
            self._set_status("Processing…")
        else:
            self._set_status("Upload your design. Optional: draw a camera mask if the cutout is wrong.")

    def _on_process_finished(self, result: ProcessingResult, job: int | None = None) -> None:
        if job is not None and job != self._job_id:
            return
        self._result = result
        self._final_key = "final_composite"
        self._set_export_enabled(True)
        self._refresh_previews(keep_mask=True)
        serious = [w for w in result.warnings if "unusually small" in w or "confidence is" in w]
        if serious:
            self._show_warning("\n".join(serious))
        self._set_status("Ready to export.")

    def _on_process_failed(self, message: str, job: int | None = None) -> None:
        if job is not None and job != self._job_id:
            return
        self._show_error(message)
        self._set_status("Processing failed. Please try another image.")

    def _on_reset(self) -> None:
        self._full_reset()

    def _on_export(self, fmt: str) -> None:
        if self.processor.design is None:
            self._show_error("Upload a cover and a design before exporting.")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if fmt == "png":
            selected, _ = QFileDialog.getSaveFileName(
                self, "Export PNG", str(OUTPUT_DIR / "cover_mockup.png"), "PNG (*.png)"
            )
        else:
            selected, _ = QFileDialog.getSaveFileName(
                self, "Export JPG", str(OUTPUT_DIR / "cover_mockup.jpg"), "JPEG (*.jpg *.jpeg)"
            )
        if not selected:
            return
        try:
            path = self.processor.export(fmt, selected)
        except CoverError as exc:
            self._show_error(str(exc))
            return
        self._set_status(f"Saved {path.name}")

    def _on_debug_clicked(self, key: str) -> None:
        self._final_key = key
        self._refresh_previews(keep_mask=True)
        label = dict(DEBUG_VIEWS).get(key, key)
        self._set_status(f"Debug view: {label}")

    def _on_tool(self, tool: str) -> None:
        if self.final_preview.tool() == tool:
            self.final_preview.set_tool(None)
            self._tool_group.setExclusive(False)
            if tool in self._tool_buttons:
                self._tool_buttons[tool].setChecked(False)
            self._tool_group.setExclusive(True)
            self._exit_mask_edit()
            self._refresh_previews(keep_mask=True)
            self._set_status("Camera editing closed.")
            return

        self._enter_mask_edit()
        self.final_preview.set_tool(tool)
        if tool in self._tool_buttons:
            self._tool_buttons[tool].setChecked(True)
        if tool == "move":
            self._set_status("Select/Move: Drag shapes, resize handles, or rotate top handle. Delete key removes shape.")
        elif tool in ("rect", "roundrect", "circle", "ellipse", "pill"):
            self._set_status(f"{tool.capitalize()}: Drag on the camera area to create a shape.")
        elif tool == "polygon":
            self._set_status("Polygon: Click to add vertices. Double-click or Enter to finish.")
        elif tool == "freeform":
            self._set_status("Freeform: Drag to draw custom outline.")
        elif tool == "brush":
            self._set_status("Brush: Paint mask area.")
        elif tool == "eraser":
            self._set_status("Eraser: Erase mask area.")

    def _on_brush(self, value: int) -> None:
        self.final_preview.brush_radius = float(value)

    def _on_shape_selected(self, shape_dict: dict | None) -> None:
        if shape_dict is None:
            self._shape_prop_group.setEnabled(False)
            return
        self._shape_prop_group.setEnabled(True)
        self._sync_shape_properties(shape_dict)

    def _on_shape_updated(self, shape_dict: dict | None) -> None:
        if shape_dict is not None:
            self._sync_shape_properties(shape_dict)

    def _sync_shape_properties(self, d: dict) -> None:
        widgets = (self.spin_x, self.spin_y, self.spin_w, self.spin_h, self.spin_rot, self.spin_rad, self.chk_aspect)
        for w in widgets:
            w.blockSignals(True)

        self.spin_x.setValue(int(round(d.get("x", 0))))
        self.spin_y.setValue(int(round(d.get("y", 0))))
        self.spin_w.setValue(int(round(d.get("width", 10))))
        self.spin_h.setValue(int(round(d.get("height", 10))))
        self.spin_rot.setValue(float(d.get("rotation", 0.0)))
        self.spin_rad.setValue(int(round(d.get("corner_radius", 0))))
        self.chk_aspect.setChecked(bool(d.get("lock_aspect", False)))

        st = d.get("shape_type", "")
        self.spin_rad.setEnabled(st in ("roundrect", "rect", "pill"))

        for w in widgets:
            w.blockSignals(False)

    def _enter_mask_edit(self) -> None:
        if self.processor.cover is None:
            return
        current = self.final_preview.camera_mask()
        if current is None or not np.any(current > 0.12):
            if self.processor.masks is not None:
                self.final_preview.set_camera_mask(self.processor.masks.camera_exclusion)
        self._mask_edit = True
        self._show_bare_cover()


    def _show_bare_cover(self) -> None:
        bare = self.processor.preview_bare_cover()
        if bare is not None:
            self.final_preview.set_image(bare)

    def _exit_mask_edit(self) -> None:
        self._mask_edit = False
        self.final_preview.set_tool(None)
        self._tool_group.setExclusive(False)
        for btn in self._tool_buttons.values():
            btn.setChecked(False)
        self._tool_group.setExclusive(True)

    def _refresh_previews(self, *, keep_mask: bool = False) -> None:
        if self.processor.cover is not None:
            self.cover_preview.set_image(self.processor.cover)
        if self.processor.design is not None:
            self.design_preview.set_image(self.processor.design)
        if self._mask_edit:
            self._show_bare_cover()
        else:
            preview = self.processor.preview_composite()
            if preview is not None and self._final_key == "final_composite":
                self.final_preview.set_image(preview)
            elif self._result is not None:
                debug = self._result.debug_images
                self.final_preview.set_image(debug.get(self._final_key, self._result.composite))
        if not keep_mask and self.processor.masks is not None:
            self.final_preview.set_camera_mask(self.processor.masks.camera_exclusion)
        if self._debug_active and self._result is not None:
            for key, thumb in self._debug_thumbs.items():
                thumb.preview.set_image(self._result.debug_images.get(key))

    def _toggle_debug_mode(self) -> None:
        self._debug_active = not self._debug_active
        if hasattr(self, "_debug_wrap"):
            self._debug_wrap.setVisible(self._debug_active)
            if self._debug_active:
                self._refresh_previews()

    def _set_export_enabled(self, enabled: bool) -> None:
        self.btn_png.setEnabled(enabled)
        self.btn_jpg.setEnabled(enabled)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.btn_cover.setEnabled(enabled)
        self.btn_design.setEnabled(enabled)
        self.btn_reset.setEnabled(True)
        self.btn_camera.setEnabled(enabled and self.processor.cover is not None)
        if enabled:
            self._set_export_enabled(self.processor.design is not None)
        else:
            self._set_export_enabled(False)

    def _on_final_zoom(self, percent: int) -> None:
        if hasattr(self, "_zoom_label"):
            self._zoom_label.setText(f"{percent}%")

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, APP_NAME, message)

    def _show_warning(self, message: str) -> None:
        QMessageBox.information(self, APP_NAME, message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #14171c;
                color: #e8eaed;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 13px;
            }
            QFrame#panel {
                background: #1e232b;
                border: 1px solid #2c333e;
                border-radius: 12px;
            }
            QGroupBox#group {
                background: transparent;
                border: 1px solid #2c333e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
                color: #f2f4f7;
            }
            QGroupBox#group::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel#panelTitle {
                font-size: 15px;
                font-weight: 600;
                color: #f2f4f7;
                letter-spacing: 0.2px;
            }
            QLabel#hint { color: #9aa3b2; font-size: 11px; }
            QLabel#muted { color: #6b7380; font-size: 11px; }
            QLabel#debugHeader { color: #8fb8ff; font-size: 12px; font-weight: 600; }
            QLabel#debugCaption { color: #9aa3b2; font-size: 11px; }
            QFrame#debugThumb {
                background: #1e232b;
                border: 1px solid #2c333e;
                border-radius: 8px;
            }
            QFrame#debugThumb:hover { border-color: #3d9cf0; }
            QLabel#zoomPct {
                color: #9aa3b2;
                font-size: 12px;
                font-weight: 600;
                min-width: 36px;
            }
            QPushButton#zoomBtn, QPushButton#toolBtn {
                padding: 5px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#toolBtn:checked {
                background: #2f7de1;
                border-color: #3d9cf0;
                color: #ffffff;
            }
            QPushButton {
                background: #2a313c;
                color: #e8eaed;
                border: 1px solid #3a4352;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #343c49; }
            QPushButton:pressed { background: #232a33; }
            QPushButton:disabled { color: #6b7380; background: #222830; }
            QPushButton#primary {
                background: #2f7de1;
                border-color: #3d9cf0;
                color: #ffffff;
            }
            QPushButton#primary:hover { background: #3d8eef; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2c333e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                height: 12px;
                margin: -5px 0;
                background: #3d9cf0;
                border-radius: 6px;
            }
            QGroupBox#subGroup {
                background: #181d24;
                border: 1px solid #2a3340;
                border-radius: 6px;
                margin-top: 8px;
                padding: 6px;
                font-size: 11px;
                font-weight: 600;
                color: #8fb8ff;
            }
            QGroupBox#subGroup::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QSpinBox, QDoubleSpinBox {
                background: #1e242d;
                color: #e8eaed;
                border: 1px solid #343f50;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #00d2ff;
            }
            QCheckBox {
                color: #c4c9d4;
                font-size: 11px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3a475a;
                border-radius: 3px;
                background: #1e242d;
            }
            QCheckBox::indicator:checked {
                background: #2f7de1;
                border-color: #00d2ff;
            }
            QStatusBar {
                background: #101317;
                color: #9aa3b2;
                border-top: 1px solid #2c333e;
            }
            QSplitter::handle { background: #14171c; width: 8px; }
            QMessageBox { background: #1e232b; }
            QScrollArea { background: transparent; }
            """
        )

