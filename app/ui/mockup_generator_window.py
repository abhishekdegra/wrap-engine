"""Studio Mockup Generator Window.

Opens a dedicated, modern commercial studio window to generate and inspect
10 photorealistic tabletop mockup scenes using the currently wrapped phone cover.
Provides live multi-threaded generation, interactive zoom/pan inspector,
parameter fine-tuning, and individual / batch export.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.core.cover_processor import CoverProcessor, ProcessingResult
from app.core.mockup_engine import (
    SceneDescriptor,
    analyze_background_scene,
    composite_mockup_on_background,
    load_all_background_paths,
)
from app.core.mockup_extractor import ExtractedPhone, extract_phone_foreground
from app.ui.image_preview import ImagePreview
from app.utils.constants import APP_NAME, MOCKUP_BG_DIR, MOCKUP_OUTPUT_DIR
from app.utils.image_utils import CoverError


class MockupWorker(QThread):
    """Asynchronous background worker that generates all 10 mockups without UI freeze."""

    progress = Signal(int, int, str)  # (current, total, message)
    scene_ready = Signal(int, str, object, object)  # (index, title, comp_bgr, scene_desc)
    finished_all = Signal(list)  # list of (index, title, comp_bgr)

    def __init__(
        self,
        extracted_phone: ExtractedPhone,
        bg_paths: List[Path],
        user_scale: float = 1.0,
        user_rotation_offset: float = 0.0,
        shadow_intensity_mult: float = 1.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.extracted_phone = extracted_phone
        self.bg_paths = bg_paths
        self.user_scale = user_scale
        self.user_rotation_offset = user_rotation_offset
        self.shadow_intensity_mult = shadow_intensity_mult
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        results = []
        total = len(self.bg_paths)

        for idx, path in enumerate(self.bg_paths):
            if self._is_cancelled:
                break

            scene_name = path.stem.replace(".png", "").replace(".jpg", "")
            self.progress.emit(idx + 1, total, f"Rendering Scene {idx + 1} of {total}: {scene_name}…")

            bgr = cv2.imread(str(path))
            if bgr is None:
                continue

            bg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            scene = analyze_background_scene(bg_rgb, scene_hint=path.name)
            comp_rgb = composite_mockup_on_background(
                bg_image=bg_rgb,
                extracted_phone=self.extracted_phone,
                scene=scene,
                user_scale=self.user_scale,
                user_rotation_offset=self.user_rotation_offset,
                shadow_intensity_mult=self.shadow_intensity_mult,
            )

            display_title = f"{scene.scene_id}. {scene.scene_name}"
            results.append((idx, display_title, comp_rgb))
            self.scene_ready.emit(idx, display_title, comp_rgb, scene)

        if not self._is_cancelled:
            self.finished_all.emit(results)


class MockupCard(QFrame):
    """Individual card for a scene in the left scroll gallery."""

    clicked = Signal(int)
    quick_save = Signal(int)

    def __init__(self, index: int, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.index = index
        self.title = title
        self._is_selected = False
        self._comp_rgb: Optional[np.ndarray] = None
        self._scene_desc: Optional[SceneDescriptor] = None

        self.setObjectName("mockupCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header with Title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        # Thumbnail display
        self.thumb_label = QLabel()
        self.thumb_label.setObjectName("cardThumb")
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setFixedSize(180, 225)
        self.thumb_label.setStyleSheet("background: #12151a; border-radius: 6px; color: #5a6473;")
        self.thumb_label.setText("Rendering…")
        layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Actions row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_inspect = QPushButton("Inspect")
        self.btn_inspect.setObjectName("cardBtn")
        self.btn_inspect.clicked.connect(lambda: self.clicked.emit(self.index))

        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("cardBtnSecondary")
        self.btn_save.clicked.connect(lambda: self.quick_save.emit(self.index))
        self.btn_save.setEnabled(False)

        btn_row.addWidget(self.btn_inspect, 1)
        btn_row.addWidget(self.btn_save, 1)
        layout.addLayout(btn_row)

    def set_mockup_image(self, comp_rgb: np.ndarray, scene_desc: SceneDescriptor) -> None:
        self._comp_rgb = comp_rgb
        self._scene_desc = scene_desc
        self.btn_save.setEnabled(True)

        # comp_rgb is already pure RGB with true artwork colors
        th_h, th_w = 225, 180
        thumb_cv = cv2.resize(comp_rgb, (th_w, th_h), interpolation=cv2.INTER_AREA)

        qimg = QImage(thumb_cv.data, th_w, th_h, 3 * th_w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.thumb_label.setPixmap(pixmap)
        self.thumb_label.setText("")

    def set_selected(self, selected: bool) -> None:
        self._is_selected = selected
        if selected:
            self.setStyleSheet(
                """
                QFrame#mockupCard {
                    background: #252d3a;
                    border: 2px solid #3d9cf0;
                    border-radius: 10px;
                }
                """
            )
            self.btn_inspect.setStyleSheet("background: #2f7de1; color: #fff;")
        else:
            self.setStyleSheet(
                """
                QFrame#mockupCard {
                    background: #1a2028;
                    border: 1px solid #2e3745;
                    border-radius: 10px;
                }
                QFrame#mockupCard:hover {
                    background: #202732;
                    border-color: #4a576d;
                }
                """
            )
            self.btn_inspect.setStyleSheet("")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class MockupGeneratorWindow(QDialog):
    """Main studio window for commercial mockup generation and inspection."""

    def __init__(
        self,
        processor: CoverProcessor,
        result: Optional[ProcessingResult] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.processor = processor
        self.result = result or processor.last_result
        self.bg_paths: List[Path] = load_all_background_paths(MOCKUP_BG_DIR)
        self.cards: List[MockupCard] = []
        self.rendered_scenes: Dict[int, Tuple[str, np.ndarray, SceneDescriptor]] = {}
        self.selected_index: int = 0
        self.worker: Optional[MockupWorker] = None

        # Fine tuning parameters
        self._scale_val: float = 1.0
        self._rotation_val: float = 0.0
        self._nudge_x: int = 0
        self._nudge_y: int = 0
        self._shadow_val: float = 1.0

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(120)
        self._debounce_timer.timeout.connect(self._re_render_selected_scene)

        self._init_window()
        self._build_ui()
        self._apply_studio_style()

        # Start extraction & generation immediately
        QTimer.singleShot(50, self._start_initial_generation)

    def _init_window(self) -> None:
        self.setWindowTitle("✨ Commercial Phone Mockup Studio — 10 Product Scenes")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(1280, 850)
        self.setMinimumSize(1000, 700)

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # 1. Top Action Bar
        top_bar = self._build_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Main Content Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("studioSplitter")

        # Left Gallery Panel
        left_panel = self._build_gallery_panel()
        splitter.addWidget(left_panel)

        # Right Inspector Panel
        right_panel = self._build_inspector_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 900])
        main_layout.addWidget(splitter, 1)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(14)

        # Title & Subtitle block
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        h1 = QLabel("✨ Commercial Mockup Studio")
        h1.setObjectName("studioH1")
        sub = QLabel("10 Photorealistic tabletop commercial scenes for Shopify, catalogs & ads")
        sub.setObjectName("studioSub")
        title_col.addWidget(h1)
        title_col.addWidget(sub)
        layout.addLayout(title_col)

        layout.addSpacing(16)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("studioProgress")
        self.progress_bar.setRange(0, 10)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Initializing…")
        self.progress_bar.setMinimumWidth(240)
        self.progress_bar.setMaximumHeight(24)
        layout.addWidget(self.progress_bar, 1)

        layout.addSpacing(10)

        # Angle preset selector
        lbl_style = QLabel("Angle:")
        lbl_style.setObjectName("studioMuted")
        layout.addWidget(lbl_style)

        self.combo_preset = QComboBox()
        self.combo_preset.setObjectName("studioCombo")
        self.combo_preset.addItems([
            "Natural Tabletop (-11°)",
            "Lying Flat (0°)",
            "Casual Angle (-16°)",
            "Dynamic Right (+12°)",
        ])
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        layout.addWidget(self.combo_preset)

        # Buttons
        self.btn_regen = QPushButton("🔄 Re-generate All")
        self.btn_regen.setObjectName("studioBtnSecondary")
        self.btn_regen.clicked.connect(self._start_generation)
        layout.addWidget(self.btn_regen)

        self.btn_export_all = QPushButton("🚀 Export All (10 Scenes)")
        self.btn_export_all.setObjectName("studioBtnPrimary")
        self.btn_export_all.clicked.connect(self._on_export_all)
        self.btn_export_all.setEnabled(False)
        layout.addWidget(self.btn_export_all)

        return bar

    def _build_gallery_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        # Gallery Title Bar
        header = QHBoxLayout()
        h_title = QLabel("Scene Gallery (10)")
        h_title.setObjectName("sectionHeader")
        header.addWidget(h_title)
        header.addStretch(1)
        self.gallery_status = QLabel("0 / 10 Ready")
        self.gallery_status.setObjectName("studioMuted")
        header.addWidget(self.gallery_status)
        layout.addLayout(header)

        # Scroll Area for Cards
        scroll = QScrollArea()
        scroll.setObjectName("galleryScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        self.cards_layout = QVBoxLayout(scroll_widget)
        self.cards_layout.setContentsMargins(4, 4, 10, 4)
        self.cards_layout.setSpacing(10)

        # Populate empty mockup cards
        for idx, path in enumerate(self.bg_paths):
            scene_name = path.stem.replace(".png", "").replace(".jpg", "").replace("background_", "Scene ")
            card = MockupCard(index=idx, title=f"Scene {idx + 1:02d}: {scene_name}", parent=scroll_widget)
            card.clicked.connect(self._on_card_selected)
            card.quick_save.connect(self._on_quick_save)
            self.cards.append(card)
            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        container.setMinimumWidth(260)
        container.setMaximumWidth(360)
        return container

    def _build_inspector_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)

        # Inspector Header
        header = QHBoxLayout()
        header.setSpacing(8)

        self.inspector_title = QLabel("Scene 01: Morning Sunlit Oak Desk")
        self.inspector_title.setObjectName("sectionHeader")
        header.addWidget(self.inspector_title, 1)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("studioMuted")
        header.addWidget(self.zoom_label)

        btn_fit = QPushButton("Fit")
        btn_fit.setObjectName("zoomToolBtn")
        btn_fit.clicked.connect(lambda: self.inspector_preview.fit_to_screen())
        header.addWidget(btn_fit)

        btn_11 = QPushButton("1:1")
        btn_11.setObjectName("zoomToolBtn")
        btn_11.clicked.connect(lambda: self.inspector_preview.reset_zoom())
        header.addWidget(btn_11)

        btn_in = QPushButton("+")
        btn_in.setObjectName("zoomToolBtn")
        btn_in.clicked.connect(lambda: self.inspector_preview.zoom_in())
        header.addWidget(btn_in)

        btn_out = QPushButton("-")
        btn_out.setObjectName("zoomToolBtn")
        btn_out.clicked.connect(lambda: self.inspector_preview.zoom_out())
        header.addWidget(btn_out)

        layout.addLayout(header)

        # Large Interactive Zoom/Pan Image Preview
        self.inspector_preview = ImagePreview("Select a scene to inspect")
        self.inspector_preview.setObjectName("inspectorCanvas")
        self.inspector_preview.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z}%"))
        layout.addWidget(self.inspector_preview, 1)

        # Fine-Tuning Controls Drawer
        tuning_box = QGroupBox("Fine-Tune Scene Placement & Shadow")
        tuning_box.setObjectName("tuningGroup")
        t_layout = QGridLayout(tuning_box)
        t_layout.setContentsMargins(12, 10, 12, 10)
        t_layout.setHorizontalSpacing(16)
        t_layout.setVerticalSpacing(8)

        # Row 0: Scale & Rotation
        t_layout.addWidget(QLabel("Scale:"), 0, 0)
        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(75, 125)
        self.slider_scale.setValue(100)
        self.val_scale = QLabel("100%")
        self.val_scale.setMinimumWidth(40)
        self.slider_scale.valueChanged.connect(self._on_tune_changed)
        t_layout.addWidget(self.slider_scale, 0, 1)
        t_layout.addWidget(self.val_scale, 0, 2)

        t_layout.addWidget(QLabel("Rotation:"), 0, 3)
        self.slider_rot = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot.setRange(-45, 45)
        self.slider_rot.setValue(0)
        self.val_rot = QLabel("0°")
        self.val_rot.setMinimumWidth(40)
        self.slider_rot.valueChanged.connect(self._on_tune_changed)
        t_layout.addWidget(self.slider_rot, 0, 4)
        t_layout.addWidget(self.val_rot, 0, 5)

        # Row 1: Nudge X & Nudge Y
        t_layout.addWidget(QLabel("Nudge X:"), 1, 0)
        self.slider_nx = QSlider(Qt.Orientation.Horizontal)
        self.slider_nx.setRange(-120, 120)
        self.slider_nx.setValue(0)
        self.val_nx = QLabel("0px")
        self.val_nx.setMinimumWidth(40)
        self.slider_nx.valueChanged.connect(self._on_tune_changed)
        t_layout.addWidget(self.slider_nx, 1, 1)
        t_layout.addWidget(self.val_nx, 1, 2)

        t_layout.addWidget(QLabel("Nudge Y:"), 1, 3)
        self.slider_ny = QSlider(Qt.Orientation.Horizontal)
        self.slider_ny.setRange(-120, 120)
        self.slider_ny.setValue(0)
        self.val_ny = QLabel("0px")
        self.val_ny.setMinimumWidth(40)
        self.slider_ny.valueChanged.connect(self._on_tune_changed)
        t_layout.addWidget(self.slider_ny, 1, 4)
        t_layout.addWidget(self.val_ny, 1, 5)

        # Row 2: Shadow Intensity & Reset Button
        t_layout.addWidget(QLabel("Shadow:"), 2, 0)
        self.slider_shadow = QSlider(Qt.Orientation.Horizontal)
        self.slider_shadow.setRange(40, 160)
        self.slider_shadow.setValue(100)
        self.val_shadow = QLabel("100%")
        self.val_shadow.setMinimumWidth(40)
        self.slider_shadow.valueChanged.connect(self._on_tune_changed)
        t_layout.addWidget(self.slider_shadow, 2, 1)
        t_layout.addWidget(self.val_shadow, 2, 2)

        btn_reset_tune = QPushButton("↺ Reset Controls")
        btn_reset_tune.setObjectName("studioBtnSecondary")
        btn_reset_tune.clicked.connect(self._reset_tune_sliders)
        t_layout.addWidget(btn_reset_tune, 2, 4, 1, 2)

        layout.addWidget(tuning_box)

        # Bottom Export This Scene Row
        export_row = QHBoxLayout()
        export_row.setSpacing(10)

        self.btn_save_png = QPushButton("💾 Save Scene as PNG")
        self.btn_save_png.setObjectName("studioBtnSecondary")
        self.btn_save_png.clicked.connect(lambda: self._on_export_current("png"))
        export_row.addWidget(self.btn_save_png)

        self.btn_save_jpg = QPushButton("💾 Save Scene as JPG")
        self.btn_save_jpg.setObjectName("studioBtnSecondary")
        self.btn_save_jpg.clicked.connect(lambda: self._on_export_current("jpg"))
        export_row.addWidget(self.btn_save_jpg)

        export_row.addStretch(1)

        self.inspector_res_label = QLabel("Resolution: 1122 × 1402 px")
        self.inspector_res_label.setObjectName("studioMuted")
        export_row.addWidget(self.inspector_res_label)

        layout.addLayout(export_row)
        return container

    def _start_initial_generation(self) -> None:
        self._extract_phone()
        if self._extracted_phone is None:
            QMessageBox.warning(self, APP_NAME, "Could not extract phone mockup from current project.")
            return
        self._start_generation()

    def _extract_phone(self) -> None:
        if self.result is None or self.result.composite is None:
            self._extracted_phone = None
            return

        outer_mask = None
        if self.result.masks is not None:
            outer_mask = self.result.masks.outer

        self._extracted_phone = extract_phone_foreground(
            composite_rgba=self.result.composite,
            outer_mask=outer_mask,
            bg_color=(255, 255, 255),
        )

    def _start_generation(self) -> None:
        if self._extracted_phone is None:
            self._extract_phone()
        if self._extracted_phone is None:
            return

        # Terminate any existing running worker
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(500)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting studio engine…")
        self.btn_export_all.setEnabled(False)
        self.btn_regen.setEnabled(False)

        # Determine rotation offset from combo preset
        rot_offset = self._get_preset_rotation_offset()

        self.worker = MockupWorker(
            extracted_phone=self._extracted_phone,
            bg_paths=self.bg_paths,
            user_scale=self._scale_val,
            user_rotation_offset=rot_offset + self._rotation_val,
            shadow_intensity_mult=self._shadow_val,
            parent=self,
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.scene_ready.connect(self._on_scene_ready)
        self.worker.finished_all.connect(self._on_worker_finished)
        self.worker.start()

    def _get_preset_rotation_offset(self) -> float:
        idx = self.combo_preset.currentIndex()
        if idx == 1:  # Lying Flat 0°
            return 11.0  # counteracts base -11°
        elif idx == 2:  # Casual Angle -16°
            return -5.0
        elif idx == 3:  # Dynamic Right +12°
            return 23.0
        return 0.0  # Natural default

    def _on_preset_changed(self) -> None:
        self._start_generation()

    def _on_worker_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current} / {total}: {message}")

    def _on_scene_ready(self, index: int, title: str, comp_bgr: np.ndarray, scene_desc: SceneDescriptor) -> None:
        self.rendered_scenes[index] = (title, comp_bgr, scene_desc)
        if 0 <= index < len(self.cards):
            self.cards[index].set_mockup_image(comp_bgr, scene_desc)

        ready_count = len(self.rendered_scenes)
        self.gallery_status.setText(f"{ready_count} / {len(self.bg_paths)} Ready")

        # Automatically display the first ready scene in the inspector
        if index == self.selected_index or (ready_count == 1 and self.selected_index == 0):
            self._display_in_inspector(index)

    def _on_worker_finished(self, results: list) -> None:
        self.progress_bar.setValue(len(self.bg_paths))
        self.progress_bar.setFormat("✨ All 10 studio scenes ready!")
        self.btn_export_all.setEnabled(True)
        self.btn_regen.setEnabled(True)

    def _on_card_selected(self, index: int) -> None:
        if index not in self.rendered_scenes:
            return
        self.selected_index = index
        for idx, card in enumerate(self.cards):
            card.set_selected(idx == index)
        self._display_in_inspector(index)

    def _display_in_inspector(self, index: int) -> None:
        if index not in self.rendered_scenes:
            return

        title, comp_rgb, scene_desc = self.rendered_scenes[index]
        self.inspector_title.setText(title)

        # comp_rgb is pure RGB; pack alpha=255 for ImagePreview
        h, w = comp_rgb.shape[:2]
        comp_rgba = np.dstack([comp_rgb, np.full((h, w), 255, dtype=np.uint8)])
        self.inspector_preview.set_image(comp_rgba)
        self.inspector_res_label.setText(f"Resolution: {w} × {h} px")

        for idx, card in enumerate(self.cards):
            card.set_selected(idx == index)

    def _on_tune_changed(self) -> None:
        self._scale_val = self.slider_scale.value() / 100.0
        self._rotation_val = float(self.slider_rot.value())
        self._nudge_x = self.slider_nx.value()
        self._nudge_y = self.slider_ny.value()
        self._shadow_val = self.slider_shadow.value() / 100.0

        self.val_scale.setText(f"{self.slider_scale.value()}%")
        self.val_rot.setText(f"{self.slider_rot.value()}°")
        self.val_nx.setText(f"{self.slider_nx.value()}px")
        self.val_ny.setText(f"{self.slider_ny.value()}px")
        self.val_shadow.setText(f"{self.slider_shadow.value()}%")

        self._debounce_timer.start()

    def _reset_tune_sliders(self) -> None:
        self.slider_scale.setValue(100)
        self.slider_rot.setValue(0)
        self.slider_nx.setValue(0)
        self.slider_ny.setValue(0)
        self.slider_shadow.setValue(100)
        self._on_tune_changed()

    def _re_render_selected_scene(self) -> None:
        """Dynamically re-composites only the currently selected scene with new slider values."""
        if self.selected_index not in self.rendered_scenes or self._extracted_phone is None:
            return

        title, old_comp, scene = self.rendered_scenes[self.selected_index]
        path = self.bg_paths[self.selected_index]
        bgr = cv2.imread(str(path))
        if bgr is None:
            return
        bg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rot_offset = self._get_preset_rotation_offset()
        new_comp_rgb = composite_mockup_on_background(
            bg_image=bg_rgb,
            extracted_phone=self._extracted_phone,
            scene=scene,
            user_scale=self._scale_val,
            user_rotation_offset=rot_offset + self._rotation_val,
            user_nudge_x=self._nudge_x,
            user_nudge_y=self._nudge_y,
            shadow_intensity_mult=self._shadow_val,
        )

        self.rendered_scenes[self.selected_index] = (title, new_comp_rgb, scene)
        self.cards[self.selected_index].set_mockup_image(new_comp_rgb, scene)
        self._display_in_inspector(self.selected_index)

    def _on_export_current(self, fmt: str) -> None:
        if self.selected_index not in self.rendered_scenes:
            return
        title, comp_rgb, _ = self.rendered_scenes[self.selected_index]
        MOCKUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        clean_name = title.replace(" ", "_").replace(":", "").replace(".", "").lower()
        default_name = str(MOCKUP_OUTPUT_DIR / f"{clean_name}.{fmt}")

        filter_str = "PNG (*.png)" if fmt == "png" else "JPEG (*.jpg *.jpeg)"
        dest, _ = QFileDialog.getSaveFileName(self, f"Save Mockup Scene as {fmt.upper()}", default_name, filter_str)
        if not dest:
            return

        comp_bgr = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)
        if fmt == "png":
            cv2.imwrite(dest, comp_bgr)
        else:
            cv2.imwrite(dest, comp_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        QMessageBox.information(self, APP_NAME, f"Successfully saved:\n{Path(dest).name}")

    def _on_quick_save(self, index: int) -> None:
        self.selected_index = index
        self._on_export_current("png")

    def _on_export_all(self) -> None:
        if not self.rendered_scenes:
            return

        dest_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for All 10 Mockup Scenes",
            str(MOCKUP_OUTPUT_DIR),
        )
        if not dest_dir:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(dest_dir) / f"mockup_studio_{timestamp}"
        target_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for idx in sorted(self.rendered_scenes.keys()):
            title, comp_rgb, _ = self.rendered_scenes[idx]
            scene_num = idx + 1
            safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)
            filename = f"mockup_{scene_num:02d}_{safe_title}.png"
            file_path = target_dir / filename
            cv2.imwrite(str(file_path), cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR))
            saved_files.append(filename)

        QMessageBox.information(
            self,
            APP_NAME,
            f"Successfully exported {len(saved_files)} mockups to:\n{target_dir.resolve()}\n\n"
            + "\n".join(f"• {f}" for f in saved_files[:5])
            + (f"\n... and {len(saved_files) - 5} more." if len(saved_files) > 5 else ""),
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(400)
        super().closeEvent(event)

    def _apply_studio_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: #101317;
                color: #e8eaed;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
                font-size: 13px;
            }
            QFrame#topBar {
                background: #181d24;
                border: 1px solid #2a3340;
                border-radius: 12px;
            }
            QLabel#studioH1 {
                font-size: 16px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: 0.3px;
            }
            QLabel#studioSub {
                font-size: 11px;
                color: #8c96a5;
            }
            QLabel#studioMuted {
                font-size: 12px;
                color: #8c96a5;
            }
            QLabel#sectionHeader {
                font-size: 14px;
                font-weight: 600;
                color: #dbe4f0;
            }
            QProgressBar#studioProgress {
                background: #14171c;
                border: 1px solid #2c333e;
                border-radius: 6px;
                text-align: center;
                color: #e0e6ed;
                font-size: 11px;
                font-weight: 600;
            }
            QProgressBar#studioProgress::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f7de1, stop:1 #00d2ff);
                border-radius: 5px;
            }
            QComboBox#studioCombo {
                background: #202732;
                border: 1px solid #364254;
                border-radius: 6px;
                padding: 4px 10px;
                color: #e8eaed;
                font-size: 12px;
            }
            QPushButton#studioBtnPrimary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2b77db, stop:1 #1d95e0);
                border: 1px solid #3d9cf0;
                border-radius: 8px;
                color: #ffffff;
                font-weight: 600;
                padding: 7px 14px;
            }
            QPushButton#studioBtnPrimary:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3884e8, stop:1 #28a3f0);
            }
            QPushButton#studioBtnPrimary:disabled {
                background: #202630;
                border-color: #2c333e;
                color: #616c7c;
            }
            QPushButton#studioBtnSecondary {
                background: #222a36;
                border: 1px solid #364254;
                border-radius: 8px;
                color: #e8eaed;
                font-weight: 600;
                padding: 7px 12px;
            }
            QPushButton#studioBtnSecondary:hover {
                background: #2b3544;
                border-color: #48576e;
            }
            QPushButton#zoomToolBtn {
                background: #1e2530;
                border: 1px solid #2f3a4b;
                border-radius: 6px;
                color: #d0d7de;
                font-size: 11px;
                font-weight: 600;
                padding: 3px 8px;
            }
            QPushButton#zoomToolBtn:hover {
                background: #2b3544;
                border-color: #3d9cf0;
            }
            QGroupBox#tuningGroup {
                background: #161b22;
                border: 1px solid #283240;
                border-radius: 10px;
                margin-top: 8px;
                padding-top: 10px;
                font-size: 12px;
                font-weight: 600;
                color: #9db2cc;
            }
            QGroupBox#tuningGroup::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QFrame#mockupCard {
                background: #1a2028;
                border: 1px solid #2e3745;
                border-radius: 10px;
            }
            QLabel#cardTitle {
                font-size: 12px;
                font-weight: 600;
                color: #e2e8f0;
            }
            QPushButton#cardBtn {
                background: #26303e;
                border: 1px solid #3a475a;
                border-radius: 6px;
                color: #c9d2de;
                font-size: 11px;
                font-weight: 600;
                padding: 4px;
            }
            QPushButton#cardBtn:hover {
                background: #334053;
                border-color: #3d9cf0;
            }
            QPushButton#cardBtnSecondary {
                background: #1c222b;
                border: 1px solid #2d3746;
                border-radius: 6px;
                color: #9aa5b5;
                font-size: 11px;
                padding: 4px;
            }
            QPushButton#cardBtnSecondary:hover {
                background: #252d3a;
                color: #ffffff;
            }
            QScrollArea#galleryScroll {
                background: transparent;
                border: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #283240;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -5px 0;
                background: #3d9cf0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #5ab1ff;
            }
            QSplitter::handle {
                background: #181d24;
                width: 8px;
            }
            """
        )
