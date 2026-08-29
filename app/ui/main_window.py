"""Main application window — UI and user interaction only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRect, QRectF, QSize, QThread, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.core.cover_processor import CoverProcessor, ProcessingResult
from app.utils.constants import (
    APP_NAME,
    APP_VERSION,
    DEBUG_MODE,
    ICONS_DIR,
    OUTPUT_DIR,
    SUPPORTED_INPUT_FILTER,
)
from app.utils.image_utils import CoverError, numpy_rgba_to_qimage


DEBUG_VIEWS = (
    ("raw_contour", "1. Raw detected contour"),
    ("cleaned_contour", "2. Cleaned contour"),
    ("printable_boundary", "3. Final printable boundary"),
    ("final_print_mask", "4. Final mask"),
    ("camera_exclusion", "5. Camera exclusion"),
    ("final_composite", "6. Final composite"),
    ("edge_exclusion", "7. Side wall / edge"),
    ("artwork_before_mask", "8. Artwork before mask"),
)


class CoverWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, processor: CoverProcessor, cover_path: str) -> None:
        super().__init__()
        self._processor = processor
        self._cover_path = cover_path

    def run(self) -> None:
        try:
            detection = self._processor.load_cover(self._cover_path)
            self.finished.emit(detection)
        except CoverError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Something went wrong while reading the cover.\n{exc}")


class DesignWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, processor: CoverProcessor, design_path: str) -> None:
        super().__init__()
        self._processor = processor
        self._design_path = design_path

    def run(self) -> None:
        try:
            result = self._processor.process_design(self._design_path)
            self.finished.emit(result)
        except CoverError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Something went wrong while processing the design.\n{exc}")


class ImagePreview(QWidget):
    """Scaled image viewer with a checkerboard behind transparent pixels."""

    region_selected = Signal(float, float, float, float)
    zoom_changed = Signal(int)

    def __init__(
        self,
        placeholder: str = "",
        parent: QWidget | None = None,
        *,
        zoomable: bool = False,
    ) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._placeholder = placeholder
        self._selectable = False
        self._zoomable = zoomable
        self._zoom_mode = "fit"
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_origin = QPointF()
        self._pan_start = QPointF()
        self._drag_origin: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self.setMinimumSize(180, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if zoomable:
            self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_selection_enabled(self, enabled: bool) -> None:
        self._selectable = enabled
        self._drag_origin = None
        self._drag_current = None
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self._zoomable:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = QPixmap()
        else:
            self._pixmap = QPixmap.fromImage(numpy_rgba_to_qimage(image))
        self.update()
        self._emit_zoom()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(1.0 / 1.25)

    def fit_to_screen(self) -> None:
        self._zoom_mode = "fit"
        self._pan = QPointF(0.0, 0.0)
        self.update()
        self._emit_zoom()

    def reset_zoom(self) -> None:
        self._zoom_mode = "free"
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()
        self._emit_zoom()

    def zoom_percent(self) -> int:
        if self._pixmap.isNull():
            return 100
        rect = self._content_rect()
        return max(1, int(round(100.0 * rect.width() / max(self._pixmap.width(), 1))))

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#1c2028"))

        if self._pixmap.isNull():
            painter.setPen(QColor("#6b7380"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        target = self._content_rect()
        self._draw_checkerboard(painter, target.toAlignedRect())
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        if self._drag_origin and self._drag_current:
            x0, y0 = self._drag_origin
            x1, y1 = self._drag_current
            box = QRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            painter.setPen(QColor("#3d9cf0"))
            painter.fillRect(box, QColor(61, 156, 240, 50))
            painter.drawRect(box)

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        if not self._zoomable or self._pixmap.isNull() or self._selectable:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self._zoom_at(event.position(), factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self._selectable and not self._pixmap.isNull() and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = (event.position().toPoint().x(), event.position().toPoint().y())
            self._drag_current = self._drag_origin
            self.update()
        elif (
            self._zoomable
            and not self._selectable
            and not self._pixmap.isNull()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._commit_fit_to_absolute()
            self._panning = True
            self._pan_origin = event.position()
            self._pan_start = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._selectable and self._drag_origin is not None:
            self._drag_current = (event.position().toPoint().x(), event.position().toPoint().y())
            self.update()
        elif self._panning:
            delta = event.position() - self._pan_origin
            self._pan = self._pan_start + delta
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._selectable and self._drag_origin and self._drag_current:
            rect = self._image_rect_from_drag()
            self._drag_origin = None
            self._drag_current = None
            self.update()
            if rect is not None:
                self.region_selected.emit(*rect)
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            if self._zoomable and not self._selectable:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._emit_zoom()

    def _image_rect_from_drag(self) -> tuple[float, float, float, float] | None:
        if self._pixmap.isNull() or self._drag_origin is None or self._drag_current is None:
            return None
        fitted = self._content_rect()
        x0, y0 = self._drag_origin
        x1, y1 = self._drag_current
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 8 or bottom - top < 8:
            return None
        ix0 = (left - fitted.x()) / max(fitted.width(), 1) * self._pixmap.width()
        iy0 = (top - fitted.y()) / max(fitted.height(), 1) * self._pixmap.height()
        ix1 = (right - fitted.x()) / max(fitted.width(), 1) * self._pixmap.width()
        iy1 = (bottom - fitted.y()) / max(fitted.height(), 1) * self._pixmap.height()
        ix0 = float(np.clip(ix0, 0, self._pixmap.width()))
        iy0 = float(np.clip(iy0, 0, self._pixmap.height()))
        ix1 = float(np.clip(ix1, 0, self._pixmap.width()))
        iy1 = float(np.clip(iy1, 0, self._pixmap.height()))
        return (ix0, iy0, max(1.0, ix1 - ix0), max(1.0, iy1 - iy0))

    def _content_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        if self._zoom_mode == "fit" or not self._zoomable:
            return QRectF(self._fitted_rect(self._pixmap.size()))
        pw = float(self._pixmap.width())
        ph = float(self._pixmap.height())
        w = pw * self._zoom
        h = ph * self._zoom
        x = (self.width() - w) / 2.0 + self._pan.x()
        y = (self.height() - h) / 2.0 + self._pan.y()
        return QRectF(x, y, w, h)

    def _commit_fit_to_absolute(self) -> None:
        if self._zoom_mode != "fit" or self._pixmap.isNull():
            return
        fitted = self._fitted_rect(self._pixmap.size())
        self._zoom = fitted.width() / max(self._pixmap.width(), 1)
        self._pan = QPointF(0.0, 0.0)
        self._zoom_mode = "free"

    def _zoom_by(self, factor: float) -> None:
        if not self._zoomable or self._pixmap.isNull():
            return
        self._zoom_at(QPointF(self.width() / 2.0, self.height() / 2.0), factor)

    def _zoom_at(self, pos: QPointF, factor: float) -> None:
        if self._pixmap.isNull():
            return
        rect = self._content_rect()
        pw = float(self._pixmap.width())
        ph = float(self._pixmap.height())
        ix = (pos.x() - rect.x()) / max(rect.width(), 1e-6) * pw
        iy = (pos.y() - rect.y()) / max(rect.height(), 1e-6) * ph
        self._commit_fit_to_absolute()
        self._zoom = float(np.clip(self._zoom * factor, 0.05, 16.0))
        new_w = pw * self._zoom
        new_h = ph * self._zoom
        new_x = pos.x() - (ix / max(pw, 1e-6)) * new_w
        new_y = pos.y() - (iy / max(ph, 1e-6)) * new_h
        self._pan = QPointF(
            new_x - (self.width() - new_w) / 2.0,
            new_y - (self.height() - new_h) / 2.0,
        )
        self._zoom_mode = "free"
        self.update()
        self._emit_zoom()

    def _emit_zoom(self) -> None:
        if self._zoomable:
            self.zoom_changed.emit(self.zoom_percent())

    def _fitted_rect(self, source: QSize) -> QRect:
        margin = 12
        avail_w = max(1, self.width() - margin * 2)
        avail_h = max(1, self.height() - margin * 2)
        scaled = source.scaled(avail_w, avail_h, Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def _draw_checkerboard(self, painter: QPainter, rect: QRect) -> None:
        if rect.width() <= 0 or rect.height() <= 0:
            return
        size = 10
        c1, c2 = QColor("#2a303a"), QColor("#232830")
        painter.save()
        painter.setClipRect(rect)
        y = rect.top()
        row = 0
        while y < rect.bottom():
            x = rect.left()
            col = 0
            while x < rect.right():
                painter.fillRect(x, y, size, size, c1 if (row + col) % 2 == 0 else c2)
                x += size
                col += 1
            y += size
            row += 1
        painter.restore()


class DebugThumb(QFrame):
    clicked = Signal(str)

    def __init__(self, key: str, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName("debugThumb")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.preview = ImagePreview("")
        self.preview.setMinimumSize(72, 110)
        self.caption = QLabel(caption)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setObjectName("debugCaption")
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.caption)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        self.clicked.emit(self.key)
        super().mousePressEvent(event)


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

        self._build_ui()
        self._apply_style()
        self._set_status("Upload a transparent phone cover.")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 8)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.cover_preview = ImagePreview("Upload a phone cover")
        self.cover_preview.region_selected.connect(self._on_camera_rect)
        self.design_preview = ImagePreview("Upload a design")
        self.final_preview = ImagePreview("Final mockup", zoomable=True)
        left = self._panel("Cover preview", self.cover_preview)
        design = self._panel("Design preview", self.design_preview)
        final = self._panel("Final mockup", self.final_preview, show_zoom=True)
        right = self._build_controls()

        splitter.addWidget(left)
        splitter.addWidget(design)
        splitter.addWidget(final)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 4)
        splitter.setStretchFactor(3, 2)
        splitter.setSizes([340, 340, 420, 280])

        root_layout.addWidget(splitter, 1)

        if DEBUG_MODE:
            header = QLabel("DEBUG MODE — click a view to inspect it in the final preview")
            header.setObjectName("debugHeader")
            wrap = QWidget()
            wrap_layout = QVBoxLayout(wrap)
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
            root_layout.addWidget(wrap)

        self.setCentralWidget(root)
        status = QStatusBar()
        self.setStatusBar(status)

    def _panel(self, title: str, widget: QWidget, show_zoom: bool = False) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setObjectName("panelTitle")
        header.addWidget(label, 1)
        if show_zoom:
            self._zoom_label = QLabel("100%")
            self._zoom_label.setObjectName("zoomPct")
            header.addWidget(self._zoom_label)
        layout.addLayout(header)
        if show_zoom:
            zoom_row = QHBoxLayout()
            zoom_row.setSpacing(6)
            btn_in = QPushButton("Zoom In")
            btn_in.setObjectName("zoomBtn")
            btn_out = QPushButton("Zoom Out")
            btn_out.setObjectName("zoomBtn")
            btn_fit = QPushButton("Fit to Screen")
            btn_fit.setObjectName("zoomBtn")
            btn_reset = QPushButton("Reset Zoom")
            btn_reset.setObjectName("zoomBtn")
            btn_in.clicked.connect(self.final_preview.zoom_in)
            btn_out.clicked.connect(self.final_preview.zoom_out)
            btn_fit.clicked.connect(self.final_preview.fit_to_screen)
            btn_reset.clicked.connect(self.final_preview.reset_zoom)
            self.final_preview.zoom_changed.connect(self._on_final_zoom)
            for b in (btn_in, btn_out, btn_fit, btn_reset):
                zoom_row.addWidget(b)
            layout.addLayout(zoom_row)
        layout.addWidget(widget, 1)
        return frame

    def _build_controls(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Actions")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        hint = QLabel(
            "1. Upload a transparent phone cover.\n"
            "2. Upload your photo or artwork.\n\n"
            "The design is clipped to the flat back panel only. "
            "Camera, bumper, and edges stay clear. Colors stay as uploaded."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        layout.addWidget(hint)

        self.btn_cover = QPushButton("Upload Cover")
        self.btn_cover.setObjectName("primary")
        self.btn_cover.clicked.connect(self._on_upload_cover)
        layout.addWidget(self.btn_cover)

        self.btn_design = QPushButton("Upload Design")
        self.btn_design.clicked.connect(self._on_upload_design)
        layout.addWidget(self.btn_design)

        self.btn_camera = QPushButton("Mark camera area")
        self.btn_camera.setEnabled(False)
        self.btn_camera.clicked.connect(self._on_mark_camera)
        layout.addWidget(self.btn_camera)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        layout.addWidget(self.btn_reset)

        layout.addSpacing(8)
        export_label = QLabel("Export")
        export_label.setObjectName("panelTitle")
        layout.addWidget(export_label)

        self.btn_png = QPushButton("Export PNG")
        self.btn_png.clicked.connect(lambda: self._on_export("png"))
        layout.addWidget(self.btn_png)

        self.btn_jpg = QPushButton("Export JPG")
        self.btn_jpg.clicked.connect(lambda: self._on_export("jpg"))
        layout.addWidget(self.btn_jpg)

        self._set_export_enabled(False)
        layout.addStretch(1)

        footer = QLabel("Offline  ·  local processing only")
        footer.setObjectName("muted")
        layout.addWidget(footer)
        return frame

    def _on_upload_cover(self) -> None:
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a phone cover image", "", SUPPORTED_INPUT_FILTER)
        if not path:
            return
        self._start_cover_load(path)

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
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status("Processing…")

        thread = QThread(self)
        worker = CoverWorker(self.processor, path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_cover_loaded)
        worker.failed.connect(self._on_process_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker_thread = thread
        self._worker = worker
        thread.start()

    def _start_design_process(self, path: str) -> None:
        self._busy = True
        self._set_buttons_enabled(False)
        self._set_status("Processing…")

        thread = QThread(self)
        worker = DesignWorker(self.processor, path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_process_finished)
        worker.failed.connect(self._on_process_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker_thread = thread
        self._worker = worker
        thread.start()

    def _clear_worker(self) -> None:
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

    def _on_mark_camera(self) -> None:
        if self.processor.cover is None:
            self._show_error("Upload a phone cover first.")
            return
        self.cover_preview.set_selection_enabled(True)
        self._set_status("Drag a box around the camera on the cover preview.")

    def _on_camera_rect(self, x: float, y: float, w: float, h: float) -> None:
        self.cover_preview.set_selection_enabled(False)
        try:
            detection = self.processor.set_manual_camera_rect(x, y, w, h)
        except CoverError as exc:
            self._show_error(str(exc))
            return
        if self.processor.last_result is not None:
            self._result = self.processor.last_result
            self._set_export_enabled(True)
        else:
            self._result = self.processor.cover_preview_result()
        self._final_key = "final_composite"
        self._refresh_previews()
        self._set_status("Camera area updated." if detection.camera_found else "Upload your design.")

    def _on_cover_loaded(self, detection) -> None:  # noqa: ANN001
        self.cover_preview.set_image(self.processor.cover)
        self.cover_preview.set_selection_enabled(False)
        self._result = self.processor.cover_preview_result()
        self._final_key = "final_composite"
        self._refresh_previews()
        self._set_export_enabled(False)
        self.btn_camera.setEnabled(True)

        serious = [w for w in detection.warnings if "unusually small" in w or "confidence is" in w]
        if serious:
            self._show_warning("\n".join(serious))

        if self._pending_design:
            self._auto_process_design = True
            self._set_status("Processing…")
        elif not detection.camera_found:
            self._set_status("Upload your design. Optional: Mark camera area if the cutout is wrong.")
        else:
            self._set_status("Upload your design.")

    def _on_process_finished(self, result: ProcessingResult) -> None:
        self._result = result
        self._final_key = "final_composite"
        self._set_export_enabled(True)
        self._refresh_previews()
        serious = [w for w in result.warnings if "unusually small" in w or "confidence is" in w]
        if serious:
            self._show_warning("\n".join(serious))
        self._set_status("Ready to export.")

    def _on_process_failed(self, message: str) -> None:
        self._show_error(message)
        self._set_status("Processing failed. Please try another image.")

    def _on_reset(self) -> None:
        if self._busy:
            return
        self.processor.reset()
        self._result = None
        self._pending_design = None
        self._auto_process_design = False
        self._final_key = "final_composite"
        self.cover_preview.set_image(None)
        self.cover_preview.set_selection_enabled(False)
        self.design_preview.set_image(None)
        self.final_preview.set_image(None)
        self.final_preview.fit_to_screen()
        self.btn_camera.setEnabled(False)
        self._set_export_enabled(False)
        if DEBUG_MODE:
            for thumb in self._debug_thumbs.values():
                thumb.preview.set_image(None)
        self._set_status("Upload a transparent phone cover.")

    def _on_export(self, fmt: str) -> None:
        if self.processor.last_result is None:
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
        self._refresh_previews()
        label = dict(DEBUG_VIEWS).get(key, key)
        self._set_status(f"Debug view: {label}")

    def _refresh_previews(self) -> None:
        if self.processor.cover is not None:
            self.cover_preview.set_image(self.processor.cover)
        if self.processor.design is not None:
            self.design_preview.set_image(self.processor.design)
        if self._result is None:
            return
        debug = self._result.debug_images
        self.final_preview.set_image(debug.get(self._final_key, self._result.composite))
        if DEBUG_MODE:
            for key, thumb in self._debug_thumbs.items():
                thumb.preview.set_image(debug.get(key))

    def _set_export_enabled(self, enabled: bool) -> None:
        self.btn_png.setEnabled(enabled)
        self.btn_jpg.setEnabled(enabled)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.btn_cover.setEnabled(enabled)
        self.btn_design.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.btn_camera.setEnabled(enabled and self.processor.cover is not None)
        if enabled:
            self._set_export_enabled(self.processor.last_result is not None)
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
            QLabel#panelTitle {
                font-size: 15px;
                font-weight: 600;
                color: #f2f4f7;
                letter-spacing: 0.2px;
            }
            QLabel#hint { color: #9aa3b2; line-height: 1.4; }
            QLabel#muted { color: #6b7380; font-size: 11px; }
            QLabel#debugHeader {
                color: #8fb8ff;
                font-size: 12px;
                font-weight: 600;
            }
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
                min-width: 48px;
            }
            QPushButton#zoomBtn {
                padding: 5px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton {
                background: #2a313c;
                color: #e8eaed;
                border: 1px solid #3a4352;
                border-radius: 8px;
                padding: 10px 14px;
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
