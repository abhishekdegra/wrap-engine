"""Zoomable mockup viewer with full manual vector and raster camera-mask editing."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from app.utils.image_utils import numpy_rgba_to_qimage

MASK_TOOLS = (
    "move",
    "rect",
    "roundrect",
    "circle",
    "ellipse",
    "pill",
    "polygon",
    "freeform",
    "brush",
    "eraser",
)
DRAW_BOX_TOOLS = ("rect", "roundrect", "circle", "ellipse", "pill")

# Handle IDs
HANDLE_NONE = 0
HANDLE_BODY = 1
HANDLE_NW = 2
HANDLE_N = 3
HANDLE_NE = 4
HANDLE_E = 5
HANDLE_SE = 6
HANDLE_S = 7
HANDLE_SW = 8
HANDLE_W = 9
HANDLE_ROT = 10
HANDLE_RADIUS = 11
HANDLE_VERTEX_BASE = 100


@dataclass
class MaskShape:
    shape_type: str  # "rect", "roundrect", "circle", "ellipse", "pill", "polygon", "freeform"
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0  # degrees around center
    corner_radius: float = 16.0  # in image pixels
    lock_aspect: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)  # for polygon/freeform in image coords

    def center(self) -> QPointF:
        return QPointF(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "rotation": round(self.rotation, 1),
            "corner_radius": round(self.corner_radius, 1),
            "lock_aspect": self.lock_aspect,
        }


class ImagePreview(QWidget):
    """Scaled image viewer with an interactive multi-shape vector + raster Camera Mask Editor."""

    region_selected = Signal(float, float, float, float)
    zoom_changed = Signal(int)
    mask_changed = Signal()
    shape_selected = Signal(object)  # emits dict of shape props or None
    shape_updated = Signal(object)  # emits live dict during drag/transform

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

        # Tool & State
        self._tool: str | None = None
        self._shapes: list[MaskShape] = []
        self._selected_index: int = -1
        self._raster_mask: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._overlay = QPixmap()

        # Drag / Transform Interaction State
        self._active_handle: int = HANDLE_NONE
        self._drag_start_img_pt: QPointF | None = None
        self._drag_start_shape: MaskShape | None = None
        self._drag_start_shapes: list[MaskShape] = []
        self._drag_start_raster: np.ndarray | None = None
        self._drag_origin_widget: tuple[int, int] | None = None
        self._drag_current_widget: tuple[int, int] | None = None

        # Drawing Path State
        self._stroke_img: list[tuple[float, float]] = []
        self._poly_pts: list[tuple[float, float]] = []
        self._freeform_pts: list[tuple[float, float]] = []

        # History for Undo / Redo
        self._history: list[tuple[list[MaskShape], np.ndarray | None]] = []
        self._history_index = -1

        self._space_down = False
        self.brush_radius = 14.0

        self.setMinimumSize(180, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if zoomable:
            self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------
    # Public API for Tools & Properties
    # ------------------------------------------------------------------

    def set_selection_enabled(self, enabled: bool) -> None:
        self._selectable = enabled
        self._drag_origin_widget = None
        self._drag_current_widget = None
        self._update_cursor()
        self.update()

    def set_tool(self, tool: str | None) -> None:
        self._tool = tool if tool in MASK_TOOLS else None
        self._drag_origin_widget = None
        self._drag_current_widget = None
        self._stroke_img = []
        self._freeform_pts = []
        self._active_handle = HANDLE_NONE

        # Auto-select the first shape when entering Move tool if none selected
        if self._tool == "move" and self._shapes and self._selected_index < 0:
            self._selected_index = 0
            self._emit_shape_selected()

        self._update_cursor()
        self.update()

    def tool(self) -> str | None:
        return self._tool

    def selected_shape(self) -> MaskShape | None:
        if 0 <= self._selected_index < len(self._shapes):
            return self._shapes[self._selected_index]
        return None

    def update_selected_shape(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        rotation: float | None = None,
        corner_radius: float | None = None,
        lock_aspect: bool | None = None,
    ) -> None:
        shape = self.selected_shape()
        if shape is None:
            return
        if x is not None:
            shape.x = float(x)
        if y is not None:
            shape.y = float(y)
        if width is not None:
            shape.width = max(2.0, float(width))
        if height is not None:
            shape.height = max(2.0, float(height))
        if rotation is not None:
            shape.rotation = float(rotation)
        if corner_radius is not None:
            shape.corner_radius = max(0.0, float(corner_radius))
        if lock_aspect is not None:
            shape.lock_aspect = bool(lock_aspect)

        if shape.shape_type == "circle":
            r = min(shape.width, shape.height)
            shape.width = r
            shape.height = r
        elif shape.shape_type == "pill":
            shape.corner_radius = min(shape.width, shape.height) / 2.0

        self._push_history()
        self._rebuild_all()
        self.shape_updated.emit(shape.to_dict())
        self.mask_changed.emit()
        self.update()

    def delete_selected_shape(self) -> None:
        if 0 <= self._selected_index < len(self._shapes):
            self._shapes.pop(self._selected_index)
            self._selected_index = len(self._shapes) - 1
            self._push_history()
            self._rebuild_all()
            self._emit_shape_selected()
            self.mask_changed.emit()
            self.update()

    def reset_selected_transform(self) -> None:
        shape = self.selected_shape()
        if shape is None:
            return
        shape.rotation = 0.0
        if shape.shape_type == "circle":
            r = min(shape.width, shape.height)
            shape.width = r
            shape.height = r
        self._push_history()
        self._rebuild_all()
        self.shape_updated.emit(shape.to_dict())
        self.mask_changed.emit()
        self.update()

    def set_camera_mask(self, mask: np.ndarray | None, *, remember: bool = True) -> None:
        if mask is None:
            self._mask = None
            self._raster_mask = None
            self._shapes = []
            self._selected_index = -1
            self._history = []
            self._history_index = -1
            self._overlay = QPixmap()
            self._emit_shape_selected()
            self.update()
            return

        arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if arr.ndim == 3:
            arr = arr[..., 0]
        self._mask = arr
        self._raster_mask = arr.copy()
        self._fit_mask_to_pixmap()

        # If no vector shapes yet, initialize from detected camera mask bounding island
        if not self._shapes and np.any(arr > 0.25):
            ys, xs = np.where(arr > 0.25)
            if ys.size > 0:
                bx = float(xs.min())
                by = float(ys.min())
                bw = float(xs.max() - xs.min() + 1)
                bh = float(ys.max() - ys.min() + 1)
                rad = max(6.0, 0.18 * min(bw, bh))
                init_shape = MaskShape(
                    shape_type="roundrect",
                    x=bx,
                    y=by,
                    width=bw,
                    height=bh,
                    rotation=0.0,
                    corner_radius=rad,
                )
                self._shapes = [init_shape]
                self._selected_index = 0
                self._raster_mask = None

        if remember:
            self._push_history()
        self._rebuild_all()
        self._emit_shape_selected()
        self.update()

    def camera_mask(self) -> np.ndarray | None:
        if not self._shapes and self._raster_mask is None:
            return None if self._mask is None else self._mask.copy()
        return self._rasterize_full_mask()

    def clear_mask(self) -> None:
        self._shapes = []
        self._selected_index = -1
        self._raster_mask = np.zeros(self._canvas_size()[::-1], dtype=np.float32) if not self._pixmap.isNull() else None
        self._poly_pts = []
        self._freeform_pts = []
        self._stroke_img = []
        self._push_history()
        self._rebuild_all()
        self._emit_shape_selected()
        self.mask_changed.emit()
        self.update()

    def undo_mask(self) -> None:
        if self._history_index <= 0:
            return
        self._history_index -= 1
        shapes_copy, raster_copy = self._history[self._history_index]
        self._shapes = [copy.deepcopy(s) for s in shapes_copy]
        self._raster_mask = raster_copy.copy() if raster_copy is not None else None
        self._selected_index = min(self._selected_index, len(self._shapes) - 1)
        self._rebuild_all()
        self._emit_shape_selected()
        self.mask_changed.emit()
        self.update()

    def redo_mask(self) -> None:
        if self._history_index < 0 or self._history_index >= len(self._history) - 1:
            return
        self._history_index += 1
        shapes_copy, raster_copy = self._history[self._history_index]
        self._shapes = [copy.deepcopy(s) for s in shapes_copy]
        self._raster_mask = raster_copy.copy() if raster_copy is not None else None
        self._selected_index = min(self._selected_index, len(self._shapes) - 1)
        self._rebuild_all()
        self._emit_shape_selected()
        self.mask_changed.emit()
        self.update()

    def reset_session(self) -> None:
        self._pixmap = QPixmap()
        self._selectable = False
        self._zoom_mode = "fit"
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_origin = QPointF()
        self._pan_start = QPointF()
        self._drag_origin_widget = None
        self._drag_current_widget = None
        self._tool = None
        self._shapes = []
        self._selected_index = -1
        self._raster_mask = None
        self._mask = None
        self._overlay = QPixmap()
        self._history = []
        self._history_index = -1
        self._stroke_img = []
        self._poly_pts = []
        self._freeform_pts = []
        self._space_down = False
        self._active_handle = HANDLE_NONE
        self._update_cursor()
        self._emit_shape_selected()
        self.update()
        self._emit_zoom()

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = QPixmap()
        else:
            self._pixmap = QPixmap.fromImage(numpy_rgba_to_qimage(image))
        self._fit_mask_to_pixmap()
        self._rebuild_all()
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

    # ------------------------------------------------------------------
    # Paint & Canvas Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1c2028"))

        if self._pixmap.isNull():
            painter.setPen(QColor("#6b7380"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        target = self._content_rect()
        self._draw_checkerboard(painter, target.toAlignedRect())
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        # Draw semi-transparent mask overlay
        if not self._overlay.isNull() and self._tool:
            painter.setOpacity(1.0)
            painter.drawPixmap(target, self._overlay, QRectF(self._overlay.rect()))

        # Draw active shapes, bounding boxes, handles, and live strokes
        if self._tool:
            self._paint_shapes_and_handles(painter, target)
            self._paint_live_draw_preview(painter, target)

    def _paint_shapes_and_handles(self, painter: QPainter, target: QRectF) -> None:
        painter.save()
        for idx, s in enumerate(self._shapes):
            is_selected = idx == self._selected_index
            self._draw_shape_outline(painter, s, target, is_selected)
            if is_selected and self._tool in ("move", "rect", "roundrect", "circle", "ellipse", "pill", "polygon", "freeform"):
                self._draw_shape_transform_box(painter, s, target)
        painter.restore()

    def _draw_shape_outline(self, painter: QPainter, s: MaskShape, target: QRectF, is_selected: bool) -> None:
        painter.save()
        c_wpt = self._image_to_widget_pt(s.center(), target)
        painter.translate(c_wpt)
        painter.rotate(s.rotation)

        w_w = self._image_px_to_widget(s.width, target)
        h_w = self._image_px_to_widget(s.height, target)
        rect_w = QRectF(-w_w / 2.0, -h_w / 2.0, w_w, h_w)

        if is_selected:
            pen = QPen(QColor("#00d2ff"), 2.0)
            fill_color = QColor(0, 210, 255, 60)
        else:
            pen = QPen(QColor("#3d9cf0"), 1.2, Qt.PenStyle.DashLine)
            fill_color = QColor(61, 156, 240, 35)

        painter.setPen(pen)
        painter.setBrush(fill_color)

        if s.shape_type == "circle":
            r_w = min(w_w, h_w) / 2.0
            painter.drawEllipse(QRectF(-r_w, -r_w, 2 * r_w, 2 * r_w))
        elif s.shape_type == "ellipse":
            painter.drawEllipse(rect_w)
        elif s.shape_type == "pill":
            rad_w = min(w_w, h_w) / 2.0
            path = QPainterPath()
            path.addRoundedRect(rect_w, rad_w, rad_w)
            painter.drawPath(path)
        elif s.shape_type == "roundrect":
            rad_w = self._image_px_to_widget(min(s.corner_radius, min(s.width, s.height) / 2.0), target)
            path = QPainterPath()
            path.addRoundedRect(rect_w, rad_w, rad_w)
            painter.drawPath(path)
        elif s.shape_type in ("polygon", "freeform") and len(s.points) >= 3:
            path = QPainterPath()
            first = True
            for px, py in s.points:
                wpt = self._image_to_widget_pt(QPointF(px, py), target)
                local_x = wpt.x() - c_wpt.x()
                local_y = wpt.y() - c_wpt.y()
                if first:
                    path.moveTo(local_x, local_y)
                    first = False
                else:
                    path.lineTo(local_x, local_y)
            path.closeSubpath()
            painter.drawPath(path)
        else:
            painter.drawRect(rect_w)

        painter.restore()

    def _draw_shape_transform_box(self, painter: QPainter, s: MaskShape, target: QRectF) -> None:
        painter.save()
        c_wpt = self._image_to_widget_pt(s.center(), target)
        painter.translate(c_wpt)
        painter.rotate(s.rotation)

        w_w = self._image_px_to_widget(s.width, target)
        h_w = self._image_px_to_widget(s.height, target)
        rect_w = QRectF(-w_w / 2.0, -h_w / 2.0, w_w, h_w)

        # Bounding box frame
        painter.setPen(QPen(QColor("#00d2ff"), 1.2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect_w)

        # Rotation stem line & handle
        rot_handle_dist = 26.0
        rot_pt = QPointF(0.0, -h_w / 2.0 - rot_handle_dist)
        painter.setPen(QPen(QColor("#00d2ff"), 1.2))
        painter.drawLine(QPointF(0.0, -h_w / 2.0), rot_pt)
        self._draw_circular_handle(painter, rot_pt, QColor("#00e5ff"), QColor("#00385c"), 4.5)

        # Corner radius handle for roundrect
        if s.shape_type == "roundrect":
            rad_w = self._image_px_to_widget(min(s.corner_radius, min(s.width, s.height) / 2.0), target)
            rad_pt = QPointF(-w_w / 2.0 + rad_w, -h_w / 2.0 + rad_w)
            self._draw_circular_handle(painter, rad_pt, QColor("#ffaa00"), QColor("#4a2c00"), 4.0)

        # 8 Box scale handles
        handles = [
            (-w_w / 2.0, -h_w / 2.0),  # NW
            (0.0, -h_w / 2.0),         # N
            (w_w / 2.0, -h_w / 2.0),   # NE
            (w_w / 2.0, 0.0),          # E
            (w_w / 2.0, h_w / 2.0),    # SE
            (0.0, h_w / 2.0),          # S
            (-w_w / 2.0, h_w / 2.0),   # SW
            (-w_w / 2.0, 0.0),         # W
        ]
        for hx, hy in handles:
            self._draw_square_handle(painter, QPointF(hx, hy), QColor("#ffffff"), QColor("#0070ba"), 3.5)

        # Vertex handles for polygon / freeform
        if s.shape_type in ("polygon", "freeform"):
            for px, py in s.points:
                wpt = self._image_to_widget_pt(QPointF(px, py), target)
                lx = wpt.x() - c_wpt.x()
                ly = wpt.y() - c_wpt.y()
                self._draw_circular_handle(painter, QPointF(lx, ly), QColor("#ffffff"), QColor("#ff0077"), 4.0)

        painter.restore()

    def _draw_square_handle(self, p: QPainter, pt: QPointF, fill: QColor, border: QColor, rad: float) -> None:
        p.setPen(QPen(border, 1.5))
        p.setBrush(fill)
        p.drawRect(QRectF(pt.x() - rad, pt.y() - rad, rad * 2, rad * 2))

    def _draw_circular_handle(self, p: QPainter, pt: QPointF, fill: QColor, border: QColor, rad: float) -> None:
        p.setPen(QPen(border, 1.5))
        p.setBrush(fill)
        p.drawEllipse(pt, rad, rad)

    def _paint_live_draw_preview(self, painter: QPainter, target: QRectF) -> None:
        painter.save()
        pen = QPen(QColor("#00d2ff"), 1.8)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 210, 255, 60))

        # Box drawing preview
        if self._drag_origin_widget and self._drag_current_widget and self._tool in DRAW_BOX_TOOLS:
            x0, y0 = self._drag_origin_widget
            x1, y1 = self._drag_current_widget
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            bx = min(x0, x1)
            by = min(y0, y1)

            if self._tool == "circle":
                side = max(w, h)
                rect = QRectF(bx, by, side, side)
                painter.drawEllipse(rect)
            elif self._tool == "ellipse":
                rect = QRectF(bx, by, w, h)
                painter.drawEllipse(rect)
            elif self._tool == "pill":
                rect = QRectF(bx, by, w, h)
                rad = min(w, h) / 2.0
                path = QPainterPath()
                path.addRoundedRect(rect, rad, rad)
                painter.drawPath(path)
            elif self._tool == "roundrect":
                rect = QRectF(bx, by, w, h)
                rad = 0.22 * min(w, h)
                path = QPainterPath()
                path.addRoundedRect(rect, rad, rad)
                painter.drawPath(path)
            else:
                rect = QRectF(bx, by, w, h)
                painter.drawRect(rect)

        # Polygon live vertex lines
        if self._tool == "polygon" and self._poly_pts:
            pts = [self._image_to_widget_pt(QPointF(x, y), target) for x, y in self._poly_pts]
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)
            for pt in pts:
                self._draw_circular_handle(painter, pt, QColor("#ffffff"), QColor("#00d2ff"), 4.0)

        # Freeform live stroke
        if self._tool == "freeform" and len(self._freeform_pts) >= 2:
            pts = [self._image_to_widget_pt(QPointF(x, y), target) for x, y in self._freeform_pts]
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

        # Brush / Eraser live stroke
        if self._tool in {"brush", "eraser"} and len(self._stroke_img) >= 1:
            color = QColor(255, 90, 90, 90) if self._tool == "eraser" else QColor(0, 210, 255, 90)
            painter.setPen(QPen(color, max(2.0, self._image_px_to_widget(self.brush_radius * 2, target))))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pts = [self._image_to_widget_pt(QPointF(x, y), target) for x, y in self._stroke_img]
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)

        painter.restore()

    # ------------------------------------------------------------------
    # Mouse & Keyboard Event Handling
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Space:
            self._space_down = True
            self._update_cursor()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_index >= 0:
                self.delete_selected_shape()
                event.accept()
                return
        if event.key() == Qt.Key.Key_Return and self._tool == "polygon" and len(self._poly_pts) >= 3:
            self._commit_polygon()
            event.accept()
            return
        if self._tool == "move" or self._selected_index >= 0:
            step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            dx, dy = 0.0, 0.0
            if event.key() == Qt.Key.Key_Left:
                dx = -step
            elif event.key() == Qt.Key.Key_Right:
                dx = step
            elif event.key() == Qt.Key.Key_Up:
                dy = -step
            elif event.key() == Qt.Key.Key_Down:
                dy = step
            if dx or dy:
                if 0 <= self._selected_index < len(self._shapes):
                    s = self._shapes[self._selected_index]
                    s.x += dx
                    s.y += dy
                    if s.points:
                        s.points = [(px + dx, py + dy) for px, py in s.points]
                    self.shape_updated.emit(s.to_dict())
                elif self._shapes:
                    for s in self._shapes:
                        s.x += dx
                        s.y += dy
                        if s.points:
                            s.points = [(px + dx, py + dy) for px, py in s.points]
                if self._raster_mask is not None:
                    self._raster_mask = _shift_mask(self._raster_mask, dx, dy)
                elif self._mask is not None:
                    self._mask = _shift_mask(self._mask, dx, dy)
                self._push_history()
                self._rebuild_all()
                self.mask_changed.emit()
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Space:
            self._space_down = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        if not self._zoomable or self._pixmap.isNull():
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
        if self._pixmap.isNull():
            super().mousePressEvent(event)
            return

        pan = (
            event.button() == Qt.MouseButton.MiddleButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_down)
            or (
                event.button() == Qt.MouseButton.LeftButton
                and self._zoomable
                and not self._selectable
                and self._tool is None
            )
        )
        if pan and self._zoomable:
            self._commit_fit_to_absolute()
            self._panning = True
            self._pan_origin = event.position()
            self._pan_start = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._tool in MASK_TOOLS:
            img_pt = self._widget_to_image_pt(event.position())
            if img_pt is None:
                super().mousePressEvent(event)
                return

            target = self._content_rect()
            
            # 1. Check handle click on currently selected shape first
            handle = self._hit_test_selected_shape_handles(event.position(), target)
            if handle != HANDLE_NONE:
                self._active_handle = handle
                self._drag_start_img_pt = img_pt
                self._drag_start_shape = copy.deepcopy(self._shapes[self._selected_index])
                self._drag_start_shapes = [copy.deepcopy(s) for s in self._shapes]
                self._drag_start_raster = self._raster_mask.copy() if self._raster_mask is not None else (self._mask.copy() if self._mask is not None else None)
                self.update()
                event.accept()
                return

            # 2. Check shape body click to select existing shape
            hit_shape_idx = self._hit_test_shape_bodies(img_pt)
            if hit_shape_idx >= 0:
                self._selected_index = hit_shape_idx
                self._active_handle = HANDLE_BODY
                self._drag_start_img_pt = img_pt
                self._drag_start_shape = copy.deepcopy(self._shapes[self._selected_index])
                self._drag_start_shapes = [copy.deepcopy(s) for s in self._shapes]
                self._drag_start_raster = self._raster_mask.copy() if self._raster_mask is not None else (self._mask.copy() if self._mask is not None else None)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._emit_shape_selected()
                self.update()
                event.accept()
                return

            # 3. If in "move" mode and user clicks ANYWHERE on canvas:
            if self._tool == "move":
                if self._shapes:
                    if self._selected_index < 0:
                        self._selected_index = 0
                        self._emit_shape_selected()
                    self._active_handle = HANDLE_BODY
                    self._drag_start_img_pt = img_pt
                    self._drag_start_shape = copy.deepcopy(self._shapes[self._selected_index])
                    self._drag_start_shapes = [copy.deepcopy(s) for s in self._shapes]
                    self._drag_start_raster = self._raster_mask.copy() if self._raster_mask is not None else (self._mask.copy() if self._mask is not None else None)
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.update()
                    event.accept()
                    return
                elif self._raster_mask is not None or self._mask is not None:
                    self._active_handle = HANDLE_BODY
                    self._drag_start_img_pt = img_pt
                    self._drag_start_shapes = []
                    self._drag_start_raster = self._raster_mask.copy() if self._raster_mask is not None else self._mask.copy()
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.update()
                    event.accept()
                    return

            # 4. Drawing tools
            if self._tool == "polygon":
                self._poly_pts.append((img_pt.x(), img_pt.y()))
                self.update()
            elif self._tool == "freeform":
                self._freeform_pts = [(img_pt.x(), img_pt.y())]
                self.update()
            elif self._tool in {"brush", "eraser"}:
                self._stroke_img = [(img_pt.x(), img_pt.y())]
                self.update()
            elif self._tool in DRAW_BOX_TOOLS:
                self._drag_origin_widget = (event.position().toPoint().x(), event.position().toPoint().y())
                self._drag_current_widget = self._drag_origin_widget
                self._selected_index = -1
                self._emit_shape_selected()
                self.update()

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._panning:
            delta = event.position() - self._pan_origin
            self._pan = self._pan_start + delta
            self.update()
            return

        img_pt = self._widget_to_image_pt(event.position())

        # Handle active interactive transformation / dragging
        if self._active_handle != HANDLE_NONE and self._drag_start_img_pt is not None and img_pt is not None:
            self._apply_drag_transform(img_pt, event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.update()
            return

        # Handle brush / eraser stroke
        if self._tool in {"brush", "eraser"} and self._stroke_img and img_pt is not None:
            self._stroke_img.append((img_pt.x(), img_pt.y()))
            self.update()
            return

        # Handle freeform stroke
        if self._tool == "freeform" and self._freeform_pts and img_pt is not None:
            self._freeform_pts.append((img_pt.x(), img_pt.y()))
            self.update()
            return

        # Handle box drawing drag
        if self._drag_origin_widget is not None:
            self._drag_current_widget = (event.position().toPoint().x(), event.position().toPoint().y())
            self.update()
            return

        # Update hover cursor over handles / shapes
        target = self._content_rect()
        hover_handle = self._hit_test_selected_shape_handles(event.position(), target)
        self._update_hover_cursor(hover_handle)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._panning and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._update_cursor()

        if event.button() == Qt.MouseButton.LeftButton:
            if self._active_handle != HANDLE_NONE:
                self._active_handle = HANDLE_NONE
                self._drag_start_img_pt = None
                self._drag_start_shape = None
                self._drag_start_shapes = []
                self._drag_start_raster = None
                self._push_history()
                self._rebuild_all()
                self.mask_changed.emit()
                self._update_cursor()
                self.update()
            elif self._drag_origin_widget and self._drag_current_widget and self._tool in DRAW_BOX_TOOLS:
                self._commit_box_tool()
                self._drag_origin_widget = None
                self._drag_current_widget = None
                self.update()
            elif self._tool == "freeform" and len(self._freeform_pts) >= 3:
                self._commit_freeform()
                self._freeform_pts = []
                self.update()
            elif self._tool in {"brush", "eraser"} and self._stroke_img:
                self._commit_brush()
                self._stroke_img = []
                self.update()

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        if self._tool == "polygon" and len(self._poly_pts) >= 3:
            self._commit_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._emit_zoom()

    # ------------------------------------------------------------------
    # Interactive Transform Engine
    # ------------------------------------------------------------------

    def _apply_drag_transform(self, img_pt: QPointF, shift_held: bool) -> None:
        if self._drag_start_img_pt is None:
            return

        handle = self._active_handle

        if handle == HANDLE_BODY:
            dx = img_pt.x() - self._drag_start_img_pt.x()
            dy = img_pt.y() - self._drag_start_img_pt.y()

            # If moving the selected shape
            if 0 <= self._selected_index < len(self._shapes) and self._drag_start_shape is not None:
                s = self._shapes[self._selected_index]
                s.x = self._drag_start_shape.x + dx
                s.y = self._drag_start_shape.y + dy
                if self._drag_start_shape.points:
                    s.points = [(px + dx, py + dy) for px, py in self._drag_start_shape.points]
                self.shape_updated.emit(s.to_dict())

            # If moving all shapes
            elif self._drag_start_shapes:
                for idx, s_init in enumerate(self._drag_start_shapes):
                    if idx < len(self._shapes):
                        s = self._shapes[idx]
                        s.x = s_init.x + dx
                        s.y = s_init.y + dy
                        if s_init.points:
                            s.points = [(px + dx, py + dy) for px, py in s_init.points]

            # If raster mask is also present
            if self._drag_start_raster is not None:
                self._raster_mask = _shift_mask(self._drag_start_raster, dx, dy)

            self._rebuild_all()
            self.mask_changed.emit()
            return

        if self._selected_index < 0 or self._drag_start_shape is None:
            return

        s = self._shapes[self._selected_index]
        s_init = self._drag_start_shape

        cx_0 = s_init.x + s_init.width / 2.0
        cy_0 = s_init.y + s_init.height / 2.0

        if handle == HANDLE_ROT:
            ang_rad = math.atan2(img_pt.y() - cy_0, img_pt.x() - cx_0)
            ang_deg = math.degrees(ang_rad) + 90.0
            if shift_held:
                ang_deg = round(ang_deg / 15.0) * 15.0
            # Normalize to -180..180
            ang_deg = (ang_deg + 180.0) % 360.0 - 180.0
            s.rotation = ang_deg
            self._rebuild_all()
            self.shape_updated.emit(s.to_dict())
            self.mask_changed.emit()
            return

        if handle == HANDLE_RADIUS:
            rad_angle = math.radians(s_init.rotation)
            dx = img_pt.x() - cx_0
            dy = img_pt.y() - cy_0
            lx = dx * math.cos(-rad_angle) - dy * math.sin(-rad_angle)
            ly = dx * math.sin(-rad_angle) + dy * math.cos(-rad_angle)
            tl_x = -s_init.width / 2.0
            tl_y = -s_init.height / 2.0
            dist_r = max(0.0, min(lx - tl_x, ly - tl_y))
            s.corner_radius = min(dist_r, min(s.width, s.height) / 2.0)
            self._rebuild_all()
            self.shape_updated.emit(s.to_dict())
            self.mask_changed.emit()
            return

        if handle >= HANDLE_VERTEX_BASE:
            v_idx = handle - HANDLE_VERTEX_BASE
            if 0 <= v_idx < len(s.points):
                s.points[v_idx] = (img_pt.x(), img_pt.y())
                xs = [p[0] for p in s.points]
                ys = [p[1] for p in s.points]
                s.x = min(xs)
                s.y = min(ys)
                s.width = max(4.0, max(xs) - s.x)
                s.height = max(4.0, max(ys) - s.y)
                self._rebuild_all()
                self.shape_updated.emit(s.to_dict())
                self.mask_changed.emit()
                return

        # 8-Handle Resize Transform
        rad_angle = math.radians(s_init.rotation)
        dx = img_pt.x() - cx_0
        dy = img_pt.y() - cy_0
        lx = dx * math.cos(-rad_angle) - dy * math.sin(-rad_angle)
        ly = dx * math.sin(-rad_angle) + dy * math.cos(-rad_angle)

        w0 = s_init.width
        h0 = s_init.height
        L0 = -w0 / 2.0
        R0 = w0 / 2.0
        T0 = -h0 / 2.0
        B0 = h0 / 2.0

        L, R, T, B = L0, R0, T0, B0

        if handle in (HANDLE_NW, HANDLE_W, HANDLE_SW):
            L = min(R0 - 4.0, lx)
        if handle in (HANDLE_NE, HANDLE_E, HANDLE_SE):
            R = max(L0 + 4.0, lx)
        if handle in (HANDLE_NW, HANDLE_N, HANDLE_NE):
            T = min(B0 - 4.0, ly)
        if handle in (HANDLE_SW, HANDLE_S, HANDLE_SE):
            B = max(T0 + 4.0, ly)

        nw = R - L
        nh = B - T

        # Aspect lock / circle rule
        lock = s_init.lock_aspect or shift_held or (s_init.shape_type == "circle")
        if lock and w0 > 0 and h0 > 0:
            aspect = w0 / h0
            if handle in (HANDLE_N, HANDLE_S):
                nw = nh * aspect
                L = (L + R) / 2.0 - nw / 2.0
                R = L + nw
            elif handle in (HANDLE_E, HANDLE_W):
                nh = nw / aspect
                T = (T + B) / 2.0 - nh / 2.0
                B = T + nh
            else:
                diag = max(nw, nh * aspect)
                nw = diag
                nh = diag / aspect
                if handle in (HANDLE_NW, HANDLE_W, HANDLE_SW):
                    L = R - nw
                else:
                    R = L + nw
                if handle in (HANDLE_NW, HANDLE_N, HANDLE_NE):
                    T = B - nh
                else:
                    B = T + nh

        if s.shape_type == "circle":
            side = max(nw, nh)
            nw = side
            nh = side
            s.corner_radius = side / 2.0
        elif s.shape_type == "pill":
            s.corner_radius = min(nw, nh) / 2.0

        # Map local center change back to image coordinates
        dlx_center = (L + R) / 2.0
        dly_center = (T + B) / 2.0

        new_cx = cx_0 + dlx_center * math.cos(rad_angle) - dly_center * math.sin(rad_angle)
        new_cy = cy_0 + dlx_center * math.sin(rad_angle) + dly_center * math.cos(rad_angle)

        s.x = new_cx - nw / 2.0
        s.y = new_cy - nh / 2.0
        s.width = nw
        s.height = nh

        self._rebuild_all()
        self.shape_updated.emit(s.to_dict())
        self.mask_changed.emit()

    # ------------------------------------------------------------------
    # Hit Testing
    # ------------------------------------------------------------------

    def _hit_test_selected_shape_handles(self, widget_pos: QPointF, target: QRectF) -> int:
        shape = self.selected_shape()
        if shape is None:
            return HANDLE_NONE

        c_wpt = self._image_to_widget_pt(shape.center(), target)
        rad_rot = math.radians(shape.rotation)

        dx = widget_pos.x() - c_wpt.x()
        dy = widget_pos.y() - c_wpt.y()
        lx_w = dx * math.cos(-rad_rot) - dy * math.sin(-rad_rot)
        ly_w = dx * math.sin(-rad_rot) + dy * math.cos(-rad_rot)

        w_w = self._image_px_to_widget(shape.width, target)
        h_w = self._image_px_to_widget(shape.height, target)
        tol = 10.0  # screen pixels tolerance

        # 1. Rotation handle
        rot_handle_dist = 26.0
        if math.hypot(lx_w - 0.0, ly_w - (-h_w / 2.0 - rot_handle_dist)) <= tol:
            return HANDLE_ROT

        # 2. Corner radius handle for roundrect
        if shape.shape_type == "roundrect":
            rad_w = self._image_px_to_widget(min(shape.corner_radius, min(shape.width, shape.height) / 2.0), target)
            if math.hypot(lx_w - (-w_w / 2.0 + rad_w), ly_w - (-h_w / 2.0 + rad_w)) <= tol:
                return HANDLE_RADIUS

        # 3. Polygon / Freeform vertex handles
        if shape.shape_type in ("polygon", "freeform"):
            for idx, (px, py) in enumerate(shape.points):
                wpt = self._image_to_widget_pt(QPointF(px, py), target)
                vlx = wpt.x() - c_wpt.x()
                vly = wpt.y() - c_wpt.y()
                if math.hypot(lx_w - vlx, ly_w - vly) <= tol:
                    return HANDLE_VERTEX_BASE + idx

        # 4. 8 Scale handles
        handles = [
            (HANDLE_NW, -w_w / 2.0, -h_w / 2.0),
            (HANDLE_N,  0.0,        -h_w / 2.0),
            (HANDLE_NE, w_w / 2.0,  -h_w / 2.0),
            (HANDLE_E,  w_w / 2.0,  0.0),
            (HANDLE_SE, w_w / 2.0,  h_w / 2.0),
            (HANDLE_S,  0.0,        h_w / 2.0),
            (HANDLE_SW, -w_w / 2.0, h_w / 2.0),
            (HANDLE_W,  -w_w / 2.0, 0.0),
        ]
        for h_id, hx, hy in handles:
            if abs(lx_w - hx) <= tol and abs(ly_w - hy) <= tol:
                return h_id

        # 5. Body hit
        if abs(lx_w) <= w_w / 2.0 and abs(ly_w) <= h_w / 2.0:
            return HANDLE_BODY

        return HANDLE_NONE

    def _hit_test_shape_bodies(self, img_pt: QPointF) -> int:
        for idx in range(len(self._shapes) - 1, -1, -1):
            s = self._shapes[idx]
            cx = s.x + s.width / 2.0
            cy = s.y + s.height / 2.0
            rad_rot = math.radians(s.rotation)
            dx = img_pt.x() - cx
            dy = img_pt.y() - cy
            lx = dx * math.cos(-rad_rot) - dy * math.sin(-rad_rot)
            ly = dx * math.sin(-rad_rot) + dy * math.cos(-rad_rot)

            # Generous body hit testing with 8px margin
            if abs(lx) <= (s.width / 2.0 + 8.0) and abs(ly) <= (s.height / 2.0 + 8.0):
                return idx
        return -1

    def _update_hover_cursor(self, handle: int) -> None:
        if self._space_down or self._panning:
            return
        if handle == HANDLE_ROT:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif handle == HANDLE_RADIUS:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif handle in (HANDLE_NW, HANDLE_SE):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in (HANDLE_NE, HANDLE_SW):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in (HANDLE_N, HANDLE_S):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in (HANDLE_E, HANDLE_W):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif handle == HANDLE_BODY or self._tool == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif handle >= HANDLE_VERTEX_BASE:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._update_cursor()

    # ------------------------------------------------------------------
    # Shape Commits from Drawing Tools
    # ------------------------------------------------------------------

    def _commit_box_tool(self) -> None:
        rect = self._image_rect_from_drag()
        if rect is None or self._tool not in DRAW_BOX_TOOLS:
            return
        x, y, w, h = rect

        if self._tool == "circle":
            side = max(w, h)
            new_shape = MaskShape("circle", x, y, side, side, lock_aspect=True)
        elif self._tool == "pill":
            new_shape = MaskShape("pill", x, y, w, h, corner_radius=min(w, h) / 2.0)
        elif self._tool == "roundrect":
            new_shape = MaskShape("roundrect", x, y, w, h, corner_radius=max(6.0, 0.22 * min(w, h)))
        elif self._tool == "ellipse":
            new_shape = MaskShape("ellipse", x, y, w, h)
        else:
            new_shape = MaskShape("rect", x, y, w, h)

        self._shapes.append(new_shape)
        self._selected_index = len(self._shapes) - 1
        self._push_history()
        self._rebuild_all()
        self._emit_shape_selected()
        self.mask_changed.emit()

    def _commit_polygon(self) -> None:
        if len(self._poly_pts) < 3:
            return
        pts = list(self._poly_pts)
        self._poly_pts = []
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bx, by = min(xs), min(ys)
        bw, bh = max(4.0, max(xs) - bx), max(4.0, max(ys) - by)
        new_shape = MaskShape("polygon", bx, by, bw, bh, points=pts)
        self._shapes.append(new_shape)
        self._selected_index = len(self._shapes) - 1
        self._push_history()
        self._rebuild_all()
        self._emit_shape_selected()
        self.mask_changed.emit()

    def _commit_freeform(self) -> None:
        if len(self._freeform_pts) < 3:
            return
        pts = list(self._freeform_pts)
        self._freeform_pts = []
        step = max(1, len(pts) // 36)
        sub_pts = pts[::step]
        if sub_pts[-1] != pts[-1]:
            sub_pts.append(pts[-1])
        xs = [p[0] for p in sub_pts]
        ys = [p[1] for p in sub_pts]
        bx, by = min(xs), min(ys)
        bw, bh = max(4.0, max(xs) - bx), max(4.0, max(ys) - by)
        new_shape = MaskShape("freeform", bx, by, bw, bh, points=sub_pts)
        self._shapes.append(new_shape)
        self._selected_index = len(self._shapes) - 1
        self._push_history()
        self._rebuild_all()
        self._emit_shape_selected()
        self.mask_changed.emit()

    def _commit_brush(self) -> None:
        if len(self._stroke_img) < 1:
            return
        w, h = self._canvas_size()
        radius = float(self.brush_radius)
        pts = list(self._stroke_img)

        def draw(p: QPainter) -> None:
            pen = QPen(
                QColor(255, 255, 255),
                radius * 2.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if len(pts) == 1:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(QPointF(pts[0][0], pts[0][1]), radius, radius)
                return
            path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
            for px, py in pts[1:]:
                path.lineTo(QPointF(px, py))
            p.drawPath(path)

        stroke = _rasterize_shape(h, w, draw, 3)
        if self._raster_mask is None:
            self._raster_mask = np.zeros((h, w), dtype=np.float32)
        elif self._raster_mask.shape != (h, w):
            self._raster_mask = cv2.resize(self._raster_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        if self._tool == "eraser":
            self._raster_mask = np.clip(self._raster_mask * (1.0 - stroke), 0.0, 1.0)
        else:
            self._raster_mask = np.clip(self._raster_mask + stroke, 0.0, 1.0)

        self._push_history()
        self._rebuild_all()
        self.mask_changed.emit()

    # ------------------------------------------------------------------
    # Rasterization & Overlay Cache
    # ------------------------------------------------------------------

    def _rasterize_full_mask(self) -> np.ndarray:
        w, h = self._canvas_size()
        if w < 1 or h < 1:
            return np.zeros((1, 1), dtype=np.float32)
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))

        for s in self._shapes:
            p.save()
            cx = s.x + s.width / 2.0
            cy = s.y + s.height / 2.0
            p.translate(cx, cy)
            p.rotate(s.rotation)
            rect = QRectF(-s.width / 2.0, -s.height / 2.0, s.width, s.height)

            if s.shape_type == "circle":
                r = min(s.width, s.height) / 2.0
                p.drawEllipse(QRectF(-r, -r, 2 * r, 2 * r))
            elif s.shape_type == "ellipse":
                p.drawEllipse(rect)
            elif s.shape_type == "pill":
                rad = min(s.width, s.height) / 2.0
                path = QPainterPath()
                path.addRoundedRect(rect, rad, rad)
                p.drawPath(path)
            elif s.shape_type == "roundrect":
                rad = min(s.corner_radius, min(s.width, s.height) / 2.0)
                path = QPainterPath()
                path.addRoundedRect(rect, rad, rad)
                p.drawPath(path)
            elif s.shape_type in ("polygon", "freeform") and len(s.points) >= 3:
                path = QPainterPath(QPointF(s.points[0][0] - cx, s.points[0][1] - cy))
                for pt in s.points[1:]:
                    path.lineTo(QPointF(pt[0] - cx, pt[1] - cy))
                path.closeSubpath()
                p.drawPath(path)
            else:
                p.drawRect(rect)
            p.restore()

        p.end()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        mask = (arr[..., 3].astype(np.float32) / 255.0).copy()

        if self._raster_mask is not None:
            if self._raster_mask.shape != (h, w):
                self._raster_mask = cv2.resize(self._raster_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask + self._raster_mask, 0.0, 1.0)

        return mask

    def _rebuild_all(self) -> None:
        if self._pixmap.isNull():
            self._overlay = QPixmap()
            return
        if self._raster_mask is not None and np.any(self._raster_mask > 0.01):
            w, h = self._canvas_size()
            if self._raster_mask.shape != (h, w):
                self._raster_mask = cv2.resize(self._raster_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = 40
            rgba[..., 1] = 170
            rgba[..., 2] = 255
            rgba[..., 3] = np.clip(self._raster_mask * 120.0, 0, 255).astype(np.uint8)
            self._overlay = QPixmap.fromImage(numpy_rgba_to_qimage(rgba))
        else:
            self._overlay = QPixmap()

    def _emit_shape_selected(self) -> None:
        shape = self.selected_shape()
        if shape is not None:
            self.shape_selected.emit(shape.to_dict())
        else:
            self.shape_selected.emit(None)

    def _push_history(self) -> None:
        shapes_copy = [copy.deepcopy(s) for s in self._shapes]
        raster_copy = self._raster_mask.copy() if self._raster_mask is not None else None
        self._history = self._history[: self._history_index + 1]
        self._history.append((shapes_copy, raster_copy))
        if len(self._history) > 32:
            self._history = self._history[-32:]
        self._history_index = len(self._history) - 1

    def _fit_mask_to_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        w, h = int(self._pixmap.width()), int(self._pixmap.height())
        if self._raster_mask is not None and self._raster_mask.shape != (h, w):
            self._raster_mask = cv2.resize(self._raster_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    def _canvas_size(self) -> tuple[int, int]:
        if not self._pixmap.isNull():
            return int(self._pixmap.width()), int(self._pixmap.height())
        return 800, 1600

    def _update_cursor(self) -> None:
        if self._space_down or (self._zoomable and not self._tool and not self._selectable):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._tool == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._tool in MASK_TOOLS or self._selectable:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Coordinate Conversions
    # ------------------------------------------------------------------

    def _widget_to_image_pt(self, pos: QPointF) -> QPointF | None:
        if self._pixmap.isNull():
            return None
        rect = self._content_rect()
        if rect.width() < 1 or rect.height() < 1:
            return None
        cw, ch = self._canvas_size()
        x = (pos.x() - rect.x()) / rect.width() * cw
        y = (pos.y() - rect.y()) / rect.height() * ch
        return QPointF(x, y)

    def _image_to_widget_pt(self, pt: QPointF, target: QRectF) -> QPointF:
        cw, ch = self._canvas_size()
        x = target.x() + (pt.x() / max(cw, 1)) * target.width()
        y = target.y() + (pt.y() / max(ch, 1)) * target.height()
        return QPointF(x, y)

    def _image_px_to_widget(self, px: float, target: QRectF) -> float:
        return float(px) * target.width() / max(self._canvas_size()[0], 1)

    def _image_rect_from_drag(self) -> tuple[float, float, float, float] | None:
        if self._pixmap.isNull() or self._drag_origin_widget is None or self._drag_current_widget is None:
            return None
        fitted = self._content_rect()
        x0, y0 = self._drag_origin_widget
        x1, y1 = self._drag_current_widget
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 4 or bottom - top < 4:
            return None
        cw, ch = self._canvas_size()
        ix0 = (left - fitted.x()) / max(fitted.width(), 1) * cw
        iy0 = (top - fitted.y()) / max(fitted.height(), 1) * ch
        ix1 = (right - fitted.x()) / max(fitted.width(), 1) * cw
        iy1 = (bottom - fitted.y()) / max(fitted.height(), 1) * ch
        ix0 = float(np.clip(ix0, 0, cw))
        iy0 = float(np.clip(iy0, 0, ch))
        ix1 = float(np.clip(ix1, 0, cw))
        iy1 = float(np.clip(iy1, 0, ch))
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


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a mask in image pixels without wrapping around the frame."""
    ox = int(round(dx))
    oy = int(round(dy))
    if ox == 0 and oy == 0:
        return mask.copy()
    h, w = mask.shape[:2]
    out = np.zeros_like(mask)
    ysrc0 = max(0, -oy)
    ysrc1 = min(h, h - oy)
    xsrc0 = max(0, -ox)
    xsrc1 = min(w, w - ox)
    if ysrc1 <= ysrc0 or xsrc1 <= xsrc0:
        return out
    ydst0 = max(0, oy)
    xdst0 = max(0, ox)
    ydst1 = ydst0 + (ysrc1 - ysrc0)
    xdst1 = xdst0 + (xsrc1 - xsrc0)
    out[ydst0:ydst1, xdst0:xdst1] = mask[ysrc0:ysrc1, xsrc0:xsrc1]
    return out


def _rasterize_shape(height: int, width: int, draw: Callable[[QPainter], None], ss: int) -> np.ndarray:
    img = QImage(width * ss, height * ss, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(ss, ss)
    draw(painter)
    painter.end()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height * ss, width * ss, 4).copy()
    alpha = arr[..., 3].astype(np.float32) / 255.0
    if ss == 1:
        return alpha
    return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)


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
