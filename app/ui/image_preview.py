"""Zoomable mockup viewer with interactive camera-mask drawing."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from app.utils.image_utils import numpy_rgba_to_qimage

MASK_TOOLS = ("move", "rect", "roundrect", "ellipse", "polygon", "brush", "eraser")
DRAW_BOX_TOOLS = ("rect", "roundrect", "ellipse")


class ImagePreview(QWidget):
    """Scaled image viewer with a checkerboard behind transparent pixels."""

    region_selected = Signal(float, float, float, float)
    zoom_changed = Signal(int)
    mask_changed = Signal()

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
        self._tool: str | None = None
        self._mask: np.ndarray | None = None
        self._overlay = QPixmap()
        self._history: list[np.ndarray] = []
        self._history_index = -1
        self._stroke_img: list[tuple[float, float]] = []
        self._poly_pts: list[tuple[float, float]] = []
        self._space_down = False
        self._moving = False
        self._move_origin: QPointF | None = None
        self._move_start_mask: np.ndarray | None = None
        self.brush_radius = 14.0
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
        self._update_cursor()
        self.update()

    def set_tool(self, tool: str | None) -> None:
        self._tool = tool if tool in MASK_TOOLS else None
        self._drag_origin = None
        self._drag_current = None
        self._stroke_img = []
        self._moving = False
        self._move_origin = None
        self._move_start_mask = None
        self._update_cursor()
        self.update()

    def tool(self) -> str | None:
        return self._tool

    def set_camera_mask(self, mask: np.ndarray | None, *, remember: bool = True) -> None:
        if mask is None:
            self._mask = None
            self._history = []
            self._history_index = -1
            self._overlay = QPixmap()
            self.update()
            return
        arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if arr.ndim == 3:
            arr = arr[..., 0]
        self._mask = arr
        self._fit_mask_to_pixmap()
        if remember:
            self._push_history()
        self._rebuild_overlay()
        self.update()

    def camera_mask(self) -> np.ndarray | None:
        return None if self._mask is None else self._mask.copy()

    def clear_mask(self) -> None:
        if self._mask is None:
            return
        self._mask = np.zeros_like(self._mask)
        self._poly_pts = []
        self._stroke_img = []
        self._push_history()
        self._rebuild_overlay()
        self.mask_changed.emit()
        self.update()

    def undo_mask(self) -> None:
        if self._history_index <= 0:
            return
        self._history_index -= 1
        self._mask = self._history[self._history_index].copy()
        self._rebuild_overlay()
        self.mask_changed.emit()
        self.update()

    def redo_mask(self) -> None:
        if self._history_index < 0 or self._history_index >= len(self._history) - 1:
            return
        self._history_index += 1
        self._mask = self._history[self._history_index].copy()
        self._rebuild_overlay()
        self.mask_changed.emit()
        self.update()

    def reset_session(self) -> None:
        """Clear image, mask editor, zoom/pan, and drawing state."""
        self._pixmap = QPixmap()
        self._selectable = False
        self._zoom_mode = "fit"
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_origin = QPointF()
        self._pan_start = QPointF()
        self._drag_origin = None
        self._drag_current = None
        self._tool = None
        self._mask = None
        self._overlay = QPixmap()
        self._history = []
        self._history_index = -1
        self._stroke_img = []
        self._poly_pts = []
        self._space_down = False
        self._moving = False
        self._move_origin = None
        self._move_start_mask = None
        self.brush_radius = 14.0
        self._update_cursor()
        self.update()
        self._emit_zoom()

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = QPixmap()
        else:
            self._pixmap = QPixmap.fromImage(numpy_rgba_to_qimage(image))
        self._fit_mask_to_pixmap()
        if self._mask is not None:
            self._rebuild_overlay()
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1c2028"))

        if self._pixmap.isNull():
            painter.setPen(QColor("#6b7380"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        target = self._content_rect()
        self._draw_checkerboard(painter, target.toAlignedRect())
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        if not self._overlay.isNull() and self._tool:
            painter.setOpacity(1.0)
            painter.drawPixmap(target, self._overlay, QRectF(self._overlay.rect()))
        self._paint_live_stroke(painter, target)
        if self._drag_origin and self._drag_current and self._tool in DRAW_BOX_TOOLS:
            x0, y0 = self._drag_origin
            x1, y1 = self._drag_current
            box = QRectF(float(min(x0, x1)), float(min(y0, y1)), float(abs(x1 - x0)), float(abs(y1 - y0)))
            painter.setPen(QPen(QColor("#3d9cf0"), 1.5))
            painter.fillRect(box, QColor(61, 156, 240, 50))
            if self._tool == "ellipse":
                painter.drawEllipse(box)
            elif self._tool == "roundrect":
                path = QPainterPath()
                rad = 0.22 * min(box.width(), box.height())
                path.addRoundedRect(box, rad, rad)
                painter.drawPath(path)
            else:
                painter.drawRect(box)

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

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Space:
            self._space_down = True
            self._update_cursor()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Return and self._tool == "polygon" and len(self._poly_pts) >= 3:
            self._commit_polygon()
            event.accept()
            return
        if self._tool == "move" and self._mask is not None:
            step = 5 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            dx, dy = 0, 0
            if event.key() == Qt.Key.Key_Left:
                dx = -step
            elif event.key() == Qt.Key.Key_Right:
                dx = step
            elif event.key() == Qt.Key.Key_Up:
                dy = -step
            elif event.key() == Qt.Key.Key_Down:
                dy = step
            if dx or dy:
                self._nudge_mask(dx, dy)
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
            self._ensure_mask()
            img_pt = self._widget_to_image(event.position())
            if img_pt is None:
                super().mousePressEvent(event)
                return
            if self._tool == "move":
                if self._mask is None or not np.any(self._mask > 0.12):
                    super().mousePressEvent(event)
                    return
                self._moving = True
                self._move_origin = img_pt
                self._move_start_mask = self._mask.copy()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            if self._tool == "polygon":
                self._poly_pts.append((img_pt.x(), img_pt.y()))
                self.update()
            elif self._tool in {"brush", "eraser"}:
                self._stroke_img = [(img_pt.x(), img_pt.y())]
                self.update()
            else:
                self._drag_origin = (event.position().toPoint().x(), event.position().toPoint().y())
                self._drag_current = self._drag_origin
                self.update()
            event.accept()
            return
        if self._selectable and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = (event.position().toPoint().x(), event.position().toPoint().y())
            self._drag_current = self._drag_origin
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._panning:
            delta = event.position() - self._pan_origin
            self._pan = self._pan_start + delta
            self.update()
        elif self._moving and self._move_origin is not None and self._move_start_mask is not None:
            img_pt = self._widget_to_image(event.position())
            if img_pt is not None:
                dx = img_pt.x() - self._move_origin.x()
                dy = img_pt.y() - self._move_origin.y()
                self._mask = _shift_mask(self._move_start_mask, dx, dy)
                self._rebuild_overlay()
                self.update()
        elif self._tool in {"brush", "eraser"} and self._stroke_img:
            img_pt = self._widget_to_image(event.position())
            if img_pt is not None:
                self._stroke_img.append((img_pt.x(), img_pt.y()))
                self.update()
        elif self._drag_origin is not None:
            self._drag_current = (event.position().toPoint().x(), event.position().toPoint().y())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._panning and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._update_cursor()
        if event.button() == Qt.MouseButton.LeftButton and self._tool == "move" and self._moving:
            self._moving = False
            self._move_origin = None
            self._move_start_mask = None
            self._push_history()
            self.mask_changed.emit()
            self._update_cursor()
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and self._tool in DRAW_BOX_TOOLS:
            if self._drag_origin and self._drag_current:
                self._commit_box_tool()
            self._drag_origin = None
            self._drag_current = None
            self.update()
        elif event.button() == Qt.MouseButton.LeftButton and self._tool in {"brush", "eraser"}:
            if self._stroke_img:
                self._commit_brush()
            self._stroke_img = []
            self.update()
        elif self._selectable and self._drag_origin and self._drag_current:
            rect = self._image_rect_from_drag()
            self._drag_origin = None
            self._drag_current = None
            self.update()
            if rect is not None:
                self.region_selected.emit(*rect)
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

    def _paint_live_stroke(self, painter: QPainter, target: QRectF) -> None:
        if not self._tool:
            return
        painter.save()
        pen = QPen(QColor("#5eb0ff"), 1.6)
        painter.setPen(pen)
        painter.setBrush(QColor(94, 176, 255, 70))
        if self._tool == "polygon" and self._poly_pts:
            pts = [self._image_to_widget(QPointF(x, y), target) for x, y in self._poly_pts]
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)
        if self._tool in {"brush", "eraser"} and len(self._stroke_img) >= 1:
            color = QColor(255, 90, 90, 90) if self._tool == "eraser" else QColor(94, 176, 255, 90)
            painter.setPen(QPen(color, max(2.0, self._image_px_to_widget(self.brush_radius * 2, target))))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pts = [self._image_to_widget(QPointF(x, y), target) for x, y in self._stroke_img]
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            painter.drawPath(path)
        painter.restore()

    def _commit_box_tool(self) -> None:
        box = self._image_rect_from_drag()
        if box is None or self._mask is None:
            return
        x, y, w, h = box
        ss = 4
        mh, mw = self._mask.shape

        def draw(p: QPainter) -> None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            rect = QRectF(x, y, w, h)
            if self._tool == "ellipse":
                p.drawEllipse(rect)
            elif self._tool == "roundrect":
                rad = 0.22 * min(w, h)
                path = QPainterPath()
                path.addRoundedRect(rect, rad, rad)
                p.drawPath(path)
            else:
                p.drawRect(rect)

        stroke = _rasterize_shape(mh, mw, draw, ss)
        self._mask = np.clip(self._mask + stroke, 0.0, 1.0)
        self._push_history()
        self._rebuild_overlay()
        self.mask_changed.emit()

    def _commit_polygon(self) -> None:
        if self._mask is None or len(self._poly_pts) < 3:
            return
        pts = list(self._poly_pts)
        self._poly_pts = []
        mh, mw = self._mask.shape

        def draw(p: QPainter) -> None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
            for x, y in pts[1:]:
                path.lineTo(QPointF(x, y))
            path.closeSubpath()
            p.drawPath(path)

        stroke = _rasterize_shape(mh, mw, draw, 4)
        self._mask = np.clip(self._mask + stroke, 0.0, 1.0)
        self._push_history()
        self._rebuild_overlay()
        self.mask_changed.emit()
        self.update()

    def _commit_brush(self) -> None:
        if self._mask is None or len(self._stroke_img) < 1:
            return
        mh, mw = self._mask.shape
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
            for x, y in pts[1:]:
                path.lineTo(QPointF(x, y))
            p.drawPath(path)

        stroke = _rasterize_shape(mh, mw, draw, 3)
        if self._tool == "eraser":
            self._mask = np.clip(self._mask * (1.0 - stroke), 0.0, 1.0)
        else:
            self._mask = np.clip(self._mask + stroke, 0.0, 1.0)
        self._push_history()
        self._rebuild_overlay()
        self.mask_changed.emit()

    def _canvas_size(self) -> tuple[int, int]:
        if self._mask is not None:
            h, w = self._mask.shape
            return w, h
        return int(self._pixmap.width()), int(self._pixmap.height())

    def _fit_mask_to_pixmap(self) -> None:
        if self._mask is None or self._pixmap.isNull():
            return
        w = int(self._pixmap.width())
        h = int(self._pixmap.height())
        if w < 1 or h < 1 or self._mask.shape == (h, w):
            return
        self._mask = cv2.resize(self._mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def _ensure_mask(self) -> None:
        if self._pixmap.isNull():
            return
        self._fit_mask_to_pixmap()
        w, h = int(self._pixmap.width()), int(self._pixmap.height())
        if w < 1 or h < 1:
            return
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.float32)
            self._history = [self._mask.copy()]
            self._history_index = 0
            self._rebuild_overlay()

    def _push_history(self) -> None:
        if self._mask is None:
            return
        self._history = self._history[: self._history_index + 1]
        self._history.append(self._mask.copy())
        if len(self._history) > 32:
            self._history = self._history[-32:]
        self._history_index = len(self._history) - 1

    def _rebuild_overlay(self) -> None:
        if self._mask is None:
            self._overlay = QPixmap()
            return
        h, w = self._mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 40
        rgba[..., 1] = 170
        rgba[..., 2] = 255
        rgba[..., 3] = np.clip(self._mask * 120.0, 0, 255).astype(np.uint8)
        self._overlay = QPixmap.fromImage(numpy_rgba_to_qimage(rgba))

    def _update_cursor(self) -> None:
        if self._space_down or (self._zoomable and not self._tool and not self._selectable):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._tool == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._tool in MASK_TOOLS or self._selectable:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _nudge_mask(self, dx: int, dy: int) -> None:
        if self._mask is None:
            return
        self._mask = _shift_mask(self._mask, float(dx), float(dy))
        self._push_history()
        self._rebuild_overlay()
        self.mask_changed.emit()
        self.update()

    def _widget_to_image(self, pos: QPointF) -> QPointF | None:
        if self._pixmap.isNull():
            return None
        rect = self._content_rect()
        if rect.width() < 1 or rect.height() < 1:
            return None
        x = (pos.x() - rect.x()) / rect.width() * self._canvas_size()[0]
        y = (pos.y() - rect.y()) / rect.height() * self._canvas_size()[1]
        return QPointF(x, y)

    def _image_to_widget(self, pt: QPointF, target: QRectF) -> QPointF:
        cw, ch = self._canvas_size()
        x = target.x() + (pt.x() / max(cw, 1)) * target.width()
        y = target.y() + (pt.y() / max(ch, 1)) * target.height()
        return QPointF(x, y)

    def _image_px_to_widget(self, px: float, target: QRectF) -> float:
        return float(px) * target.width() / max(self._canvas_size()[0], 1)

    def _image_rect_from_drag(self) -> tuple[float, float, float, float] | None:
        if self._pixmap.isNull() or self._drag_origin is None or self._drag_current is None:
            return None
        fitted = self._content_rect()
        x0, y0 = self._drag_origin
        x1, y1 = self._drag_current
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
        return mask
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
    import cv2

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
