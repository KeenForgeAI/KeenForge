# src/ui/canvas.py
import math
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsTextItem, QGraphicsItem, QMenu, QStyle)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPen, QColor, QBrush, QFont, QCursor, QPainter, QFontMetrics, QFontMetricsF

class BoxItem(QGraphicsRectItem):
    """
    Custom Graphics Item representing a bounding box.
    Handles mouse interactions, resizing, and visual styles.
    """
    handle_size = 18.0
    margin = 35.0  # Sensitivity margin for mouse hover
    # Visual (screen-pixel) constant sizes - independent of view zoom
    visual_handle = 10.0     # corner handle edge length in screen px
    visual_margin = 15.0     # corner hover hit area in screen px

    def __init__(self, rect, cls_idx, class_names, parent=None,
                 is_ai=False, conf=None):
        super().__init__(rect, parent)
        self.cls_idx = cls_idx
        self.class_names = class_names
        # is_ai=True marks a MODEL PREDICTION the user has not reviewed yet.
        # Predictions are drawn in cyan, are never written to the label file,
        # and are promoted to normal human boxes as soon as the user edits them.
        self.is_ai = bool(is_ai)
        self.conf = float(conf) if conf is not None else None

        # Visual styles
        self.default_pen = QPen(self._base_color(), 5)
        self.selected_pen = QPen(QColor(255, 0, 0), 7)

        self.setPen(self.default_pen)
        self.setBrush(QBrush(self._fill_color(80)))

        self.setFlags(QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemIsMovable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # Label: painted in paint() like X-AnyLabeling (tightBoundingRect + baseline).
        # Keep hidden child items only for API compatibility.
        self.text_item = QGraphicsTextItem(self.get_label_text(), self)
        self.text_item.setDefaultTextColor(QColor(Qt.black))
        font = QFont("Arial", 10, QFont.Bold)
        self.text_item.setFont(font)
        self.text_item.setVisible(False)
        self.text_bg = QGraphicsRectItem(self.text_item.boundingRect(), self)
        self.text_bg.setBrush(QBrush(QColor(0, 255, 0, 200)))
        self.text_bg.setPen(QPen(Qt.NoPen))
        self.text_bg.setVisible(False)
        self.resizing = False
        self.resize_corner = None

    # ── Model-prediction support ────────────────────────────────────────
    def _base_color(self):
        """Cyan for an unreviewed model prediction, green for a human box."""
        return QColor(0, 188, 212) if self.is_ai else QColor(0, 255, 0)

    def _fill_color(self, alpha=80):
        c = self._base_color()
        c.setAlpha(alpha)
        return c

    def promote_to_human(self):
        """Mark this box as human-reviewed so it becomes saveable.

        Called as soon as the user edits a model prediction (move or resize):
        an edited prediction has effectively been reviewed and accepted.
        """
        if not self.is_ai:
            return
        self.is_ai = False
        self.setPen(QPen(self._base_color(), 5))
        self.setBrush(QBrush(self._fill_color(80)))

    def setRect(self, rect):
        # Resizing a prediction counts as reviewing it -> promote it.
        if getattr(self, 'is_ai', False) and self.rect() != rect:
            self.promote_to_human()
        super().setRect(rect)

    def itemChange(self, change, value):
        # Handle selection change
        if change == QGraphicsItem.ItemSelectedChange:
            if value == True:
                self.text_bg.setBrush(QBrush(QColor(255, 0, 0, 200)))
                self.setZValue(100)
            else:
                self.text_bg.setBrush(QBrush(self._fill_color(200)))
                self.setZValue(10)

        # Moving a model prediction counts as reviewing it -> promote it so it
        # becomes a normal, saveable human box.
        if change == QGraphicsItem.ItemPositionChange and getattr(self, 'is_ai', False):
            self.promote_to_human()

        # Constrain movement within image boundaries
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_pos = value
            try:
                rect = self.rect()
                max_w = 0
                max_h = 0
                for item in self.scene().items():
                    if isinstance(item, QGraphicsPixmapItem):
                        max_w = item.pixmap().width()
                        max_h = item.pixmap().height()
                        break
                if max_w == 0: return super().itemChange(change, value)

                if new_pos.x() + rect.left() < 0: new_pos.setX(-rect.left())
                elif new_pos.x() + rect.right() > max_w: new_pos.setX(max_w - rect.right())

                if new_pos.y() + rect.top() < 0: new_pos.setY(-rect.top())
                elif new_pos.y() + rect.bottom() > max_h: new_pos.setY(max_h - rect.bottom())

                return new_pos
            except:
                pass

        return super().itemChange(change, value)

    def boundingRect(self):
        rect = self.rect()
        extra = self.handle_size + 5.0
        return rect.adjusted(-extra, -extra, extra, extra)

    def get_label_text(self):
        if self.class_names and 0 <= self.cls_idx < len(self.class_names):
            name = self.class_names[self.cls_idx]
        else:
            name = str(self.cls_idx)
        # Model predictions carry a confidence; showing it lets the reviewer see
        # at a glance how sure the model was about each proposed box.
        if getattr(self, 'is_ai', False) and self.conf is not None:
            return f"{name} {self.conf:.2f}"
        return f"{name}"

    def _fit_label_font(self):
        """Shrink the label font until the RENDERED text + padding fits inside the box,
        so the text is always covered by its background pill (min 6pt, like X-AnyLabeling)."""
        r = self.rect()
        if r.width() < 14 or r.height() < 10:
            return
        size = 11
        while True:
            self.text_item.setFont(QFont("Arial", size))
            tb = self.text_item.boundingRect()
            if tb.width() + 8.0 <= r.width() and tb.height() + 4.0 <= r.height():
                break
            if size <= 6:
                break
            size -= 1
    def _label_bg_rect(self):
        """Label background rect: RENDERED text bounds + padding, clamped INSIDE the box,
        so the pill never overflows the annotation rectangle."""
        r = self.rect()
        tb = self.text_item.boundingRect()
        w = min(tb.width() + 8.0, max(0.0, r.width()))
        h = min(tb.height() + 4.0, max(0.0, r.height()))
        # rect is in the child item's local coords (origin 0,0); position is set via setPos
        return QRectF(0, 0, w, h)

    def update_classes(self):
        self.text_item.setPlainText(self.get_label_text())
        self._fit_label_font()
        self.text_bg.setRect(self._label_bg_rect())

    def update_text_pos(self):
        if self.rect().isValid():
            r = self.rect()
            self._fit_label_font()
            self.text_item.setPos(r.left(), r.top())
            # X-AnyLabeling-style label pill: text + 4/2px padding,
            # always kept inside the box bounds
            self.text_bg.setRect(self._label_bg_rect())
            self.text_bg.setPos(r.left(), r.top())

    def _view_scale(self):
        """Current view zoom factor (scene units per screen px), for constant-size handles."""
        try:
            if self.scene() and self.scene().views():
                return max(self.scene().views()[0].transform().m11(), 0.001)
        except Exception:
            pass
        return 1.0

    def check_cursor_on_corner(self, pos):
        rect = self.rect()
        x, y = pos.x(), pos.y()
        s = self._view_scale()
        m = self.visual_margin / s  # screen-constant hit area
        dist_tl = math.hypot(x - rect.left(), y - rect.top())
        dist_tr = math.hypot(x - rect.right(), y - rect.top())
        dist_bl = math.hypot(x - rect.left(), y - rect.bottom())
        dist_br = math.hypot(x - rect.right(), y - rect.bottom())

        if dist_tl < m: return 'TL'
        if dist_tr < m: return 'TR'
        if dist_bl < m: return 'BL'
        if dist_br < m: return 'BR'
        return None

    def hoverMoveEvent(self, event):
        if self.resizing: return
        try:
            corner = self.check_cursor_on_corner(event.pos())
            if corner:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.SizeAllCursor)
        except:
            pass
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            try:
                corner = self.check_cursor_on_corner(event.pos())
                if corner:
                    self.resize_corner = corner
                    self.resizing = True
                    self.setSelected(True)
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                else:
                    self.resizing = False
                    self.resize_corner = None
                    super().mousePressEvent(event)
            except:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing and self.resize_corner:
            try:
                self.prepareGeometryChange()
                max_w, max_h = 99999, 99999
                if self.scene() and len(self.scene().views()) > 0:
                    view = self.scene().views()[0]
                    if hasattr(view, 'image_w') and view.image_w > 0:
                        max_w, max_h = view.image_w, view.image_h

                mouse_scene = event.scenePos()
                mx = max(0, min(mouse_scene.x(), max_w))
                my = max(0, min(mouse_scene.y(), max_h))

                scene_shape = self.mapRectToScene(self.rect())
                curr_scene_rect = scene_shape.boundingRect() if hasattr(scene_shape, "boundingRect") else scene_shape
                l, r, t, b = curr_scene_rect.left(), curr_scene_rect.right(), curr_scene_rect.top(), curr_scene_rect.bottom()

                if self.resize_corner == 'TL': l, t = mx, my
                elif self.resize_corner == 'TR': r, t = mx, my
                elif self.resize_corner == 'BL': l, b = mx, my
                elif self.resize_corner == 'BR': r, b = mx, my

                if l > r: l, r = r, l
                if t > b: t, b = b, t

                new_scene_rect = QRectF(QPointF(l, t), QPointF(r, b)).normalized()
                local_shape = self.mapFromScene(new_scene_rect)
                new_local_rect = local_shape.boundingRect() if hasattr(local_shape, "boundingRect") else local_shape

                if new_local_rect.width() > 5 and new_local_rect.height() > 5:
                    self.setRect(new_local_rect)
                    self.update_text_pos()
            except:
                pass
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.resize_corner = None
        try:
            if self.check_cursor_on_corner(event.pos()):
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        except:
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def paint(self, painter, option, widget=None):
        option.state &= ~QStyle.State_Selected
        s = self._view_scale()
        # Screen-constant line width via cosmetic pens: exact device-pixel
        # width at any zoom level (avoids int-rounding thickness variation).
        pen = QPen()
        pen.setCosmetic(True)
        if self.isSelected():
            pen.setColor(QColor(255, 0, 0))
            pen.setWidthF(3.0)
            brush = QBrush(QColor(255, 0, 0, 80))
        else:
            pen.setColor(self._base_color())
            pen.setWidthF(2.0)
            brush = QBrush(self._fill_color(50))
            if getattr(self, 'is_ai', False):
                # Dashed outline keeps predictions distinguishable even in a
                # greyscale screenshot or for colour-blind users.
                pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(self.rect())

        if self.isSelected():
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(Qt.black)
            r = self.rect()
            hs = self.visual_handle / (2 * s)   # half edge in scene units
            edge = self.visual_handle / s       # screen-constant edge
            for p in [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]:
                painter.drawRect(QRectF(p.x() - hs, p.y() - hs, edge, edge))

        # Label pill (X-AnyLabeling standard: paint-drawn, tightBoundingRect)
        try:
            painter.save()
            painter.resetTransform()
            view = None
            if self.scene() and self.scene().views():
                view = self.scene().views()[0]
            if view is not None:
                label_text = self.get_label_text()
                font = QFont("Arial", 10, QFont.Bold)
                painter.setFont(font)
                fm = QFontMetrics(font)
                tr = fm.tightBoundingRect(label_text)
                pw, ph = 4, 2
                rw = tr.width() + 2 * pw
                rh = fm.height() + 2 * ph
                tl = view.mapFromScene(self.mapToScene(self.rect().topLeft()))
                bg_rect = QRectF(tl.x(), tl.y(), rw, rh)
                painter.fillRect(
                    bg_rect,
                    QColor(255, 0, 0, 200) if self.isSelected() else self._fill_color(200))
                painter.setPen(Qt.black)
                painter.drawText(
                    QPointF(tl.x() + pw, tl.y() + rh - ph - fm.descent()), label_text)
            painter.restore()
        except Exception:
            try:
                painter.restore()
            except Exception:
                pass

class GraphicsCanvas(QGraphicsView):
    """
    Custom Graphics View for displaying images and handling drawing events.

    Interaction modes (mutually exclusive):
        MODE_CREATE : drag to draw a new box (boxes are never moved here)
        MODE_EDIT   : select / move / resize existing boxes
    """
    MODE_CREATE = 0
    MODE_EDIT = 1
    ZOOM_FIT = 0          # equal margins (fit whole image)
    ZOOM_MANUAL = 1       # user zoomed with Ctrl+wheel
    ZOOM_FIT_HEIGHT = 2   # top/bottom edges flush with the view

    new_box_created = pyqtSignal(object)
    mode_changed = pyqtSignal(int)          # MODE_CREATE / MODE_EDIT
    cursor_pos = pyqtSignal(float, float)   # live image coords for status bar

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Light background (X-AnyLabeling style); vertical scrollbar disabled
        # because FIT_HEIGHT keeps the image fully inside the viewport
        self.setBackgroundBrush(QBrush(QColor(232, 232, 232)))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        self.mode = self.MODE_EDIT
        self.zoom_mode = self.ZOOM_FIT_HEIGHT
        self.setCursor(Qt.ArrowCursor)
        self.pixmap_item = None
        self.drawing_start_pos = None
        self.current_temp_rect = None
        self.image_w = 0
        self.image_h = 0
        self.classes = []
        self.history = []
        self.clipboard = []
        # Cross-line (green dashed) follow mouse
        self.cross_line_show = True
        self.prev_move_point = QPointF()
        # Space + drag panning (like X-AnyLabeling)
        self._space_pressed = False
        self._space_panning = False
        self._space_pan_prev = None

    def load_pixmap(self, pixmap, classes):
        self.drawing_start_pos = None
        self.current_temp_rect = None
        self.last_pan_pos = None
        self.history.clear()
        self.scene.clear()
        self.classes = classes
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)
        self.image_w = pixmap.width()
        self.image_h = pixmap.height()
        margin = 2000
        self.scene.setSceneRect(-margin, -margin, self.image_w + margin * 2, self.image_h + margin * 2)
        # Fit: reset transform FIRST, scale, then center on the image
        # (order matters - resetTransform would otherwise undo the centering)
        self.zoom_mode = self.ZOOM_FIT_HEIGHT
        self._fit_image()

    def _fit_image(self):
        """Reset transform and fit the image into the view per zoom_mode, centered."""
        if not self.pixmap_item or self.image_w <= 0 or self.image_h <= 0:
            return
        self.resetTransform()
        vw, vh = self.viewport().width(), self.viewport().height()
        if vw > 0 and vh > 0:
            if self.zoom_mode == self.ZOOM_FIT_HEIGHT:
                # Near-flush fit: only 4px visual margin top & bottom so edge
                # boxes (2px outline + corner handles) stay fully visible
                # without needing a vertical scrollbar
                scale_factor = (vh - 8) / self.image_h
                # 横图(宽>高)只按高度适配会溢出左右画布边界，且水平滚动条已禁用；
                # 这里加一个宽度上限，确保整图始终完整可见（contain 适配）
                if scale_factor * self.image_w > vw:
                    scale_factor = (vw - 8) / self.image_w
            else:  # ZOOM_FIT: equal margins around the image
                scale_factor = min(vw / (self.image_w + 100), vh / (self.image_h + 100))
            scale_factor = max(scale_factor, 0.01)
            self.scale(scale_factor, scale_factor)
        # center on the image AFTER the transform is applied
        self.centerOn(self.pixmap_item)

    def resizeEvent(self, event):
        """Re-fit when the view is resized while in a FIT mode."""
        super().resizeEvent(event)
        if self.zoom_mode in (self.ZOOM_FIT, self.ZOOM_FIT_HEIGHT) and self.pixmap_item is not None:
            self._fit_image()

    def add_box(self, cls_idx, x, y, w, h, record_history=True,
                is_ai=False, conf=None):
        """Add a box to the canvas.

        is_ai=True marks a model prediction: drawn in cyan, and never written
        to the label file unless the user edits it (see BoxItem.promote_to_human).
        Returns the created BoxItem.
        """
        rect = QRectF(float(x), float(y), float(w), float(h))
        box = BoxItem(rect, cls_idx, self.classes, is_ai=is_ai, conf=conf)
        box.setZValue(10)
        self.scene.addItem(box)
        if record_history:
            self.history.append({'action': 'add', 'item': box})
        return box

    def delete_selected_boxes(self):
        selected_items = self.scene.selectedItems()
        if not selected_items: return
        for item in selected_items:
            if isinstance(item, BoxItem):
                self.scene.removeItem(item)
                self.history.append({'action': 'delete', 'item': item})

    def undo_action(self):
        if not self.history: return
        last_action = self.history.pop()
        action_type = last_action['action']
        item = last_action['item']
        if action_type == 'add':
            if item.scene() == self.scene: self.scene.removeItem(item)
        elif action_type == 'delete':
            self.scene.addItem(item)

    def copy_selection(self):
        selected = self.scene.selectedItems()
        self.clipboard = []
        for item in selected:
            if isinstance(item, BoxItem):
                self.clipboard.append({'cls_idx': item.cls_idx, 'rect': item.rect()})

    def paste_selection(self):
        if not self.clipboard: return
        offset = 20
        for data in self.clipboard:
            r = data['rect']
            self.add_box(data['cls_idx'], r.x() + offset, r.y() + offset, r.width(), r.height(), record_history=True)

    def duplicate_selection(self):
        selected = self.scene.selectedItems()
        if not selected: return
        offset = 20
        to_create = []
        for item in selected:
            if isinstance(item, BoxItem):
                to_create.append({'cls_idx': item.cls_idx, 'rect': item.rect()})
        self.scene.clearSelection()
        for data in to_create:
            r = data['rect']
            self.add_box(data['cls_idx'], r.x() + offset, r.y() + offset, r.width(), r.height(), record_history=True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_pressed = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z: self.undo_action(); return
            if event.key() == Qt.Key_C: self.copy_selection(); return
            if event.key() == Qt.Key_V: self.paste_selection(); return
            if event.key() == Qt.Key_D: self.duplicate_selection(); return
        if event.key() == Qt.Key_Delete: self.delete_selected_boxes(); return
        if event.key() == Qt.Key_R: self.toggle_draw_mode(); return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            self._space_panning = False
            self._space_pan_prev = None
            self.viewport().setCursor(
                Qt.CrossCursor if self.mode == self.MODE_CREATE else Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def drawForeground(self, painter, rect):
        """Green dashed cross-line following the mouse (X-AnyLabeling style)."""
        if not self.cross_line_show or self.pixmap_item is None or self.image_w <= 0:
            return
        p = self.prev_move_point
        if not (0 <= p.x() <= self.image_w and 0 <= p.y() <= self.image_h):
            return
        pen = QPen(QColor(0, 255, 0, 130))
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(p.x(), 0), QPointF(p.x(), self.image_h))
        painter.drawLine(QPointF(0, p.y()), QPointF(self.image_w, p.y()))

    def toggle_draw_mode(self):
        """Toggle between CREATE (draw) and EDIT modes. Returns the new mode."""
        new_mode = self.MODE_EDIT if self.mode == self.MODE_CREATE else self.MODE_CREATE
        self.set_mode(new_mode)
        return self.mode

    def set_mode(self, mode):
        """Switch interaction mode. CREATE = draw new boxes only; EDIT = move/edit only."""
        if mode == self.MODE_CREATE:
            self.mode = self.MODE_CREATE
            self.setCursor(Qt.CrossCursor)
            self.scene.clearSelection()
        else:
            self.mode = self.MODE_EDIT
            self.setCursor(Qt.ArrowCursor)
        self.mode_changed.emit(self.mode)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Ctrl+wheel = zoom (only wheel function; plain wheel is ignored
            # so the image never scrolls vertically)
            self.zoom_mode = self.ZOOM_MANUAL
            zoom = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom, zoom)
            else:
                self.scale(1 / zoom, 1 / zoom)
            event.accept()
        elif event.modifiers() & Qt.ShiftModifier:
            # Shift+wheel = horizontal scroll (wide images)
            delta = event.angleDelta().y()
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - int(delta))
            event.accept()
        else:
            # EDIT 模式 + 恰好选中 1 个矩形 → 滚轮调整框 (X-AnyLabeling 风格)
            # 鼠标在框内: 整体缩放; 鼠标在框外: 移动最近的一条边
            selected = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
            if self.mode == self.MODE_EDIT and len(selected) == 1:
                box = selected[0]
                sp = self.mapToScene(event.pos())
                local_pos = box.mapFromScene(sp)
                wheel_up = event.angleDelta().y() > 0
                if box.rect().contains(local_pos):
                    self._scale_box(box, wheel_up)
                else:
                    self._adjust_box_edge(box, local_pos, wheel_up)
                event.accept()
                return
            # Plain wheel: do nothing (image is anchored, no vertical scrolling)
            event.accept()

    def _scale_box(self, box, scale_up):
        """整体缩放选中框(以中心为锚点), 边界钳制在图片内 (X-AnyLabeling 风格)。"""
        scene_rect = box.mapRectToScene(box.rect())
        if scene_rect.width() < 5 or scene_rect.height() < 5:
            return
        center = scene_rect.center()
        factor = max(0.1, 1.05 if scale_up else 0.95)
        new_w = scene_rect.width() * factor
        new_h = scene_rect.height() * factor
        new_scene_rect = QRectF(center.x() - new_w / 2, center.y() - new_h / 2, new_w, new_h)
        # 边界钳制：用两点构造 QRectF(左上, 右下) 对每条边独立取边界值。
        # 注意：QRectF(x,y,w,h) 是 (x,y,宽,高)，不能直接传 (left,top,right,bottom)!
        new_scene_rect = QRectF(
            QPointF(max(0.0, new_scene_rect.left()),
                    max(0.0, new_scene_rect.top())),
            QPointF(min(float(self.image_w), new_scene_rect.right()),
                    min(float(self.image_h), new_scene_rect.bottom())),
        )
        if new_scene_rect.width() < 5 or new_scene_rect.height() < 5:
            return
        local = box.mapFromScene(new_scene_rect)
        local_rect = local.boundingRect() if hasattr(local, "boundingRect") else local
        box.prepareGeometryChange()
        box.setRect(local_rect)
        box.update_text_pos()

    def _adjust_box_edge(self, box, local_pos, move_outward):
        """移动选中框最近的一条边(鼠标在框外滚轮): 滚轮上=边向外移(框变大), 滚轮下=向内移。"""
        scene_rect = box.mapRectToScene(box.rect())
        left, right = scene_rect.left(), scene_rect.right()
        top, bottom = scene_rect.top(), scene_rect.bottom()
        sp = box.mapToScene(local_pos)
        # 空间区域判定最近边 (X-AnyLabeling 逻辑, 非像素阈值)
        if sp.x() < left and top <= sp.y() <= bottom:
            edge = 'left'
        elif sp.x() > right and top <= sp.y() <= bottom:
            edge = 'right'
        elif sp.y() < top and left <= sp.x() <= right:
            edge = 'top'
        elif sp.y() > bottom and left <= sp.x() <= right:
            edge = 'bottom'
        else:
            dists = {'left': abs(sp.x() - left), 'right': abs(sp.x() - right),
                     'top': abs(sp.y() - top), 'bottom': abs(sp.y() - bottom)}
            edge = min(dists, key=dists.get)
        step = 2.0 if move_outward else -2.0
        new_rect = QRectF(scene_rect)
        if edge == 'left': new_rect.setLeft(max(0.0, left - step))
        elif edge == 'right': new_rect.setRight(min(float(self.image_w), right + step))
        elif edge == 'top': new_rect.setTop(max(0.0, top - step))
        elif edge == 'bottom': new_rect.setBottom(min(float(self.image_h), bottom + step))
        if new_rect.width() < 5 or new_rect.height() < 5:
            return
        local = box.mapFromScene(new_rect)
        local_rect = local.boundingRect() if hasattr(local, "boundingRect") else local
        box.prepareGeometryChange()
        box.setRect(local_rect)
        box.update_text_pos()
    def mousePressEvent(self, event):
        # Image is anchored: middle-button pan is removed (like X-AnyLabeling,
        # panning only via Space+drag or scrollbars)
        if event.button() == Qt.LeftButton and self._space_pressed:
            self._space_panning = True
            self._space_pan_prev = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        sp = self.mapToScene(event.pos())
        if self.mode == self.MODE_CREATE:
            # CREATE mode: only draw new boxes (existing boxes are never touched)
            if event.button() == Qt.LeftButton:
                if self.pixmap_item:
                    raw_sp = self.mapToScene(event.pos())
                    sx = max(0, min(raw_sp.x(), self.image_w))
                    sy = max(0, min(raw_sp.y(), self.image_h))
                    self.drawing_start_pos = QPointF(sx, sy)
                    self.current_temp_rect = QGraphicsRectItem()
                    temp_pen = QPen(Qt.yellow)
                    temp_pen.setStyle(Qt.DashLine)
                    temp_pen.setCosmetic(True)
                    temp_pen.setWidthF(2.0)
                    self.current_temp_rect.setPen(temp_pen)
                    self.current_temp_rect.setZValue(100)
                    self.scene.addItem(self.current_temp_rect)
                return
        else:
            # EDIT mode: only select / move existing boxes
            item = self.scene.itemAt(sp, self.transform())
            is_box = isinstance(item, BoxItem) or (
                        item and item.parentItem() and isinstance(item.parentItem(), BoxItem))
            if is_box:
                super().mousePressEvent(event)
            else:
                self.scene.clearSelection()
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Live cursor position for the cross-line + status bar
        if self.pixmap_item is not None:
            sp = self.mapToScene(event.pos())
            self.prev_move_point = sp
            if 0 <= sp.x() <= self.image_w and 0 <= sp.y() <= self.image_h:
                self.cursor_pos.emit(sp.x(), sp.y())
            self.viewport().update()
        # Space + drag panning
        if self._space_panning and self._space_pan_prev is not None:
            delta = event.pos() - self._space_pan_prev
            self._space_pan_prev = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        try:
            if self.drawing_start_pos and self.current_temp_rect:
                current_pos = self.mapToScene(event.pos())
                mx = max(0, min(current_pos.x(), self.image_w))
                my = max(0, min(current_pos.y(), self.image_h))
                rect = QRectF(self.drawing_start_pos, QPointF(mx, my)).normalized()
                self.current_temp_rect.setRect(rect)
                return
        except:
            pass
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._space_panning:
            self._space_panning = False
            self._space_pan_prev = None
            self.viewport().setCursor(Qt.CrossCursor if self.mode == self.MODE_CREATE else Qt.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.drawing_start_pos:
            if self.current_temp_rect:
                rect = self.current_temp_rect.rect()
                if self.current_temp_rect.scene() == self.scene: self.scene.removeItem(self.current_temp_rect)
                self.current_temp_rect = None
                self.drawing_start_pos = None
                if rect.width() > 5 and rect.height() > 5:
                    self.new_box_created.emit(rect)
                    # Auto-return to EDIT mode so the user can immediately
                    # fine-tune the box they just drew (X-AnyLabeling behavior)
                    self.set_mode(self.MODE_EDIT)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        sp = self.mapToScene(event.pos())
        item = self.scene.itemAt(sp, self.transform())
        selected_box = None
        if isinstance(item, BoxItem):
            selected_box = item
        elif item and item.parentItem() and isinstance(item.parentItem(), BoxItem):
            selected_box = item.parentItem()

        if selected_box:
            menu = QMenu()
            # Use main_window's translation method
            action_edit = menu.addAction(self.main_window.tr_text("CTX_EDIT_LABEL"))
            action_delete = menu.addAction(self.main_window.tr_text("CTX_DEL_BOX"))
            res = menu.exec_(event.globalPos())
            if res == action_delete:
                self.scene.removeItem(selected_box)
                self.history.append({'action': 'delete', 'item': selected_box})
            elif res == action_edit:
                self.main_window.edit_label(selected_box)