# src/ui/canvas.py
import math
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsTextItem, QGraphicsItem, QMenu, QStyle)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPen, QColor, QBrush, QFont, QCursor, QPainter

class BoxItem(QGraphicsRectItem):
    """
    Custom Graphics Item representing a bounding box.
    Handles mouse interactions, resizing, and visual styles.
    """
    handle_size = 18.0
    margin = 35.0  # Sensitivity margin for mouse hover

    def __init__(self, rect, cls_idx, class_names, parent=None):
        super().__init__(rect, parent)
        self.cls_idx = cls_idx
        self.class_names = class_names

        # Visual styles
        self.default_pen = QPen(QColor(0, 255, 0), 5)
        self.selected_pen = QPen(QColor(255, 0, 0), 7)

        self.setPen(self.default_pen)
        self.setBrush(QBrush(QColor(0, 255, 0, 80)))

        self.setFlags(QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemIsMovable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # Label text setup
        self.text_item = QGraphicsTextItem(self.get_label_text(), self)
        self.text_item.setDefaultTextColor(QColor(Qt.black))
        font = QFont("Arial", 26, QFont.Bold)
        self.text_item.setFont(font)
        self.text_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        # Text background
        self.text_bg = QGraphicsRectItem(self.text_item.boundingRect(), self)
        self.text_bg.setBrush(QBrush(QColor(0, 255, 0, 200)))
        self.text_bg.setPen(QPen(Qt.NoPen))
        self.text_bg.stackBefore(self.text_item)
        self.text_bg.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        self.update_text_pos()
        self.resizing = False
        self.resize_corner = None

    def itemChange(self, change, value):
        # Handle selection change
        if change == QGraphicsItem.ItemSelectedChange:
            if value == True:
                self.text_bg.setBrush(QBrush(QColor(255, 0, 0, 200)))
                self.setZValue(100)
            else:
                self.text_bg.setBrush(QBrush(QColor(0, 255, 0, 200)))
                self.setZValue(10)

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
        return f"{name}"

    def update_classes(self):
        self.text_item.setPlainText(self.get_label_text())
        self.text_bg.setRect(self.text_item.boundingRect())

    def update_text_pos(self):
        if self.rect().isValid():
            r = self.rect()
            self.text_item.setPos(r.left(), r.top())
            self.text_bg.setRect(self.text_item.boundingRect())
            self.text_bg.setPos(r.left(), r.top())

    def check_cursor_on_corner(self, pos):
        rect = self.rect()
        x, y = pos.x(), pos.y()
        dist_tl = math.hypot(x - rect.left(), y - rect.top())
        dist_tr = math.hypot(x - rect.right(), y - rect.top())
        dist_bl = math.hypot(x - rect.left(), y - rect.bottom())
        dist_br = math.hypot(x - rect.right(), y - rect.bottom())

        if dist_tl < self.margin: return 'TL'
        if dist_tr < self.margin: return 'TR'
        if dist_bl < self.margin: return 'BL'
        if dist_br < self.margin: return 'BR'
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
        if self.isSelected():
            painter.setPen(self.selected_pen)
            brush = QBrush(QColor(255, 0, 0, 80))
        else:
            painter.setPen(self.default_pen)
            brush = QBrush(QColor(0, 255, 0, 50))
        painter.setBrush(brush)
        painter.drawRect(self.rect())

        if self.isSelected():
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(Qt.black)
            r = self.rect()
            hs = self.handle_size / 2
            for p in [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]:
                painter.drawRect(QRectF(p.x() - hs, p.y() - hs, self.handle_size, self.handle_size))


class GraphicsCanvas(QGraphicsView):
    """
    Custom Graphics View for displaying images and handling drawing events.
    """
    new_box_created = pyqtSignal(object)

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
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))

        self.is_draw_mode = False
        self.setCursor(Qt.ArrowCursor)
        self.pixmap_item = None
        self.drawing_start_pos = None
        self.current_temp_rect = None
        self.last_pan_pos = None
        self.image_w = 0
        self.image_h = 0
        self.classes = []
        self.history = []
        self.clipboard = []

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
        self.centerOn(self.pixmap_item)
        self.resetTransform()
        if self.width() > 0 and self.height() > 0:
            scale_factor = min(self.width() / (self.image_w + 100), self.height() / (self.image_h + 100))
            self.scale(scale_factor, scale_factor)

    def add_box(self, cls_idx, x, y, w, h, record_history=True):
        rect = QRectF(float(x), float(y), float(w), float(h))
        box = BoxItem(rect, cls_idx, self.classes)
        box.setZValue(10)
        self.scene.addItem(box)
        if record_history:
            self.history.append({'action': 'add', 'item': box})

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
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z: self.undo_action(); return
            if event.key() == Qt.Key_C: self.copy_selection(); return
            if event.key() == Qt.Key_V: self.paste_selection(); return
            if event.key() == Qt.Key_D: self.duplicate_selection(); return
        if event.key() == Qt.Key_Delete: self.delete_selected_boxes(); return
        if event.key() == Qt.Key_R: self.toggle_draw_mode(); return
        super().keyPressEvent(event)

    def toggle_draw_mode(self):
        self.is_draw_mode = not self.is_draw_mode
        if self.is_draw_mode:
            self.setCursor(Qt.CrossCursor)
            self.scene.clearSelection()
        else:
            self.setCursor(Qt.ArrowCursor)
        return self.is_draw_mode

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            zoom = 1.15
            if event.angleDelta().y() > 0:
                self.scale(zoom, zoom)
            else:
                self.scale(1 / zoom, 1 / zoom)
        elif event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - int(delta))
        else:
            delta = event.angleDelta().y()
            v_bar = self.verticalScrollBar()
            v_bar.setValue(v_bar.value() - int(delta))

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.last_pan_pos = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        sp = self.mapToScene(event.pos())
        if self.is_draw_mode:
            if event.button() == Qt.LeftButton:
                if self.pixmap_item:
                    raw_sp = self.mapToScene(event.pos())
                    sx = max(0, min(raw_sp.x(), self.image_w))
                    sy = max(0, min(raw_sp.y(), self.image_h))
                    self.drawing_start_pos = QPointF(sx, sy)
                    self.current_temp_rect = QGraphicsRectItem()
                    self.current_temp_rect.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
                    self.current_temp_rect.setZValue(100)
                    self.scene.addItem(self.current_temp_rect)
                return
        else:
            item = self.scene.itemAt(sp, self.transform())
            is_box = isinstance(item, BoxItem) or (
                        item and item.parentItem() and isinstance(item.parentItem(), BoxItem))
            if is_box:
                super().mousePressEvent(event)
            else:
                self.scene.clearSelection()
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.last_pan_pos is not None:
            delta = event.pos() - self.last_pan_pos
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self.last_pan_pos = event.pos()
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
        if event.button() == Qt.MiddleButton:
            self.last_pan_pos = None
            self.viewport().setCursor(Qt.CrossCursor if self.is_draw_mode else Qt.ArrowCursor)
            return
        if event.button() == Qt.LeftButton and self.drawing_start_pos:
            if self.current_temp_rect:
                rect = self.current_temp_rect.rect()
                if self.current_temp_rect.scene() == self.scene: self.scene.removeItem(self.current_temp_rect)
                self.current_temp_rect = None
                self.drawing_start_pos = None
                if rect.width() > 5 and rect.height() > 5:
                    self.new_box_created.emit(rect)
                    self.is_draw_mode = False
                    self.setCursor(Qt.ArrowCursor)
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