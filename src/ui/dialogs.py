# src/ui/dialogs.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                             QCheckBox, QListWidget, QListWidgetItem, QAbstractItemView, QLabel, QComboBox,
                             QDialogButtonBox, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt
from src.config import TRANS, COCO_CLASSES

class LabelDialog(QDialog):
    def __init__(self, parent=None, class_list=[], current_label=""):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(self.tr_text("DLG_LABEL_TITLE"))
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        input_layout = QHBoxLayout()
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText(self.tr_text("DLG_PH_CLASS"))
        self.edit_label.setText(current_label)
        self.edit_label.setMinimumHeight(30)
        self.btn_group = QPushButton(self.tr_text("DLG_BTN_GROUP"))
        input_layout.addWidget(self.edit_label)
        input_layout.addWidget(self.btn_group)
        layout.addLayout(input_layout)

        self.edit_desc = QLineEdit()
        self.edit_desc.setPlaceholderText(self.tr_text("DLG_PH_DESC"))
        layout.addWidget(self.edit_desc)

        btn_layout = QHBoxLayout()
        self.cb_difficult = QCheckBox(self.tr_text("DLG_CHK_DIFFICULT"))
        self.btn_ok = QPushButton(self.tr_text("DLG_BTN_OK"))
        self.btn_ok.setStyleSheet("background-color: #FFA500; font-weight: bold;")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton(self.tr_text("DLG_BTN_CANCEL"))
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.cb_difficult)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.addItems(class_list)
        layout.addWidget(self.list_widget)

        self.list_widget.itemClicked.connect(self.list_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self.list_item_double_clicked)

        if current_label:
            items = self.list_widget.findItems(current_label, Qt.MatchExactly)
            if items: self.list_widget.setCurrentItem(items[0])

        self.edit_label.setFocus()
        self.edit_label.selectAll()

    def tr_text(self, key):
        if hasattr(self.parent_window, 'tr_text'):
            return self.parent_window.tr_text(key)
        return TRANS['zh'].get(key, key)

    def list_item_clicked(self, item):
        self.edit_label.setText(item.text())

    def list_item_double_clicked(self, item):
        self.edit_label.setText(item.text())
        self.accept()

    def get_label(self):
        return self.edit_label.text().strip()


class ClassSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(self.tr_text("DLG_CLS_TITLE"))
        self.resize(400, 250)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.tr_text("DLG_CLS_MSG")))
        layout.addWidget(QLabel(self.tr_text("DLG_CLS_HINT")))

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(self.tr_text("DLG_QUICK_ADD")))
        self.combo = QComboBox()
        self.combo.addItem(self.tr_text("DLG_COMBO_SELECT"))
        self.combo.addItems(COCO_CLASSES)
        self.combo.currentIndexChanged.connect(self.on_combo_change)
        hbox.addWidget(self.combo)
        layout.addLayout(hbox)

        self.input_line = QLineEdit()
        self.input_line.setText("person")
        self.input_line.setPlaceholderText(self.tr_text("DLG_PH_CLASS_EXAMPLES"))
        layout.addWidget(self.input_line)

        self.btn_clear = QPushButton(self.tr_text("DLG_BTN_CLEAR"))
        self.btn_clear.clicked.connect(self.input_line.clear)
        layout.addWidget(self.btn_clear)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText(self.tr_text("DLG_BTN_OK"))
        self.button_box.button(QDialogButtonBox.Cancel).setText(self.tr_text("DLG_BTN_CANCEL"))
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def tr_text(self, key):
        if hasattr(self.parent_window, 'tr_text'):
            return self.parent_window.tr_text(key)
        return TRANS['zh'].get(key, key)

    def on_combo_change(self, index):
        if index <= 0: return
        selected_cls = self.combo.currentText()
        current_text = self.input_line.text().strip()
        parts = [p.strip() for p in current_text.split(',') if p.strip()]
        if selected_cls in parts:
            self.combo.setCurrentIndex(0)
            return
        if current_text:
            new_text = current_text + ", " + selected_cls if not current_text.endswith(
                ',') else current_text + " " + selected_cls
        else:
            new_text = selected_cls
        self.input_line.setText(new_text)
        self.combo.setCurrentIndex(0)

    def get_classes(self):
        text = self.input_line.text().strip()
        if not text: return ["person"]
        return [t.strip() for t in text.split(',') if t.strip()]


class LabelSetupDialog(QDialog):
    """
    Label setup dialog with full class management:
      - reorder (move up/down), rename, delete, add classes
      - toggle AI-assisted pre-labeling (YOLO-World / trained model)
    """

    def __init__(self, parent=None, current_classes=None, has_trained_model: bool = False):
        super().__init__(parent)
        self.parent_window = parent
        self.current_classes = list(current_classes) if current_classes else ["person"]
        self.setWindowTitle(self.tr_text("LBLST_TITLE"))
        self.resize(560, 500)

        layout = QVBoxLayout(self)

        # ── Header ──
        header = QLabel(self.tr_text("LBLST_HEADER"))
        header.setWordWrap(True)
        header.setStyleSheet(
            "background: #e8f4fd; border: 1px solid #b8d4f0; "
            "border-radius: 4px; padding: 8px; color: #1a5276;")
        layout.addWidget(header)

        # ── Detection result: trained model takes priority ──
        if has_trained_model:
            self._recommended = True
            det_txt = self.tr_text("LBLST_DET_TRAINED")
            det_style = (
                "background: #e8f5e9; border: 1px solid #a5d6a7; "
                "border-radius: 4px; padding: 8px; color: #1b5e20;")
        else:
            from src.utils.coco_matcher import match_coco_overlap, recommend_ai_assist
            matched, total, ratio = match_coco_overlap(self.current_classes)
            self._recommended = recommend_ai_assist(self.current_classes)
            if matched:
                det_txt = self.tr_text("LBLST_DET_COCO").format(total, len(matched), ratio * 100, ', '.join(matched))
                det_style = (
                    "background: #e8f5e9; border: 1px solid #a5d6a7; "
                    "border-radius: 4px; padding: 8px; color: #1b5e20;")
            else:
                det_txt = self.tr_text("LBLST_DET_NOMATCH").format('、'.join(self.current_classes))
                det_style = (
                    "background: #fdf6e3; border: 1px solid #f0d9a0; "
                    "border-radius: 4px; padding: 8px; color: #7a5c00;")
        self.lbl_detect = QLabel(det_txt)
        self.lbl_detect.setWordWrap(True)
        self.lbl_detect.setStyleSheet(det_style)
        layout.addWidget(self.lbl_detect)

        # ── AI assist checkbox ──
        self.chk_ai = QCheckBox(self.tr_text("LBLST_CHK_AI"))
        self.chk_ai.setChecked(self._recommended)
        layout.addWidget(self.chk_ai)

        # ── Class list (editable: reorder / rename / delete / add) ──
        layout.addWidget(QLabel(self.tr_text("LBLST_LBL_CLASSES")))
        self.list_classes = QListWidget()
        for i, name in enumerate(self.current_classes):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, i)      # stable internal id = original index
            self.list_classes.addItem(item)
        self.list_classes.setSelectionMode(QAbstractItemView.SingleSelection)
        if self.list_classes.count():
            self.list_classes.setCurrentRow(0)
        layout.addWidget(self.list_classes, 1)

        # ── Manage buttons row ──
        manage_row = QHBoxLayout()
        self.btn_move_up = QPushButton(self.tr_text("LBLST_BTN_MOVE_UP"))
        self.btn_move_up.clicked.connect(self._move_up)
        self.btn_move_down = QPushButton(self.tr_text("LBLST_BTN_MOVE_DOWN"))
        self.btn_move_down.clicked.connect(self._move_down)
        self.btn_rename = QPushButton(self.tr_text("LBLST_BTN_RENAME"))
        self.btn_rename.clicked.connect(self._rename)
        self.btn_delete = QPushButton(self.tr_text("LBLST_BTN_DELETE"))
        self.btn_delete.clicked.connect(self._delete)
        for b in (self.btn_move_up, self.btn_move_down, self.btn_rename, self.btn_delete):
            manage_row.addWidget(b)
        manage_row.addStretch()
        layout.addLayout(manage_row)

        # ── Add new class row ──
        add_row = QHBoxLayout()
        self.input_add = QLineEdit()
        self.input_add.setPlaceholderText(self.tr_text("LBLST_PH_CLASSES"))
        self.input_add.returnPressed.connect(self._add_class)
        self.btn_add = QPushButton(self.tr_text("LBLST_BTN_ADD"))
        self.btn_add.clicked.connect(self._add_class)
        add_row.addWidget(self.input_add, 1)
        add_row.addWidget(self.btn_add)
        layout.addLayout(add_row)

        # ── COCO quick add ──
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(self.tr_text("LBLST_LBL_COCO")))
        self.combo = QComboBox()
        self.combo.addItem(self.tr_text("DLG_COMBO_SELECT"))
        self.combo.addItems(COCO_CLASSES)
        self.combo.currentIndexChanged.connect(self.on_combo_change)
        hbox.addWidget(self.combo, 1)
        layout.addLayout(hbox)

        # ── Buttons ──
        self.btn_ok = QPushButton(self.tr_text("LBLST_BTN_START"))
        self.btn_ok.setStyleSheet("background-color: #FFA500; font-weight: bold;")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel = QPushButton(self.tr_text("LBLST_BTN_CANCEL"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    def tr_text(self, key):
        if hasattr(self.parent_window, 'tr_text'):
            return self.parent_window.tr_text(key)
        return TRANS['zh'].get(key, key)

    def _selected_row(self):
        return self.list_classes.currentRow()

    def _names(self):
        return [self.list_classes.item(i).text().strip()
                for i in range(self.list_classes.count())]

    def _move_up(self):
        row = self._selected_row()
        if row <= 0: return
        item = self.list_classes.takeItem(row)
        self.list_classes.insertItem(row - 1, item)
        self.list_classes.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._selected_row()
        if row < 0 or row >= self.list_classes.count() - 1: return
        item = self.list_classes.takeItem(row)
        self.list_classes.insertItem(row + 1, item)
        self.list_classes.setCurrentRow(row + 1)

    def _rename(self):
        row = self._selected_row()
        if row < 0: return
        old = self.list_classes.item(row).text()
        new_name, ok = QInputDialog.getText(self, self.tr_text("MSG_RENAME_TITLE"),
                                            self.tr_text("MSG_RENAME_PROMPT"), text=old)
        if not ok: return
        new_name = new_name.strip()
        if not new_name: return
        if new_name != old and new_name in self._names():
            QMessageBox.warning(self, self.tr_text("MSG_CLASS_EXISTS"),
                                self.tr_text("MSG_CLASS_EXISTS"))
            return
        self.list_classes.item(row).setText(new_name)

    def _delete(self):
        row = self._selected_row()
        if row < 0: return
        name = self.list_classes.item(row).text()
        reply = QMessageBox.question(
            self, self.tr_text("LBLST_BTN_DELETE"),
            self.tr_text("MSG_DELETE_CLASS_CONFIRM").format(name),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self.list_classes.takeItem(row)
        if self.list_classes.count():
            self.list_classes.setCurrentRow(min(row, self.list_classes.count() - 1))

    def _add_class(self):
        name = self.input_add.text().strip()
        if not name: return
        if name in self._names():
            QMessageBox.warning(self, self.tr_text("MSG_CLASS_EXISTS"),
                                self.tr_text("MSG_CLASS_EXISTS"))
            return
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, -1)   # brand-new class
        self.list_classes.addItem(item)
        self.list_classes.setCurrentRow(self.list_classes.count() - 1)
        self.input_add.clear()

    def on_combo_change(self, index):
        if index <= 0: return
        selected_cls = self.combo.currentText()
        if selected_cls not in self._names():
            item = QListWidgetItem(selected_cls)
            item.setData(Qt.UserRole, -1)
            self.list_classes.addItem(item)
        self.combo.setCurrentIndex(0)

    def _on_ok(self):
        if not self._names():
            QMessageBox.warning(self, self.tr_text("MSG_CLASS_EMPTY"),
                                self.tr_text("MSG_CLASS_EMPTY"))
            return
        self.accept()

    def get_classes(self):
        names = self._names()
        if not names: return ["person"]
        return names

    def get_mapping(self):
        """Return {old_index: new_index} for kept classes (deleted ones absent)."""
        mapping = {}
        for new_idx in range(self.list_classes.count()):
            internal_id = self.list_classes.item(new_idx).data(Qt.UserRole)
            if internal_id is not None and internal_id >= 0:
                mapping[internal_id] = new_idx
        return mapping

    def get_ai_assist(self):
        return self.chk_ai.isChecked()
