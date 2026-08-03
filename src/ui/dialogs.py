# src/ui/dialogs.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                             QCheckBox, QListWidget, QAbstractItemView, QLabel, QComboBox,
                             QDialogButtonBox)
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
        self.edit_label.setPlaceholderText("Class Name")
        self.edit_label.setText(current_label)
        self.edit_label.setMinimumHeight(30)
        self.btn_group = QPushButton("Group ID")
        input_layout.addWidget(self.edit_label)
        input_layout.addWidget(self.btn_group)
        layout.addLayout(input_layout)

        self.edit_desc = QLineEdit()
        self.edit_desc.setPlaceholderText("Description (Optional)")
        layout.addWidget(self.edit_desc)

        btn_layout = QHBoxLayout()
        self.cb_difficult = QCheckBox("useDifficult")
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
        self.combo.addItem("--- Select ---")
        self.combo.addItems(COCO_CLASSES)
        self.combo.currentIndexChanged.connect(self.on_combo_change)
        hbox.addWidget(self.combo)
        layout.addLayout(hbox)

        self.input_line = QLineEdit()
        self.input_line.setText("person")
        self.input_line.setPlaceholderText("person, car, ...")
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