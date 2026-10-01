# src/ui/training_dialog.py
"""
Training setup dialog – shown when a training round is triggered.

    Round 1   : after all representative samples are annotated.
    Round 2+  : after training, the remaining unlabeled images are
                re-clustered into new representative clusters.

Shows: model, epochs, freeze.

Usage from MainWindow:

    dialog = TrainingSetupDialog(self, round_number=1, ...)
    if dialog.exec_() == QDialog.Accepted:
        params = dialog.get_params()
        # params = {model, epochs, freeze}
        self.start_background_training(params)
"""

import os

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QComboBox, QSpinBox, QGroupBox,
                              QFormLayout, QFrame, QApplication, QFileDialog)
from PyQt5.QtCore import Qt

from src.config import TRANS


class TrainingSetupDialog(QDialog):
    """Modal dialog to review/confirm training parameters before a round."""

    # Model list must mirror the main window's combo_yolo_model items
    MODEL_OPTIONS = [
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
        "rtdetr-l", "rtdetr-x", "yolov8s-worldv2", "yolov8m-worldv2",
        "fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "ssd300_vgg16",
    ]

    def __init__(self, parent=None, round_number: int = 1,
                 current_model: str = "yolov8n",
                 current_epochs: int = 30, current_freeze: int = 0,
                 annotated_count: int = 0, representative_count: int = 0,
                 dataset_size: int = 0, total_annotated: int = 0,
                 last_model_name: str = "",
                 last_epochs: int = 0, last_map: float = 0.0,
                 locked_model: bool = False, trained_models=None):
        super().__init__(parent)
        self.parent_window = parent
        self._params = {}
        self._round_number = round_number

        self.setWindowTitle(self.tr_text("TD_TITLE"))
        self.resize(460, 420)
        self._build_ui(round_number, current_model, current_epochs,
                       current_freeze,
                       annotated_count, representative_count, dataset_size,
                       total_annotated, last_model_name, last_epochs, last_map,
                       locked_model, trained_models)

    # ------------------------------------------------------------------
    def _build_ui(self, round_number, model, epochs, freeze,
                  annotated_count, representative_count, dataset_size,
                  total_annotated, last_model_name, last_epochs, last_map,
                  locked_model, trained_models=None):
        layout = QVBoxLayout(self)

        # ── Status header (round-dependent) ──
        if round_number == 1:
            header_text = self.tr_text("TD_HEADER_FIRST").format(
                total_annotated or annotated_count, dataset_size)
            header_style = ("background: #e8f5e9; border: 1px solid #a5d6a7; "
                            "border-radius: 4px; padding: 8px; color: #1b5e20; "
                            "font-weight: bold;")
        else:
            header_text = self.tr_text("TD_HEADER_ROUND").format(
                round_number, annotated_count,
                total_annotated or annotated_count, dataset_size)
            if last_model_name:
                mAP50_part = self.tr_text("FMT_MAP50_SUFFIX").format(last_map) if last_map else ""
                epochs_part = self.tr_text("FMT_EPOCHS_SUFFIX").format(last_epochs) if last_epochs else ""
                header_text += self.tr_text("TD_HEADER_LAST").format(last_model_name, mAP50_part, epochs_part)
            header_style = ("background: #fdf6e3; border: 1px solid #f0d9a0; "
                            "border-radius: 4px; padding: 8px; color: #7a5c00; "
                            "font-weight: bold;")
        header = QLabel(header_text)
        header.setWordWrap(True)
        header.setStyleSheet(header_style)
        layout.addWidget(header)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line1)

        # ── Model ──
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(self.tr_text("TD_LBL_MODEL")))
        self.combo_model = QComboBox()
        hint = None
        if trained_models:
            # Iterative training: pick which archived round model to continue from
            # (best mAP first, selected by default)
            for display, value in trained_models:
                self.combo_model.addItem(display, value)
            # Base detection models stay available (e.g. start fresh with yolo11n)
            self.combo_model.insertSeparator(self.combo_model.count())
            for _m in self.MODEL_OPTIONS:
                self.combo_model.addItem(_m)
            self.combo_model.setCurrentIndex(0)
            hint = QLabel(self.tr_text("TD_HINT_MODEL_PICK"))
            hint.setStyleSheet("color: #7a5c00; font-size: 11px;")
            hint.setWordWrap(True)
        elif locked_model and last_model_name:
            # Legacy fallback: locked to the previous round's trained model
            self.combo_model.addItem(last_model_name)
            self.combo_model.setEnabled(False)
            hint = QLabel(self.tr_text("TD_HINT_ITERATE"))
            hint.setStyleSheet("color: #7a5c00; font-size: 11px;")
            hint.setWordWrap(True)
        else:
            self.combo_model.addItems(self.MODEL_OPTIONS)
            if model in self.MODEL_OPTIONS:
                self.combo_model.setCurrentText(model)
        model_row.addWidget(self.combo_model, 1)
        # Custom weights: browse for ANY .pt/.pth file instead of the presets above
        self.btn_browse_model = QPushButton(self.tr_text("TD_BTN_BROWSE"))
        self.btn_browse_model.setToolTip(self.tr_text("TD_TIP_BROWSE"))
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        model_row.addWidget(self.btn_browse_model)
        layout.addLayout(model_row)
        if hint is not None:
            layout.addWidget(hint)

        # ── Parameters ──
        param_group = QGroupBox(self.tr_text("TD_GRP_PARAMS"))
        form = QFormLayout(param_group)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(epochs)

        self.spin_freeze = QSpinBox()
        self.spin_freeze.setRange(0, 50)
        self.spin_freeze.setValue(freeze)

        form.addRow(self.tr_text("TD_LBL_EPOCHS"), self.spin_epochs)
        form.addRow(self.tr_text("TD_LBL_FREEZE"), self.spin_freeze)
        layout.addWidget(param_group)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_train = QPushButton(self.tr_text("TD_BTN_TRAIN"))
        self.btn_train.setStyleSheet("background-color: #FFA500; font-weight: bold;")
        self.btn_train.clicked.connect(self._on_confirm)
        btn_row.addWidget(self.btn_train)
        self.btn_cancel = QPushButton(self.tr_text("TD_BTN_CANCEL"))
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def showEvent(self, event):
        """Center the dialog each time it is shown."""
        super().showEvent(event)
        try:
            screen = QApplication.primaryScreen().availableGeometry()
            x = screen.x() + (screen.width() - self.width()) // 2
            y = screen.y() + (screen.height() - self.height()) // 2
            self.move(x, y)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def tr_text(self, key):
        if hasattr(self.parent_window, 'tr_text'):
            return self.parent_window.tr_text(key)
        return TRANS.get('zh', {}).get(key, key)

    # ------------------------------------------------------------------
    def _on_confirm(self):
        self._params = {
            "model": self.combo_model.currentData() or self.combo_model.currentText(),
            "epochs": self.spin_epochs.value(),
            "freeze": self.spin_freeze.value(),
        }
        self.accept()

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return dict(self._params)

    def get_round_number(self) -> int:
        return self._round_number

    # ------------------------------------------------------------------
    def _on_browse_model(self):
        """Let the user train from ANY weights file (.pt/.pth), not just the presets."""
        start_dir = os.getcwd()
        try:
            root = getattr(self.parent_window, 'dataset_root', None)
            if root:
                cand = os.path.join(root, 'training_runs', 'loop_train', 'weights')
                if os.path.isdir(cand):
                    start_dir = cand
        except Exception:
            pass
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr_text("TTL_SELECT_MODEL"), start_dir,
            "Model Weights (*.pt *.pth);;All Files (*)")
        if not file_path:
            return
        file_path = os.path.abspath(file_path)
        # Reuse the entry when the same file is picked again
        for i in range(self.combo_model.count()):
            if self.combo_model.itemData(i) == file_path:
                self.combo_model.setCurrentIndex(i)
                return
        label = self.tr_text("TD_CUSTOM_PREFIX").format(os.path.basename(file_path))
        self.combo_model.insertItem(0, label, file_path)
        self.combo_model.setCurrentIndex(0)
