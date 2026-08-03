# src/ui/main_window.py
import os
import shutil
import yaml
import traceback
import time
from multiprocessing import Queue

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QListWidget, QMessageBox, QAction, QDialog, QLineEdit,
                             QRadioButton, QButtonGroup, QSpinBox, QFormLayout, QGroupBox,
                             QProgressBar, QTextEdit, QSplitter, QScrollArea, QFrame, QComboBox, QApplication)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSettings
from PyQt5.QtGui import QPixmap, QIcon, QCursor

# Import from other modules
from src.config import TRANS, COCO_CLASSES
from src.ui.canvas import GraphicsCanvas, BoxItem
from src.ui.dialogs import LabelDialog, ClassSelectionDialog
from src.utils.threads import ModelLoaderThread

# Import core logic
from src.core.training_worker import TrainingProcess
from src.core.inference_engine import InferenceEngine
from src.core.label_converter import LabelConverter

# ==========================================
# Custom Widgets for Better Scroll Experience
# ==========================================
class FocusScrollSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class FocusScrollComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

# ==========================================
# Main Window Class
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyCompany", "AutoLabelPro")
        self.curr_lang = self.settings.value("language", "zh")

        if os.path.exists("logo.png"): self.setWindowIcon(QIcon("logo.png"))

        # Adaptive initial size for 1080P screens
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.80), int(screen.height() * 0.80))
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        self.image_list = []
        self.current_index = -1
        self.classes = ["object"]
        self.verified_count_since_train = 0
        self.train_threshold = 15
        self.dataset_root = ""
        self.train_start_time = None
        self.current_model_path = "yolov8n.pt"
        self.is_training = False
        self.is_restoring = False

        # Initialize Inference Engine
        self.inference_engine = InferenceEngine(self.current_model_path)
        self.inference_engine.prediction_ready.connect(self.on_inference_result)
        self.inference_engine.model_loaded.connect(
            lambda p: self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(p))))
        self.inference_engine.inference_info.connect(self.update_inference_label)

        # Training Queue and Timer
        self.train_queue = Queue()
        self.train_timer = QTimer()
        self.train_timer.timeout.connect(self.check_training_status)
        self.train_timer.start(1000)

        self.train_loop_count = 0
        self.total_images_in_model = 0
        self.current_batch_count = 0

        self.setup_ui()
        self.retranslate_ui()
        QTimer.singleShot(0, self.restore_state)

    def tr_text(self, key):
        lang_dict = TRANS.get(self.curr_lang, TRANS["zh"])
        return lang_dict.get(key, key)

    def switch_language(self, lang_code):
        if lang_code == self.curr_lang: return
        self.curr_lang = lang_code
        self.settings.setValue("language", lang_code)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr_text("APP_TITLE"))
        self.menu_lang.setTitle(self.tr_text("MENU_LANG"))
        self.menu_import.setTitle(self.tr_text("MENU_IMPORT"))
        self.menu_export.setTitle(self.tr_text("MENU_EXPORT"))
        self.action_import_coco.setText(self.tr_text("ACT_IMPORT_COCO"))
        self.action_import_voc.setText(self.tr_text("ACT_IMPORT_VOC"))
        self.action_export_voc.setText(self.tr_text("ACT_EXPORT_VOC"))
        self.action_export_coco.setText(self.tr_text("ACT_EXPORT_COCO"))
        self.btn_load.setText(self.tr_text("BTN_LOAD_DIR"))
        self.btn_load_model.setText(self.tr_text("BTN_LOAD_MODEL"))
        if self.is_training:
            self.btn_force_train.setText(self.tr_text("BTN_STOP_TRAIN"))
        else:
            self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
        self.lbl_info.setText(
            self.tr_text("FMT_PROGRESS").format(self.verified_count_since_train, self.train_threshold))
        self.lbl_total_stats.setText(
            self.tr_text("FMT_TOTAL_STATS").format(self.train_loop_count, self.total_images_in_model))
        self.status_label.setText(self.tr_text("TIP_READY"))
        self.lbl_model_title.setText(self.tr_text("LBL_BASE_MODEL"))
        if hasattr(self, 'lbl_prompt_title'):
            self.lbl_prompt_title.setText(self.tr_text("LBL_PROMPT_TITLE"))
            self.txt_prompt.setToolTip(self.tr_text("TIP_PROMPT_INPUT"))
            self.txt_prompt.setPlaceholderText(self.tr_text("PH_PROMPT"))
        self.combo_yolo_model.setToolTip(self.tr_text("TIP_MODEL_SELECT"))
        self.lbl_mode.setText(self.tr_text("LBL_STRATEGY"))
        self.radio_transfer.setText(self.tr_text("MODE_TRANSFER"))
        self.radio_transfer.setToolTip(self.tr_text("TIP_TRANSFER"))
        self.radio_recursive.setText(self.tr_text("MODE_RECURSIVE"))
        self.radio_recursive.setToolTip(self.tr_text("TIP_RECURSIVE"))
        self.radio_scratch.setText(self.tr_text("MODE_SCRATCH"))
        self.radio_scratch.setToolTip(self.tr_text("TIP_SCRATCH"))
        self.param_group.setTitle(self.tr_text("GRP_PARAMS"))
        self.lbl_epochs_title.setText(self.tr_text("LBL_EPOCHS"))
        self.lbl_freeze_title.setText(self.tr_text("LBL_FREEZE"))
        self.lbl_thresh_title.setText(self.tr_text("LBL_THRESH"))
        self.monitor_group.setTitle(self.tr_text("GRP_MONITOR"))
        self.lbl_timer.setText(self.tr_text("TIP_READY"))
        self.tips_label.setText(self.tr_text("TIPS_HTML"))
        if "AI" in self.lbl_data_source.text():
            self.lbl_data_source.setText(self.tr_text("STATUS_AI"))
        else:
            self.lbl_data_source.setText(self.tr_text("STATUS_MANUAL"))

    def setup_ui(self):
        menubar = self.menuBar()
        self.menu_lang = menubar.addMenu("")
        action_zh = QAction("简体中文", self)
        action_zh.triggered.connect(lambda: self.switch_language("zh"))
        self.menu_lang.addAction(action_zh)
        action_en = QAction("English", self)
        action_en.triggered.connect(lambda: self.switch_language("en"))
        self.menu_lang.addAction(action_en)

        self.menu_import = menubar.addMenu("")
        self.action_import_coco = QAction("", self)
        self.action_import_coco.triggered.connect(self.import_coco_labels)
        self.menu_import.addAction(self.action_import_coco)
        self.action_import_voc = QAction("", self)
        self.action_import_voc.triggered.connect(self.import_voc_labels)
        self.menu_import.addAction(self.action_import_voc)

        self.menu_export = menubar.addMenu("")
        self.action_export_voc = QAction("", self)
        self.action_export_voc.triggered.connect(self.export_voc)
        self.menu_export.addAction(self.action_export_voc)
        self.action_export_coco = QAction("", self)
        self.action_export_coco.triggered.connect(self.export_coco)
        self.menu_export.addAction(self.action_export_coco)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === Left Sidebar ===
        self.left_content = QWidget()
        left_layout = QVBoxLayout(self.left_content)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # 1. Top Buttons
        self.btn_load = QPushButton("")
        self.btn_load.setObjectName("btn_normal")
        self.btn_load.clicked.connect(self.load_folder)

        self.btn_load_model = QPushButton("")
        self.btn_load_model.setObjectName("btn_small")
        self.btn_load_model.clicked.connect(self.load_custom_model)

        top_btns_layout = QHBoxLayout()
        top_btns_layout.addWidget(self.btn_load, 2)
        top_btns_layout.addWidget(self.btn_load_model, 1)
        left_layout.addLayout(top_btns_layout)

        self.btn_force_train = QPushButton("")
        self.btn_force_train.setObjectName("btn_train")
        self.btn_force_train.clicked.connect(self.on_train_btn_clicked)
        left_layout.addWidget(self.btn_force_train)

        # 2. Status Info
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("lbl_info")

        self.lbl_total_stats = QLabel("")
        self.lbl_total_stats.setObjectName("lbl_stats")

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")

        left_layout.addWidget(self.lbl_info)
        left_layout.addWidget(self.lbl_total_stats)
        left_layout.addWidget(self.status_label)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setStyleSheet("background-color: #ddd;")
        left_layout.addWidget(line1)

        # 3. Model Settings
        model_group = QWidget()
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)

        self.lbl_model_title = QLabel("")
        self.lbl_model_title.setObjectName("lbl_section_title")

        # 【修改】使用自定义的 FocusScrollComboBox
        self.combo_yolo_model = FocusScrollComboBox()
        self.combo_yolo_model.addItems([
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
            "rtdetr-l", "rtdetr-x", "yolov8s-worldv2", "yolov8m-worldv2",
            "fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "ssd300_vgg16"
        ])
        self.combo_yolo_model.currentTextChanged.connect(self.on_base_model_changed)

        model_layout.addWidget(self.lbl_model_title)
        model_layout.addWidget(self.combo_yolo_model)

        self.lbl_prompt_title = QLabel("")
        self.lbl_prompt_title.setObjectName("lbl_section_title")
        model_layout.addWidget(self.lbl_prompt_title)

        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("e.g. giant panda")
        self.txt_prompt.setText("giant panda")
        model_layout.addWidget(self.txt_prompt)

        left_layout.addWidget(model_group)

        # 4. Strategy Selection
        self.lbl_mode = QLabel("")
        self.lbl_mode.setObjectName("lbl_section_title")
        left_layout.addWidget(self.lbl_mode)

        self.radio_transfer = QRadioButton("")
        self.radio_recursive = QRadioButton("")
        self.radio_recursive.setChecked(True)
        self.radio_scratch = QRadioButton("")

        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.addButton(self.radio_transfer)
        self.btn_group_mode.addButton(self.radio_recursive)
        self.btn_group_mode.addButton(self.radio_scratch)

        left_layout.addWidget(self.radio_transfer)
        left_layout.addWidget(self.radio_recursive)
        left_layout.addWidget(self.radio_scratch)

        # 5. Parameters
        self.param_group = QGroupBox("")
        param_layout = QFormLayout()
        param_layout.setContentsMargins(5, 10, 5, 5)
        param_layout.setSpacing(5)

        # 【修改】使用自定义的 FocusScrollSpinBox
        self.spin_epochs = FocusScrollSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(30)
        self.lbl_epochs_title = QLabel("")
        param_layout.addRow(self.lbl_epochs_title, self.spin_epochs)

        # 【修改】使用自定义的 FocusScrollSpinBox
        self.spin_freeze = FocusScrollSpinBox()
        self.spin_freeze.setRange(0, 50)
        self.spin_freeze.setValue(10)
        self.lbl_freeze_title = QLabel("")
        param_layout.addRow(self.lbl_freeze_title, self.spin_freeze)

        # 【修改】使用自定义的 FocusScrollSpinBox
        self.spin_threshold = FocusScrollSpinBox()
        self.spin_threshold.setRange(1, 500)
        self.spin_threshold.setValue(self.train_threshold)
        self.spin_threshold.valueChanged.connect(self.update_threshold_logic)
        self.lbl_thresh_title = QLabel("")
        param_layout.addRow(self.lbl_thresh_title, self.spin_threshold)

        self.param_group.setLayout(param_layout)
        left_layout.addWidget(self.param_group)

        self.radio_transfer.toggled.connect(self.sync_params_with_mode)
        self.radio_recursive.toggled.connect(self.sync_params_with_mode)
        self.radio_scratch.toggled.connect(self.sync_params_with_mode)

        # 6. Monitor
        self.monitor_group = QGroupBox("")
        monitor_layout = QVBoxLayout()
        monitor_layout.setSpacing(5)
        monitor_layout.setContentsMargins(5, 10, 5, 5)

        progress_layout = QHBoxLayout()
        self.lbl_timer = QLabel("")
        self.lbl_timer.setStyleSheet("font-size: 11px; color: #555;")
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setTextVisible(True)
        progress_layout.addWidget(self.train_progress)
        monitor_layout.addLayout(progress_layout)
        monitor_layout.addWidget(self.lbl_timer, 0, Qt.AlignRight)

        metrics_layout = QHBoxLayout()
        self.lbl_map = QLabel("mAP50\n0.000")
        self.lbl_map.setObjectName("lbl_metric")
        self.lbl_map.setAlignment(Qt.AlignCenter)

        self.lbl_loss = QLabel("Loss\n0.00")
        self.lbl_loss.setObjectName("lbl_metric_loss")
        self.lbl_loss.setAlignment(Qt.AlignCenter)

        metrics_layout.addWidget(self.lbl_map)
        metrics_layout.addWidget(self.lbl_loss)
        monitor_layout.addLayout(metrics_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        monitor_layout.addWidget(self.log_text)
        self.monitor_group.setLayout(monitor_layout)
        left_layout.addWidget(self.monitor_group)

        left_layout.addStretch()

        # 7. Tips
        self.tips_label = QLabel("")
        self.tips_label.setObjectName("lbl_tips")
        self.tips_label.setTextFormat(Qt.RichText)
        self.tips_label.setWordWrap(True)
        left_layout.addWidget(self.tips_label)

        self.lbl_data_source = QLabel("")
        self.lbl_data_source.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_data_source)

        self.lbl_infer_result = QLabel("")
        self.lbl_infer_result.setWordWrap(True)
        self.lbl_infer_result.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_infer_result)

        # Scroll Area
        left_scroll = QScrollArea()
        left_scroll.setWidget(self.left_content)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(280)

        # Canvas and List
        self.canvas = GraphicsCanvas(self)
        self.canvas.new_box_created.connect(self.handle_new_box)
        self.canvas.setMinimumWidth(400)

        self.file_list_widget = QListWidget()
        self.file_list_widget.currentRowChanged.connect(self.change_image)
        self.file_list_widget.setMinimumWidth(180)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_scroll)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.file_list_widget)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([300, 1300, 200])
        self.splitter.splitterMoved.connect(self.update_dynamic_styles)

        main_layout.addWidget(self.splitter)
        self.setCentralWidget(main_widget)
        QTimer.singleShot(100, self.update_dynamic_styles)

    def update_dynamic_styles(self, pos=None, index=None):
        if not hasattr(self, 'left_content'): return

        style_sheet = """
            QWidget { font-family: "Segoe UI", "Microsoft YaHei"; font-size: 13px; color: #333; }
            QPushButton { border: 1px solid #ccc; border-radius: 4px; background-color: #f8f9fa; padding: 4px; min-height: 28px; }
            QPushButton:hover { background-color: #e2e6ea; }
            QPushButton:pressed { background-color: #dae0e5; }
            QPushButton#btn_normal { font-weight: bold; }
            QPushButton#btn_small { font-size: 12px; }
            QPushButton#btn_train { background-color: #e0e0e0; font-size: 16px; font-weight: bold; min-height: 45px; border-radius: 6px; margin-top: 5px; margin-bottom: 5px; }
            QComboBox, QSpinBox, QLineEdit { min-height: 28px; border: 1px solid #ccc; border-radius: 4px; padding-left: 5px; background-color: white; }
            QLineEdit { font-size: 14px; }
            QLabel#lbl_section_title { font-size: 14px; font-weight: bold; color: #444; margin-top: 8px; }
            QLabel#lbl_info { font-size: 15px; font-weight: bold; color: #0056b3; }
            QLabel#lbl_stats { font-size: 12px; color: #666; }
            QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; left: 10px; color: #555; }
            QLabel#lbl_metric { background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 4px; color: #2e7d32; font-weight: bold; font-size: 16px; padding: 4px; }
            QLabel#lbl_metric_loss { background-color: #ffebee; border: 1px solid #ffcdd2; border-radius: 4px; color: #c62828; font-weight: bold; font-size: 16px; padding: 4px; }
            QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; min-height: 18px; max-height: 18px; font-size: 11px; }
            QProgressBar::chunk { background-color: #28a745; }
            QTextEdit { font-family: Consolas; font-size: 11px; border: 1px solid #ddd; background-color: #fafafa; }
            QLabel#lbl_tips { font-size: 11px; color: #666; background-color: #f1f1f1; border-radius: 4px; padding: 6px; margin-top: 5px; }
        """
        self.left_content.setStyleSheet(style_sheet)

    def on_train_btn_clicked(self):
        if self.is_training:
            reply = QMessageBox.question(self, 'Stop Training', self.tr_text("MSG_CONFIRM_STOP"),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_training()
        else:
            self.start_background_training()

    def stop_training(self):
        if hasattr(self, 'training_worker') and self.training_worker.is_alive():
            self.training_worker.terminate()
            self.training_worker.join()
        self.is_training = False
        self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
        self.btn_force_train.setStyleSheet("")
        self.btn_force_train.setObjectName("btn_train")
        self.lbl_info.setText(self.tr_text("STATUS_TRAIN_STOPPED"))
        self.train_progress.setFormat("Stopped")
        self.log_text.append("!!! User stopped training !!!")
        while not self.train_queue.empty():
            try:
                self.train_queue.get_nowait()
            except:
                pass
        self.update_dynamic_styles()

    def export_voc(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Error", self.tr_text("MSG_NO_DATASET"))
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select Export Dir")
        if not save_dir: return
        try:
            self.status_label.setText("Exporting XML...")
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count = converter.to_pascal_voc(save_dir)
            QMessageBox.information(self, "Success", self.tr_text("MSG_EXPORT_SUCCESS").format(count, save_dir))
            self.status_label.setText("Export XML Done")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export Failed: {str(e)}")

    def export_coco(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Error", self.tr_text("MSG_NO_DATASET"))
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save COCO JSON", "", "JSON Files (*.json)")
        if not save_path: return
        try:
            self.status_label.setText("Exporting JSON...")
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count = converter.to_coco_json(save_path)
            QMessageBox.information(self, "Success", self.tr_text("MSG_EXPORT_SUCCESS").format(count, save_path))
            self.status_label.setText("Export JSON Done")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export Failed: {str(e)}")

    def import_coco_labels(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Error", self.tr_text("MSG_NO_DATASET"))
            return
        json_path, _ = QFileDialog.getOpenFileName(self, "Select JSON", "", "JSON Files (*.json)")
        if not json_path: return
        reply = QMessageBox.question(self, "Confirm", self.tr_text("MSG_IMPORT_CONFIRM"),
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        try:
            self.status_label.setText("Importing...")
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count, new_classes = converter.from_coco_json(json_path)
            if new_classes:
                self.classes.extend(new_classes)
                QMessageBox.information(self, "Classes Updated", f"New classes: {', '.join(new_classes)}")
            if self.current_index >= 0: self.change_image(self.current_index)
            QMessageBox.information(self, "Success", self.tr_text("MSG_IMPORT_SUCCESS").format(count))
            self.status_label.setText(f"Import Done: {count}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Import Failed: {str(e)}")

    def import_voc_labels(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Error", self.tr_text("MSG_NO_DATASET"))
            return
        xml_dir = QFileDialog.getExistingDirectory(self, "Select XML Folder")
        if not xml_dir: return
        reply = QMessageBox.question(self, "Confirm", self.tr_text("MSG_IMPORT_CONFIRM"),
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        try:
            self.status_label.setText("Importing XML...")
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count, new_classes = converter.from_pascal_voc(xml_dir)
            if new_classes:
                self.classes.extend(new_classes)
                QMessageBox.information(self, "Classes Updated", f"New classes: {', '.join(new_classes)}")
            if self.current_index >= 0: self.change_image(self.current_index)
            QMessageBox.information(self, "Success", self.tr_text("MSG_IMPORT_SUCCESS").format(count))
            self.status_label.setText(f"Import Done: {count}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Import Failed: {str(e)}")

    def on_base_model_changed(self, text):
        if not hasattr(self, 'inference_engine'): return
        if any(x in text for x in ["rcnn", "ssd", "retina"]):
            ext = ".pth"
        else:
            ext = ".pt"
        project_weights_dir = os.path.join(os.getcwd(), "runs", "detect", "loop_train", "weights")
        possible_trained_name = f"best_{text}{ext}"
        path_a = os.path.join(project_weights_dir, possible_trained_name)
        path_b = f"{text}{ext}"
        if os.path.exists(path_a):
            target_path = path_a
            display_msg = f"Loading Trained: {possible_trained_name} ..."
            self._pending_mode_transfer = "recursive"
        else:
            target_path = path_b
            display_msg = f"Downloading/Loading Base: {target_path} ..."
            self._pending_mode_transfer = "transfer"
        self.current_model_path = target_path
        self.combo_yolo_model.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.status_label.setText(display_msg)
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()
        self.loader_thread = ModelLoaderThread(self.inference_engine, target_path)
        self.loader_thread.finished_signal.connect(self.on_model_load_finished)
        self.loader_thread.start()

    def on_model_load_finished(self, success, msg):
        self.combo_yolo_model.setEnabled(True)
        self.btn_load.setEnabled(True)
        if success:
            self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(msg))
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            if hasattr(self, '_pending_mode_transfer'):
                if self._pending_mode_transfer == "recursive" and hasattr(self, 'radio_recursive'):
                    self.radio_recursive.setChecked(True)
                elif self._pending_mode_transfer == "transfer" and hasattr(self, 'radio_transfer'):
                    self.radio_transfer.setChecked(True)
            self.sync_params_with_mode()
            if self.current_index >= 0: self.change_image(self.current_index)
        else:
            self.status_label.setText(self.tr_text("MSG_MODEL_FAIL").format(msg))
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Load Failed:\n{msg}")

    def update_inference_label(self, text):
        self.lbl_infer_result.setText(text)
        if "(no detections)" in text:
            self.lbl_infer_result.setStyleSheet("font-size: 14px; color: gray; margin-bottom: 5px;")
        else:
            self.lbl_infer_result.setStyleSheet(
                "font-size: 14px; color: #00008b; font-weight: bold; margin-bottom: 5px;")

    def load_custom_model(self):
        start_dir = os.path.join(os.getcwd(), "runs", "detect", "loop_train", "weights")
        if not os.path.exists(start_dir): start_dir = os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", start_dir,
                                                   "Model Files (*.pt *.onnx *.pth *.yaml);;All Files (*)")
        if file_path:
            self.current_model_path = file_path
            self.inference_engine.load_model(file_path)
            file_name = os.path.basename(file_path)
            self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(file_name))
            is_onnx = file_path.lower().endswith(".onnx")
            if is_onnx:
                self.btn_force_train.setEnabled(False)
                self.btn_force_train.setText("ONNX (No Train)")
                QMessageBox.information(self, "ONNX Mode", self.tr_text("MSG_ONNX_MODE"))
            else:
                self.btn_force_train.setEnabled(True)
                self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
                if hasattr(self, 'radio_recursive'): self.radio_recursive.setChecked(True)
                QMessageBox.information(self, "Success", f"Model Loaded: {file_name}\nMode: Recursive")
            if self.current_index >= 0: self.change_image(self.current_index)

    def move_to_junk(self):
        if self.current_index < 0 or self.current_index >= len(self.image_list): return
        img_path = self.image_list[self.current_index]
        label_path = self.get_label_path(img_path)
        file_name = os.path.basename(img_path)
        junk_dir = os.path.join(self.dataset_root, "junk")
        if not os.path.exists(junk_dir): os.makedirs(junk_dir)
        try:
            if os.path.exists(img_path): shutil.move(img_path, os.path.join(junk_dir, file_name))
            if os.path.exists(label_path): shutil.move(label_path, os.path.join(junk_dir, os.path.basename(label_path)))
            self.remove_from_train_txt(img_path)
            self.status_label.setText(f"Moved to Junk: {file_name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Move failed:\n{e}")
            return
        self.file_list_widget.takeItem(self.current_index)
        self.image_list.pop(self.current_index)
        if self.current_index >= len(self.image_list): self.current_index = len(self.image_list) - 1
        if self.current_index >= 0:
            self.file_list_widget.setCurrentRow(self.current_index)
        else:
            self.canvas.scene.clear()
            self.status_label.setText("No images left")

    def remove_from_train_txt(self, img_path_to_remove):
        list_path = os.path.join(self.dataset_root, "train_list.txt")
        if not os.path.exists(list_path): return
        try:
            with open(list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            target_path = os.path.abspath(img_path_to_remove).strip()
            new_lines = [line for line in lines if os.path.abspath(line.strip()) != target_path]
            with open(list_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_dynamic_styles()

    def update_threshold_logic(self, value):
        self.train_threshold = value
        self.lbl_info.setText(
            self.tr_text("FMT_PROGRESS").format(self.verified_count_since_train, self.train_threshold))

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if folder: self.load_dataset_from_path(folder)

    def load_dataset_from_path(self, folder):
        if not os.path.exists(folder): return
        self.dataset_root = folder
        yaml_path = os.path.join(folder, "data.yaml")
        if os.path.exists(yaml_path):
            self.load_classes()
        else:
            dialog = ClassSelectionDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                self.classes = dialog.get_classes()
            else:
                self.classes = ["person"]
            self.save_data_yaml(yaml_path)

        img_dir = os.path.join(folder, "images")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(folder, "train", "images")
            if not os.path.exists(img_dir): img_dir = folder

        if os.path.exists(img_dir) and os.path.isdir(img_dir):
            self.image_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                               f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.file_list_widget.clear()
            self.file_list_widget.addItems([os.path.basename(f) for f in self.image_list])
            self.train_loop_count = 0
            self.verified_count_since_train = 0
            self.lbl_info.setText(self.tr_text("FMT_PROGRESS").format(0, self.train_threshold))
            self.lbl_total_stats.setText(self.tr_text("LBL_TOTAL_RESET"))
            self.check_and_load_local_model()
            if self.image_list: self.file_list_widget.setCurrentRow(0)
        else:
            QMessageBox.warning(self, "Error", "Images folder not found!")

    def save_data_yaml(self, yaml_path):
        data = {'path': self.dataset_root, 'train': 'train_list.txt', 'val': 'train_list.txt', 'nc': len(self.classes),
                'names': {i: name for i, name in enumerate(self.classes)}}
        with open(yaml_path, 'w', encoding='utf-8') as f: yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    def check_and_load_local_model(self):
        current_base_model = self.combo_yolo_model.currentText()
        ext = ".pth" if any(x in current_base_model for x in ['ssd', 'faster', 'retina']) else ".pt"
        local_weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
        local_best_model = os.path.join(local_weights_dir, f"best_{current_base_model}{ext}")
        if os.path.exists(local_best_model):
            self.current_model_path = local_best_model
            self.inference_engine.load_model(local_best_model)
            self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(local_best_model)))
            self.radio_recursive.setChecked(True)
        else:
            base_model_path = f"{current_base_model}{ext}"
            if not os.path.exists(base_model_path) and ext == ".pt": base_model_path = f"{current_base_model}.pt"
            self.current_model_path = base_model_path
            self.inference_engine.load_model(base_model_path)
            self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(current_base_model))
            self.radio_transfer.setChecked(True)

    def closeEvent(self, event):
        settings = QSettings("MyCompany", "AutoLabelPro")
        settings.setValue("dataset_root", self.dataset_root)
        settings.setValue("current_index", self.current_index)
        settings.setValue("model_path", self.current_model_path)
        mode_id = 2
        if self.radio_transfer.isChecked():
            mode_id = 1
        elif self.radio_recursive.isChecked():
            mode_id = 2
        elif self.radio_scratch.isChecked():
            mode_id = 3
        settings.setValue("train_mode", mode_id)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("base_model_index", self.combo_yolo_model.currentIndex())
        super().closeEvent(event)

    def restore_state(self):
        self.is_restoring = True
        try:
            settings = QSettings("MyCompany", "AutoLabelPro")
            idx = settings.value("base_model_index", 0, type=int)
            if 0 <= idx < self.combo_yolo_model.count(): self.combo_yolo_model.setCurrentIndex(idx)
            if settings.value("geometry"): self.restoreGeometry(settings.value("geometry"))
            if settings.value("windowState"): self.restoreState(settings.value("windowState"))
            last_root = settings.value("dataset_root")
            if last_root and isinstance(last_root, str) and os.path.exists(last_root):
                self.load_dataset_from_path(last_root)
                last_index = settings.value("current_index", 0, type=int)
                if 0 <= last_index < len(self.image_list): self.file_list_widget.setCurrentRow(last_index)
            last_model = settings.value("model_path")
            if last_model and isinstance(last_model, str) and os.path.exists(last_model):
                self.current_model_path = last_model
                self.inference_engine.load_model(self.current_model_path)
                self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(last_model)))
            mode_id = settings.value("train_mode", 2, type=int)
            if mode_id == 1:
                self.radio_transfer.setChecked(True)
            elif mode_id == 2:
                self.radio_recursive.setChecked(True)
            elif mode_id == 3:
                self.radio_scratch.setChecked(True)
            self.sync_params_with_mode()
        finally:
            self.is_restoring = False

    def sync_params_with_mode(self):
        if self.radio_transfer.isChecked():
            self.spin_epochs.setValue(40)
            self.spin_freeze.setValue(10)
            self.spin_freeze.setEnabled(True)
        elif self.radio_recursive.isChecked():
            self.spin_epochs.setValue(30)
            self.spin_freeze.setValue(0)
            self.spin_freeze.setEnabled(True)
        elif self.radio_scratch.isChecked():
            self.spin_epochs.setValue(100)
            self.spin_freeze.setValue(0)
            self.spin_freeze.setEnabled(False)

    def load_classes(self):
        yaml_path = os.path.join(self.dataset_root, "data.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    names = data.get('names', [])
                    if isinstance(names, dict):
                        self.classes = list(names.values())
                    elif isinstance(names, list):
                        self.classes = names
            except:
                pass
        if not self.classes: self.classes = ["object"]

    def change_image(self, index):
        if index < 0 or index >= len(self.image_list): return
        self.current_index = index
        img_path = self.image_list[index]
        pixmap = QPixmap(img_path)
        if pixmap.isNull(): return
        self.canvas.load_pixmap(pixmap, self.classes)
        label_path = self.get_label_path(img_path)
        if os.path.exists(label_path):
            self.lbl_data_source.setText(self.tr_text("STATUS_MANUAL"))
            self.lbl_data_source.setStyleSheet("font-size: 14px; color: green; font-weight: bold; margin-top: 5px;")
            self.load_existing_labels(label_path, pixmap.width(), pixmap.height())
            self.lbl_infer_result.setText("")
        else:
            self.lbl_data_source.setText(self.tr_text("STATUS_AI"))
            self.lbl_data_source.setStyleSheet("font-size: 14px; color: blue; margin-top: 5px;")
            # === 新增：获取 Prompt 并传递 ===
            prompt_text = self.txt_prompt.text().strip()
            # 传给引擎
            self.inference_engine.run_inference(img_path, prompt=prompt_text)
        self.canvas.setFocus()

    def get_label_path(self, img_path):
        path_obj = os.path.dirname(img_path)
        file_name = os.path.basename(img_path)
        label_name = os.path.splitext(file_name)[0] + ".txt"
        parent_dir = os.path.dirname(path_obj)
        if os.path.basename(path_obj) == "images":
            label_dir = os.path.join(parent_dir, "labels")
        else:
            label_dir = path_obj
        return os.path.join(label_dir, label_name)

    def load_existing_labels(self, path, img_w, img_h):
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_n, y_n, w_n, h_n = map(float, parts[1:5])
                        w = w_n * img_w
                        h = h_n * img_h
                        x = (x_n * img_w) - (w / 2)
                        y = (y_n * img_h) - (h / 2)
                        while cls >= len(self.classes): self.classes.append(str(len(self.classes)))
                        self.canvas.add_box(cls, x, y, w, h)
        except:
            pass

    @pyqtSlot(list)
    def on_inference_result(self, predictions):
        img_w = self.canvas.image_w
        img_h = self.canvas.image_h
        if img_w == 0 or img_h == 0 or not predictions: return
        target_classes = self.classes
        for pred in predictions:
            raw_cls_id = int(pred[0])
            x_n, y_n, w_n, h_n = pred[1:]
            final_cls_idx = -1
            detected_name = COCO_CLASSES[raw_cls_id] if raw_cls_id < len(COCO_CLASSES) else "unknown"
            if detected_name in target_classes:
                final_cls_idx = target_classes.index(detected_name)
            elif len(target_classes) == 1 and target_classes[0] not in COCO_CLASSES:
                final_cls_idx = 0
            else:
                continue
            if final_cls_idx >= 0:
                w = w_n * img_w
                h = h_n * img_h
                x = (x_n * img_w) - (w / 2)
                y = (y_n * img_h) - (h / 2)
                self.canvas.add_box(final_cls_idx, x, y, w, h)

    def handle_new_box(self, rect):
        QTimer.singleShot(0, lambda: self._show_class_dialog(rect))

    def _show_class_dialog(self, rect):
        current_selection = self.classes[0] if self.classes else ""
        dialog = LabelDialog(self, self.classes, current_selection)
        self.smart_move_dialog(dialog)
        if dialog.exec_() == QDialog.Accepted:
            new_label = dialog.get_label()
            if not new_label: return
            if new_label not in self.classes: self.classes.append(new_label)
            cls_idx = self.classes.index(new_label)
            self.canvas.add_box(cls_idx, rect.x(), rect.y(), rect.width(), rect.height())

    def edit_label(self, box_item):
        current_name = box_item.get_label_text().split(": ")[-1]
        dialog = LabelDialog(self, self.classes, current_name)
        self.smart_move_dialog(dialog)
        if dialog.exec_() == QDialog.Accepted:
            new_label = dialog.get_label()
            if not new_label: return
            if new_label not in self.classes: self.classes.append(new_label)
            box_item.cls_idx = self.classes.index(new_label)
            box_item.update_classes()

    def smart_move_dialog(self, dialog):
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos).availableGeometry()
        d_width, d_height = dialog.width(), dialog.height()
        target_x, target_y = cursor_pos.x() + 10, cursor_pos.y() + 10
        if target_y + d_height > screen.bottom(): target_y = cursor_pos.y() - d_height - 10
        if target_x + d_width > screen.right(): target_x = cursor_pos.x() - d_width - 10
        dialog.move(target_x, target_y)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.canvas.toggle_draw_mode()
        elif event.key() == Qt.Key_D:
            if self.current_index >= 0:
                self.save_current_annotation()
                self.verified_count_since_train += 1
                self.lbl_info.setText(
                    self.tr_text("FMT_PROGRESS").format(self.verified_count_since_train, self.train_threshold))
                self.check_trigger_training()
                if self.current_index < self.file_list_widget.count() - 1:
                    self.file_list_widget.setCurrentRow(self.current_index + 1)
                else:
                    QMessageBox.information(self, "Info", "End of list")
        elif event.key() == Qt.Key_A:
            if self.current_index > 0: self.file_list_widget.setCurrentRow(self.current_index - 1)
        elif event.key() == Qt.Key_J:
            if self.current_index >= 0: self.move_to_junk()
        elif event.key() == Qt.Key_Delete:
            self.canvas.delete_selected_boxes()

    def save_current_annotation(self):
        if not self.image_list: return
        img_path = self.image_list[self.current_index]
        label_path = self.get_label_path(img_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        img_w, img_h = self.canvas.image_w, self.canvas.image_h
        if img_w == 0 or img_h == 0: return
        try:
            with open(label_path, 'w') as f:
                for item in self.canvas.scene.items():
                    if isinstance(item, BoxItem):
                        scene_rect = item.mapRectToScene(item.rect())
                        x_center = (scene_rect.x() + scene_rect.width() / 2) / img_w
                        y_center = (scene_rect.y() + scene_rect.height() / 2) / img_h
                        norm_w = scene_rect.width() / img_w
                        norm_h = scene_rect.height() / img_h
                        f.write(f"{item.cls_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        except:
            pass

    def generate_train_list(self):
        valid_images = []
        for img_path in self.image_list:
            if os.path.exists(img_path):
                label_path = self.get_label_path(img_path)
                if os.path.exists(label_path): valid_images.append(os.path.abspath(img_path))
        if not valid_images: return None
        list_path = os.path.join(self.dataset_root, "train_list.txt")
        try:
            with open(list_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(valid_images))
            return list_path
        except:
            return None

    def update_data_yaml(self, train_list_path):
        yaml_path = os.path.join(self.dataset_root, "data.yaml")
        try:
            data = {}
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f: data = yaml.safe_load(f) or {}
            data['names'] = {i: name for i, name in enumerate(self.classes)}
            data['nc'] = len(self.classes)
            data['train'] = train_list_path
            data['val'] = train_list_path
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            return True
        except:
            return False

    def check_trigger_training(self):
        if self.verified_count_since_train >= self.train_threshold:
            self.verified_count_since_train = 0
            self.lbl_info.setText(self.tr_text("FMT_STARTING").format(self.train_threshold))
            self.start_background_training()

    def start_background_training(self):
        if self.is_training: return
        if not self.dataset_root:
            QMessageBox.warning(self, "Error", self.tr_text("MSG_NO_DATASET"))
            return
        list_path = self.generate_train_list()
        if not list_path:
            QMessageBox.warning(self, "Warning", "No annotated data.")
            return
        try:
            with open(list_path, 'r', encoding='utf-8') as f:
                self.current_batch_count = len([l for l in f if l.strip()])
        except:
            self.current_batch_count = 0
        if not self.update_data_yaml(list_path): return

        self.is_training = True
        self.btn_force_train.setText(self.tr_text("BTN_STOP_TRAIN"))
        self.btn_force_train.setStyleSheet("background-color: #ff4444; color: white; border: none;")
        self.lbl_info.setText("Training...")

        dataset_runs_dir = os.path.join(self.dataset_root, "training_runs")
        weights_dir = os.path.join(dataset_runs_dir, "loop_train", "weights")
        selected_model_name = self.combo_yolo_model.currentText()
        is_classic = any(x in selected_model_name.lower() for x in ['ssd', 'faster', 'retina', 'rcnn'])
        ext = ".pth" if is_classic else ".pt"
        expected_best_model_path = os.path.join(weights_dir, f"best_{selected_model_name}{ext}")

        if self.radio_transfer.isChecked():
            model_to_use = f"{selected_model_name}{ext}" if is_classic else f"{selected_model_name}.pt"
        elif self.radio_recursive.isChecked():
            if os.path.exists(expected_best_model_path):
                model_to_use = expected_best_model_path
            else:
                model_to_use = f"{selected_model_name}{ext}" if is_classic else f"{selected_model_name}.pt"
        else:
            if "rtdetr" in selected_model_name.lower() or "world" in selected_model_name.lower():
                model_to_use = f"{selected_model_name}.pt"
            else:
                model_to_use = f"{selected_model_name}.yaml"

        yaml_path = os.path.join(self.dataset_root, "data.yaml")
        self.training_worker = TrainingProcess(
            yaml_path, model_to_use, self.train_queue,
            epochs=self.spin_epochs.value(),
            freeze=self.spin_freeze.value(),
            project_dir=dataset_runs_dir
        )
        self.training_worker.start()

    def check_training_status(self):
        while not self.train_queue.empty():
            status, msg = self.train_queue.get()
            if status == "start":
                self.train_start_time = time.time()
                self.train_progress.setValue(0)
                self.train_progress.setFormat("Initializing...")
                self.lbl_timer.setText("Starting...")
                self.log_text.clear()
                self.log_text.append(">>> Training process connected...")
            elif status == "step":
                if isinstance(msg, dict):
                    epoch, total = msg.get('epoch', 0), msg.get('total', 100)
                    self.train_progress.setValue(int((epoch / total) * 100) if total > 0 else 0)
                    self.train_progress.setFormat(f"Epoch {epoch}/{total}")
                    if self.train_start_time: self.lbl_timer.setText(
                        f"Runtime: {(time.time() - self.train_start_time) / 3600:.4f} h")
                    self.lbl_map.setText(f"mAP50\n{msg.get('map50', 0.0):.3f}")
                    self.lbl_loss.setText(f"Loss\n{msg.get('loss', 0.0):.3f}")
                    self.log_text.append(
                        f"[Ep {epoch}/{total}] mAP: {msg.get('map50', 0.0):.3f} | Loss: {msg.get('loss', 0.0):.3f}")
                    self.log_text.moveCursor(self.log_text.textCursor().End)
            elif status == "success":
                self.is_training = False
                self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
                self.btn_force_train.setStyleSheet("")
                self.btn_force_train.setObjectName("btn_train")
                self.current_model_path = msg
                self.inference_engine.load_model(msg)
                self.train_loop_count += 1
                self.total_images_in_model = self.current_batch_count
                self.lbl_total_stats.setText(
                    self.tr_text("FMT_TOTAL_STATS").format(self.train_loop_count, self.total_images_in_model))
                duration = (time.time() - self.train_start_time) / 3600 if self.train_start_time else 0
                self.train_progress.setValue(100)
                self.train_progress.setFormat("Done")
                self.lbl_timer.setText(f"Done in {duration:.4f} h")
                self.lbl_timer.setStyleSheet("color: #28a745; font-weight: bold;")
                self.log_text.append(f">>> Model saved: {os.path.basename(msg)}")
                QMessageBox.information(self, "Update", f"Training Done!\nSaved: {os.path.basename(msg)}")
                self.lbl_info.setText(self.tr_text("FMT_PROGRESS").format(0, self.train_threshold))
                self.update_dynamic_styles()
                if self.current_index >= 0:
                    img_path = self.image_list[self.current_index]
                    if not os.path.exists(self.get_label_path(img_path)):
                        self.canvas.load_pixmap(QPixmap(img_path), self.classes)
                        # === 新增：获取 Prompt 并传递 ===
                        prompt_text = self.txt_prompt.text().strip()
                        # 传给引擎
                        self.inference_engine.run_inference(img_path, prompt=prompt_text)
            elif status == "error":
                self.is_training = False
                self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
                self.btn_force_train.setStyleSheet("")
                self.btn_force_train.setObjectName("btn_train")
                self.train_progress.setFormat("Error")
                self.lbl_timer.setText("Failed")
                self.log_text.append(f"!!! Error: {msg}")
                self.update_dynamic_styles()