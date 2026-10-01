# src/ui/main_window.py
import os
import shutil
import yaml
import traceback
import time
import re
import json
import glob
from multiprocessing import Queue

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QListWidget, QMessageBox, QAction, QDialog, QLineEdit, QCheckBox,
                             QRadioButton, QButtonGroup, QSpinBox, QFormLayout, QGroupBox,
                             QProgressBar, QTextEdit, QSplitter, QScrollArea, QFrame, QComboBox, QApplication,
                             QListWidgetItem, QToolBar, QToolButton, QMenu, QShortcut)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSettings
from PyQt5.QtGui import QPixmap, QIcon, QCursor, QColor, QKeySequence

# Import from other modules
from src.config import TRANS, COCO_CLASSES
from src.ui.canvas import GraphicsCanvas, BoxItem
from src.ui.dialogs import LabelDialog, ClassSelectionDialog, LabelSetupDialog
from src.utils.threads import ModelLoaderThread
from src.utils.app_id import resolve_icon_path
from src.utils.image_io import load_qpixmap
from src.ui.cleaning_dialog import CleaningDialog

# Import core logic
from src.core.training_worker import TrainingProcess
from src.core.inference_engine import InferenceEngine
from src.core.label_converter import LabelConverter

# Model catalog shared by the hidden sidebar combo and the toolbar "Load Model" menu.
BASE_MODELS = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "rtdetr-l", "rtdetr-x",
]
WORLD_MODELS = ["yolov8s-worldv2", "yolov8m-worldv2"]
CLASSIC_MODELS = ["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "ssd300_vgg16"]
ALL_MODELS = BASE_MODELS + WORLD_MODELS + CLASSIC_MODELS

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
        self.settings = QSettings("KeenForgeAI", "KeenForge")
        self.curr_lang = self.settings.value("language", "zh")

        _icon_path = resolve_icon_path()
        if _icon_path:
            self.setWindowIcon(QIcon(_icon_path))

        # Adaptive initial size for 1080P screens
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.80), int(screen.height() * 0.80))
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

        self.image_list = []
        self.current_index = -1
        self.classes = ["object"]
        self.verified_count_since_train = 0
        self.dataset_root = ""
        self.train_start_time = None
        self.current_model_path = "yolov8n.pt"
        self.is_training = False
        self.is_restoring = False
        self._ai_assist_enabled = True   # AI-assisted pre-labeling toggle

        # Initialize Inference Engine
        self.inference_engine = InferenceEngine(self.current_model_path)
        self.inference_engine.prediction_ready.connect(self.on_inference_result)
        self.inference_engine.model_loaded.connect(
            lambda p: self._show_model_status(p))
        self.inference_engine.inference_info.connect(self.update_inference_label)

        # Training Queue and Timer
        self.train_queue = Queue()
        self.train_timer = QTimer()
        self.train_timer.timeout.connect(self.check_training_status)
        self.train_timer.start(1000)

        self.train_loop_count = 0
        self.total_images_in_model = 0
        self.current_batch_count = 0
        self.last_model_name = ""
        self._active_train_model = ""  # basename of the model actually used for the last training
        self.last_epochs = 0
        self.last_map = 0.0

        # === Round-based clustering state ===
        self._full_image_list: list = []         # always contains ALL images (for re-clustering)
        self._round_is_active = False           # True during labeling phase (showing reps only)
        self._round_rep_count = 0               # number of representatives in current round
        self._round_rep_done = 0                # how many reps have been annotated
        self._round_threshold = 0.90            # similarity threshold for current round
        self._target_cluster_pct = 0.05         # 5% of remaining → target cluster count
        self._pending_round_ready = False       # quota met while a training run is still active → defer next-round training

        # === Round clustering config (from docs) ===
        self._cluster_initial_threshold = 0.90  # starting granularity
        self._cluster_min_clusters = 5          # clamp minimum clusters
        self._cluster_max_clusters = 50         # clamp maximum clusters
        self._badge_state = "ready"  # monitor badge state, kept for language refresh


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
        # Top toolbar buttons keep their emoji+text; refresh translated text
        self.tb_load.setText(self.tr_text("BTN_LOAD_DIR"))
        self.tb_load.setToolTip(self.tr_text("BTN_LOAD_DIR"))
        self.tb_dedup.setText(self.tr_text("BTN_DEDUP"))
        self.tb_dedup.setToolTip(self.tr_text("TIP_DEDUP"))
        self.tb_cluster.setText(self.tr_text("BTN_CLUSTER"))
        self.tb_cluster.setToolTip(self.tr_text("TIP_CLUSTER"))
        self.tb_label.setText(self.tr_text("BTN_LABEL_SETUP"))
        self.tb_model.setText(self.tr_text("BTN_MODEL"))
        self.tb_summary.setText(self.tr_text("BTN_SUMMARY"))
        self.tb_import.setText(self.tr_text("BTN_IMPORT"))
        self.tb_export.setText(self.tr_text("BTN_EXPORT"))
        self.tb_lang.setText(self.tr_text("BTN_LANG"))
        if self.is_training:
            self.btn_force_train.setText(self.tr_text("BTN_STOP_TRAIN"))
        else:
            self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
        self.lbl_info.setText(
            self.tr_text("FMT_PROGRESS").format(self.verified_count_since_train))
        # Recompute from the current image list so a language switch can never
        # resurrect the previous dataset's trained-image count.
        self._refresh_total_stats()
        self.status_label.setText(self.tr_text("TIP_READY"))
        self.chk_ai_assist.setText(self.tr_text("LBL_AI_ASSIST"))
        self.chk_ai_assist.setToolTip(self.tr_text("TIP_AI_ASSIST"))
        if hasattr(self, "chk_compare"):
            self.chk_compare.setText(self.tr_text("LBL_COMPARE"))
            self.chk_compare.setToolTip(self.tr_text("TIP_COMPARE"))
        if hasattr(self, 'lbl_prompt_title'):
            self.lbl_prompt_title.setText(self.tr_text("LBL_PROMPT_TITLE"))
            self.txt_prompt.setToolTip(self.tr_text("TIP_PROMPT_INPUT"))
            self.txt_prompt.setPlaceholderText(self.tr_text("PH_PROMPT"))
        self.combo_yolo_model.setToolTip(self.tr_text("TIP_MODEL_SELECT"))
        if hasattr(self, 'lbl_model_title'):
            self.lbl_model_title.setText(self.tr_text("LBL_BASE_MODEL"))
        if hasattr(self, 'lbl_mode'):
            self.lbl_mode.setText(self.tr_text("LBL_STRATEGY"))
        if hasattr(self, 'radio_transfer'):
            self.radio_transfer.setText(self.tr_text("MODE_TRANSFER"))
            self.radio_transfer.setToolTip(self.tr_text("TIP_TRANSFER"))
            self.radio_recursive.setText(self.tr_text("MODE_RECURSIVE"))
            self.radio_recursive.setToolTip(self.tr_text("TIP_RECURSIVE"))
            self.radio_scratch.setText(self.tr_text("MODE_SCRATCH"))
            self.radio_scratch.setToolTip(self.tr_text("TIP_SCRATCH"))
        if hasattr(self, 'param_group'):
            self.param_group.setTitle(self.tr_text("GRP_PARAMS"))
        if hasattr(self, 'lbl_epochs_title'):
            self.lbl_epochs_title.setText(self.tr_text("LBL_EPOCHS"))
        self.monitor_group.setTitle(self.tr_text("GRP_MONITOR"))
        self.lbl_timer.setText(self.tr_text("TIP_READY"))
        self.tips_label.setText(self.tr_text("TIPS_HTML"))
        if "AI" in self.lbl_data_source.text():
            self.lbl_data_source.setText(self.tr_text("STATUS_AI"))
        else:
            self.lbl_data_source.setText(self.tr_text("STATUS_MANUAL"))
        # Refresh dynamic content after language switch (file list + counts)
        try:
            idx = self.current_index
            self.file_list_widget.blockSignals(True)
            self.populate_file_list()
            self.file_list_widget.blockSignals(False)
            if 0 <= idx < self.file_list_widget.count():
                self.file_list_widget.setCurrentRow(idx)
                self.current_index = idx
        except Exception:
            pass
        # Refresh monitor card titles with the new language
        try:
            self.lbl_monitor_title.setText(self.tr_text("GRP_MONITOR"))
            self.lbl_map_title.setText(self.tr_text("MON_MAP50"))
            self.lbl_loss_title.setText(self.tr_text("MON_LOSS"))
            self.lbl_epoch_title.setText(self.tr_text("MON_EPOCH"))
            self._set_training_badge(self._badge_state)
        except Exception:
            pass
        # Refresh the persistent model-info label with the new language
        try:
            self._show_model_status(getattr(self, 'current_model_path', None))
        except Exception:
            pass

    def setup_ui(self):
        # Menus are owned by their QToolButton (setMenu takes ownership);
        # create them WITHOUT a parent to avoid double-delete aborts.
        self.menu_lang = QMenu()
        action_zh = QAction("简体中文", self)
        action_zh.triggered.connect(lambda: self.switch_language("zh"))
        self.menu_lang.addAction(action_zh)
        action_en = QAction("English", self)
        action_en.triggered.connect(lambda: self.switch_language("en"))
        self.menu_lang.addAction(action_en)

        self.menu_import = QMenu()
        self.action_import_coco = QAction("", self)
        self.action_import_coco = QAction("", self)
        self.action_import_coco.triggered.connect(self.import_coco_labels)
        self.menu_import.addAction(self.action_import_coco)
        self.action_import_voc = QAction("", self)
        self.action_import_voc.triggered.connect(self.import_voc_labels)
        self.menu_import.addAction(self.action_import_voc)

        self.menu_export = QMenu()
        self.action_export_voc = QAction("", self)
        self.action_export_voc = QAction("", self)
        self.action_export_voc.triggered.connect(self.export_voc)
        self.menu_export.addAction(self.action_export_voc)
        self.action_export_coco = QAction("", self)
        self.action_export_coco.triggered.connect(self.export_coco)
        self.menu_export.addAction(self.action_export_coco)

        # === Top action toolbar: grouped workflow, larger readable font ===
        self.toolbar = QToolBar("MainToolBar", self)
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(self.toolbar)

        def _mk_btn(text):
            b = QToolButton()
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            return b

        # ---- workflow: dataset -> cleaning -> clustering ----
        self.tb_load = _mk_btn(self.tr_text("BTN_LOAD_DIR"))
        self.tb_load.clicked.connect(self.load_folder)
        self.toolbar.addWidget(self.tb_load)

        self.toolbar.addSeparator()

        self.tb_dedup = _mk_btn(self.tr_text("BTN_DEDUP"))
        self.tb_dedup.setToolTip(self.tr_text("TIP_DEDUP"))
        self.tb_dedup.clicked.connect(self.open_dedup_dialog)
        self.toolbar.addWidget(self.tb_dedup)

        self.tb_cluster = _mk_btn(self.tr_text("BTN_CLUSTER"))
        self.tb_cluster.setToolTip(self.tr_text("TIP_CLUSTER"))
        self.tb_cluster.clicked.connect(self.open_cluster_dialog)
        self.toolbar.addWidget(self.tb_cluster)

        self.toolbar.addSeparator()

        # ---- annotation / model controls ----
        self.tb_label = _mk_btn(self.tr_text("BTN_LABEL_SETUP"))
        self.tb_label.clicked.connect(self.open_label_setup_dialog)
        self.toolbar.addWidget(self.tb_label)

        self.tb_model = _mk_btn(self.tr_text("BTN_MODEL"))
        self.tb_model.setPopupMode(QToolButton.InstantPopup)
        self.menu_model = QMenu()
        self.tb_model.setMenu(self.menu_model)
        self.menu_model.aboutToShow.connect(self._build_model_menu)
        self.toolbar.addWidget(self.tb_model)
        self._build_model_menu()

        self.tb_summary = _mk_btn(self.tr_text("BTN_SUMMARY"))
        self.tb_summary.clicked.connect(self._show_training_summary)
        self.toolbar.addWidget(self.tb_summary)

        self.toolbar.addSeparator()

        # ---- data I/O ----
        self.tb_import = QToolButton()
        self.tb_import.setText(self.tr_text("BTN_IMPORT"))
        self.tb_import.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tb_import.setAutoRaise(True)
        self.tb_import.setCursor(Qt.PointingHandCursor)
        self.tb_import.setPopupMode(QToolButton.InstantPopup)
        self.tb_import.setMenu(self.menu_import)
        self.toolbar.addWidget(self.tb_import)

        self.tb_export = QToolButton()
        self.tb_export.setText(self.tr_text("BTN_EXPORT"))
        self.tb_export.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tb_export.setAutoRaise(True)
        self.tb_export.setCursor(Qt.PointingHandCursor)
        self.tb_export.setPopupMode(QToolButton.InstantPopup)
        self.tb_export.setMenu(self.menu_export)
        self.toolbar.addWidget(self.tb_export)

        self.toolbar.addSeparator()

        # ---- settings ----
        self.tb_lang = QToolButton()
        self.tb_lang.setText(self.tr_text("BTN_LANG"))
        self.tb_lang.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.tb_lang.setAutoRaise(True)
        self.tb_lang.setCursor(Qt.PointingHandCursor)
        self.tb_lang.setPopupMode(QToolButton.InstantPopup)
        self.tb_lang.setMenu(self.menu_lang)
        self.toolbar.addWidget(self.tb_lang)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # === Left Sidebar ===

        self.left_content = QWidget()

        left_layout = QVBoxLayout(self.left_content)

        left_layout.setContentsMargins(8, 8, 8, 8)

        left_layout.setSpacing(6)



        # (icon buttons moved to the top toolbar)
        # Train button stays as the prominent left-side action

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
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")

        left_layout.addWidget(self.lbl_info)
        left_layout.addWidget(self.lbl_total_stats)

        # AI 预标注辅助开关（与标签设置对话框里的开关同步）
        self.chk_ai_assist = QCheckBox("")
        self.chk_ai_assist.setChecked(True)
        self.chk_ai_assist.toggled.connect(self._on_ai_assist_toggled)
        left_layout.addWidget(self.chk_ai_assist)

        # 模型比对开关：对已标注的图片也跑一遍模型，预测框用青色虚线显示，
        # 便于人工复核模型与人工标注的差异（预测框不会被写入标签文件）
        self.chk_compare = QCheckBox("")
        self.chk_compare.setChecked(False)
        self.chk_compare.setToolTip(self.tr_text("TIP_COMPARE"))
        self.chk_compare.toggled.connect(self._on_compare_toggled)
        left_layout.addWidget(self.chk_compare)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setStyleSheet("background-color: #ddd;")
        left_layout.addWidget(line1)

        # 3. Model Settings
        # 3. Model + strategy + params: settings live in TrainingSetupDialog now.
        #    The objects below are still REQUIRED (their .currentText()/.value()/
        #    .isChecked() feed training defaults + closeEvent persistence), so we
        #    keep them as plain attributes but NEVER add them to the layout —
        #    no visible widgets, no setVisible(False) bookkeeping.
        self.combo_yolo_model = FocusScrollComboBox()
        self.combo_yolo_model.addItems(ALL_MODELS)
        self.combo_yolo_model.currentTextChanged.connect(self.on_base_model_changed)

        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText(self.tr_text("PH_PROMPT"))
        self.txt_prompt.setText("giant panda")

        self.radio_transfer = QRadioButton("")
        self.radio_recursive = QRadioButton("")
        self.radio_recursive.setChecked(True)
        self.radio_scratch = QRadioButton("")

        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.addButton(self.radio_transfer)
        self.btn_group_mode.addButton(self.radio_recursive)
        self.btn_group_mode.addButton(self.radio_scratch)

        self.spin_epochs = FocusScrollSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(30)
        self.spin_freeze = FocusScrollSpinBox()
        self.spin_freeze.setRange(0, 50)
        self.spin_freeze.setValue(10)

        # 6. Monitor
        self.monitor_group = QGroupBox("")
        monitor_layout = QVBoxLayout()
        monitor_layout.setSpacing(5)
        monitor_layout.setContentsMargins(5, 10, 5, 5)

        # Header: title + status badge
        monitor_header = QHBoxLayout()
        self.lbl_monitor_title = QLabel(self.tr_text("GRP_MONITOR"))
        self.lbl_monitor_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #444;")
        self.lbl_badge = QLabel(self.tr_text("MON_BADGE_READY"))
        self.lbl_badge.setStyleSheet(
            "background: #9e9e9e; color: white; border-radius: 8px; "
            "padding: 2px 8px; font-size: 11px; font-weight: bold;")
        monitor_header.addWidget(self.lbl_monitor_title)
        monitor_header.addStretch()
        monitor_header.addWidget(self.lbl_badge)
        monitor_layout.addLayout(monitor_header)

        # Progress + elapsed time
        progress_layout = QHBoxLayout()
        self.lbl_timer = QLabel("")
        self.lbl_timer.setStyleSheet("font-size: 11px; color: #555;")
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setTextVisible(True)
        progress_layout.addWidget(self.train_progress, 1)
        progress_layout.addWidget(self.lbl_timer, 0, Qt.AlignRight)
        monitor_layout.addLayout(progress_layout)

        # Metric cards: mAP50 / Loss / Epoch
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(6)
        card_map, self.lbl_map, self.lbl_map_title = self._make_metric_card(
            "metric_val_map", "MON_MAP50", "#0056b3")
        card_loss, self.lbl_loss, self.lbl_loss_title = self._make_metric_card(
            "metric_val_loss", "MON_LOSS", "#c62828")
        card_epoch, self.lbl_epoch, self.lbl_epoch_title = self._make_metric_card(
            "metric_val_epoch", "MON_EPOCH", "#37474f")
        self.lbl_epoch.setText("0 / 0")
        for card in (card_map, card_loss, card_epoch):
            metrics_layout.addWidget(card)
        monitor_layout.addLayout(metrics_layout)

        # Training log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        monitor_layout.addWidget(self.log_text)
        self.monitor_group.setLayout(monitor_layout)
        left_layout.addWidget(self.monitor_group)

        # 数据状态（人工标注 / 模型推理）紧跟训练监控下方
        self.data_group = QGroupBox("")
        data_layout = QVBoxLayout(self.data_group)
        data_layout.setContentsMargins(8, 8, 8, 8)
        data_layout.setSpacing(4)

        self.lbl_data_source = QLabel("")
        self.lbl_data_source.setObjectName("lbl_data_source")
        self.lbl_data_source.setAlignment(Qt.AlignCenter)
        data_layout.addWidget(self.lbl_data_source)

        self.lbl_infer_result = QLabel("")
        self.lbl_infer_result.setObjectName("lbl_infer_result")
        self.lbl_infer_result.setWordWrap(True)
        self.lbl_infer_result.setAlignment(Qt.AlignCenter)
        data_layout.addWidget(self.lbl_infer_result)

        left_layout.addWidget(self.data_group)


        left_layout.addStretch()

        # tips 留在底部（导航 / 画布操作 / 快捷键说明）
        self.tips_label = QLabel("")
        self.tips_label.setObjectName("lbl_tips")
        self.tips_label.setTextFormat(Qt.RichText)
        self.tips_label.setWordWrap(True)
        left_layout.addWidget(self.tips_label)

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
        # Right-side list panel: stats label + file list
        list_panel = QWidget()
        list_panel_layout = QVBoxLayout(list_panel)
        list_panel_layout.setContentsMargins(4, 4, 4, 4)
        list_panel_layout.setSpacing(4)
        # ── Find image: collapsed to a 🔍 button until the user opens it ──
        # Rows are HIDDEN (never removed), so the row index stays parallel to
        # self.image_list and every index-based operation keeps working.
        self._find_query = ""
        self.lbl_list_stats = QLabel("")
        self.lbl_list_stats.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #1a5276; "
            "background: #e8f4fd; border-radius: 4px; padding: 4px;")

        self.btn_find = QToolButton()
        self.btn_find.setText("🔍")   # always visible: opens/closes the search box
        self.btn_find.setToolTip(self.tr_text("TIP_FIND_IMAGE"))
        self.btn_find.setCheckable(True)
        self.btn_find.setAutoRaise(True)
        self.btn_find.setCursor(Qt.PointingHandCursor)
        self.btn_find.setFocusPolicy(Qt.NoFocus)
        self.btn_find.setStyleSheet(
            "QToolButton { border: 1px solid #ccc; border-radius: 4px; "
            "padding: 1px 6px; font-size: 14px; background-color: white; }"
            "QToolButton:hover { background: #e2e6ea; }"
            "QToolButton:checked { background: #d0e4ff; border-color: #0056b3; }")
        self.btn_find.clicked.connect(self._toggle_find_box)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)
        header_row.addWidget(self.lbl_list_stats, 1)
        header_row.addWidget(self.btn_find, 0)
        list_panel_layout.addLayout(header_row)

        self.txt_find = QLineEdit()
        self.txt_find.setPlaceholderText(self.tr_text("PH_FIND_IMAGE"))
        self.txt_find.setToolTip(self.tr_text("TIP_FIND_IMAGE"))
        self.txt_find.setClearButtonEnabled(True)
        self.txt_find.setStyleSheet(
            "QLineEdit { min-height: 26px; border: 1px solid #ccc; border-radius: 4px; "
            "padding-left: 6px; background-color: white; }")
        self.txt_find.textChanged.connect(self._apply_file_filter)
        self.txt_find.returnPressed.connect(self._jump_to_first_match)
        self.txt_find.setVisible(False)        # hidden until 🔍 is clicked
        list_panel_layout.addWidget(self.txt_find)
        # Esc inside the box collapses it again
        self._find_esc = QShortcut(QKeySequence(Qt.Key_Escape), self.txt_find)
        self._find_esc.setContext(Qt.WidgetShortcut)
        self._find_esc.activated.connect(self._collapse_find_box)
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentRowChanged.connect(self.change_image)
        self.file_list_widget.setMinimumWidth(180)
        list_panel_layout.addWidget(self.file_list_widget, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_scroll)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(list_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([300, 1300, 200])
        self.splitter.splitterMoved.connect(self.update_dynamic_styles)

        main_layout.addWidget(self.splitter)
        self.setCentralWidget(main_widget)

        # Status bar: status_label lives at the bottom now (standard location)
        self.statusBar().addWidget(self.status_label, 1)
        # Live cursor coordinate (right side of the status bar)
        # Persistent model info (middle of the status bar, never overwritten)
        self.lbl_model_status = QLabel("")
        self.lbl_model_status.setStyleSheet("color: #0056b3; font-size: 12px; padding: 0 8px;")
        self.statusBar().addWidget(self.lbl_model_status)
        self.lbl_cursor = QLabel("  ")
        self.lbl_cursor = QLabel("  ")
        self.lbl_cursor.setStyleSheet("color: #333; font-size: 12px; padding: 0 6px;")
        self.statusBar().addPermanentWidget(self.lbl_cursor)
        # Canvas mode feedback
        self.canvas.mode_changed.connect(self._on_canvas_mode_changed)
        self.canvas.cursor_pos.connect(self._on_canvas_cursor_pos)
        QTimer.singleShot(100, self.update_dynamic_styles)
        QTimer.singleShot(150, lambda: self._on_canvas_mode_changed(self.canvas.mode))

    def _on_canvas_cursor_pos(self, x, y):
        """Live image coordinates in the status bar (X-AnyLabeling style)."""
        self.lbl_cursor.setText(f" X: {int(x)}, Y: {int(y)} ")

    def _on_canvas_mode_changed(self, mode):
        """Show the current canvas interaction mode in the status bar."""
        if mode == getattr(self.canvas, 'MODE_CREATE', 0):
            self.status_label.setText(self.tr_text("MSG_STATUS_DRAW_MODE"))
        else:
            self.status_label.setText(self.tr_text("MSG_STATUS_EDIT_MODE"))

    def update_dynamic_styles(self, pos=None, index=None):
        if not hasattr(self, 'left_content'): return

        style_sheet = """
            QWidget { font-family: "Segoe UI", "Microsoft YaHei"; font-size: 12px; color: #333; }
            QPushButton { border: 1px solid #ccc; border-radius: 4px; background-color: #f8f9fa; padding: 4px; min-height: 28px; }
            QPushButton:hover { background-color: #e2e6ea; }
            QPushButton:pressed { background-color: #dae0e5; }
            QPushButton#btn_normal { font-weight: bold; }
            QPushButton#btn_small { font-size: 12px; }
            QPushButton#btn_train { background-color: #e0e0e0; font-size: 16px; font-weight: bold; min-height: 45px; border-radius: 6px; margin-top: 5px; margin-bottom: 5px; }
            QComboBox, QSpinBox, QLineEdit { min-height: 28px; border: 1px solid #ccc; border-radius: 4px; padding-left: 5px; background-color: white; }
            QLineEdit { font-size: 13px; }
            QLabel#lbl_section_title { font-size: 13px; font-weight: bold; color: #444; margin-top: 8px; }
            QLabel#lbl_info { font-size: 14px; font-weight: bold; color: #0056b3; }
            QLabel#lbl_stats { font-size: 12px; color: #666; }
            QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 6px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; left: 10px; color: #555; }
            QLabel#metric_val_map { font-size: 18px; font-weight: bold; }
            QLabel#metric_val_loss { font-size: 18px; font-weight: bold; }
            QLabel#metric_val_epoch { font-size: 18px; font-weight: bold; }
            QLabel#metric_title { font-size: 11px; color: #888; }
            QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; min-height: 22px; max-height: 22px; font-size: 12px; }
            QProgressBar::chunk { background-color: #28a745; }
            QTextEdit { font-family: Consolas; font-size: 11px; border: 1px solid #ddd; background-color: #fafafa; }
            QLabel#lbl_tips { font-size: 11px; color: #555; background-color: #f7f8fa; border: 1px solid #e8eaed; border-radius: 6px; padding: 8px; margin-top: 5px; }
            QLabel#lbl_data_source { font-size: 12px; color: #555; }
            QLabel#lbl_infer_result { font-size: 13px; font-weight: bold; }
        """
        self.left_content.setStyleSheet(style_sheet)
        # Toolbar style: clean grouped workflow, larger readable font
        try:
            self.toolbar.setStyleSheet(
                "QToolBar { background: #f5f6f8; border-bottom: 1px solid #dfe3e8; "
                "spacing: 6px; padding: 6px 10px; }"
                "QToolBar QToolButton { border: none; border-radius: 6px; padding: 6px 12px; "
                "font-size: 14px; color: #333; background: transparent; }"
                "QToolBar QToolButton:hover { background: #e4e9f0; }"
                "QToolBar QToolButton:pressed, QToolBar QToolButton:checked { "
                "background: #d0e4ff; color: #0056b3; }"
                "QToolBar::separator { background: #d0d4da; width: 1px; margin: 4px 6px; }")
        except Exception:
            pass

    def _make_metric_card(self, value_obj_name, title_key, color):
        """Build a metric card (small title + bold value); returns (card, value_label, title_label)."""
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: white; border: 1px solid #e0e0e0; border-radius: 6px; }")
        v = QVBoxLayout(card)
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(0)
        title = QLabel(self.tr_text(title_key))
        title.setStyleSheet("font-size: 10px; color: #888;")
        title.setAlignment(Qt.AlignCenter)
        value = QLabel("0.000")
        value.setObjectName(value_obj_name)
        value.setAlignment(Qt.AlignCenter)
        value.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: %s;" % color)
        v.addWidget(title)
        v.addWidget(value)
        return card, value, title

    def _append_log(self, text, level="info"):
        """Append a timestamped, color-coded line to the training log."""
        try:
            ts = time.strftime("[%H:%M:%S] ")
            colors = {"info": "#333333", "success": "#1a7f37", "error": "#cf222e"}
            color = colors.get(level, "#333333")
            self.log_text.append(
                '<span style="color:%s;">%s%s</span>' % (color, ts, text))
        except Exception:
            pass

    def _set_training_badge(self, state):
        """state: ready | training | done | failed"""
        styles = {
            "ready": ("#9e9e9e", "MON_BADGE_READY"),
            "training": ("#1565c0", "MON_BADGE_TRAINING"),
            "done": ("#2e7d32", "MON_BADGE_DONE"),
            "failed": ("#c62828", "MON_BADGE_FAILED"),
        }
        bg, key = styles.get(state, styles["ready"])
        self._badge_state = state if state in styles else "ready"
        try:
            self.lbl_badge.setText(self.tr_text(key))
            self.lbl_badge.setStyleSheet(
                "background: %s; color: white; border-radius: 8px; "
                "padding: 2px 8px; font-size: 11px; font-weight: bold;" % bg)
        except Exception:
            pass

    def on_train_btn_clicked(self):
        if self.is_training:
            reply = QMessageBox.question(self, self.tr_text("TTL_STOP_TRAINING"), self.tr_text("MSG_CONFIRM_STOP"),
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_training()
        else:
            # Manual training trigger: show setup dialog to pick model & params
            if not self.dataset_root:
                QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_DATASET"))
                return
            from src.ui.training_dialog import TrainingSetupDialog
            done = sum(1 for p in self.image_list
                      if os.path.exists(self.get_label_path(p)))
            last_model = self.last_model_name or self._get_last_trained_model()
            dialog = TrainingSetupDialog(
                self,
                round_number=self.train_round + 1,
                current_model=self.combo_yolo_model.currentText(),
                current_epochs=self.spin_epochs.value(),
                current_freeze=self.spin_freeze.value(),
                annotated_count=done,
                representative_count=done,
                dataset_size=len(self.image_list),
                total_annotated=done,
                last_model_name=last_model,
                last_epochs=self.last_epochs,
                last_map=self.last_map,
                trained_models=self._list_trained_models(),
            )
            if dialog.exec_() == QDialog.Accepted:
                self.start_background_training(dialog.get_params())

    def stop_training(self):
        if hasattr(self, 'training_worker') and self.training_worker.is_alive():
            self.training_worker.terminate()
            self.training_worker.join()
        self.is_training = False
        self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
        self.btn_force_train.setStyleSheet("")
        self.btn_force_train.setObjectName("btn_train")
        self.lbl_info.setText(self.tr_text("STATUS_TRAIN_STOPPED"))
        self.train_progress.setFormat(self.tr_text("MSG_STATUS_STOPPED"))
        self._append_log(self.tr_text("MSG_USER_STOPPED"), "info")
        self._set_training_badge("ready")
        while not self.train_queue.empty():
            try:
                self.train_queue.get_nowait()
            except:
                pass
        self.update_dynamic_styles()

    def export_voc(self):
        if not self.dataset_root:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_DATASET"))
            return
        save_dir = QFileDialog.getExistingDirectory(self, self.tr_text("TTL_SELECT_EXPORT_DIR"))
        if not save_dir: return
        try:
            self.status_label.setText(self.tr_text("MSG_STATUS_EXPORTING_XML"))
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count = converter.to_pascal_voc(save_dir)
            QMessageBox.information(self, self.tr_text("TTL_SUCCESS"), self.tr_text("MSG_EXPORT_SUCCESS").format(count, save_dir))
            self.status_label.setText(self.tr_text("MSG_STATUS_EXPORT_XML_DONE"))
        except Exception as e:
            QMessageBox.critical(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_EXPORT_FAILED").format(str(e)))

    def export_coco(self):
        if not self.dataset_root:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_DATASET"))
            return
        save_path, _ = QFileDialog.getSaveFileName(self, self.tr_text("TTL_SAVE_COCO"), "", "JSON Files (*.json)")
        if not save_path: return
        try:
            self.status_label.setText(self.tr_text("MSG_STATUS_EXPORTING_JSON"))
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count = converter.to_coco_json(save_path)
            QMessageBox.information(self, self.tr_text("TTL_SUCCESS"), self.tr_text("MSG_EXPORT_SUCCESS").format(count, save_path))
            self.status_label.setText(self.tr_text("MSG_STATUS_EXPORT_JSON_DONE"))
        except Exception as e:
            QMessageBox.critical(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_EXPORT_FAILED").format(str(e)))

    def import_coco_labels(self):
        if not self.dataset_root:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_DATASET"))
            return
        json_path, _ = QFileDialog.getOpenFileName(self, self.tr_text("TTL_SELECT_JSON"), "", "JSON Files (*.json)")
        if not json_path: return
        reply = QMessageBox.question(self, self.tr_text("TTL_CONFIRM"), self.tr_text("MSG_IMPORT_CONFIRM"),
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        try:
            self.status_label.setText(self.tr_text("MSG_STATUS_IMPORTING"))
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count, new_classes = converter.from_coco_json(json_path)
            if new_classes:
                self.classes.extend(new_classes)
                QMessageBox.information(self, self.tr_text("TTL_CLASSES_UPDATED"), self.tr_text("MSG_NEW_CLASSES").format(', '.join(new_classes)))
            if self.current_index >= 0: self.change_image(self.current_index)
            QMessageBox.information(self, self.tr_text("TTL_SUCCESS"), self.tr_text("MSG_IMPORT_SUCCESS").format(count))
            self.status_label.setText(self.tr_text("MSG_STATUS_IMPORT_DONE").format(count))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_IMPORT_FAILED").format(str(e)))

    def import_voc_labels(self):
        if not self.dataset_root:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_DATASET"))
            return
        xml_dir = QFileDialog.getExistingDirectory(self, self.tr_text("TTL_SELECT_XML_DIR"))
        if not xml_dir: return
        reply = QMessageBox.question(self, self.tr_text("TTL_CONFIRM"), self.tr_text("MSG_IMPORT_CONFIRM"),
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No: return
        try:
            self.status_label.setText(self.tr_text("MSG_STATUS_IMPORTING"))
            QApplication.processEvents()
            converter = LabelConverter(self.dataset_root, self.classes)
            count, new_classes = converter.from_pascal_voc(xml_dir)
            if new_classes:
                self.classes.extend(new_classes)
                QMessageBox.information(self, self.tr_text("TTL_CLASSES_UPDATED"), self.tr_text("MSG_NEW_CLASSES").format(', '.join(new_classes)))
            # 持久化本次发现的类别名，保证每个标签的类别索引下次加载时能映射回真实名字
            # （此前 VOC 导入不写 data.yaml，类别名会丢失）。
            self.save_data_yaml(os.path.join(self.dataset_root, "data.yaml"))
            skipped = getattr(converter, 'skipped_classes', [])
            if skipped:
                QMessageBox.warning(self, self.tr_text("TTL_WARNING"),
                                    self.tr_text("MSG_SKIPPED_CLASSES").format(', '.join(skipped)))
            if self.current_index >= 0: self.change_image(self.current_index)
            QMessageBox.information(self, self.tr_text("TTL_SUCCESS"), self.tr_text("MSG_IMPORT_SUCCESS").format(count))
            self.status_label.setText(self.tr_text("MSG_STATUS_IMPORT_DONE").format(count))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_IMPORT_FAILED").format(str(e)))

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
            display_msg = self.tr_text("MSG_LOADING_TRAINED").format(possible_trained_name)
            self._pending_mode_transfer = "recursive"
        else:
            target_path = path_b
            display_msg = self.tr_text("MSG_LOADING_BASE").format(target_path)
            self._pending_mode_transfer = "transfer"
        self.current_model_path = target_path
        self.combo_yolo_model.setEnabled(False)
        self.tb_load.setEnabled(False)
        self.status_label.setText(display_msg)
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()
        self.loader_thread = ModelLoaderThread(self.inference_engine, target_path)
        self.loader_thread.finished_signal.connect(self.on_model_load_finished)
        self.loader_thread.start()

    def on_model_load_finished(self, success, msg):
        self.combo_yolo_model.setEnabled(True)
        self.tb_load.setEnabled(True)
        if success:
            self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(msg))
            self._show_model_status(msg)
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
            QMessageBox.critical(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_LOAD_FAILED").format(msg))

    def update_inference_label(self, text):
        display_text = text.replace("(no detections)", self.tr_text("INF_NO_DETECTIONS"))
        self.lbl_infer_result.setText(display_text)
        if "(no detections)" in text:
            self.lbl_infer_result.setStyleSheet("font-size: 14px; color: gray; margin-bottom: 5px;")
        else:
            self.lbl_infer_result.setStyleSheet(
                "font-size: 14px; color: #00008b; font-weight: bold; margin-bottom: 5px;")

    def load_custom_model(self):
        start_dir = os.path.join(os.getcwd(), "runs", "detect", "loop_train", "weights")
        if not os.path.exists(start_dir): start_dir = os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr_text("TTL_SELECT_MODEL"), start_dir,
                                                   "Model Files (*.pt *.onnx *.pth *.yaml);;All Files (*)")
        if file_path:
            self._apply_loaded_model(file_path, notify=True)

    def _apply_loaded_model(self, file_path, notify=True):
        """Load a model file and update UI/state. notify=False suppresses the success popup."""
        self.current_model_path = file_path
        self.inference_engine.load_model(file_path)
        file_name = os.path.basename(file_path)
        self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(file_name))
        self._show_model_status(file_path)
        if file_path.lower().endswith(".onnx"):
            self.btn_force_train.setEnabled(False)
            self.btn_force_train.setText(self.tr_text("MSG_BTN_ONNX_NOTRAIN"))
            if notify:
                QMessageBox.information(self, self.tr_text("TTL_ONNX"), self.tr_text("MSG_ONNX_MODE"))
        else:
            self.btn_force_train.setEnabled(True)
            self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
            if hasattr(self, 'radio_recursive'): self.radio_recursive.setChecked(True)
            if notify:
                QMessageBox.information(self, self.tr_text("TTL_SUCCESS"), self.tr_text("MSG_MODEL_LOADED_SUCCESS").format(file_name))
        if self.current_index >= 0: self.change_image(self.current_index)

    def _resolve_world_model_path(self):
        """Return a YOLO-World model path to load (prefer an existing local file)."""
        for name in WORLD_MODELS:
            p = f"{name}.pt"
            if os.path.exists(p):
                return p
        return f"{WORLD_MODELS[0]}.pt"

    def _list_trained_models(self):
        """Return [(full_path, basename), ...] of trained round models, newest first."""
        if not self.dataset_root:
            return []
        weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
        if not os.path.isdir(weights_dir):
            return []
        try:
            cands = [f for f in os.listdir(weights_dir)
                     if f.startswith("best_") and f.endswith((".pt", ".pth"))]
            cands.sort(key=lambda f: os.path.getmtime(os.path.join(weights_dir, f)), reverse=True)
            return [(os.path.join(weights_dir, f), f) for f in cands]
        except Exception:
            return []

    def _select_and_load_model(self, name):
        """Programmatically select + load a named base/world model."""
        idx = self.combo_yolo_model.findText(name)
        if idx >= 0:
            self.combo_yolo_model.blockSignals(True)
            self.combo_yolo_model.setCurrentIndex(idx)
            self.combo_yolo_model.blockSignals(False)
        self.on_base_model_changed(name)

    def _build_model_menu(self):
        """(Re)build the 'Load Model' toolbar menu: own-trained / base / world."""
        self.menu_model.clear()

        own_menu = QMenu(self.tr_text("BTN_MODEL_OWN"), self)
        trained = self._list_trained_models()
        if trained:
            for full_path, base in trained:
                act = own_menu.addAction(base)
                act.triggered.connect(lambda checked, p=full_path: self._apply_loaded_model(p, notify=False))
        else:
            no_act = own_menu.addAction(self.tr_text("MSG_NO_TRAINED_MODEL"))
            no_act.setEnabled(False)
        own_menu.addSeparator()
        act_file = own_menu.addAction(self.tr_text("BTN_MODEL_OWN_FILE"))
        act_file.triggered.connect(self.load_custom_model)
        self.menu_model.addMenu(own_menu)

        base_menu = QMenu(self.tr_text("BTN_MODEL_BASE"), self)
        for name in BASE_MODELS:
            act = base_menu.addAction(name)
            act.triggered.connect(lambda checked, n=name: self._select_and_load_model(n))
        self.menu_model.addMenu(base_menu)

        world_menu = QMenu(self.tr_text("BTN_MODEL_WORLD"), self)
        for name in WORLD_MODELS:
            act = world_menu.addAction(name)
            act.triggered.connect(lambda checked, n=name: self._select_and_load_model(n))
        self.menu_model.addMenu(world_menu)

    def _prune_cluster_map(self, paths):
        """Remove the given paths from cluster_map.json (single load/save)."""
        if not paths:
            return
        try:
            from src.core.data_cleaner import ClusterManager
            cm = ClusterManager(self.dataset_root)
            if not cm.load():
                return
            changed = False
            for p in paths:
                if cm.remove_path(p):
                    changed = True
            if changed:
                cm.save()
        except Exception:
            pass

    def _reset_monitor_state(self):
        """Clear the training-monitor sidebar (round counters, metrics, progress,
        timer, badge, log) so switching datasets never shows the training data of
        the previously loaded dataset."""
        self.total_images_in_model = 0
        self.current_batch_count = 0
        self.last_model_name = ""
        self._active_train_model = ""
        self.last_epochs = 0
        self.last_map = 0.0
        self.train_start_time = None
        self.is_training = False
        try:
            self.log_text.clear()
            self.train_progress.setValue(0)
            self.train_progress.setFormat("")
            self.lbl_timer.setText(self.tr_text("TIP_READY"))
            self.lbl_timer.setStyleSheet("font-size: 11px; color: #555;")
            self.lbl_map.setText("0.000")
            self.lbl_loss.setText("0.000")
            self.lbl_epoch.setText("0 / 0")
            self._set_training_badge("ready")
            self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
            self.btn_force_train.setStyleSheet("")
            self.btn_force_train.setObjectName("btn_train")
        except Exception:
            pass

    def _refresh_total_stats(self):
        """Recompute the cumulative stats label from the current image list."""
        try:
            total_ann = sum(1 for p in self.image_list
                            if os.path.exists(self.get_label_path(p)))
            self.lbl_total_stats.setText(
                self.tr_text("FMT_TOTAL_STATS").format(
                    getattr(self, 'train_round', 0), total_ann))
        except Exception:
            pass

    def _after_image_removed(self, img_path):
        """Sync all lists/counts after an image was moved out of the dataset
        (junk / cleaning removal): prune _full_image_list + cluster_map.json and
        refresh the visible statistics."""
        try:
            if self._full_image_list:
                self._full_image_list = [p for p in self._full_image_list if p != img_path]
        except Exception:
            pass
        self._prune_cluster_map([img_path])
        try:
            self._update_list_stats()
        except Exception:
            pass
        self._refresh_total_stats()

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
            self.status_label.setText(self.tr_text("MSG_STATUS_MOVED_JUNK").format(file_name))
        except Exception as e:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_MOVE_FAILED").format(e))
            return
        self.file_list_widget.takeItem(self.current_index)
        self.image_list.pop(self.current_index)
        if self.current_index >= len(self.image_list): self.current_index = len(self.image_list) - 1
        if self.current_index >= 0:
            self.file_list_widget.setCurrentRow(self.current_index)
        else:
            self.canvas.scene.clear()
            self.status_label.setText(self.tr_text("MSG_STATUS_NO_IMAGES"))
        # Keep every list / counter / cluster map in sync, then re-check whether
        # the round is complete — a deleted rep must not block training.
        self._after_image_removed(img_path)
        self.check_trigger_training()

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


    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr_text("TTL_SELECT_DATASET"))
        if folder: self.load_dataset_from_path(folder)

    def load_dataset_from_path(self, folder):
        if not os.path.exists(folder): return
        self.dataset_root = folder

        # ── 1. Resolve images dir & build image list ──
        img_dir = os.path.join(folder, "images")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(folder, "train", "images")
            if not os.path.exists(img_dir): img_dir = folder

        if not (os.path.exists(img_dir) and os.path.isdir(img_dir)):
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_IMAGES_FOLDER_NOT_FOUND"))
            return
        self.image_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if
                           f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self._full_image_list = list(self.image_list)  # FULL copy for re-clustering
        self._round_is_active = False
        self._round_rep_count = 0
        self._round_rep_done = 0
        self._pending_round_ready = False       # new dataset → drop any deferred training state

        if not self.image_list:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_IMAGES_FOUND"))
            return

        # ── Reset the training monitor so the left sidebar never shows the
        #    previous dataset's training data (round / mAP / loss / log) ──
        self._reset_monitor_state()

        # ── 2. Data cleaning is OPT-IN (no longer forced on load) ──
        # The wizard used to auto-open here, which blocked inspecting a
        # freshly loaded dataset's ORIGINAL labels. Cleaning is now started
        # explicitly from the toolbar: 除重 (dedup) or 聚类 (clustering).

        # ── 3. Class / label setup is OPT-IN ──
        # LabelSetupDialog used to be forced open here. Users often just want to
        # LOOK at a dataset, so it is no longer auto-opened: classes are read from
        # data.yaml (or default to "person") and the dialog is available from the
        # toolbar (标签设置 / open_label_setup_dialog) whenever the user wants it.
        yaml_path = os.path.join(folder, "data.yaml")
        if os.path.exists(yaml_path):
            self.load_classes()
        else:
            self.classes = ["person"]
        self._load_ai_settings()

        # ── 4. Populate list (cluster-coloured if available) ──
        self.populate_file_list()
        self.train_loop_count = 0
        self.verified_count_since_train = 0
        self.lbl_info.setText(self.tr_text("FMT_PROGRESS").format(0))
        total_ann = sum(1 for p in self.image_list
                        if os.path.exists(self.get_label_path(p)))
        self.lbl_total_stats.setText(
            self.tr_text("FMT_TOTAL_STATS").format(self.train_round, total_ann))
        self.check_and_load_local_model()
        if self.image_list: self.file_list_widget.setCurrentRow(0)
        # NOTE: Do NOT auto-enter round mode on restart.
        # Round mode is only entered via:
        #   1. Initial clustering (cleaning wizard → _save_cluster_map_from_dialog)
        #   2. User clicking "重新聚类并开始标注" in next-round dialog
    def save_data_yaml(self, yaml_path):
        data = {'path': self.dataset_root, 'train': 'train_list.txt', 'val': 'train_list.txt', 'nc': len(self.classes),
                'names': {i: name for i, name in enumerate(self.classes)}}
        with open(yaml_path, 'w', encoding='utf-8') as f: yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    # ── AI-assist settings persistence ──
    def _ai_settings_path(self):
        return os.path.join(self.dataset_root, "annotation_settings.json")

    def _load_ai_settings(self):
        """Load AI-assist toggle + training round from dataset settings."""
        self._ai_assist_enabled = True
        self._compare_mode = False
        self.train_round = 0
        try:
            import json as _json
            with open(self._ai_settings_path(), 'r', encoding='utf-8') as f:
                data = _json.load(f)
            self._ai_assist_enabled = bool(data.get('ai_assist', True))
            self._compare_mode = bool(data.get('compare_mode', False))
            self.train_round = int(data.get('train_round', 0) or 0)
        except Exception:
            pass
        self._sync_ai_assist_checkbox()
        self._sync_compare_checkbox()

    def _save_ai_settings(self):
        """Persist AI-assist toggle + training round to dataset settings."""
        import json as _json
        try:
            with open(self._ai_settings_path(), 'w', encoding='utf-8') as f:
                _json.dump({
                    'ai_assist': bool(self._ai_assist_enabled),
                    'compare_mode': bool(getattr(self, '_compare_mode', False)),
                    'train_round': int(getattr(self, 'train_round', 0)),
                }, f, indent=2)
        except Exception:
            pass

    def _sync_ai_assist_checkbox(self):
        """同步左侧栏 AI 辅助复选框与 _ai_assist_enabled（阻断信号防止循环）。"""
        if hasattr(self, 'chk_ai_assist'):
            self.chk_ai_assist.blockSignals(True)
            self.chk_ai_assist.setChecked(bool(self._ai_assist_enabled))
            self.chk_ai_assist.blockSignals(False)

    def _sync_compare_checkbox(self):
        """同步左侧栏「模型比对」复选框与 _compare_mode（阻断信号防止循环）。"""
        if hasattr(self, 'chk_compare'):
            self.chk_compare.blockSignals(True)
            self.chk_compare.setChecked(bool(getattr(self, '_compare_mode', False)))
            self.chk_compare.blockSignals(False)

    def _on_compare_toggled(self, checked):
        """模型比对开关：对已标注图片也跑模型，预测框用青色虚线显示。"""
        self._compare_mode = bool(checked)
        if not self.dataset_root:
            return
        self._save_ai_settings()
        # 刷新当前图，开关立即生效
        if 0 <= self.current_index < len(self.image_list):
            self.change_image(self.current_index)

    def _on_ai_assist_toggled(self, checked):
        """左侧栏 AI 预标注辅助开关：应用 + 保存 + 刷新当前图。"""
        self._ai_assist_enabled = bool(checked)
        if not self.dataset_root:
            return
        self._save_ai_settings()
        if checked and not self._has_trained_model():
            # 开启且无训练模型、当前不是 World 模型 → 自动切到 World 零样本模型
            if not getattr(self.inference_engine, 'is_world_model', False):
                self.check_and_load_local_model()
        if 0 <= self.current_index < len(self.image_list):
            self.change_image(self.current_index)

    def _get_last_trained_model(self) -> str:
        """Return basename of the most recent locally-trained model, or ''."""
        if not self.dataset_root:
            return ''
        weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
        if not os.path.isdir(weights_dir):
            return ''
        try:
            candidates = [f for f in os.listdir(weights_dir)
                          if f.startswith("best_") and f.endswith((".pt", ".pth"))]
            if not candidates:
                return ''
            candidates.sort(
                key=lambda f: os.path.getmtime(os.path.join(weights_dir, f)), reverse=True)
            return candidates[0]
        except Exception:
            return ''

    def _has_trained_model(self) -> bool:
        """True if this dataset already has a locally-trained model (best_*.pt)."""
        return bool(self._get_last_trained_model())

    def _round_model_info(self, model_name: str) -> str:
        """Append round/mAP info. Parses round-named files first (best_x_r7_m0938.pt),
        then falls back to last_round.json for legacy names."""
        if not model_name or not self.dataset_root:
            return ""
        import re as _re
        m = _re.search(r"_r(\d+)_m(\d{3,4})", str(model_name))
        if m:
            info = self.tr_text("FMT_ROUND_PRODUCT").format(int(m.group(1)))
            if m.group(2):
                info += self.tr_text("FMT_ROUND_MAP").format(int(m.group(2)) / 1000.0)
            return info
        try:
            import json as _json
            with open(os.path.join(self.dataset_root, "last_round.json"), 'r',
                      encoding='utf-8') as f:
                meta = _json.load(f)
            if meta.get("model") == model_name:
                info = self.tr_text("FMT_ROUND_PRODUCT").format(meta.get('round', '?'))
                if meta.get("mAP"):
                    info += self.tr_text("FMT_ROUND_MAP").format(float(meta['mAP']))
                return info
        except Exception:
            pass
        return ""

    def _load_training_history(self) -> list:
        """Load training_history.json -> list of {round, model, mAP, ...}."""
        try:
            import json as _json
            with open(os.path.join(self.dataset_root, "training_history.json"), 'r',
                      encoding='utf-8') as f:
                data = _json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_training_history(self, history):
        try:
            import json as _json
            with open(os.path.join(self.dataset_root, "training_history.json"), 'w',
                      encoding='utf-8') as f:
                _json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _list_trained_models(self):
        """Return [(display_text, filename)] of archived round models, best mAP first."""
        result = []
        try:
            weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
            if not os.path.isdir(weights_dir):
                return result
            entries = []
            seen = set()
            for e in self._load_training_history():
                nm = e.get("model", "")
                if nm and nm not in seen and os.path.exists(os.path.join(weights_dir, nm)):
                    seen.add(nm)
                    entries.append((nm, float(e.get("mAP", 0.0) or 0.0)))
            for p in glob.glob(os.path.join(weights_dir, "best_*.pt")):
                nm = os.path.basename(p)
                if re.search(r"_r\d+_m\d{3,4}\.pt$", nm) and nm not in seen:
                    seen.add(nm)
                    entries.append((nm, 0.0))
            entries.sort(key=lambda t: t[1], reverse=True)
            for nm, _mp in entries:
                info = self._round_model_info(nm)
                result.append(((nm + " " + info).strip() if info else nm, nm))
        except Exception:
            pass
        return result

    def _show_model_status(self, model_path):
        """Persist current model info in the status bar (never overwritten by mode hints)."""
        try:
            name = os.path.basename(str(model_path))
            self.lbl_model_status.setText(
                name + self._round_model_info(name))
        except Exception:
            pass
    def _resolve_best_round_model(self, weights_dir):
        """Return the absolute path of the best trained .pt model:
        highest mAP in training_history.json -> last_round.json record ->
        newest round-named file. Raw intermediates (best.pt / unsuffixed
        best_{model}.pt) are ignored. Returns None when nothing is found."""
        if not os.path.isdir(weights_dir):
            return None
        try:
            candidates = [p for p in glob.glob(os.path.join(weights_dir, "best_*.pt"))
                          if re.search(r"_r\d+_m\d{3,4}\.pt$", os.path.basename(p))]
        except Exception:
            return None
        if not candidates:
            return None
        chosen = None
        best_map = -1.0
        try:
            with open(os.path.join(self.dataset_root, "training_history.json"), 'r',
                      encoding='utf-8') as f:
                hist = json.load(f)
            if isinstance(hist, list):
                for e in hist:
                    nm = e.get("model", "")
                    if e.get("mAP", 0) > best_map and os.path.exists(
                            os.path.join(weights_dir, nm)):
                        best_map = e.get("mAP", 0)
                        chosen = os.path.join(weights_dir, nm)
        except Exception:
            chosen = None
        if chosen is None:
            meta_model = None
            try:
                with open(os.path.join(self.dataset_root, "last_round.json"), 'r',
                          encoding='utf-8') as f:
                    meta_model = json.load(f).get("model")
            except Exception:
                pass
            if meta_model:
                for p in candidates:
                    if os.path.basename(p) == meta_model:
                        chosen = p
                        break
        if chosen is None:
            chosen = max(candidates, key=os.path.getmtime)
        return chosen

    def check_and_load_local_model(self):
        current_base_model = self.combo_yolo_model.currentText()
        ext = ".pth" if any(x in current_base_model for x in ['ssd', 'faster', 'retina']) else ".pt"
        local_weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
        if ext == ".pt":
            # Deploy the highest-mAP round model (history -> last_round -> newest).
            # Raw intermediates (best.pt / unsuffixed best_{model}.pt) are never preferred.
            chosen = self._resolve_best_round_model(local_weights_dir)
            if chosen:
                self.current_model_path = chosen
                self.inference_engine.load_model(chosen)
                self.status_label.setText(
                    self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(chosen))
                    + self._round_model_info(os.path.basename(chosen)))
                self._show_model_status(chosen)
                self.radio_recursive.setChecked(True)
                return
        else:
            local_best_model = os.path.join(local_weights_dir, f"best_{current_base_model}{ext}")
            if os.path.exists(local_best_model):
                self.current_model_path = local_best_model
                self.inference_engine.load_model(local_best_model)
                self.status_label.setText(
                    self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(local_best_model))
                    + self._round_model_info(os.path.basename(local_best_model)))
                self._show_model_status(local_best_model)
                self.radio_recursive.setChecked(True)
                return
        # No local trained model → first round. Zero-shot via YOLO-World when AI assist is on.
        if getattr(self, '_ai_assist_enabled', True):
            world_path = self._resolve_world_model_path()
            self.current_model_path = world_path
            self.inference_engine.load_model(world_path)
            self.status_label.setText(
                self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(world_path)))
            self._show_model_status(world_path)
            self.radio_transfer.setChecked(True)
            return
        base_model_path = f"{current_base_model}{ext}"
        if not os.path.exists(base_model_path) and ext == ".pt": base_model_path = f"{current_base_model}.pt"
        self.current_model_path = base_model_path
        self.inference_engine.load_model(base_model_path)
        self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(current_base_model))
        self._show_model_status(base_model_path)
        self.radio_transfer.setChecked(True)

    def closeEvent(self, event):
        settings = QSettings("KeenForgeAI", "KeenForge")
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
            settings = QSettings("KeenForgeAI", "KeenForge")
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
                # 若数据集已有本地训练模型且 QSettings 记录的是基础模型,
                # 保留 check_and_load_local_model 已加载的训练模型(带轮次meta),不再覆盖
                is_trained = "best_" in os.path.basename(last_model) or "training_runs" in str(last_model)
                if is_trained or (not self._has_trained_model()
                                  and not getattr(self, '_ai_assist_enabled', True)):
                    self.current_model_path = last_model
                    self.inference_engine.load_model(self.current_model_path)
                    self.status_label.setText(self.tr_text("MSG_MODEL_LOADED").format(os.path.basename(last_model))
                                            + self._round_model_info(os.path.basename(last_model)))
                    self._show_model_status(last_model)
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
        pixmap = load_qpixmap(img_path)
        if pixmap.isNull(): return
        self.canvas.load_pixmap(pixmap, self.classes)
        label_path = self.get_label_path(img_path)
        if os.path.exists(label_path):
            self.lbl_data_source.setText(self.tr_text("STATUS_MANUAL"))
            self.lbl_data_source.setStyleSheet("font-size: 14px; color: green; font-weight: bold; margin-top: 5px;")
            self.load_existing_labels(label_path, pixmap.width(), pixmap.height())
            self.lbl_infer_result.setText("")
            # Model-comparison mode: also run the deployed model on this already
            # annotated image, so the reviewer can see where the model disagrees
            # with the human labels. Its boxes are drawn in cyan and are never
            # saved unless the user edits them.
            if getattr(self, '_compare_mode', False):
                prompt_text = self._effective_prompt()
                self.inference_engine.run_inference(img_path, prompt=prompt_text)
        else:
            self.lbl_data_source.setText(self.tr_text("STATUS_AI"))
            self.lbl_data_source.setStyleSheet("font-size: 14px; color: blue; margin-top: 5px;")
            # AI-assisted pre-labeling: only when the toggle is enabled
            if getattr(self, '_ai_assist_enabled', True):
                prompt_text = self._effective_prompt()
                self.inference_engine.run_inference(img_path, prompt=prompt_text)

        # Keyboard focus must stay on the canvas, otherwise plain A/D
        # (prev / next image) are swallowed by whatever widget holds focus
        # (e.g. the file list's type-ahead search) and the image never advances.
        # NOTE: this call used to sit inside _effective_prompt() *after* its
        # return statement, so it was unreachable dead code and A/D were dead.
        self.canvas.setFocus()

    def _effective_prompt(self):
        """Return the zero-shot prompt: always the dataset classes (prompt field is hidden)."""
        return ", ".join(self.classes)

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
                        while cls >= len(self.classes):
                            # 绝不按索引编造类别名（曾把 1..9 写成 "1".."9" 存进 data.yaml）；
                            # 这里只放明确标记的占位名，用户需在「标签设置」里改成真实类别名。
                            self.classes.append(f"class_{len(self.classes)}")
                        self.canvas.add_box(cls, x, y, w, h)
        except:
            pass

    @pyqtSlot(list)
    def on_inference_result(self, predictions):
        img_w = self.canvas.image_w
        img_h = self.canvas.image_h
        if img_w == 0 or img_h == 0 or not predictions: return
        custom = getattr(self.inference_engine, 'is_custom_trained', False)
        is_world = getattr(self.inference_engine, 'is_world_model', False)
        unmatched = set()
        for pred in predictions:
            raw_cls_id = int(pred[0])
            # [cls_id, xc, yc, w, h] with an optional confidence as element 6.
            x_n, y_n, w_n, h_n = pred[1:5]
            conf = float(pred[5]) if len(pred) > 5 else 1.0
            final_cls_idx = -1
            if custom or is_world:
                # Locally-trained / YOLO-World model: map via the model's OWN class names
                # (trained order) to the CURRENT dataset classes by NAME,
                # so reordering/renaming classes stays consistent.
                names_map = None
                try:
                    names_map = getattr(self.inference_engine.model, 'names', None)
                except Exception:
                    names_map = None
                if isinstance(names_map, dict):
                    detected_name = names_map.get(raw_cls_id)
                elif isinstance(names_map, (list, tuple)) and raw_cls_id < len(names_map):
                    detected_name = names_map[raw_cls_id]
                else:
                    detected_name = None
                if detected_name and detected_name in self.classes:
                    final_cls_idx = self.classes.index(detected_name)
                elif 0 <= raw_cls_id < len(self.classes):
                    final_cls_idx = raw_cls_id
                elif self.classes:
                    # Unmatched custom-model class -> suggestion box (first class)
                    final_cls_idx = 0
                    if detected_name:
                        unmatched.add(str(detected_name))
            else:
                # Standard COCO model: map COCO name to target classes
                detected_name = COCO_CLASSES[raw_cls_id] if raw_cls_id < len(COCO_CLASSES) else "unknown"
                if detected_name in self.classes:
                    final_cls_idx = self.classes.index(detected_name)
                elif self.classes:
                    # Unmatched class (e.g. "bear" on a raccoon dataset) -> still draw
                    # a suggestion box so the user only needs to fix the label
                    final_cls_idx = 0
                    if detected_name and detected_name != "unknown":
                        unmatched.add(str(detected_name))
            if final_cls_idx >= 0:
                w = w_n * img_w
                h = h_n * img_h
                x = (x_n * img_w) - (w / 2)
                y = (y_n * img_h) - (h / 2)
                # When comparing against an already-annotated image, tag these as
                # model predictions: cyan, dashed, and never saved unless edited.
                # In compare mode EVERY prediction is the model's opinion, so it
                # is always an AI box (cyan, dashed, never auto-saved) - even on
                # an image that has no labels of its own yet. Outside compare mode
                # it stays a normal pre-label that behaves as before.
                self.canvas.add_box(final_cls_idx, x, y, w, h,
                                    is_ai=bool(getattr(self, '_compare_mode', False)),
                                    conf=conf)
        if unmatched:
            try:
                self.status_label.setText(
                    self.tr_text("FMT_SUGGEST_BOX").format(', '.join(sorted(unmatched))))
            except Exception:
                pass

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
        try:
            from PyQt5.QtWidgets import QDesktopWidget
            desktop = QDesktopWidget()
            screen_num = desktop.screenNumber(cursor_pos)
            screen = desktop.screenGeometry(screen_num)
        except Exception:
            # Fallback: primary screen geometry
            screen = QApplication.primaryScreen().availableGeometry()
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
                    self.tr_text("FMT_PROGRESS").format(self.verified_count_since_train))
                self.check_trigger_training()
                if self.current_index < self.file_list_widget.count() - 1:
                    self.file_list_widget.setCurrentRow(self.current_index + 1)
                else:
                    QMessageBox.information(self, self.tr_text("TTL_INFO"), self.tr_text("MSG_END_OF_LIST"))
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
                        # NEVER write unreviewed model predictions into the
                        # label file. A prediction becomes saveable only after
                        # the user edits it (BoxItem.promote_to_human sets
                        # is_ai=False), so unreviewed predictions stay out.
                        if getattr(item, 'is_ai', False):
                            continue
                        scene_rect = item.mapRectToScene(item.rect())
                        # Clamp to the image bounds so labels never go out of
                        # range (YOLO requires 0 <= coords <= 1)
                        x1 = max(0.0, min(scene_rect.x(), img_w))
                        y1 = max(0.0, min(scene_rect.y(), img_h))
                        x2 = max(0.0, min(scene_rect.x() + scene_rect.width(), img_w))
                        y2 = max(0.0, min(scene_rect.y() + scene_rect.height(), img_h))
                        if x2 - x1 < 1 or y2 - y1 < 1:
                            continue
                        x_center = (x1 + x2) / 2 / img_w
                        y_center = (y1 + y2) / 2 / img_h
                        norm_w = (x2 - x1) / img_w
                        norm_h = (y2 - y1) / img_h
                        f.write(f"{item.cls_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        except:
            pass
        # Refresh the list stats counter
        try:
            self._update_list_stats()
        except Exception:
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
            return True, ""
        except Exception as e:
            try:
                print(traceback.format_exc())
            except Exception:
                pass
            return False, str(e)

    def _round_progress(self):
        """Quota-based round progress. Returns (progress, target, rep_done, extras)
        or None when there is no cluster map.
        progress = labeled reps + extra labels saved after the map was created;
        target = original representative count (persisted in cluster_map.json)."""
        try:
            from src.core.data_cleaner import ClusterManager
            cm = ClusterManager(self.dataset_root)
            if not cm.load():
                return None
            clusters = cm.get_clusters()
            if not clusters:
                return None
            ids = [c.get("cluster_id", 0) for c in clusters]
            rep_target = cm.data.get("rep_target")
            if not rep_target:
                # Legacy map: ids are assigned 0..N-1, so original count = max+1
                rep_target = (max(ids) + 1) if ids else len(clusters)
                try:
                    cm.data["rep_target"] = rep_target
                    cm.save()
                except Exception:
                    pass
            reps = [c.get("representative", "") for c in clusters]
            reps = [r for r in reps if r and os.path.exists(r)]
            rep_done = sum(1 for r in reps if os.path.exists(self.get_label_path(r)))
            try:
                created = time.mktime(time.strptime(str(cm.data.get("created", "")),
                                                     "%Y-%m-%d %H:%M:%S"))
            except Exception:
                try:
                    created = os.path.getmtime(cm.cluster_map_path)
                except Exception:
                    created = 0
            extras = 0
            try:
                rep_set = set(os.path.normpath(r) for r in reps)
                for f in os.listdir(self.dataset_root):
                    if not f.endswith(".txt"):
                        continue
                    base = os.path.splitext(f)[0]
                    img = None
                    for _ext in (".jpg", ".jpeg", ".png", ".bmp"):
                        cand = os.path.join(self.dataset_root, base + _ext)
                        if os.path.exists(cand):
                            img = cand
                            break
                    if img is None or os.path.normpath(img) in rep_set:
                        continue
                    if os.path.getmtime(os.path.join(self.dataset_root, f)) > created:
                        extras += 1
            except Exception:
                pass
            return (rep_done + extras, rep_target, rep_done, extras)
        except Exception as e:
            print(f"_round_progress error: {e}")
            return None
    def check_trigger_training(self):
        """Trigger training:
        - When all representatives in the current round are annotated → auto-advance.
        - Falls back to threshold-based trigger if no cluster_map exists.
        """
        from src.ui.training_dialog import TrainingSetupDialog

        # Quota-based round progress: labeled reps + extra labels vs target
        prog = self._round_progress()
        round_number = self.train_round + 1
        if not prog:
            return
        progress, rep_target, rep_done, extras = prog
        self._round_rep_count = rep_target
        self._round_rep_done = progress
        if progress < rep_target:
            self.lbl_info.setText(
                self.tr_text("FMT_ROUND_QUOTA").format(
                    round_number, progress, rep_target, rep_done, extras,
                    rep_target - progress))
            return

        # Quota met, but a training run is still active: notify ONCE and defer
        # instead of popping a dialog whose confirm would silently no-op. The
        # pending flag is consumed in the training-success branch to auto-open
        # the next-round setup as soon as the current run actually finishes.
        if self.is_training:
            if not getattr(self, '_pending_round_ready', False):
                self._pending_round_ready = True
                QMessageBox.information(
                    self, self.tr_text("TTL_INFO"),
                    self.tr_text("MSG_ROUND_WAIT_TRAINING").format(
                        round_number, round_number - 1))
            self.status_label.setText(
                self.tr_text("FMT_ROUND_WAIT_TRAINING").format(round_number))
            return

        # Quota met and no active run → restore full list, then show the dialog
        # ── Next-round gate: called when a round's quota is met with no active
        # training run. Restores the full list and opens the setup dialog. Kept
        # as a separate method so the training-success branch can re-trigger it
        # once a deferred round becomes eligible.
        self._restore_full_list()
        self.lbl_info.setText(
            self.tr_text("FMT_ROUND_REPS_DONE").format(round_number))
        last_model = self.last_model_name or self._get_last_trained_model()
        total_annotated = sum(1 for p in self._full_image_list
                              if os.path.exists(self.get_label_path(p)))
        dialog = TrainingSetupDialog(
            self,
            round_number=round_number,
            current_model=self.combo_yolo_model.currentText(),
            current_epochs=self.spin_epochs.value(),
            current_freeze=self.spin_freeze.value(),
            annotated_count=progress,
            representative_count=rep_target,
            dataset_size=len(self._full_image_list) if self._full_image_list else len(self.image_list),
            total_annotated=total_annotated,
            last_model_name=last_model,
            last_epochs=self.last_epochs,
            last_map=self.last_map,
            trained_models=self._list_trained_models(),
        )
        if dialog.exec_() == QDialog.Accepted:
            self.verified_count_since_train = 0
            self.start_background_training(dialog.get_params())

    def start_background_training(self, params: dict = None):
        """Start training. If params is given (from TrainingSetupDialog), it overrides UI."""
        if self.is_training: return
        if not self.dataset_root:
            QMessageBox.warning(self, self.tr_text("TTL_ERROR"), self.tr_text("MSG_NO_DATASET"))
            return

        params = params or {}
        selected_model_name = params.get("model") or self.combo_yolo_model.currentText()

        # A custom weights file picked via "浏览…" in the setup dialog: the combo
        # stores an ABSOLUTE path to an existing .pt/.pth. It bypasses the model
        # catalog entirely and becomes the training starting point for any round.
        custom_model_path = None
        if (selected_model_name and selected_model_name.lower().endswith((".pt", ".pth"))
                and os.path.isfile(selected_model_name)):
            custom_model_path = os.path.abspath(selected_model_name)

        # Normalize model key: strip extension, best_ prefix and round suffix
        # (e.g. "best_yolo11n_r6_m0993.pt" -> key "yolo11n"; for a custom file the
        # key comes from its basename so round-named outputs stay well formed)
        _key_src = os.path.basename(custom_model_path) if custom_model_path else selected_model_name
        base_model_key = _key_src
        if base_model_key.endswith((".pt", ".pth")):
            base_model_key = os.path.splitext(base_model_key)[0]
        if base_model_key.startswith("best_"):
            base_model_key = base_model_key[5:]
        base_model_key = re.sub(r"_r\d+_m\d{3,4}$", "", base_model_key)

        # Training strategy is fixed to recursive (iterative enhancement)
        mode_id = 2

        # Model chosen in the setup dialog (an archived round file) takes precedence
        weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
        params_model_path = None
        if custom_model_path:
            # Custom file wins outright: train from exactly what the user chose
            params_model_path = custom_model_path
        elif selected_model_name.endswith((".pt", ".pth")):
            _cand = os.path.join(weights_dir, selected_model_name)
            if os.path.exists(_cand):
                params_model_path = _cand

        epochs = params.get("epochs") or self.spin_epochs.value()
        freeze = params.get("freeze") or self.spin_freeze.value()

        list_path = self.generate_train_list()
        if not list_path:
            QMessageBox.warning(self, self.tr_text("TTL_WARNING"), self.tr_text("MSG_NO_ANNOTATED_DATA"))
            return
        try:
            with open(list_path, 'r', encoding='utf-8') as f:
                self.current_batch_count = len([l for l in f if l.strip()])
        except:
            self.current_batch_count = 0
        _yaml_ok, _yaml_err = self.update_data_yaml(list_path)
        if not _yaml_ok:
            self._append_log("update_data_yaml failed: %s" % _yaml_err, "error")
            QMessageBox.critical(self, self.tr_text("TTL_ERROR"),
                                 self.tr_text("MSG_DATA_YAML_FAILED").format(_yaml_err))
            return

        self.is_training = True
        self.btn_force_train.setText(self.tr_text("BTN_STOP_TRAIN"))
        self.btn_force_train.setStyleSheet("background-color: #ff4444; color: white; border: none;")
        self.lbl_info.setText(self.tr_text("MSG_STATUS_TRAINING"))
        # Sync sidebar model selector (hidden) + status bar so the displayed
        # model matches the one actually being trained
        try:
            self.combo_yolo_model.blockSignals(True)
            self.combo_yolo_model.setCurrentText(selected_model_name)
            self.combo_yolo_model.blockSignals(False)
        except Exception:
            pass
        self.status_label.setText(self.tr_text("FMT_STATUS_TRAINING").format(selected_model_name, epochs, freeze))

        dataset_runs_dir = os.path.join(self.dataset_root, "training_runs")
        weights_dir = os.path.join(dataset_runs_dir, "loop_train", "weights")
        is_classic = any(x in base_model_key.lower() for x in ['ssd', 'faster', 'retina', 'rcnn'])
        ext = ".pth" if is_classic else ".pt"
        expected_best_model_path = os.path.join(weights_dir, f"best_{base_model_key}{ext}")

        if mode_id == 1:  # transfer
            model_to_use = f"{base_model_key}{ext}" if is_classic else f"{base_model_key}.pt"
        elif mode_id == 2:  # recursive (iterative on the best weights so far)
            if is_classic:
                if os.path.exists(expected_best_model_path):
                    model_to_use = expected_best_model_path
                else:
                    model_to_use = f"{base_model_key}{ext}"
            else:
                if params_model_path:
                    model_to_use = params_model_path
                else:
                    best_round_model = self._resolve_best_round_model(weights_dir)
                    if best_round_model:
                        model_to_use = best_round_model
                    else:
                        model_to_use = f"{base_model_key}.pt"
        else:  # scratch
            if "rtdetr" in base_model_key.lower() or "world" in base_model_key.lower():
                model_to_use = f"{base_model_key}.pt"
            else:
                model_to_use = f"{base_model_key}.yaml"

        # Custom weights: use the exact file the user chose, whatever the strategy
        if custom_model_path:
            model_to_use = custom_model_path

        # Resolve model path to absolute (prevent ultralytics download attempts)
        if not os.path.isabs(model_to_use):
            # Check in project root first (where .pt files usually live)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate = os.path.join(project_root, model_to_use)
            if os.path.exists(candidate):
                model_to_use = candidate
            elif os.path.exists(model_to_use):
                model_to_use = os.path.abspath(model_to_use)

        self._active_train_model = os.path.basename(model_to_use)
        yaml_path = os.path.join(self.dataset_root, "data.yaml")
        self.training_worker = TrainingProcess(
            yaml_path, model_to_use, self.train_queue,
            epochs=epochs,
            freeze=freeze,
            project_dir=dataset_runs_dir
        )
        self.training_worker.start()

    def check_training_status(self):
        while not self.train_queue.empty():
            status, msg = self.train_queue.get()
            if status == "start":
                self.train_start_time = time.time()
                self.train_progress.setValue(0)
                self.train_progress.setFormat(self.tr_text("MSG_STATUS_INITIALIZING"))
                self.lbl_timer.setText(self.tr_text("MSG_STATUS_STARTING"))
                self.log_text.clear()
                self._append_log(self.tr_text("MSG_LOG_TRAIN_CONNECTED"), "info")
                self._set_training_badge("training")
            elif status == "step":
                if isinstance(msg, dict):
                    epoch, total = msg.get('epoch', 0), msg.get('total', 100)
                    self.train_progress.setValue(int((epoch / total) * 100) if total > 0 else 0)
                    self.train_progress.setFormat(self.tr_text("FMT_EPOCH").format(epoch, total))
                    if self.train_start_time: self.lbl_timer.setText(
                        self.tr_text("FMT_RUNTIME").format((time.time() - self.train_start_time) / 3600))
                    self.lbl_map.setText(f"{msg.get('map50', 0.0):.3f}")
                    self.lbl_loss.setText(f"{msg.get('loss', 0.0):.3f}")
                    self.lbl_epoch.setText(
                        self.tr_text("MON_EPOCH_FMT").format(epoch, total))
                    self._append_log(
                        self.tr_text("FMT_LOG_EPOCH").format(epoch, total, msg.get('map50', 0.0), msg.get('loss', 0.0)), "info")
            elif status == "success":
                self.is_training = False
                self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
                self.btn_force_train.setStyleSheet("")
                self.btn_force_train.setObjectName("btn_train")
                self.current_model_path = msg
                self.train_loop_count += 1
                self.train_round += 1  # global round counter, persisted
                self._save_ai_settings()  # persist round count
                # Record last round info for next round's setup dialog
                self.last_epochs = self.spin_epochs.value()
                self.last_map = float(str(self.lbl_map.text()).strip() or 0)

                # --- Round-named model file + training history + best-score deployment ---
                weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
                base_key = (getattr(self, "_active_train_model", "")
                            or self.last_model_name or self.combo_yolo_model.currentText())
                if base_key.startswith("best_"): base_key = base_key[5:]
                for ext_ in (".pt", ".pth", ".yaml"):
                    if base_key.endswith(ext_): base_key = base_key[:-len(ext_)]
                base_key = re.sub(r"_r\d+_m\d{3,4}$", "", base_key)
                round_model_name = "best_%s_r%d_m%04d.pt" % (
                    base_key, self.train_round, int(round(self.last_map * 1000)))
                round_model_path = os.path.join(weights_dir, round_model_name)
                try:
                    if os.path.exists(msg) and not os.path.exists(round_model_path):
                        import shutil as _shutil
                        _shutil.copy2(msg, round_model_path)
                    # Drop the raw ultralytics intermediate once archived
                    if os.path.exists(round_model_path) and os.path.basename(msg) == "best.pt":
                        try:
                            os.remove(msg)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Next round's setup dialog shows the archived round file (with round/mAP)
                if os.path.exists(round_model_path):
                    self.last_model_name = round_model_name
                # Append to training_history.json (one entry per round)
                try:
                    history = self._load_training_history()
                    history.append({
                        "round": self.train_round,
                        "model": round_model_name,
                        "mAP": round(float(self.last_map), 4),
                        "epochs": self.last_epochs,
                        "time": time.strftime("%Y-%m-%d %H:%M"),
                        "images": self.current_batch_count,
                        "duration_min": round(
                            ((time.time() - self.train_start_time) / 60.0)
                            if self.train_start_time else 0.0, 1),
                        "mAP50_95": self._read_last_map50_95(),
                    })
                    seen = {}
                    for e in history:
                        seen[e.get("round")] = e
                    self._save_training_history([seen[k] for k in sorted(seen)])
                except Exception:
                    pass
                # Deploy the highest-mAP model from history
                try:
                    deploy_name = round_model_name
                    best_map = float(self.last_map)
                    for e in self._load_training_history():
                        if e.get("mAP", 0) > best_map and os.path.exists(
                                os.path.join(weights_dir, e.get("model", ""))):
                            best_map = e.get("mAP", 0)
                            deploy_name = e.get("model", deploy_name)
                    deploy_path = os.path.join(weights_dir, deploy_name)
                    self.current_model_path = deploy_path
                    self.inference_engine.load_model(deploy_path)
                    self._show_model_status(deploy_path)
                except Exception:
                    pass
                # Plan C: persist round meta so the UI can show which round a
                # best_*.pt belongs to (last_round.json in the dataset root)
                try:
                    import json as _json
                    with open(os.path.join(self.dataset_root, "last_round.json"), 'w',
                              encoding='utf-8') as _f:
                        _json.dump({
                            "round": self.train_round,
                            "model": round_model_name,
                            "mAP": self.last_map,
                            "epochs": self.last_epochs,
                        }, _f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                self.total_images_in_model = self.current_batch_count
                self.lbl_total_stats.setText(
                    self.tr_text("FMT_TOTAL_STATS").format(self.train_round, self.total_images_in_model))
                duration = (time.time() - self.train_start_time) / 3600 if self.train_start_time else 0
                self.train_progress.setValue(100)
                self.train_progress.setFormat(self.tr_text("MSG_STATUS_DONE"))
                self.lbl_timer.setText(self.tr_text("FMT_DONE_IN").format(duration))
                self.lbl_timer.setStyleSheet("color: #28a745; font-weight: bold;")
                self._append_log(self.tr_text("MSG_LOG_MODEL_SAVED").format(round_model_name), "success")
                self._set_training_badge("done")
                # Training finished. Deliberately NON-BLOCKING: no modal report
                # and no auto-opened clustering wizard. Training runs in its own
                # process, so the user may be mid-clustering or mid-annotation —
                # interrupting them (and rewriting their file list) is worse than
                # reporting the result passively.
                _report = self.tr_text("MSG_TRAIN_REPORT").format(
                    self.train_round, self.total_images_in_model,
                    float(self.last_map), duration * 60.0, round_model_name)
                _oneline = " ".join(_report.split())
                self._append_log(_oneline, "success")
                self.status_label.setText(_oneline)
                self.lbl_info.setText(self.tr_text("FMT_PROGRESS").format(0))
                self.update_dynamic_styles()
                # A round's quota was met while this run was still active and
                # got deferred (_pending_round_ready). Now that training just
                # finished, re-arm the trigger so the next-round setup dialog
                # appears — deferred to a fresh event-loop turn so this
                # callback unwinds before a modal exec_() opens.
                if getattr(self, '_pending_round_ready', False):
                    self._pending_round_ready = False
                    QTimer.singleShot(350, self.check_trigger_training)
                # Only restore the full list when the user is actually in round
                # mode; otherwise leave whatever view they are working in alone.
                if self._round_is_active:
                    self._restore_full_list()
                # Next-round clustering is now triggered by the user from the
                # toolbar (聚类) instead of popping up here.
                if self.current_index >= 0:
                    img_path = self.image_list[self.current_index]
                    if not os.path.exists(self.get_label_path(img_path)):
                        self.canvas.load_pixmap(load_qpixmap(img_path), self.classes)
                        # Post-training pre-labeling (only when AI assist is enabled)
                        if getattr(self, '_ai_assist_enabled', True):
                            prompt_text = self._effective_prompt()
                            self.inference_engine.run_inference(img_path, prompt=prompt_text)
            elif status == "error":
                self.is_training = False
                self.btn_force_train.setText(self.tr_text("BTN_TRAIN"))
                self.btn_force_train.setStyleSheet("")
                self.btn_force_train.setObjectName("btn_train")
                self.train_progress.setFormat(self.tr_text("MSG_STATUS_ERROR"))
                self.lbl_timer.setText(self.tr_text("MSG_STATUS_FAILED"))
                self._append_log(self.tr_text("MSG_LOG_ERROR").format(msg), "error")
                self._set_training_badge("failed")

    def open_dedup_dialog(self):
        """Toolbar 除重: detect and remove duplicate images only (no clustering)."""
        if not self.dataset_root or not self.image_list:
            QMessageBox.warning(self, self.tr_text("TTL_INFO"), self.tr_text("MSG_NO_DATASET"))
            return
        try:
            dialog = CleaningDialog(self, self.dataset_root, list(self.image_list),
                                    mode="dedup")
            if dialog.exec_() == QDialog.Accepted:
                removed = dialog.get_removed_paths()
                if removed:
                    self._apply_cleaning_removals(removed)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            with open('keenforge_crash.log', 'w', encoding='utf-8') as f:
                f.write(tb)
            QMessageBox.critical(self, self.tr_text("TTL_CRASH"), self.tr_text("MSG_CRASH_OPEN_LOG").format("keenforge_crash.log", tb[-500:]))

    def open_cluster_dialog(self):
        """Toolbar 聚类: diversity clustering only (no dedup).

        Once labeling has started (any label file exists, or a trained model is
        present) only the still-unlabeled images are clustered (follow-up
        round); on a brand-new dataset every loaded image is clustered.
        """
        if not self.dataset_root or not self.image_list:
            QMessageBox.warning(self, self.tr_text("TTL_INFO"), self.tr_text("MSG_NO_DATASET"))
            return
        try:
            # Follow-up round detection must NOT depend on the trained model file:
            # during a training run best_*.pt is only written at the very end, so
            # gating on _has_trained_model() used to cluster the FULL list (e.g.
            # 1256 images) instead of the still-unlabeled ones (e.g. 1055).
            # Labeling state is the ground truth — round 1 has no labels at all,
            # every later round has at least one.
            full_list = self._full_image_list or self.image_list
            labeled = sum(1 for p in full_list
                          if os.path.exists(self.get_label_path(p)))
            if labeled or self._has_trained_model():   # follow-up round
                remaining = [p for p in full_list
                             if not os.path.exists(self.get_label_path(p))]
                if len(remaining) == 0:
                    # Everything is labeled — show the training recap
                    self._show_training_summary()
                    return
                if len(remaining) == 1:
                    QMessageBox.information(
                        self, self.tr_text("TTL_INFO"),
                        self.tr_text("FMT_REMAINING_NO_CLUSTER").format(len(remaining)))
                    return
                base = list(remaining)
            else:
                base = list(self.image_list)
            dialog = CleaningDialog(self, self.dataset_root, base, mode="cluster")
            if dialog.exec_() == QDialog.Accepted:
                diverse = dialog.get_diverse_order()
                if diverse:
                    self._apply_diverse_order(diverse)
                if dialog.get_diversity_report() is not None:
                    self._save_cluster_map_from_dialog(dialog)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            with open('keenforge_crash.log', 'w', encoding='utf-8') as f:
                f.write(tb)
            QMessageBox.critical(self, self.tr_text("TTL_CRASH"), self.tr_text("MSG_CRASH_OPEN_LOG").format("keenforge_crash.log", tb[-500:]))

    def open_label_setup_dialog(self):
        """Open label setup dialog anytime: edit classes + toggle AI-assist."""
        if not self.dataset_root:
            QMessageBox.warning(self, self.tr_text("TTL_INFO"), self.tr_text("MSG_NO_DATASET"))
            return
        dialog = LabelSetupDialog(self, current_classes=self.classes,
                                 has_trained_model=self._has_trained_model())
        self.smart_move_dialog(dialog)
        if dialog.exec_() != QDialog.Accepted:
            return
        old_classes = list(self.classes)
        new_classes = dialog.get_classes()
        mapping = dialog.get_mapping()
        self._ai_assist_enabled = dialog.get_ai_assist()
        self._sync_ai_assist_checkbox()
        classes_changed = (new_classes != old_classes)
        if classes_changed:
            files_remapped, boxes_removed = self._apply_class_mapping(
                old_classes, new_classes, mapping)
        self.classes = new_classes
        self._save_ai_settings()
        self.save_data_yaml(os.path.join(self.dataset_root, "data.yaml"))
        # Save any unsaved annotation, then reload current image so the canvas
        # reflects the new class mapping
        self.save_current_annotation()
        if 0 <= self.current_index < len(self.image_list):
            self.change_image(self.current_index)
        if classes_changed:
            self.status_label.setText(
                self.tr_text("MSG_CLASS_MAPPED").format(files_remapped, boxes_removed))
            # Classes changed -> the trained model's id order no longer matches.
            # Semantic mapping in on_inference_result handles it, but retraining
            # is strongly recommended before relying on AI pre-labeling.
            if self._has_trained_model():
                QMessageBox.information(
                    self, self.tr_text("TTL_INFO"),
                    self.tr_text("MSG_CLASS_CHANGED_TRAIN"))
        else:
            self.status_label.setText(
                self.tr_text("FMT_LABEL_SETUP_UPDATED").format(
                    f"{len(self.classes)} 个类别 · AI 辅助{'开' if self._ai_assist_enabled else '关'}"))

    def _apply_class_mapping(self, old_classes, new_classes, mapping):
        """Remap cls_id in all label txt files after class reorder/rename/delete."""
        files_remapped = 0
        boxes_removed = 0
        for img_path in self.image_list:
            label_path = self.get_label_path(img_path)
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                changed = False
                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        old_id = int(parts[0])
                    except ValueError:
                        continue
                    new_id = mapping.get(old_id)
                    if new_id is None:
                        boxes_removed += 1
                        changed = True
                        continue
                    if new_id != old_id:
                        changed = True
                    new_lines.append(" ".join([str(new_id)] + parts[1:]) + "\n")
                if changed:
                    with open(label_path, 'w') as f:
                        f.writelines(new_lines)
                    files_remapped += 1
            except Exception:
                continue
        return files_remapped, boxes_removed
    def _save_cluster_map_from_dialog(self, dialog):
        """Persist the dialog's diversity report as cluster_map.json."""
        try:
            from src.core.data_cleaner import ClusterManager
            cm = ClusterManager(self.dataset_root)
            div = dialog.get_diversity_report()
            k = len(div.selected_indices)
            cm.build_from_report(div, dialog.get_cluster_base_paths(),
                                 backend=dialog.get_engine_name(), k=k)
            self.cluster_manager = cm
            self.status_label.setText(self.tr_text("FMT_CLUSTER_SAVED").format(k))
            self.populate_file_list()
            self._activate_round_if_clustered()  # enter round mode after initial clustering
        except Exception as e:
            print(f"Cluster map save failed: {e}")

    def _activate_round_if_clustered(self):
        """If cluster_map.json exists and reps are not all done,
        filter the display to show only representatives.
        If all reps are already annotated, restore the full list instead."""
        try:
            from src.core.data_cleaner import ClusterManager
            cm = ClusterManager(self.dataset_root)
            if cm.load():
                reps_list = [c.get("representative", "") for c in cm.get_clusters()]
                reps_list = [r for r in reps_list if r and os.path.exists(r)]
                if not reps_list:
                    return
                reps_set = set(reps_list)
                # Quota-based completion check (reps + extra labels vs target)
                prog = self._round_progress()
                if not prog:
                    return
                progress, rep_target, rep_done, extras = prog
                if progress >= rep_target:
                    # All reps already labeled → restore full list
                    self._restore_full_list()
                    self.status_label.setText(
                        self.tr_text("FMT_ROUND_ALL_DONE").format(self.train_round + 1))
                    return

                # Keep ALL images visible; populate_file_list sorts reps to the top
                # (● representative mark + cluster colour) so the full dataset shows.
                if not self._full_image_list:
                    self._full_image_list = list(self.image_list)
                self.image_list = list(self._full_image_list)
                self._round_is_active = True
                self._round_rep_count = rep_target
                self._round_rep_done = progress
                self.populate_file_list()
                if self.image_list:
                    self.file_list_widget.setCurrentRow(0)
                self.status_label.setText(
                    self.tr_text("FMT_ROUND_ACTIVE").format(self.train_round + 1, len(reps_set)))
        except Exception as e:
            print(f"_activate_round_if_clustered error: {e}")

    def _restore_full_list(self):
        """
        Restore the full image list (all images, not just reps).
        Used when round representatives are all done — user can see
        remaining unlabeled images and optionally continue labeling.
        """
        if not self._full_image_list:
            return
        self.image_list = list(self._full_image_list)
        self._round_is_active = False
        self.populate_file_list()
        self.status_label.setText(
            self.tr_text("FMT_FULL_LIST_RESTORED").format(len(self._full_image_list)))
        if self.image_list:
            # Keep current selection if valid, otherwise go to first
            if self.current_index >= 0 and self.current_index < self.file_list_widget.count():
                self.file_list_widget.setCurrentRow(self.current_index)
            elif self.file_list_widget.count() > 0:
                self.file_list_widget.setCurrentRow(0)

    def _show_next_round_cleaning_dialog(self):
        """
        After training completes, show the cleaning wizard (start_step=1:
        similarity threshold -> treemap preview) for the REMAINING unlabeled
        images. User confirms the clusters, then round-2 labeling starts.
        """
        if not self.dataset_root or not self._full_image_list:
            return
        remaining = [p for p in self._full_image_list
                     if not os.path.exists(self.get_label_path(p))]
        if len(remaining) == 0:
            # Everything is labeled — show the full labeling/training recap.
            # Deferred: no nested exec_() inside the training-status timer callback.
            QTimer.singleShot(300, self._show_training_summary)
            return
        if len(remaining) == 1:
            QMessageBox.information(
                self, self.tr_text("TTL_INFO"),
                self.tr_text("FMT_REMAINING_MANUAL").format(len(remaining)))
            return
        # Defer the dialog: opening a matplotlib-backed dialog with nested
        # exec_() inside the training-status QTimer callback causes native
        # aborts (Qt/matplotlib). A singleShot lets the callback unwind first.
        QTimer.singleShot(300, lambda: self._open_next_round_cleaning(list(remaining)))

    def _open_next_round_cleaning(self, remaining):
        """Actually open the round-N clustering wizard (called from a fresh event-loop turn)."""
        try:
            dialog = CleaningDialog(
                self, self.dataset_root, list(remaining),
                wizard_mode=True, start_step=1,
            )
            if dialog.exec_() == QDialog.Accepted:
                diverse = dialog.get_diverse_order()
                if diverse:
                    self._apply_diverse_order(diverse)
                if dialog.get_diversity_report() is not None:
                    self._save_cluster_map_from_dialog(dialog)
        except Exception:
            import traceback
            traceback.print_exc()

    def _read_last_map50_95(self):
        """mAP50-95 of the last epoch from the current run's results.csv (or None)."""
        try:
            import csv as _csv
            csv_path = os.path.join(self.dataset_root, "training_runs", "loop_train", "results.csv")
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(_csv.reader(f))
            if len(rows) < 2:
                return None
            header = [h.strip() for h in rows[0]]
            idx = None
            for i, h in enumerate(header):
                if "mAP50-95" in h:
                    idx = i
                    break
            if idx is None:
                return None
            last = rows[-1]
            if len(last) <= idx:
                return None
            return round(float(last[idx]), 4)
        except Exception:
            return None

    def _count_boxes_by_class(self, img_list):
        """Count YOLO annotation boxes per class across img_list.

        Returns (total_boxes, [(class_name, count), ...]) sorted by count desc.
        """
        counts = {}
        total = 0
        for p in img_list:
            label_path = self.get_label_path(p)
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            cid = int(parts[0])
                        except ValueError:
                            continue
                        counts[cid] = counts.get(cid, 0) + 1
                        total += 1
            except Exception:
                continue
        rows = []
        for cid, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            name = self.classes[cid] if 0 <= cid < len(self.classes) else "class_%d" % cid
            rows.append((name, cnt))
        return total, rows

    def _build_training_summary_html(self):
        """Build the rich-text recap of the whole labeling + training process."""
        history = self._load_training_history()
        img_list = self._full_image_list if self._full_image_list else self.image_list
        total = len(img_list)
        total_boxes, per_class = self._count_boxes_by_class(img_list)
        weights_dir = os.path.join(self.dataset_root, "training_runs", "loop_train", "weights")
        rounds = sorted([e for e in history if e.get("round") is not None],
                        key=lambda e: e.get("round", 0))
        best = None
        for e in rounds:
            nm = e.get("model", "")
            if nm and os.path.exists(os.path.join(weights_dir, nm)):
                if best is None or float(e.get("mAP", 0) or 0) > float(best.get("mAP", 0) or 0):
                    best = e
        rows = ""
        prev_map = None
        for e in rounds:
            style = " style='color:#1a7f37;font-weight:bold;'" if (best is not None and e is best) else ""
            mp = float(e.get("mAP", 0) or 0)
            mp_txt = "%.3f" % mp
            if prev_map is not None:
                mp_txt += " (%+.3f)" % (mp - prev_map)
            prev_map = mp
            mp95 = e.get("mAP50_95")
            mp95_txt = ("%.3f" % float(mp95)) if mp95 not in (None, "") else "—"
            imgs = e.get("images")
            imgs_txt = str(imgs) if imgs not in (None, "") else "—"
            dur = e.get("duration_min")
            dur_txt = ("%.1f" % float(dur)) if dur not in (None, "") else "—"
            rows += ("<tr%s><td align='center'>%s</td><td align='center'>%s</td>"
                     "<td align='center'>%s</td><td align='center'>%s</td>"
                     "<td align='center'>%s</td><td align='center'>%s</td><td>%s</td></tr>") % (
                style, e.get("round", ""), mp_txt, mp95_txt, imgs_txt, dur_txt,
                e.get("time", ""), e.get("model", ""))
        header = ""
        try:
            header += self.tr_text("SUM_DATASET").format(total) + "<br/>"
            if rounds:
                header += self.tr_text("SUM_ROUNDS").format(
                    len(rounds), rounds[0].get("round", ""), rounds[-1].get("round", "")) + "<br/>"
            total_min = 0.0
            for e in rounds:
                try:
                    total_min += float(e.get("duration_min") or 0)
                except Exception:
                    pass
            if total_min > 0:
                header += self.tr_text("SUM_TOTAL_TIME").format("%.0f" % total_min) + "<br/>"
            if best is not None:
                header += self.tr_text("SUM_BEST").format(
                    best.get("model", ""), best.get("round", ""), float(best.get("mAP", 0) or 0)) + "<br/>"
            header += self.tr_text("SUM_DEPLOYED").format(
                os.path.basename(str(self.current_model_path or "")))
        except Exception:
            pass
        box_table = ""
        try:
            if total_boxes > 0:
                box_table += "<p><b>" + self.tr_text("SUM_BOXES").format(
                    total_boxes, len(per_class)) + "</b></p>"
            if per_class:
                brows = ""
                for _name, _cnt in per_class:
                    _share = (_cnt * 100.0 / total_boxes) if total_boxes else 0.0
                    brows += ("<tr><td>%s</td><td align='center'>%s</td>"
                              "<td align='center'>%.1f%%</td></tr>") % (_name, _cnt, _share)
                box_table += ("<p><b>%s</b></p>"
                             "<table border='1' cellspacing='0' cellpadding='4' width='100%%'>"
                             "<tr><th>%s</th><th>%s</th><th>%s</th></tr>%s</table>") % (
                    self.tr_text("SUM_BOXES_TITLE"), self.tr_text("SUM_TH_CLASS"),
                    self.tr_text("SUM_TH_BOXES"), self.tr_text("SUM_TH_SHARE"), brows)
        except Exception:
            pass
        html = ("<p><b>%s</b></p>"
                "<table border='1' cellspacing='0' cellpadding='4' width='100%%'>"
                "<tr><th>%s</th><th>mAP50</th><th>mAP50-95</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th></tr>%s</table>"
                "<p>%s</p>") % (
            header, self.tr_text("SUM_TH_ROUND"), self.tr_text("SUM_TH_IMAGES"),
            self.tr_text("SUM_TH_DURATION"), self.tr_text("SUM_TH_TIME"),
            self.tr_text("SUM_TH_MODEL"), rows, self.tr_text("SUM_FOOTER"))
        return html + box_table

    def _show_training_summary(self):
        """Show the full labeling/training recap dialog (scrollable)."""
        try:
            if not self._load_training_history():
                QMessageBox.information(
                    self, self.tr_text("TTL_INFO"), self.tr_text("SUM_NO_HISTORY"))
                return
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle(self.tr_text("TTL_TRAIN_SUMMARY"))
            dlg.resize(940, 560)
            lay = QVBoxLayout(dlg)
            tb = QTextBrowser()
            tb.setHtml(self._build_training_summary_html())
            lay.addWidget(tb)
            btn = QPushButton(self.tr_text("SUM_BTN_CLOSE"))
            btn.clicked.connect(dlg.accept)
            lay.addWidget(btn)
            dlg.exec_()
        except Exception:
            import traceback
            traceback.print_exc()
    def _show_next_round_dialog(self):
        """
        After training completes, show a dialog letting the user configure
        the next round's cluster target count, then optionally enter round mode.
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QPushButton, QHBoxLayout

        # Count remaining unlabeled images
        remaining = [p for p in self._full_image_list
                     if not os.path.exists(self.get_label_path(p))]
        remaining_count = len(remaining)

        if remaining_count <= self._cluster_min_clusters:
            QMessageBox.information(
                self, self.tr_text("TTL_INFO"),
                self.tr_text("FMT_REMAINING_TOO_FEW").format(remaining_count, self._cluster_min_clusters))
            return

        # Default target
        default_target = max(
            self._cluster_min_clusters,
            min(self._cluster_max_clusters,
                int(remaining_count * self._target_cluster_pct))
        )

        # Build custom dialog
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr_text("FMT_ROUND_DONE_TITLE").format(self.train_round + 1))
        dlg.resize(420, 200)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            f"<b>剩余未标注: {remaining_count} 张</b><br><br>"
            f"系统推荐目标簇数: <b>{default_target}</b> （{int(self._target_cluster_pct * 100)}% × 剩余 = {int(remaining_count * self._target_cluster_pct)}，"
            f"限制 {self._cluster_min_clusters}-{self._cluster_max_clusters}）<br>"
            f"可手动调整后点击「重新聚类并开始标注」进入下一轮。<br>"
            f"点击「查看全部图片」跳过，保持完整列表。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Target cluster count spin box
        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel(self.tr_text("FMT_TARGET_CLUSTERS")))
        spin_target = QSpinBox()
        spin_target.setRange(self._cluster_min_clusters, self._cluster_max_clusters)
        spin_target.setValue(default_target)
        spin_layout.addWidget(spin_target)
        spin_layout.addStretch()
        layout.addLayout(spin_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_recluster = QPushButton(self.tr_text("MSG_BTN_RECLUSTER"))
        btn_recluster.setStyleSheet(
            "background-color: #28a745; color: white; padding: 8px 16px; border-radius: 4px;")
        btn_view_all = QPushButton(self.tr_text("MSG_BTN_VIEW_ALL"))
        btn_view_all.setStyleSheet(
            "padding: 8px 16px;")
        btn_layout.addWidget(btn_recluster)
        btn_layout.addWidget(btn_view_all)
        layout.addLayout(btn_layout)

        # Result tracking
        dlg._user_choice = None  # 'recluster' or 'view_all'
        dlg._target = default_target

        def on_recluster():
            dlg._user_choice = 'recluster'
            dlg._target = spin_target.value()
            dlg.accept()

        def on_view_all():
            dlg._user_choice = 'view_all'
            dlg.reject()

        btn_recluster.clicked.connect(on_recluster)
        btn_view_all.clicked.connect(on_view_all)

        # Center dialog on parent
        dlg.move(
            self.x() + (self.width() - dlg.width()) // 2,
            self.y() + (self.height() - dlg.height()) // 2,
        )

        if dlg.exec_() == QDialog.Accepted and dlg._user_choice == 'recluster':
            # Run reclustering with user's target
            target = dlg._target
            self.status_label.setText(self.tr_text("FMT_RECLUSTER_STATUS").format(target))
            n_clusters, err = self._recluster_remaining(target_override=target)
            if n_clusters > 0:
                QMessageBox.information(
                    self, self.tr_text("TTL_CLUSTER_DONE"),
                    self.tr_text("FMT_CLUSTER_SAVED").format(n_clusters))
                self._activate_round_if_clustered()
            else:
                QMessageBox.warning(
                    self, self.tr_text("TTL_CLUSTER_FAIL"),
                    self.tr_text("FMT_RECLUSTER_FAILED").format(err))
        # else: user chose 'view_all' or closed dialog — stay in full list mode

    def _recluster_remaining(self, target_override: int = 0):
        """
        Re-cluster the REMAINING (unlabeled) images for the next round.
        1. Restore full image list from _full_image_list
        2. Compute target cluster count: clamp(5% × remaining, 5, 50)
        3. Extract features for remaining images
        4. Binary-search similarity threshold for target count
        5. Build new cluster_map.json
        6. Activate round mode (reps only)

        If ``target_override`` is provided, skip the default 5% calculation.

        Returns (n_clusters, None) on success, (0, error_msg) on failure.
        """
        if not self.dataset_root or not self._full_image_list:
            return (0, self.tr_text("MSG_NO_DATASET_LOADED"))

        # Restore full list first
        self.image_list = list(self._full_image_list)
        self._round_is_active = False

        # Determine remaining (unlabeled) images
        remaining = [p for p in self._full_image_list
                     if not os.path.exists(self.get_label_path(p))]

        if len(remaining) <= self._cluster_min_clusters:
            msg = self.tr_text("FMT_REMAINING_TOO_FEW").format(len(remaining), self._cluster_min_clusters)
            self.status_label.setText(msg)
            self.populate_file_list()
            return (0, msg)

        try:
            from src.core.data_cleaner import (
                FeatureExtractor, DiversitySelector, ClusterManager
            )
            import torch as _torch

            # Target cluster count: use override or default (clamp(5% × remaining, min, max))
            if target_override and target_override > 0:
                target = max(
                    self._cluster_min_clusters,
                    min(self._cluster_max_clusters, target_override)
                )
            else:
                target = max(
                    self._cluster_min_clusters,
                    min(
                        self._cluster_max_clusters,
                        int(len(remaining) * self._target_cluster_pct)
                    )
                )

            # Extract features for remaining images only
            device = 'cuda' if _torch.cuda.is_available() else 'cpu'
            extractor = FeatureExtractor(
                backend="mobilenet_v3", device=device
            )
            self.status_label.setText(
                self.tr_text("FMT_RECLUSTER_FEATURES").format(target))
            feats = extractor.extract_batch(remaining, batch_size=32)

            # Binary-search for the right threshold
            selector = DiversitySelector(feats)
            threshold = selector.find_threshold_for_target_clusters(
                target, search_range=(0.80, 0.98), max_iterations=8
            )
            self._round_threshold = threshold

            # Re-cluster and save cluster_map.json
            self.status_label.setText(
                self.tr_text("FMT_RECLUSTER_THRESHOLD").format(threshold))
            cm = ClusterManager(self.dataset_root)
            cm.recluster_paths(remaining, feats, threshold)
            self.cluster_manager = cm

            n_clusters = len(cm.get_clusters())
            self.status_label.setText(
                self.tr_text("FMT_ROUND_CLUSTERED").format(self.train_round + 2, threshold, n_clusters))

            return (n_clusters, None)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.setText(self.tr_text("FMT_RECLUSTER_FAILED").format(e))
            self.populate_file_list()
            return (0, str(e))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.setText(self.tr_text("FMT_RECLUSTER_FAILED").format(e))
            self.populate_file_list()


    def _apply_cleaning_removals(self, removed_paths: list):
        """Move duplicate images to _removed_duplicates/ folder."""
        import shutil
        removed_dir = os.path.join(self.dataset_root, "_removed_duplicates")
        os.makedirs(removed_dir, exist_ok=True)

        count = 0
        moved_paths = []
        for img_path in removed_paths:
            if not os.path.exists(img_path):
                continue
            try:
                fname = os.path.basename(img_path)
                dest = os.path.join(removed_dir, fname)
                # Avoid overwriting: append a suffix if needed
                if os.path.exists(dest):
                    base, ext = os.path.splitext(fname)
                    dest = os.path.join(removed_dir, f"{base}_dup{ext}")
                shutil.move(img_path, dest)

                # Also move label file
                label_path = self.get_label_path(img_path)
                if os.path.exists(label_path):
                    lname = os.path.basename(label_path)
                    shutil.move(label_path, os.path.join(removed_dir, lname))

                # Remove from train_list.txt
                self.remove_from_train_txt(img_path)
                moved_paths.append(img_path)
                count += 1
            except Exception as e:
                print(f"Failed to move {img_path}: {e}")

        # Keep cluster_map.json in sync with the removals
        self._prune_cluster_map(moved_paths)
        # Refresh image list
        self.image_list = [p for p in self.image_list if os.path.exists(p)]
        self._full_image_list = list(self.image_list)  # keep full copy in sync
        self.populate_file_list()
        self._refresh_total_stats()

        self.status_label.setText(self.tr_text("FMT_REMOVED_DUPS").format(count, len(self.image_list)))
        if self.current_index >= len(self.image_list):
            self.current_index = len(self.image_list) - 1
        if self.current_index >= 0:
            self.file_list_widget.setCurrentRow(self.current_index)

    def _apply_diverse_order(self, diverse_order: list):
        """Re-order file list to show diverse samples first."""
        diverse_set = set(diverse_order)
        rest = [p for p in self.image_list if p not in diverse_set]
        self.image_list = diverse_order + rest
        self.populate_file_list()
        self.status_label.setText(
            self.tr_text("FMT_DIVERSE_ORDER").format(len(diverse_order)))
        if self.image_list:
            self.file_list_widget.setCurrentRow(0)
            self.current_index = 0
    def populate_file_list(self):
        """Fill the right-side file list. Representatives (manual-label) come first."""
        self.file_list_widget.clear()
        cluster_colors = {}
        cluster_reps = set()
        cluster_ids = {}
        try:
            from src.core.data_cleaner import ClusterManager
            cm = ClusterManager(self.dataset_root)
            if cm.load():
                self.cluster_manager = cm
                for p in self.image_list:
                    c = cm.color_of(p)
                    if c:
                        cluster_colors[p] = c
                    if cm.is_representative(p):
                        cluster_reps.add(p)
                    cid = cm.cluster_of(p)
                    if cid is not None:
                        cluster_ids[p] = cid
        except Exception as e:
            print(f"populate_file_list cluster error: {e}")

        # Reorder: representatives first, then cluster members, then ungrouped
        # (stable partition keeps relative order within each group)
        ordered = []
        if cluster_reps:
            reps = [p for p in self.image_list if p in cluster_reps]
            reps.sort(key=lambda p: cluster_ids.get(p, 9999))
            members = [p for p in self.image_list if p not in cluster_reps and p in cluster_colors]
            members.sort(key=lambda p: cluster_ids.get(p, 9999))
            ungrouped = [p for p in self.image_list if p not in cluster_reps and p not in cluster_colors]
            ordered = reps + members + ungrouped
            if ordered != self.image_list:
                self.image_list = ordered
        else:
            ordered = self.image_list

        self.current_index = 0

        for p in ordered:
            fname = os.path.basename(p)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, p)  # used by the find filter (basename match)
            if p in cluster_reps:
                cid = cluster_ids.get(p, "")
                item.setText(self.tr_text("FMT_FILE_REP").format(cid, fname))
                item.setToolTip(self.tr_text("FMT_FILE_REP_TIP").format(cid))
                item.setForeground(QColor(cluster_colors[p]) if p in cluster_colors else QColor("#333"))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            elif p in cluster_colors:
                cid = cluster_ids.get(p, "")
                item.setText(self.tr_text("FMT_FILE_AI").format(cid, fname))
                item.setToolTip(self.tr_text("FMT_FILE_AI_TIP").format(cid))
                item.setForeground(QColor(cluster_colors[p]))
            else:
                # Not part of any cluster: show the plain filename in the normal
                # colour, exactly as before clustering was ever introduced. Users
                # may simply be browsing the dataset.
                item.setText(fname)
            self.file_list_widget.addItem(item)

        # Re-apply an active find filter to the freshly rebuilt list
        # (this also refreshes the stats label).
        self._apply_file_filter(self.txt_find.text() if hasattr(self, 'txt_find') else '')

    def _update_list_stats(self):
        """Show total / annotated counts above the file list."""
        total = len(self.image_list)
        done = sum(1 for p in self.image_list if os.path.exists(self.get_label_path(p)))
        pending = total - done
        text = self.tr_text("FMT_LIST_STATS").format(total, done, pending)
        # When a find filter is active, also report how many rows are visible
        if getattr(self, '_find_query', ''):
            text += " · " + self.tr_text("FMT_LIST_FILTER").format(self._visible_count(), total)
        self.lbl_list_stats.setText(text)

    def _visible_count(self):
        """Number of list rows currently visible (not hidden by the find filter)."""
        try:
            return sum(1 for r in range(self.file_list_widget.count())
                       if not self.file_list_widget.item(r).isHidden())
        except Exception:
            return self.file_list_widget.count()

    def _apply_file_filter(self, text):
        """Hide right-panel rows whose file name doesn't contain the search text.

        Rows are hidden rather than removed, so the row index stays parallel to
        ``self.image_list``; every index-based operation (change_image,
        move_to_junk's takeItem/pop, cluster ordering) keeps working unchanged.
        """
        self._find_query = (text or "").strip().lower()
        query = self._find_query
        was_blocked = self.file_list_widget.blockSignals(True)
        try:
            for row in range(self.file_list_widget.count()):
                item = self.file_list_widget.item(row)
                if not query:
                    item.setHidden(False)
                    continue
                path = item.data(Qt.UserRole)
                name = os.path.basename(path).lower() if path else ""
                haystack = (item.text() or "").lower()
                item.setHidden(query not in name and query not in haystack)
        finally:
            self.file_list_widget.blockSignals(was_blocked)
        self._update_list_stats()

    def _jump_to_first_match(self):
        """Enter in the find box: select the first visible (matching) image."""
        if not getattr(self, '_find_query', ''):
            return
        for row in range(self.file_list_widget.count()):
            if not self.file_list_widget.item(row).isHidden():
                self.file_list_widget.setCurrentRow(row)
                return

    def _toggle_find_box(self):
        """🔍 button: reveal the search box, or collapse it and clear the filter."""
        if self.btn_find.isChecked():
            self.txt_find.setVisible(True)
            self.txt_find.setFocus()
            self.txt_find.selectAll()
        else:
            self._collapse_find_box()

    def _collapse_find_box(self):
        """Hide the search box and drop any active filter so no row stays hidden."""
        self.btn_find.setChecked(False)
        if self.txt_find.text():
            # Clearing re-runs _apply_file_filter("") -> every row visible again
            self.txt_find.clear()
        self.txt_find.setVisible(False)
        self.file_list_widget.setFocus()