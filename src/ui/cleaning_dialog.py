# src/ui/cleaning_dialog.py
"""
Data cleaning wizard (3-step guided flow).

    Step 1 – 移除完全重复的图片 (fixed perceptual hash, no model choice)
    Step 2 – 设置聚类分组 (choose feature model + similarity threshold)
    Step 3 – 聚类结果预览 (Pareto coverage + representative list)

Matches existing dialog patterns (LabelDialog, ClassSelectionDialog):
    - QDialog with QVBoxLayout root
    - tr_text() helper delegating to parent
    - smart_move_dialog positioning
    - exec_() return pattern
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QComboBox, QGroupBox, QStackedWidget,
                              QListWidget, QProgressBar, QMessageBox,
                              QListWidgetItem, QAbstractItemView, QFrame,
                              QDoubleSpinBox, QSlider, QWidget, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QColor

from src.config import TRANS, t

# ---------------------------------------------------------------------------
# Colour palette for cluster visualisation (must match ClusterManager.DEFAULT_COLORS)
# ---------------------------------------------------------------------------
CLUSTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60",
]


# ===========================================================================
# Background workers
# ===========================================================================
class CleaningWorker(QThread):
    """Step 1: dedup scan (fixed perceptual hash)."""
    progress = pyqtSignal(int, int, str)
    duplicates_ready = pyqtSignal(object)     # DuplicateReport
    error_occurred = pyqtSignal(str)

    def __init__(self, engine, image_paths):
        super().__init__()
        self._engine = engine
        self._paths = image_paths

    def run(self):
        try:
            report = self._engine.scan(
                self._paths,
                progress_cb=lambda c, t, phase: self.progress.emit(c, t, phase),
            )
            self.duplicates_ready.emit(report)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class ClusterWorker(QThread):
    """Step 2: clustering on the final image set (after removal decision)."""
    progress = pyqtSignal(int, int, str)
    diversity_ready = pyqtSignal(object)      # DiversityReport
    error_occurred = pyqtSignal(str)

    def __init__(self, image_paths, similarity_threshold=0.90, backend="mobilenet_v3",
                 feats=None, target_clusters=None):
        super().__init__()
        self._paths = image_paths
        self._sim_threshold = similarity_threshold
        self._backend = backend
        self._feats = feats
        self._target_clusters = target_clusters
        self.feats = None        # exposed for reuse on re-cluster
        self.selector = None     # exposed for threshold recommendation
        self.used_threshold = None  # actual threshold used (auto mode)

    def run(self):
        try:
            if self._sim_threshold <= 0 or len(self._paths) <= 1:
                return
            from src.core.data_cleaner import FeatureExtractor, DiversitySelector
            if self._feats is not None:
                # Reuse cached features (fast re-cluster after threshold change)
                feats = self._feats
                self.progress.emit(0, len(self._paths),
                                   t("CLN_PHASE_CACHE"))
            else:
                import torch as _torch
                device = 'cuda' if _torch.cuda.is_available() else 'cpu'
                extractor = FeatureExtractor(backend=self._backend, device=device)
                self.progress.emit(0, len(self._paths),
                                   t("CLN_PHASE_EXTRACT"))
                feats = extractor.extract_batch(
                    self._paths, batch_size=32,
                    progress_cb=lambda c, t2: self.progress.emit(
                        c, t2, t("CLN_PHASE_EXTRACT")),
                )
            self.feats = feats
            selector = DiversitySelector(feats)
            self.selector = selector
            # Auto mode: binary-search the threshold that yields ~target clusters
            if self._target_clusters is not None and self._target_clusters > 0:
                self.used_threshold = selector.find_threshold_for_target_clusters(
                    self._target_clusters, search_range=(0.80, 0.98), max_iterations=8)
                self.progress.emit(0, len(self._paths),
                                   t("CLN_PHASE_THRESH").format(self.used_threshold))
                th = self.used_threshold
            else:
                self.used_threshold = self._sim_threshold
                th = self._sim_threshold
            div_report = selector.cluster_by_similarity(th)
            div_report.selected_paths = [self._paths[i]
                                          for i in div_report.selected_indices]
            div_report.base_paths = list(self._paths)
            self.diversity_ready.emit(div_report)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


# ===========================================================================
# CleaningDialog – 3-step wizard
# ===========================================================================
class CleaningDialog(QDialog):
    """Guided 3-step data cleaning wizard."""

    EMBEDDING_OPTIONS = [
        ("mobilenet_v3", "CLN_EMBED_MOBILE"),
        ("efficientnet_b0", "CLN_EMBED_EFFICIENT"),
        ("resnet50", "CLN_EMBED_RESNET"),
    ]

    def __init__(self, parent=None, dataset_root: str = "",
                 image_list: list = None, wizard_mode: bool = False,
                 start_step: int = 0, mode: str = "full"):
        super().__init__(parent)
        self.parent_window = parent
        self.dataset_root = dataset_root
        self.image_list = image_list or []
        self.wizard_mode = wizard_mode
        # "full"    = dedup -> cluster setup -> preview (default wizard)
        # "dedup"   = duplicate removal only (single page)
        # "cluster" = diversity clustering only (starts at the cluster step)
        self.mode = mode if mode in ("full", "dedup", "cluster") else "full"
        if self.mode == "cluster":
            start_step = max(start_step, 1)

        self._report = None
        self._div_report = None
        self._worker = None
        self._removed_paths: list = []
        self._diverse_order: list = []
        self._cluster_base_paths: list = []
        self._engine_name = "mobilenet_v3"
        self._current_step = max(0, min(start_step, 2))
        self._start_step = max(0, min(start_step, 2))
        # Feature cache for fast re-cluster
        self._cached_feats = None
        self._cached_backend = None

        title_key = {"dedup": "CLN_TITLE_DEDUP",
                     "cluster": "CLN_TITLE_CLUSTER"}.get(self.mode, "CLN_TITLE")
        self.setWindowTitle(self.tr_text(title_key))
        self.resize(780, 660)
        self.setup_ui()
        self._update_nav_buttons()
        self._detect_hardware_and_recommend()

        # No auto-scan: user clicks "开始扫描" manually after loading dataset

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------
    def get_removed_paths(self) -> list:
        return self._removed_paths

    def get_diverse_order(self) -> list:
        return self._diverse_order

    def get_report(self):
        return self._report

    def get_cluster_base_paths(self) -> list:
        return list(self._cluster_base_paths)

    def get_engine_name(self) -> str:
        return self._engine_name

    def get_diversity_report(self):
        return self._div_report

    # ------------------------------------------------------------------
    # tr_text helper
    # ------------------------------------------------------------------
    def tr_text(self, key):
        if hasattr(self.parent_window, 'tr_text'):
            return self.parent_window.tr_text(key)
        return TRANS.get('zh', {}).get(key, key)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Step indicator ──
        self.lbl_step = QLabel("")
        self.lbl_step.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #0056b3; padding: 4px;")
        layout.addWidget(self.lbl_step)

        # ── Stacked pages ──
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._build_page1()  # dedup removal
        self._build_page2()  # clustering setup
        self._build_page3()  # result preview

        # ── Bottom navigation ──
        nav_row = QHBoxLayout()
        nav_row.addStretch()
        self.btn_prev = QPushButton(self.tr_text("CLN_BTN_PREV"))
        self.btn_prev.clicked.connect(self._go_prev)
        nav_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton(self.tr_text("CLN_BTN_NEXT"))
        self.btn_next.setObjectName("btn_train")
        self.btn_next.clicked.connect(self._go_next)
        nav_row.addWidget(self.btn_next)

        self.btn_finish = QPushButton(self.tr_text("CLN_BTN_FINISH"))
        self.btn_finish.setStyleSheet("background-color: #FFA500; font-weight: bold;")
        self.btn_finish.setVisible(False)
        self.btn_finish.clicked.connect(self._on_ok)
        nav_row.addWidget(self.btn_finish)

        self.btn_cancel = QPushButton(self.tr_text("CLN_BTN_CANCEL"))
        self.btn_cancel.clicked.connect(self.reject)
        nav_row.addWidget(self.btn_cancel)
        layout.addLayout(nav_row)

    # ------------------------------------------------------------------
    # Page 1: dedup removal
    # ------------------------------------------------------------------
    def _build_page1(self):
        page = QWidget()
        v = QVBoxLayout(page)

        header = QLabel(self.tr_text("CLN_STEP1_TITLE"))
        header.setWordWrap(True)
        header.setStyleSheet("background: #e8f4fd; border: 1px solid #b8d4f0; "
                             "border-radius: 4px; padding: 8px; color: #1a5276;")
        v.addWidget(header)

        self.btn_scan = QPushButton(self.tr_text("CLN_BTN_SCAN"))
        self.btn_scan.setObjectName("btn_train")
        self.btn_scan.clicked.connect(self._start_scan)
        v.addWidget(self.btn_scan)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        v.addWidget(self.progress_bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #666; font-size: 12px;")
        v.addWidget(self.lbl_status)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        v.addWidget(self.lbl_summary)

        self.list_results = QListWidget()
        self.list_results.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_results.itemChanged.connect(self._update_apply_buttons)
        v.addWidget(self.list_results, 1)

        btn_row = QHBoxLayout()
        self.btn_apply_remove = QPushButton(self.tr_text("CLN_BTN_CONFIRM_REMOVE"))
        self.btn_apply_remove.setEnabled(False)
        self.btn_apply_remove.clicked.connect(self._on_apply_remove)
        btn_row.addWidget(self.btn_apply_remove)

        self.btn_skip_remove = QPushButton(self.tr_text("CLN_BTN_SKIP"))
        self.btn_skip_remove.setEnabled(False)
        self.btn_skip_remove.clicked.connect(self._on_skip_remove)
        btn_row.addWidget(self.btn_skip_remove)
        btn_row.addStretch()
        v.addLayout(btn_row)

        self.stack.addWidget(page)

    # ------------------------------------------------------------------
    # Page 2: clustering setup
    # ------------------------------------------------------------------
    def _build_page2(self):
        page = QWidget()
        v = QVBoxLayout(page)

        header = QLabel(self.tr_text("CLN_STEP2_TITLE"))
        header.setWordWrap(True)
        header.setStyleSheet("background: #fdf6e3; border: 1px solid #f0d9a0; "
                             "border-radius: 4px; padding: 8px; color: #7a5c00;")
        v.addWidget(header)

        # Remaining count
        self.lbl_remaining = QLabel("")
        v.addWidget(self.lbl_remaining)

        # Hardware detection hint (blue box, above model selection)
        self.lbl_hardware = QLabel("")
        self.lbl_hardware.setWordWrap(True)
        self.lbl_hardware.setStyleSheet(
            "background: #e8f4fd; border: 1px solid #b8d4f0; "
            "border-radius: 4px; padding: 6px; color: #1a5276; font-size: 12px;")
        v.addWidget(self.lbl_hardware)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(self.tr_text("CLN_LBL_FEATURE_MODEL")))
        self.combo_model = QComboBox()
        for _, label_key in self.EMBEDDING_OPTIONS:
            self.combo_model.addItem(self.tr_text(label_key))
        self.combo_model.setCurrentIndex(0)
        model_row.addWidget(self.combo_model, 1)
        v.addLayout(model_row)

        # Reference suggestion (green box, above target cluster count)
        self.lbl_cluster_ref = QLabel("")
        self.lbl_cluster_ref.setWordWrap(True)
        self.lbl_cluster_ref.setStyleSheet(
            "background: #e8f5e9; border: 1px solid #a5d6a7; "
            "border-radius: 4px; padding: 6px; color: #1b5e20; font-size: 12px;")
        v.addWidget(self.lbl_cluster_ref)

        # Similarity threshold (original clustering scheme)
        th_row = QHBoxLayout()
        th_row.addWidget(QLabel(self.tr_text("CLN_LBL_THRESHOLD")))
        self.spin_diverse = QDoubleSpinBox()
        self.spin_diverse.setRange(0.50, 0.99)
        self.spin_diverse.setSingleStep(0.01)
        self.spin_diverse.setValue(0.90)
        th_row.addWidget(self.spin_diverse)
        th_row.addWidget(QLabel(self.tr_text("CLN_LBL_THRESHOLD_HINT")))
        th_row.addStretch()
        v.addLayout(th_row)

        hint = QLabel(self.tr_text("CLN_THRESHOLD_EXPLAIN"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        v.addWidget(hint)

        self.btn_cluster = QPushButton(self.tr_text("CLN_BTN_CLUSTER"))
        self.btn_cluster.setObjectName("btn_train")
        self.btn_cluster.clicked.connect(self._start_clustering_from_page2)
        v.addWidget(self.btn_cluster)

        self.lbl_diversity = QLabel("")
        self.lbl_diversity.setWordWrap(True)
        v.addWidget(self.lbl_diversity)

        self.stack.addWidget(page)

    # ------------------------------------------------------------------
    # Page 3: result preview
    # ------------------------------------------------------------------
    def _build_page3(self):
        page = QWidget()
        v = QVBoxLayout(page)

        header = QLabel(self.tr_text("CLN_STEP3_TITLE"))
        header.setStyleSheet("background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 4px; padding: 8px; color: #1b5e20; font-weight: bold;")
        v.addWidget(header)

        # Treemap canvas (cluster size distribution)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.treemap_fig = Figure(figsize=(7, 3.2), dpi=100)
        self.treemap_canvas = FigureCanvasQTAgg(self.treemap_fig)
        self.treemap_canvas.setVisible(False)
        v.addWidget(self.treemap_canvas, 2)

        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setRange(0, 100)
        self.progress_bar2.setValue(0)
        self.progress_bar2.setVisible(False)
        v.addWidget(self.progress_bar2)

        self.lbl_cluster_status = QLabel("")
        self.lbl_cluster_status.setWordWrap(True)
        v.addWidget(self.lbl_cluster_status)

        # Legend: cluster color -> representative
        self.lbl_legend = QLabel("")
        self.lbl_legend.setWordWrap(True)
        self.lbl_legend.setStyleSheet("font-size: 11px; color: #555;")
        v.addWidget(self.lbl_legend)

        self.list_diverse = QListWidget()
        self.list_diverse.setVisible(False)
        v.addWidget(self.list_diverse, 1)

        self.stack.addWidget(page)

    # ------------------------------------------------------------------
    # Hardware detection & model recommendation
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

    def _detect_hardware_and_recommend(self):
        """Detect GPU/RAM and recommend the best default clustering model."""
        try:
            import torch as _torch
            has_cuda = _torch.cuda.is_available()
            gpu_name = _torch.cuda.get_device_name(0) if has_cuda else self.tr_text("CLN_GPU_NONE")
        except Exception:
            has_cuda = False
            gpu_name = self.tr_text("CLN_GPU_UNKNOWN")

        # RAM detection
        try:
            import psutil as _psutil
            ram_gb = _psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            ram_gb = 0.0

        self._has_cuda = has_cuda
        self._ram_gb = ram_gb

        # Recommendation logic
        if has_cuda:
            rec_idx = 2  # ResNet-50 (GPU available)
            rec_note = self.tr_text("CLN_REC_GPU")
        elif ram_gb >= 16:
            rec_idx = 1  # EfficientNet-B0
            rec_note = self.tr_text("CLN_REC_RAM")
        else:
            rec_idx = 0  # MobileNetV3
            rec_note = self.tr_text("CLN_REC_LOWMEM")

        # Display hardware info
        self.lbl_hardware.setText(
            self.tr_text("CLN_HARDWARE_FMT").format(gpu_name, ram_gb, rec_note))

        # Auto-select recommended model
        if 0 <= rec_idx < self.combo_model.count():
            self.combo_model.setCurrentIndex(rec_idx)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _update_nav_buttons(self):
        if self.mode == "dedup":
            # Dedup-only: single page, "Finish" acts as the confirm button.
            self.lbl_step.setText(self.tr_text("CLN_STEP1_ONLY"))
            self.stack.setCurrentIndex(0)
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.btn_finish.setText(self.tr_text("CLN_BTN_FINISH_DEDUP"))
            self.btn_finish.setVisible(True)
            self.btn_cancel.setVisible(True)
            return
        self.btn_finish.setText(self.tr_text("CLN_BTN_FINISH"))
        if self._start_step >= 1:
            # Follow-up rounds: no dedup step -> only 2 steps (index 0 unused)
            step_titles = [
                "",
                self.tr_text("CLN_STEP2_1"),
                self.tr_text("CLN_STEP2_2"),
            ]
        else:
            step_titles = [
                self.tr_text("CLN_STEP3_1"),
                self.tr_text("CLN_STEP3_2"),
                self.tr_text("CLN_STEP3_3"),
            ]
        self.lbl_step.setText(step_titles[self._current_step])
        self.stack.setCurrentIndex(self._current_step)

        self.btn_prev.setVisible(self._current_step > self._start_step)
        self.btn_next.setVisible(self._current_step < 2)
        self.btn_finish.setVisible(self._current_step == 2)
        self.btn_cancel.setVisible(True)

    def _go_prev(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._update_nav_buttons()

    def _go_next(self):
        if self._current_step == 0:
            # Step 1 → 2: require removal decision made
            if self._removed_paths is None:
                QMessageBox.information(self, self.tr_text("TTL_INFO"), self.tr_text("CLN_MSG_FIRST_REMOVE"))
                return
            self._current_step = 1
            self._update_nav_buttons()
            self._update_remaining_label()
        elif self._current_step == 1:
            if self._div_report is None:
                QMessageBox.information(self, self.tr_text("TTL_INFO"), self.tr_text("CLN_MSG_FIRST_CLUSTER"))
                return
            self._current_step = 2
            self._update_nav_buttons()

    # ------------------------------------------------------------------
    # Step 1: dedup scan
    # ------------------------------------------------------------------
    def _start_scan(self):
        if not self.image_list:
            QMessageBox.warning(self, self.tr_text("TTL_INFO"), self.tr_text("CLN_MSG_LOAD_DATASET"))
            return

        # Fixed perceptual hash engine (no model choice in step 1)
        from src.core.data_cleaner import DedupEngine
        engine = DedupEngine(method="pdqhash", hash_threshold=5,
                             similarity_threshold=0.95)

        self._report = None
        self._div_report = None
        self._removed_paths = None
        self._diverse_order = []
        self.list_results.clear()
        self.lbl_summary.setText("")
        self.btn_apply_remove.setEnabled(False)
        self.btn_skip_remove.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(self.tr_text("CLN_STATUS_SCANNING"))

        self._worker = CleaningWorker(engine, list(self.image_list))
        self._worker.progress.connect(self._on_progress)
        self._worker.duplicates_ready.connect(self._on_duplicates_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, phase: str):
        pct = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(min(pct, 100))
        self.lbl_status.setText(f"{phase}  ({current}/{total})")

    def _on_scan_finished(self):
        self.btn_scan.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(self.tr_text("CLN_STATUS_SCAN_DONE"))

    def _on_duplicates_ready(self, report):
        self._report = report
        self.list_results.clear()

        # ── Classified statistics (L1/L2/L3) ──
        layer_stats = {"L1": [0, 0], "L2": [0, 0], "L3": [0, 0]}  # [张数, 组数]
        for g in report.duplicate_groups:
            layer = g.layer if g.layer in layer_stats else "L2"
            layer_stats[layer][0] += len(g.paths) - 1
            layer_stats[layer][1] += 1

        total_remove = sum(v[0] for v in layer_stats.values())
        total_groups = sum(v[1] for v in layer_stats.values())

        lines = []
        if layer_stats["L1"][0]:
            lines.append(f"{self.tr_text('CLN_L1')}: {layer_stats['L1'][0]} 张 ({layer_stats['L1'][1]} 组)")
        if layer_stats["L2"][0]:
            lines.append(f"{self.tr_text('CLN_L2')}: {layer_stats['L2'][0]} 张 ({layer_stats['L2'][1]} 组)")
        if layer_stats["L3"][0]:
            lines.append(f"{self.tr_text('CLN_L3')}: {layer_stats['L3'][0]} 张 ({layer_stats['L3'][1]} 组)")

        summary_text = (self.tr_text("CLN_STATS_TOTAL").format(report.total_images, report.elapsed_seconds) + "\n"
                        + self.tr_text("CLN_DUP_FOUND").format(total_groups) + " — " + "  |  ".join(lines))
        if total_remove:
            summary_text += "\n" + self.tr_text("CLN_DUP_FOUND_TAIL").format(total_remove, total_groups)
        else:
            summary_text += "\n" + self.tr_text("CLN_NO_DUP")
        self.lbl_summary.setText(summary_text)

        # ── Detail list grouped by layer ──
        layer_labels = {"L1": self.tr_text("CLN_L1"), "L2": self.tr_text("CLN_L2"), "L3": self.tr_text("CLN_L3")}
        for layer in ["L1", "L2", "L3"]:
            layer_groups = [g for g in report.duplicate_groups if g.layer == layer]
            if not layer_groups:
                continue
            header = QListWidgetItem(f"── {layer_labels[layer]} ({len(layer_groups)} 组) ──")
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setFlags(header.flags() & ~Qt.ItemIsUserCheckable)
            self.list_results.addItem(header)

            for g in layer_groups:
                rep = g.representative or g.paths[0]
                rep_base = rep.replace("\\", "/").split("/")[-1] if rep else "?"
                dups = [p.replace("\\", "/").split("/")[-1] for p in g.paths[1:]]
                reason = g.reasons[1] if len(g.reasons) > 1 else ""
                text = self.tr_text("CLN_GROUP_FMT").format(g.group_id, rep_base, ', '.join(dups))
                if reason:
                    text += f"  [{reason}]"

                item = QListWidgetItem(text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, g)
                self.list_results.addItem(item)

        # Blurry
        if report.blurry_images:
            item = QListWidgetItem(
                self.tr_text("CLN_BLURRY").format(len(report.blurry_images)))
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            self.list_results.addItem(item)

        self.btn_apply_remove.setEnabled(total_remove > 0)
        self.btn_skip_remove.setEnabled(total_remove > 0)

        # 无重复图片时自动跳过（_removed_paths 置空），否则「下一步」会被
        # 「请先确认移除或跳过」拦截，而两个按钮又都被禁用，导致卡死。
        if total_remove == 0:
            self._removed_paths = []

    # ------------------------------------------------------------------
    # Step 1 actions
    # ------------------------------------------------------------------
    def _collect_checked_removals(self):
        paths = []
        for i in range(self.list_results.count()):
            item = self.list_results.item(i)
            if item.checkState() == Qt.Checked:
                group = item.data(Qt.UserRole)
                if group is not None:
                    paths.extend(group.paths[1:])
        return paths

    def _on_apply_remove(self):
        paths = self._collect_checked_removals()
        count = len(paths)
        if count == 0:
            QMessageBox.information(self, self.tr_text("TTL_INFO"), self.tr_text("CLN_MSG_NO_SELECTION"))
            return

        stats = {"L1": 0, "L2": 0, "L3": 0}
        for i in range(self.list_results.count()):
            item = self.list_results.item(i)
            if item.checkState() == Qt.Checked:
                group = item.data(Qt.UserRole)
                if group is not None:
                    layer = group.layer if group.layer in stats else "L2"
                    stats[layer] += len(group.paths) - 1

        parts = []
        if stats['L1']:
            parts.append(f"  {self.tr_text('CLN_L1')}: {stats['L1']} 张")
        if stats['L2']:
            parts.append(f"  {self.tr_text('CLN_L2')}: {stats['L2']} 张")
        if stats['L3']:
            parts.append(f"  {self.tr_text('CLN_L3')}: {stats['L3']} 张")
        breakdown = "\n".join(parts)

        reply = QMessageBox.question(
            self, self.tr_text("TTL_REMOVE_CONFIRM"),
            self.tr_text("CLN_REMOVE_CONFIRM_BODY").format(count, breakdown),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self._removed_paths = paths
            for i in range(self.list_results.count()):
                item = self.list_results.item(i)
                if item.checkState() == Qt.Checked:
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    item.setText(item.text() + self.tr_text("CLN_REMOVED_MARK"))
            self.btn_apply_remove.setEnabled(False)
            self.btn_skip_remove.setEnabled(False)
            self.lbl_status.setText(self.tr_text("CLN_REMOVED_STATUS").format(count))

    def _on_skip_remove(self):
        self._removed_paths = []
        self.btn_apply_remove.setEnabled(False)
        self.btn_skip_remove.setEnabled(False)
        self.lbl_status.setText(self.tr_text("CLN_SKIPPED_STATUS"))

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------
    def _update_remaining_label(self):
        remaining = self.image_list
        if self._removed_paths:
            removed_set = set(self._removed_paths)
            remaining = [p for p in self.image_list if p not in removed_set]
        self._cluster_input_paths = remaining
        self.lbl_remaining.setText(
            self.tr_text("CLN_REMAINING_FMT").format(len(remaining))
            + (self.tr_text("CLN_REMAINING_REMOVED").format(len(self._removed_paths)) if self._removed_paths else ""))
        # Reference suggestion: ~5% of the dataset as first-round cluster count
        n = len(remaining)
        k_ref = max(1, round(n * 0.05))
        self.lbl_cluster_ref.setText(
            self.tr_text("CLN_CLUSTER_REF").format(n, k_ref))

    def _start_clustering_from_page2(self):
        if not hasattr(self, '_cluster_input_paths') or not self._cluster_input_paths:
            self._update_remaining_label()
        paths = self._cluster_input_paths
        if len(paths) <= 1:
            QMessageBox.information(self, self.tr_text("TTL_INFO"), self.tr_text("CLN_MSG_TOO_FEW"))
            return

        idx = self.combo_model.currentIndex()
        backend = self.EMBEDDING_OPTIONS[idx][0]
        self._engine_name = backend
        self._sim_threshold = self.spin_diverse.value()
        self._auto_mode = False
        target_clusters = None

        # Use cached features if available (same backend) to speed up re-cluster
        cached_feats = getattr(self, '_cached_feats', None)
        if cached_feats is not None and self._cached_backend != backend:
            cached_feats = None  # features are model-specific, must re-extract

        self.btn_cluster.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.progress_bar2.setVisible(True)
        self.progress_bar2.setValue(0)
        self.lbl_diversity.setText(self.tr_text("CLN_STATUS_CLUSTERING"))

        self._cluster_worker = ClusterWorker(
            paths, similarity_threshold=self._sim_threshold, backend=backend,
            feats=cached_feats, target_clusters=target_clusters)
        self._cluster_worker.progress.connect(
            lambda c, t, ph: self._on_cluster_progress(c, t))
        self._cluster_worker.diversity_ready.connect(self._on_diversity_ready)
        self._cluster_worker.error_occurred.connect(self._on_error)
        self._cluster_worker.finished.connect(self._on_cluster_finished)
        self._cluster_worker.start()

    def _on_cluster_progress(self, current, total):
        pct = int((current / max(total, 1)) * 100)
        self.progress_bar2.setValue(min(pct, 100))
        self.lbl_diversity.setText(self.tr_text("CLN_STATUS_CLUSTERING_PROG").format(current, total))

    def _on_cluster_finished(self):
        self.btn_cluster.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.progress_bar2.setVisible(False)


    # ------------------------------------------------------------------
    # Step 3: result preview
    # ------------------------------------------------------------------
    def _on_diversity_ready(self, div_report):
        self._div_report = div_report
        self._cluster_base_paths = list(getattr(div_report, "base_paths", []))

        # Cache features + selector from the worker for fast re-cluster / recommendation
        worker = getattr(self, '_cluster_worker', None)
        if worker is not None:
            wf = getattr(worker, 'feats', None)
            if wf is not None:
                self._cached_feats = wf
                self._cached_backend = self._engine_name

        # Auto-advance to step 3
        self._current_step = 2
        self._update_nav_buttons()

        self.list_diverse.setVisible(True)
        self.list_diverse.clear()

        n_clusters = len(div_report.selected_indices)
        total = len(self._cluster_base_paths)

        # Cluster sizes
        from collections import Counter
        size_counter = Counter(div_report.cluster_assignments)
        cluster_sizes = [(c, size_counter.get(c, 0)) for c in range(n_clusters)]
        cluster_sizes.sort(key=lambda x: -x[1])

        # Largest cluster share
        largest_share = cluster_sizes[0][1] / max(total, 1) * 100 if cluster_sizes else 0

        # ── Reference cluster count (~5% of dataset) ──
        k_ref = max(1, round(total * 0.05))
        status_text = (
            self.tr_text("CLN_STATUS_TEXT_HEAD").format(n_clusters, total) + "\n"
            + self.tr_text("CLN_STATUS_TEXT_MAX").format(largest_share) + "\n"
            + self.tr_text("CLN_STATUS_TEXT_REF").format(k_ref, total))
        # Deviation hint if cluster count is far from the ~5% reference
        if n_clusters > 1 and total > 1:
            dev = n_clusters / max(k_ref, 1)
            if dev < 0.5 or dev > 2.0:
                status_text += self.tr_text("CLN_STATUS_TEXT_DEVIATION").format(n_clusters, k_ref)

        self.lbl_cluster_status.setText(status_text)

        # Draw treemap (cluster size distribution)
        self._draw_treemap(div_report, cluster_sizes, total)

        # Legend + representative list
        legend_parts = []
        for rank, (c, sz) in enumerate(cluster_sizes):
            rep_idx = div_report.selected_indices[c]
            path = div_report.selected_paths[c] if c < len(div_report.selected_paths) else f"#{rep_idx}"
            fname = path.replace("\\", "/").split("/")[-1]
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            legend_parts.append(
                f"<span style='color:{color};'>■</span> 簇{c}: {fname} ({sz}张)")
            item = QListWidgetItem(f"● 簇{c}: 代表 {fname}  ({sz} 张成员)")
            item.setForeground(QColor(color))
            self.list_diverse.addItem(item)

        self.lbl_legend.setText(" | ".join(legend_parts[:8]) + ("  …" if len(legend_parts) > 8 else ""))
        self.lbl_diversity.setText(self.tr_text("MSG_STATUS_DONE"))

    def _draw_treemap(self, div_report, cluster_sizes, total):
        """Draw a treemap with adaptive labels (big clusters get full text, small get none)."""
        try:
            import squarify
            import matplotlib
            from matplotlib.patches import Rectangle
            matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
            matplotlib.rcParams["axes.unicode_minus"] = False

            sizes = [max(sz, 0.1) for _, sz in cluster_sizes]
            colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c, _ in cluster_sizes]

            self.treemap_fig.clear()
            ax = self.treemap_fig.add_subplot(111)

            # Compute rectangles manually for per-rectangle label control
            normed = squarify.normalize_sizes(sizes, 1.0, 1.0)
            rects = squarify.squarify(normed, 0, 0, 1, 1)

            for r, (c, sz), color in zip(rects, cluster_sizes, colors):
                x, y, w, h = r["x"], r["y"], r["dx"], r["dy"]
                ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="white",
                                       linewidth=1.5, alpha=0.8))
                # Adaptive label: only show text if rectangle is big enough
                share = sz / max(total, 1)
                if w * h < 0.012:  # too small -> no text (like small-cap logo)
                    continue
                if share >= 0.10:
                    text = self.tr_text("CLN_TREEMAP_ITEM_3").format(c, sz, share)
                    fs = 10
                elif share >= 0.05:
                    text = self.tr_text("CLN_TREEMAP_ITEM_2").format(c, share)
                    fs = 8
                else:
                    text = self.tr_text("CLN_TREEMAP_ITEM_1").format(c)
                    fs = 7
                ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                        fontsize=fs, color="white", fontweight="bold",
                        linespacing=1.2)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("auto")
            ax.set_axis_off()
            ax.set_title(self.tr_text("CLN_TREEMAP_TITLE").format(len(cluster_sizes), total), fontsize=10)
            self.treemap_fig.tight_layout()
            self.treemap_canvas.draw()
            self.treemap_canvas.setVisible(True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.treemap_canvas.setVisible(False)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _on_error(self, msg: str):
        QMessageBox.critical(self, self.tr_text("TTL_ERROR"), self.tr_text("CLN_MSG_ERROR").format(msg))

    def _update_apply_buttons(self):
        checked = sum(1 for i in range(self.list_results.count())
                      if self.list_results.item(i).checkState() == Qt.Checked)
        self.btn_apply_remove.setEnabled(checked > 0)

    def _on_ok(self):
        """Finish: auto-apply diverse order if clustering ran."""
        if self.mode == "dedup" and self._removed_paths is None:
            # Scan finished but the user never confirmed or skipped the removal.
            QMessageBox.information(self, self.tr_text("TTL_INFO"),
                                    self.tr_text("CLN_MSG_FIRST_REMOVE"))
            return
        if self._div_report is not None and not self._diverse_order:
            self._diverse_order = list(self._div_report.selected_paths)
        self.accept()
