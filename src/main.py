# src/main.py
import sys
import os
import ctypes
import multiprocessing
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

# Ensure the src directory is in the python path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix for OpenMP conflict - MUST be before any torch/hnswlib import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# hnswlib must import before torch to avoid OpenMP runtime clash
try:
    import hnswlib  # noqa: F401
except ImportError:
    pass

from src.ui.main_window import MainWindow

from src.utils.app_id import APP_ID, register_app_id

# Set App ID + register the taskbar display name/icon so Windows shows
# KeenForge's own logo instead of a generic python.exe fallback.
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
except Exception:
    pass
_APP_ICON = register_app_id()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. Enable High DPI Scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    if hasattr(Qt, 'AA_Use96Dpi'):
        QApplication.setAttribute(Qt.AA_Use96Dpi)

    app = QApplication(sys.argv)

    if _APP_ICON:
        app.setWindowIcon(QIcon(_APP_ICON))

    # 2. Set Global Font (Segoe UI for Windows clarity)
    font = QFont("Segoe UI", 14)
    font.setBold(True)
    font.setWeight(75)
    app.setFont(font)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())