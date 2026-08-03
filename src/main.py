# src/main.py
import sys
import os
import ctypes
import multiprocessing
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Ensure the src directory is in the python path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.main_window import MainWindow

# Set App ID for Windows Taskbar Icon
try:
    myappid = 'mycompany.autolabelpro.v5.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

# Fix for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. Enable High DPI Scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    if hasattr(Qt, 'AA_Use96Dpi'):
        QApplication.setAttribute(Qt.AA_Use96Dpi)

    app = QApplication(sys.argv)

    # 2. Set Global Font (Segoe UI for Windows clarity)
    font = QFont("Segoe UI", 14)
    font.setBold(True)
    font.setWeight(75)
    app.setFont(font)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())