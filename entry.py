# Root entry point for PyInstaller packaging
import sys
import os
import ctypes
import multiprocessing

# Ensure the project root is on sys.path so `src` package is importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication  # noqa: E402
from PyQt5.QtGui import QFont  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402

# Set App ID for Windows Taskbar Icon
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('com.keenforgeai.keenforge.v1.0')
except Exception:
    pass

# Fix for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Enable High DPI Scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    if hasattr(Qt, 'AA_Use96Dpi'):
        QApplication.setAttribute(Qt.AA_Use96Dpi)

    app = QApplication(sys.argv)

    # Set Global Font
    font = QFont("Segoe UI", 14)
    font.setBold(True)
    font.setWeight(75)
    app.setFont(font)

    # Import MainWindow AFTER sys.path is set (inside main block)
    from src.ui.main_window import MainWindow

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())
