# Root entry point for PyInstaller packaging
import faulthandler
faulthandler.enable()
import sys
import os
import ctypes
import multiprocessing

# Ensure the project root is on sys.path so `src` package is importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import QApplication  # noqa: E402
from PyQt5.QtGui import QFont, QIcon  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402

from src.utils.app_id import APP_ID, register_app_id  # noqa: E402

# Set App ID + register the taskbar display name/icon so Windows shows
# KeenForge's own logo instead of a generic python.exe fallback.
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
except Exception:
    pass
_APP_ICON = register_app_id()

# PyQt5 5.9 aborts (qFatal -> native abort) on unhandled exceptions inside
# slots. Installing a custom sys.excepthook makes PyQt log them instead and
# keeps the app alive. Full traceback -> app_stderr.log + app_crash.log.
def _install_excepthook():
    import traceback as _tb

    def _hook(exc_type, exc_value, exc_tb):
        try:
            txt = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
            sys.stderr.write(txt + "\n")
            sys.stderr.flush()
            with open(os.path.join(_ROOT, "app_crash.log"), "a", encoding="utf-8") as f:
                f.write(txt + "\n")
        except Exception:
            pass

    sys.excepthook = _hook


_install_excepthook()

# Fix for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# hnswlib must import before torch to avoid OpenMP runtime clash
try:
    import hnswlib  # noqa: F401
except ImportError:
    pass

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Enable High DPI Scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    if hasattr(Qt, 'AA_Use96Dpi'):
        QApplication.setAttribute(Qt.AA_Use96Dpi)

    app = QApplication(sys.argv)

    if _APP_ICON:
        app.setWindowIcon(QIcon(_APP_ICON))

    # Set Global Font (X-AnyLabeling style: system default size, not oversized)
    font = QFont("Segoe UI", 11)
    app.setFont(font)

    # Import MainWindow AFTER sys.path is set (inside main block)
    from src.ui.main_window import MainWindow

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec_())
