# src/utils/threads.py
import os
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

class ModelLoaderThread(QThread):
    """
    Background thread for loading heavy deep learning models
    to prevent freezing the main UI.
    """
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, engine, model_path):
        super().__init__()
        self.engine = engine
        self.model_path = model_path

    def run(self):
        try:
            self.engine.load_model(self.model_path)
            self.finished_signal.emit(True, f"Loaded: {os.path.basename(self.model_path)}")
        except Exception as e:
            traceback.print_exc()
            self.finished_signal.emit(False, str(e))