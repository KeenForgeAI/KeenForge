# src/core/inference_engine.py

import os
import torch
import traceback
from PyQt5.QtCore import QObject, pyqtSignal

# Try importing Ultralytics libraries for YOLO/RT-DETR models
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    print("Warning: Ultralytics library not found. YOLO models will not be available.")

# Import custom wrapper for classic torchvision models
# Using relative import assuming this file and model_wrappers.py are in the same package (src.core)
from .model_wrappers import ClassicWrapper


class InferenceEngine(QObject):
    """
    Backend engine for handling model loading and inference.
    Supports:
    1. Ultralytics models: YOLOv8, YOLOv11, RT-DETR, YOLO-World.
    2. Classic Torchvision models: Faster-RCNN, SSD, RetinaNet (via ClassicWrapper).
    """

    # Signals to communicate with the UI thread
    prediction_ready = pyqtSignal(list)  # Emits list of [cls_id, x, y, w, h]
    model_loaded = pyqtSignal(str)  # Emits the path of the loaded model
    inference_info = pyqtSignal(str)  # Emits logs/speed info (e.g., "15ms")

    def __init__(self, model_path: str = "yolov8n.pt"):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.model_type = "yolo"  # Options: 'yolo', 'classic'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, new_path: str = None):
        """
        Load a model from the specified path.
        Automatically detects if it's a YOLO model or a Classic Torchvision model.
        """
        if new_path:
            self.model_path = new_path

        print(f"Loading model: {self.model_path} on device: {self.device}")

        try:
            model_name_lower = self.model_path.lower()

            # === Branch 1: Classic Torchvision Models (SSD, Faster-RCNN, RetinaNet) ===
            if any(x in model_name_lower for x in ['ssd', 'faster', 'retina', 'rcnn']):
                self.model_type = "classic"
                self.model = ClassicWrapper(model_name=self.model_path, weights_path=self.model_path)

            # === Branch 2: Ultralytics Models (YOLO, RT-DETR) ===
            else:
                self.model_type = "yolo"
                if "rtdetr" in model_name_lower:
                    self.model = RTDETR(self.model_path)
                else:
                    self.model = YOLO(self.model_path)

            # Reset classes to default if supported (clears previous custom prompts)
            if hasattr(self.model, 'reset_classes'):
                try:
                    self.model.reset_classes()
                except Exception:
                    pass

            self.model_loaded.emit(self.model_path)
            print(f"Model loaded successfully. Type: {self.model_type}")

        except Exception as e:
            error_msg = f"Model Load Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.inference_info.emit(error_msg)

    def run_inference(self, image_path: str, prompt: str = None):
        """
        Run inference on a single image.

        Args:
            image_path (str): Path to the image file.
            prompt (str, optional): Comma-separated class names for YOLO-World Zero-Shot detection.
        """
        if self.model is None:
            self.load_model()

        try:
            predictions = []
            log_message = ""

            # === Logic for YOLO / RT-DETR / YOLO-World ===
            if self.model_type == "yolo":

                # Check if it is a YOLO-World model
                is_world_model = "world" in self.model_path.lower()

                # Handle Zero-Shot Prompting for YOLO-World
                if is_world_model and prompt and isinstance(prompt, str) and prompt.strip():
                    # 1. Parse Prompt: "giant panda, bear" -> ["giant panda", "bear"]
                    custom_classes = [c.strip() for c in prompt.split(',') if c.strip()]

                    # 2. Set custom classes (Persist until reset)
                    # This enables the model to detect open-vocabulary classes
                    self.model.set_classes(custom_classes)

                    # 3. Lower threshold for zero-shot tasks usually yields better recall
                    conf_thres = 0.15
                else:
                    # Standard detection (COCO classes or pre-trained custom classes)
                    conf_thres = 0.25

                # Run prediction
                # verbose=False prevents printing to stdout, we capture it manually
                results = self.model.predict(
                    image_path,
                    conf=conf_thres,
                    iou=0.5,
                    device=self.device,
                    verbose=False
                )

                for r in results:
                    # Process bounding boxes
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        # Normalized coordinates: [x_center, y_center, width, height]
                        x, y, w, h = box.xywhn[0].tolist()
                        predictions.append([cls_id, x, y, w, h])

                    # Extract inference speed and stats
                    speed = r.speed['inference']
                    # r.verbose() returns a string like "2 persons, 1 car"
                    class_stats = r.verbose().strip() or "(no detections)"
                    log_message = f"YOLO: {class_stats} | {speed:.1f}ms"

            # === Logic for Classic Torchvision Models ===
            elif self.model_type == "classic":
                # The ClassicWrapper handles the inference logic internally
                preds, msg = self.model.infer(image_path, conf_thres=0.3)
                predictions = preds
                log_message = msg

            # Emit results to UI
            self.prediction_ready.emit(predictions)
            self.inference_info.emit(log_message)

        except Exception as e:
            print(f"Inference Error: {e}")
            traceback.print_exc()
            self.inference_info.emit("Inference Failed")