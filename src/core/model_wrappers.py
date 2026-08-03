# src/core/model_wrappers.py

import torch
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Union

# Import Torchvision models and weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights
)


# =========================================================
# 1. Classic Wrapper (Torchvision Models)
# =========================================================
class ClassicWrapper:
    """
    Wrapper for classic Torchvision object detection models.
    Supports: Faster R-CNN, RetinaNet, SSD.
    """

    def __init__(self, model_name: str = "fasterrcnn", weights_path: str = None):
        """
        Initialize the model and load weights.

        Args:
            model_name (str): Name of the model architecture (e.g., 'faster', 'ssd').
            weights_path (str, optional): Path to custom .pth weights.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name.lower()

        print(f"[ClassicWrapper] Initializing model: {model_name} on {self.device}")

        # --- Load Model Architecture ---
        if "faster" in self.model_name:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        elif "retina" in self.model_name:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights)
        elif "ssd" in self.model_name:
            weights = SSD300_VGG16_Weights.DEFAULT
            self.model = ssd300_vgg16(weights=weights)
        else:
            print("Warning: Unknown model name, defaulting to Faster R-CNN")
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

        # --- Load Custom Weights if provided ---
        if weights_path and weights_path.endswith((".pth", ".pt")):
            if "yolo" not in weights_path.lower():  # Prevent loading YOLO weights by mistake
                print(f"Loading custom weights: {weights_path}")
                try:
                    state_dict = torch.load(weights_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Failed to load weights: {e}")

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def infer(self, image_path: str, conf_thres: float = 0.5) -> Tuple[List[List[float]], str]:
        """
        Run inference on an image.

        Args:
            image_path (str): Path to the input image.
            conf_thres (float): Confidence threshold for filtering predictions.

        Returns:
            Tuple: (List of predictions [cls_id, x, y, w, h], Log message)
        """
        # 1. Read Image
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            return [], "Load Error"
        h_raw, w_raw = img_raw.shape[:2]

        # 2. Preprocess (BGR -> RGB -> Tensor 0-1)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            preds = self.model(img_tensor)[0]

        # 4. Post-process
        final_results = []
        boxes = preds['boxes'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()

        count = 0
        for i, score in enumerate(scores):
            if score > conf_thres:
                x1, y1, x2, y2 = boxes[i]
                # COCO usually starts from 1, YOLO starts from 0. Adjusting by -1.
                cls_id = int(labels[i]) - 1
                if cls_id < 0: cls_id = 0

                # Convert xyxy to normalized xywh
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2

                final_results.append([cls_id, cx / w_raw, cy / h_raw, w / w_raw, h / h_raw])
                count += 1

        return final_results, f"Classic Found: {count}"


# =========================================================
# 2. ONNX Wrapper (Optimized for YOLO Exports)
# =========================================================
class OnnxWrapper:
    """
    Wrapper for running ONNX models using ONNX Runtime.
    Optimized for YOLOv8/v11 exported models.
    """

    def __init__(self, onnx_path: str):
        """
        Initialize ONNX session.
        """
        # Prioritize CUDA execution
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            print("ONNX Runtime: CUDA not available, falling back to CPU.")
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.img_size = 640  # Default YOLO input size

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Letterbox Resize (Resize with aspect ratio + Padding).
        Standard preprocessing for YOLO models.
        """
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (self.img_size, self.img_size)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # Add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = img[None]  # Add batch dimension
        return img, r, (dw, dh)

    def infer(self, image_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45) -> Tuple[
        List[List[float]], str]:
        """
        Run inference on an image using ONNX Runtime.
        """
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            return [], "Load Error"
        h_raw, w_raw = img_raw.shape[:2]

        # 1. Preprocess
        blob, ratio, (dw, dh) = self.preprocess(img_raw)

        # 2. Inference
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]

        # 3. Post-process (Handle YOLOv8/v11 output format [1, 84, 8400])
        # Transpose: [1, 84, 8400] -> [1, 8400, 84]
        outputs = np.transpose(outputs, (0, 2, 1))
        prediction = outputs[0]

        # Extract data
        # prediction lines: [x, y, w, h, class1, class2, ...]
        x = prediction[:, 0]
        y = prediction[:, 1]
        w = prediction[:, 2]
        h = prediction[:, 3]
        scores = prediction[:, 4:]

        # Get max class score and index
        max_scores = np.max(scores, axis=1)
        max_indices = np.argmax(scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores > conf_thres

        x = x[mask]
        y = y[mask]
        w = w[mask]
        h = h[mask]
        confidences = max_scores[mask]
        class_ids = max_indices[mask]

        # Convert to NMS format (x1, y1, w, h)
        boxes_nms = []
        for i in range(len(x)):
            boxes_nms.append([int((x[i] - w[i] / 2)), int((y[i] - h[i] / 2)), int(w[i]), int(h[i])])

        # 4. NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes_nms, confidences.tolist(), conf_thres, iou_thres)

        final_results = []
        for i in indices:
            idx = i if isinstance(i, int) else i[0]

            # Get box in 640 scale
            cx_pad = x[idx]
            cy_pad = y[idx]
            w_pad = w[idx]
            h_pad = h[idx]
            cls = class_ids[idx]

            # Restore coordinates to original image scale
            # (coord - padding) / ratio
            cx = (cx_pad - dw) / ratio
            cy = (cy_pad - dh) / ratio
            w_res = w_pad / ratio
            h_res = h_pad / ratio

            # Normalize (0-1)
            final_results.append([
                int(cls),
                min(max(cx / w_raw, 0), 1),
                min(max(cy / h_raw, 0), 1),
                min(max(w_res / w_raw, 0), 1),
                min(max(h_res / h_raw, 0), 1)
            ])

        return final_results, f"ONNX Found: {len(final_results)}"