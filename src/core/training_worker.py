# src/core/training_worker.py

import multiprocessing
import shutil
import os
import time
import cv2
import numpy as np
import yaml
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any

# --- Import torchmetrics for mAP calculation ---
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    print("Warning: torchmetrics not installed. mAP will remain 0.")
    MeanAveragePrecision = None

# --- Import Ultralytics ---
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    pass

# --- Import Torchvision ---
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
)
from torchvision import transforms as T


# =========================================================
# 1. Dataset Class
# =========================================================
class YoloFormatDataset(Dataset):
    """
    Custom Dataset for loading images and labels in YOLO format.
    Used for training Torchvision models.
    """
    def __init__(self, list_path: str, img_size: int = 640):
        self.image_paths = []
        if os.path.exists(list_path):
            with open(list_path, 'r', encoding='utf-8') as f:
                self.image_paths = [x.strip() for x in f.readlines() if x.strip()]
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        h_orig, w_orig = img.shape[:2]

        # Infer label path from image path
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.basename(os.path.dirname(img_path)) == "images":
            label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        # Convert normalized coordinates to absolute
                        cx *= w_orig
                        cy *= h_orig
                        w *= w_orig
                        h *= h_orig

                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2

                        # Clip coordinates to image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w_orig, x2)
                        y2 = min(h_orig, y2)

                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls + 1)  # Torchvision background is 0

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================================================
# 2. Training Process
# =========================================================
class TrainingProcess(multiprocessing.Process):
    """
    Background process for training models.
    Supports both Ultralytics YOLO and Classic Torchvision models.
    """
    def __init__(self, data_yaml_path: str, model_path: str, status_queue: multiprocessing.Queue, 
                 epochs: int = 30, freeze: int = 10, project_dir: str = None):
        super().__init__()
        self.data_yaml_path = data_yaml_path
        self.base_model_path = model_path
        self.status_queue = status_queue
        self.epochs = epochs
        self.freeze = freeze

        if project_dir:
            self.project_dir = project_dir
        else:
            self.project_dir = os.path.abspath("runs/detect")

        self.run_name = "loop_train"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def cleanup_weights_dir(self, weights_dir: str, extension: str = ".pt", max_backups: int = 3):
        """
        Clean up the weights directory:
        1. Delete 'last.pt' / 'last.pth'
        2. Limit the number of backup files to 'max_backups'
        """
        try:
            print(f"Cleaning up weights in {weights_dir}...")

            # 1. Delete last model
            last_model = os.path.join(weights_dir, f"last{extension}")
            if os.path.exists(last_model):
                try:
                    os.remove(last_model)
                    print(f"Deleted: {last_model}")
                except Exception as e:
                    print(f"Failed to delete last model: {e}")

            # 2. Limit backups
            search_pattern = os.path.join(weights_dir, f"*{extension}")
            all_files = glob.glob(search_pattern)

            backup_files = []
            for f in all_files:
                filename = os.path.basename(f)
                # Skip the main 'best' file
                if filename == f"best{extension}":
                    continue
                # Skip the main 'best_modelname' file
                if filename.startswith("best_") and not any(char.isdigit() for char in filename):
                    continue

                # Assume remaining files are backups
                backup_files.append(f)

            # Sort by modification time (oldest first)
            backup_files.sort(key=os.path.getmtime)

            # Delete old backups if limit exceeded
            if len(backup_files) > max_backups:
                files_to_delete = backup_files[:-max_backups]
                for f in files_to_delete:
                    try:
                        os.remove(f)
                        print(f"Deleted old backup: {os.path.basename(f)}")
                    except Exception as e:
                        print(f"Failed to delete {f}: {e}")

        except Exception as e:
            print(f"Cleanup error: {e}")

    def run(self):
        try:
            self.status_queue.put(("start", "Initializing training environment..."))

            model_name = os.path.basename(self.base_model_path).lower()

            # Branch based on model type
            if any(x in model_name for x in ['ssd', 'faster', 'retina', 'rcnn']):
                self.train_classic_torchvision()
            else:
                self.train_ultralytics_yolo()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_queue.put(("error", str(e)))

    # -----------------------------------------------------------------
    # Branch A: Classic Torchvision (with mAP calculation)
    # -----------------------------------------------------------------
    def train_classic_torchvision(self):
        # Extract model name
        filename_with_ext = os.path.basename(self.base_model_path)
        model_name = os.path.splitext(filename_with_ext)[0]

        # Prevent recursive naming (best_best_...)
        if model_name.startswith("best_"):
            model_name = model_name[5:]
        self.status_queue.put(("start", f"Starting PyTorch Training: {model_name}"))

        # 1. Load Model
        print(f"Loading Torchvision model: {model_name}")
        model = None
        if "ssd" in model_name:
            # Try loading local weights first
            local_pth = "ssd300_vgg16_coco-b556d3b4.pth"
            if os.path.exists(local_pth):
                model = ssd300_vgg16(weights=None)
                model.load_state_dict(torch.load(local_pth))
            else:
                model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

        elif "retina" in model_name:
            model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

        if self.freeze > 0:
            for param in model.backbone.parameters():
                param.requires_grad = False

        model.to(self.device)

        # 2. Prepare Data
        with open(self.data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        train_list_path = data_cfg.get('train', '')

        dataset = YoloFormatDataset(train_list_path, img_size=640)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                                num_workers=0, collate_fn=collate_fn)

        # 3. Optimizer & Metrics
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Initialize mAP calculator
        map_metric = None
        if MeanAveragePrecision:
            map_metric = MeanAveragePrecision(iou_type="bbox")
            map_metric.to(self.device)

        print(f"Start training for {self.epochs} epochs...")

        # 4. Training Loop
        for epoch in range(self.epochs):
            # --- Phase 1: Training (Loss) ---
            model.train()
            epoch_loss = 0.0
            steps = 0

            for images, targets in dataloader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += loss_value
                steps += 1

            lr_scheduler.step()
            avg_loss = epoch_loss / max(steps, 1)

            # --- Phase 2: Validation (mAP) ---
            current_map50 = 0.0

            if map_metric:
                try:
                    model.eval()
                    map_metric.reset()

                    with torch.no_grad():
                        # Evaluate on training set for active learning feedback loop
                        for val_imgs, val_targets in dataloader:
                            val_imgs = [img.to(self.device) for img in val_imgs]
                            val_targets = [{k: v.to(self.device) for k, v in t.items()} for t in val_targets]

                            predictions = model(val_imgs)
                            map_metric.update(predictions, val_targets)

                    map_result = map_metric.compute()
                    current_map50 = map_result['map_50'].item()
                except Exception as e:
                    print(f"mAP Calc Error: {e}")
                    current_map50 = 0.0

            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f} | mAP50 = {current_map50:.4f}")

            # Send progress to UI
            progress_data = {
                "epoch": epoch + 1,
                "total": self.epochs,
                "map50": current_map50,
                "loss": avg_loss
            }
            self.status_queue.put(("step", progress_data))

        # =========================================================
        # 5. Save Results (.pth)
        # =========================================================
        save_dir = os.path.join(self.project_dir, self.run_name, "weights")
        os.makedirs(save_dir, exist_ok=True)

        # 1. Generate final filename
        best_name = f"best_{model_name}.pth"
        best_path_named = os.path.join(save_dir, best_name)

        # 2. Backup existing model
        if os.path.exists(best_path_named):
            timestamp = int(time.time())
            backup_name = f"backup_{timestamp}_{best_name}"
            backup_path = os.path.join(save_dir, backup_name)
            try:
                shutil.copy(best_path_named, backup_path)
                print(f"Backup created: {backup_name}")
            except Exception as e:
                print(f"Backup failed: {e}")

        # 3. Save current model
        save_path = os.path.join(save_dir, "temp_last.pth")
        torch.save(model.state_dict(), save_path)

        # 4. Overwrite best model
        shutil.copy(save_path, best_path_named)

        if os.path.exists(save_path):
            os.remove(save_path)

        # 5. Cleanup old backups
        self.cleanup_weights_dir(save_dir, extension=".pth", max_backups=3)

        # Send success signal
        self.status_queue.put(("success", best_path_named))

    # -----------------------------------------------------------------
    # Branch B: Ultralytics YOLO
    # -----------------------------------------------------------------
    def train_ultralytics_yolo(self):
        # 1. Preparation
        filename_with_ext = os.path.basename(self.base_model_path)
        model_name = os.path.splitext(filename_with_ext)[0]
        if model_name.startswith("best_"):
            model_name = model_name[5:]

        self.status_queue.put(("start", f"Starting YOLO Training: {model_name}"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load Model
        if "rtdetr" in self.base_model_path.lower():
            model = RTDETR(self.base_model_path)
        else:
            model = YOLO(self.base_model_path)

        # 2. Define Callback
        def on_train_epoch_end(trainer):
            try:
                current_epoch = trainer.epoch + 1
                total = trainer.epochs
                metrics = trainer.metrics
                map50 = metrics.get("metrics/mAP50(B)", 0.0)

                train_loss = 0.0
                if hasattr(trainer, 'loss_items'):
                    loss_items = trainer.loss_items
                    if loss_items is not None:
                        if isinstance(loss_items, torch.Tensor):
                            train_loss = loss_items.sum().item()
                        else:
                            train_loss = float(loss_items)

                progress_data = {
                    "epoch": current_epoch,
                    "total": total,
                    "map50": map50,
                    "loss": train_loss
                }
                self.status_queue.put(("step", progress_data))
            except:
                pass

        model.add_callback("on_fit_epoch_end", on_train_epoch_end)

        # 3. Start Training
        model.train(
            data=self.data_yaml_path,
            epochs=self.epochs,
            imgsz=640,
            batch=4,
            workers=0,
            freeze=self.freeze,
            project=self.project_dir,
            name=self.run_name,
            exist_ok=True,
            verbose=False,
            patience=0
        )

        # =========================================================
        # 4. Result Handling
        # =========================================================

        generated_best = os.path.join(self.project_dir, self.run_name, "weights", "best.pt")
        weights_dir = os.path.dirname(generated_best)

        if os.path.exists(generated_best):
            # A. Target filename
            target_best_name = f"best_{model_name}.pt"
            target_best_path = os.path.join(weights_dir, target_best_name)

            # B. Backup existing
            if os.path.exists(target_best_path):
                timestamp = int(time.time())
                backup_name = f"backup_{timestamp}_{target_best_name}"
                backup_path = os.path.join(weights_dir, backup_name)
                try:
                    shutil.copy(target_best_path, backup_path)
                    print(f"Backup created: {backup_name}")
                except Exception as e:
                    print(f"Backup failed: {e}")

            # C. Overwrite
            shutil.copy(generated_best, target_best_path)

            # D. Cleanup
            self.cleanup_weights_dir(weights_dir, extension=".pt", max_backups=3)

            # E. Success
            self.status_queue.put(("success", target_best_path))

        else:
            self.status_queue.put(("error", "Training failed: best.pt not found"))