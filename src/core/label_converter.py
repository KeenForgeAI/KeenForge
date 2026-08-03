# src/core/label_converter.py

import os
import json
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict, Any


class LabelConverter:
    """
    Utility class for converting label formats between YOLO, Pascal VOC, and COCO.
    Handles directory structure detection and coordinate normalization.
    """

    def __init__(self, dataset_root: str, classes: List[str]):
        """
        Initialize the converter with dataset root and class list.
        Automatically detects 'images' and 'labels' directories.

        Args:
            dataset_root (str): Path to the dataset root directory.
            classes (List[str]): List of class names.
        """
        self.dataset_root = dataset_root
        self.classes = classes

        # ========================================================
        # Intelligent Path Detection Logic
        # ========================================================
        # 1. Priority: Standard structure (root/images)
        path_v1 = os.path.join(dataset_root, "images")
        # 2. Secondary: Training structure (root/train/images)
        path_v2 = os.path.join(dataset_root, "train", "images")

        if os.path.exists(path_v1) and os.path.isdir(path_v1):
            self.images_dir = path_v1
            self.labels_dir = path_v1.replace("images", "labels")
        elif os.path.exists(path_v2) and os.path.isdir(path_v2):
            self.images_dir = path_v2
            self.labels_dir = path_v2.replace("images", "labels")
        else:
            # 3. Fallback: Assume images are in the root directory
            print(f"Warning: Standard 'images' directory not found. Using root: {dataset_root}")
            self.images_dir = dataset_root

            # Check if 'labels' folder exists in root, otherwise assume mixed content
            possible_labels = os.path.join(dataset_root, "labels")
            if os.path.exists(possible_labels):
                self.labels_dir = possible_labels
            else:
                self.labels_dir = dataset_root

    def _read_yolo_txt(self, txt_path: str, img_w: int, img_h: int) -> List[List[float]]:
        """
        Read YOLO format .txt file and convert to absolute coordinates [cls_id, xmin, ymin, xmax, ymax].
        """
        boxes = []
        if not os.path.exists(txt_path):
            return boxes

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Convert normalized (0-1) to absolute pixel coordinates
                    x_center = cx * img_w
                    y_center = cy * img_h
                    box_w = w * img_w
                    box_h = h * img_h

                    x_min = max(0, x_center - box_w / 2)
                    y_min = max(0, y_center - box_h / 2)
                    x_max = min(img_w, x_center + box_w / 2)
                    y_max = min(img_h, y_center + box_h / 2)

                    boxes.append([cls_id, x_min, y_min, x_max, y_max])
                except ValueError:
                    continue
        return boxes

    def to_pascal_voc(self, save_dir: str) -> int:
        """
        Export current YOLO labels to Pascal VOC (.xml) format.

        Args:
            save_dir (str): Directory to save the .xml files.

        Returns:
            int: Number of files processed.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(self.images_dir):
            print(f"Error: Image directory not found: {self.images_dir}")
            return 0

        image_files = [f for f in os.listdir(self.images_dir)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        if not image_files:
            print(f"Error: No images found in {self.images_dir}")
            return 0

        count = 0
        for img_file in image_files:
            img_path = os.path.join(self.images_dir, img_file)

            # Find corresponding txt file
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(self.labels_dir, txt_file)

            # Need image dimensions for conversion
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, c = img.shape

            boxes = self._read_yolo_txt(txt_path, w, h)

            # Build XML tree
            root = ET.Element('annotation')
            ET.SubElement(root, 'folder').text = os.path.basename(self.images_dir)
            ET.SubElement(root, 'filename').text = img_file

            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(w)
            ET.SubElement(size, 'height').text = str(h)
            ET.SubElement(size, 'depth').text = str(c)

            for box in boxes:
                cls_id, xmin, ymin, xmax, ymax = box
                if cls_id >= len(self.classes):
                    obj_name = str(cls_id)
                else:
                    obj_name = self.classes[cls_id]

                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = obj_name
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'

                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
                ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
                ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
                ET.SubElement(bndbox, 'ymax').text = str(int(ymax))

            # Format XML string
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            save_path = os.path.join(save_dir, os.path.splitext(img_file)[0] + ".xml")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(xml_str)
            count += 1

        return count

    def to_coco_json(self, save_path: str) -> int:
        """
        Export current YOLO labels to COCO Detection (.json) format.

        Args:
            save_path (str): Path to save the .json file.

        Returns:
            int: Number of images processed.
        """
        coco_dict = {
            "info": {"description": "Exported from AutoLabelPro", "year": 2025},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        for idx, name in enumerate(self.classes):
            coco_dict["categories"].append({"id": idx + 1, "name": name, "supercategory": "object"})

        if not os.path.exists(self.images_dir):
            print(f"Error: Image directory not found: {self.images_dir}")
            return 0

        image_files = [f for f in os.listdir(self.images_dir)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not image_files:
            print(f"Error: No images found in {self.images_dir}")
            return 0

        ann_id_counter = 1

        for img_id, img_file in enumerate(image_files):
            img_path = os.path.join(self.images_dir, img_file)
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(self.labels_dir, txt_file)

            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape

            coco_dict["images"].append({
                "id": img_id + 1,
                "file_name": img_file,
                "width": w,
                "height": h
            })

            boxes = self._read_yolo_txt(txt_path, w, h)

            for box in boxes:
                cls_id, xmin, ymin, xmax, ymax = box
                box_w = xmax - xmin
                box_h = ymax - ymin

                coco_dict["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": img_id + 1,
                    "category_id": cls_id + 1,
                    "bbox": [xmin, ymin, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id_counter += 1

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, indent=4)

        return len(image_files)

    def from_coco_json(self, json_path: str) -> Tuple[int, List[str]]:
        """
        Import labels from a COCO JSON file and convert to YOLO .txt format.

        Args:
            json_path (str): Path to the source JSON file.

        Returns:
            Tuple[int, List[str]]: (Number of processed images, List of newly added class names)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"JSON Load Error: {e}")
            return 0, []

        if not os.path.exists(self.labels_dir):
            os.makedirs(self.labels_dir)

        # 1. Parse Categories
        cat_id_to_name = {}
        new_classes_found = []

        for cat in data.get('categories', []):
            cat_id = cat['id']
            name = cat['name']
            cat_id_to_name[cat_id] = name

            if name not in self.classes:
                if name not in new_classes_found:
                    new_classes_found.append(name)

        # Temporary class list to maintain ID consistency during this import
        temp_classes = self.classes + new_classes_found

        # 2. Parse Images
        img_id_map = {}
        for img in data.get('images', []):
            img_id_map[img['id']] = {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }

        # 3. Parse Annotations
        annotations_by_img = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_img:
                annotations_by_img[img_id] = []
            annotations_by_img[img_id].append(ann)

        count = 0

        # Process each image
        for img_id, anns in annotations_by_img.items():
            if img_id not in img_id_map: continue

            img_info = img_id_map[img_id]
            file_name = img_info['file_name']
            w_img = img_info['width']
            h_img = img_info['height']

            # Construct txt path
            txt_name = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
            txt_path = os.path.join(self.labels_dir, txt_name)

            lines = []
            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox']  # [x_min, y_min, w, h]

                if cat_id not in cat_id_to_name: continue
                name = cat_id_to_name[cat_id]

                if name in temp_classes:
                    cls_idx = temp_classes.index(name)
                else:
                    continue

                # --- Convert: COCO (xywh absolute) -> YOLO (xywh normalized) ---
                x_min, y_min, box_w, box_h = bbox

                x_center = x_min + box_w / 2
                y_center = y_min + box_h / 2

                norm_x = x_center / w_img
                norm_y = y_center / h_img
                norm_w = box_w / w_img
                norm_h = box_h / h_img

                # Clamp to 0-1
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                lines.append(f"{cls_idx} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            if lines:
                with open(txt_path, 'w') as f:
                    f.writelines(lines)
                count += 1

        return count, new_classes_found

    def from_pascal_voc(self, xml_dir: str) -> Tuple[int, List[str]]:
        """
        Import labels from Pascal VOC XML files and convert to YOLO .txt format.

        Args:
            xml_dir (str): Directory containing .xml files.

        Returns:
            Tuple[int, List[str]]: (Number of processed files, List of newly added class names)
        """
        if not os.path.exists(xml_dir):
            return 0, []

        if not os.path.exists(self.labels_dir):
            os.makedirs(self.labels_dir)

        xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith('.xml')]

        count = 0
        new_classes_found = []
        temp_classes = list(self.classes)

        for xml_file in xml_files:
            xml_path = os.path.join(xml_dir, xml_file)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # 1. Get Image Size
                size = root.find('size')
                if size is not None:
                    w_img = int(size.find('width').text)
                    h_img = int(size.find('height').text)
                else:
                    print(f"Skipping {xml_file}: No size info.")
                    continue

                if w_img == 0 or h_img == 0: continue

                # 2. Prepare TXT content
                txt_name = os.path.splitext(xml_file)[0] + ".txt"
                txt_path = os.path.join(self.labels_dir, txt_name)

                lines = []

                # 3. Iterate objects
                for obj in root.findall('object'):
                    name = obj.find('name').text

                    if name not in temp_classes:
                        temp_classes.append(name)
                        if name not in new_classes_found:
                            new_classes_found.append(name)

                    cls_idx = temp_classes.index(name)

                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # --- Convert: VOC (xyxy absolute) -> YOLO (xywh normalized) ---
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    x_center = xmin + box_w / 2
                    y_center = ymin + box_h / 2

                    norm_x = x_center / w_img
                    norm_y = y_center / h_img
                    norm_w = box_w / w_img
                    norm_h = box_h / h_img

                    # Clamp to 0-1
                    norm_x = max(0, min(1, norm_x))
                    norm_y = max(0, min(1, norm_y))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))

                    lines.append(f"{cls_idx} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

                if lines:
                    with open(txt_path, 'w') as f:
                        f.writelines(lines)
                    count += 1

            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue

        return count, new_classes_found