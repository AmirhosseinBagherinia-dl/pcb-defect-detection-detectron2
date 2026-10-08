import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm

CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper"
]

def convert_yolo_to_coco(images_dir, labels_dir, output_json_path):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    }

    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    ann_id = 0
    for img_id, img_path in enumerate(tqdm(image_paths, desc=f"Converting {Path(images_dir).parent.name}")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width, _ = img.shape

        coco_format["images"].append({
            "id": img_id,
            "file_name": str(img_path),
            "width": width,
            "height": height
        })

        txt_path = Path(labels_dir) / (img_path.stem + ".txt")
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, bw, bh = map(float, parts[1:5])

                        abs_w = bw * width
                        abs_h = bh * height
                        abs_x = (x_center * width) - (abs_w / 2.0)
                        abs_y = (y_center * height) - (abs_h / 2.0)

                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [abs_x, abs_y, abs_w, abs_h],
                            "area": abs_w * abs_h,
                            "iscrowd": 0
                        })
                        ann_id += 1

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_format, f)