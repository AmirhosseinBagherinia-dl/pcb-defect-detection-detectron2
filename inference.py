import os
import argparse
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def main():
    parser = argparse.ArgumentParser(description="Inference for PCB Defect Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default="model_final.pth", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", default="output_pred.jpg", help="Path to save result")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    metadata = MetadataCatalog.get("pcb_inference")
    metadata.set(thing_classes=["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"])

    predictor = DefaultPredictor(cfg)
    img = cv2.imread(args.image)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(args.output, out.get_image()[:, :, ::-1])
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()