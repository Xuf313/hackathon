"""Build the YOLO-World (open-vocabulary) model with our apartment vocabulary.

Run once:  python make_world_model.py
Downloads yolov8s-worldv2.pt (~25 MB) + the CLIP text encoder (~340 MB, first time only) and writes
yolo_world_apartment.pt, which tb3_sar loads (no CLIP / internet needed at runtime).
"""
import os

from ultralytics import YOLOWorld

VOCAB = ["chair", "sofa", "armchair", "bed", "cabinet", "table", "desk", "toilet", "bathtub", "sink",
         "refrigerator", "oven", "washing machine", "fire extinguisher", "television", "computer monitor",
         "laptop", "potted plant", "flowers", "book", "bottle", "wine glass", "plate", "bowl", "clock",
         "painting", "radiator", "door", "stairs", "cardboard box", "cat", "rubber duck",
         "red apple", "green apple", "orange", "soccer ball", "person"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    m = YOLOWorld(os.path.join(here, "yolov8s-worldv2.pt"))
    m.set_classes(VOCAB)
    m.save(os.path.join(here, "yolo_world_apartment.pt"))
    print("saved yolo_world_apartment.pt with", len(VOCAB), "classes")
