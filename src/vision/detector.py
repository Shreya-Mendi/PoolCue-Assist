"""
Ball detector using YOLOv11 model trained on merged Roboflow dataset.
Loads best.pt from models/pool_vision/ and runs inference on a camera frame.
Returns a dict of {ball_label: (cx, cy)} in pixel coordinates.

Class names are read directly from the model — no need to hardcode them.
After training, the last cell in train/colab_train.ipynb prints the class order.
"""

import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parents[2] / "models" / "pool_vision" / "best.pt"


class BallDetector:
    def __init__(self, conf_threshold=0.45):
        from ultralytics import YOLO
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {MODEL_PATH}. "
                "Train in Colab (train/colab_train.ipynb) and place best.pt there."
            )
        self.model = YOLO(str(MODEL_PATH))
        self.conf = conf_threshold
        # Class names come from the model itself — set during training
        self.class_names = self.model.names  # dict: {0: "cue", 1: "1", ...}

    def detect(self, frame) -> dict:
        """
        Run inference on a BGR frame (numpy array from cv2).
        Returns dict: {"cue": (cx, cy), "4": (cx, cy), ...}
        Only includes balls detected above confidence threshold.
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        balls = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            conf = float(box.conf[0])
            # Keep highest-confidence detection per ball label
            if label not in balls or conf > balls[label][2]:
                balls[label] = (cx, cy, conf)

        # Strip confidence from output — callers only need positions
        return {label: (cx, cy) for label, (cx, cy, _) in balls.items()}

    def draw(self, frame, balls: dict) -> np.ndarray:
        """Draw bounding boxes and labels on frame for debugging."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        annotated = results.plot()
        return annotated
