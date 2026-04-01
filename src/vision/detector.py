"""
Ball detector using YOLOv11 ONNX model trained on merged Roboflow dataset.
Loads best.onnx from models/pool_vision/ and runs inference via onnxruntime.
Returns a dict of {ball_label: (cx, cy)} in pixel coordinates.

ONNX output shape: [1, 20, 8400]
  - 20 = 4 box coords (cx,cy,w,h) + 16 class scores
  - 8400 = anchor grid for 640x640 input
"""

import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parents[2] / "models" / "pool_vision" / "best.onnx"

# Class names in training order — update if retrained with different class order
# 16 classes: cue + balls 1-15
CLASS_NAMES = {
    0: "cue",
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "10", 11: "11", 12: "12", 13: "13", 14: "14", 15: "15",
}


def _preprocess(frame, imgsz):
    """Resize + normalize frame to YOLO input tensor [1,3,imgsz,imgsz]."""
    img = cv2.resize(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))        # HWC -> CHW
    img = np.expand_dims(img, axis=0)          # -> [1,3,H,W]
    return img


def _nms(boxes, scores, iou_thresh=0.45):
    """Simple NMS. boxes: [[x1,y1,x2,y2], ...], scores: [float, ...]"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


class BallDetector:
    def __init__(self, conf_threshold=0.45, imgsz=320):  # imgsz ignored — ONNX model fixed at 640x640
        import onnxruntime as ort
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {MODEL_PATH}. "
                "Export from best.pt: yolo export model=best.pt format=onnx"
            )
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        self.conf = conf_threshold
        self.imgsz = 640  # ONNX model input is fixed at 640x640
        self.class_names = CLASS_NAMES
        self._input_name = self.session.get_inputs()[0].name

    def detect(self, frame) -> dict:
        """
        Run inference on a BGR frame (numpy array from cv2).
        Returns dict: {"cue": (cx, cy), "4": (cx, cy), ...}
        Only includes balls detected above confidence threshold.
        """
        orig_h, orig_w = frame.shape[:2]
        inp = _preprocess(frame, self.imgsz)

        outputs = self.session.run(None, {self._input_name: inp})
        # output0: [1, 20, 8400] -> squeeze to [20, 8400] -> transpose to [8400, 20]
        preds = outputs[0][0].T  # [8400, 20]

        # Split into boxes and class scores
        boxes_xywh = preds[:, :4]   # cx, cy, w, h (in imgsz coords)
        class_scores = preds[:, 4:] # [8400, num_classes]

        best_scores = class_scores.max(axis=1)
        best_cls    = class_scores.argmax(axis=1)

        mask = best_scores >= self.conf
        boxes_xywh  = boxes_xywh[mask]
        best_scores = best_scores[mask]
        best_cls    = best_cls[mask]

        if len(boxes_xywh) == 0:
            return {}

        # Convert cx,cy,w,h -> x1,y1,x2,y2 (still in imgsz space)
        cx, cy, w, h = boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,2], boxes_xywh[:,3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Per-class NMS, keep highest-conf detection per label
        balls = {}
        unique_cls = np.unique(best_cls)
        for cls_id in unique_cls:
            cls_mask = best_cls == cls_id
            cls_boxes  = xyxy[cls_mask]
            cls_scores = best_scores[cls_mask]
            keep = _nms(cls_boxes, cls_scores)
            if not keep:
                continue
            best_idx = keep[np.argmax(cls_scores[keep])]
            b = cls_boxes[best_idx]
            score = cls_scores[best_idx]

            # Scale back to original frame coords
            scale_x = orig_w / self.imgsz
            scale_y = orig_h / self.imgsz
            cx_px = int((b[0] + b[2]) / 2 * scale_x)
            cy_px = int((b[1] + b[3]) / 2 * scale_y)

            label = self.class_names.get(int(cls_id), str(cls_id))
            if label not in balls or score > balls[label][2]:
                balls[label] = (cx_px, cy_px, float(score))

        return {label: (cx, cy) for label, (cx, cy, _) in balls.items()}

    def draw(self, frame, balls: dict) -> np.ndarray:
        """Draw detected balls on frame for debugging."""
        out = frame.copy()
        for label, (cx, cy) in balls.items():
            cv2.circle(out, (cx, cy), 15, (0, 255, 0), 2)
            cv2.putText(out, label, (cx - 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out
