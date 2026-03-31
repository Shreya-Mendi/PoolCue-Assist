"""
Pocket coordinate map for a standard 9-foot pool table.
On first run, click each pocket on the camera frame to calibrate.
Saves pixel coordinates to config/pocket_coords.json.
Subsequent runs load from file — no recalibration needed unless camera moves.
"""

import json
import cv2
import numpy as np
from pathlib import Path

COORDS_FILE = Path(__file__).parents[2] / "config" / "pocket_coords.json"

# Standard 6-pocket names in order (used in LLM prompts and audio)
POCKET_NAMES = [
    "top-left", "top-center", "top-right",
    "bottom-left", "bottom-center", "bottom-right"
]

# Fallback: estimated pocket positions as fractions of frame (w, h)
# Used if calibration file is missing — less accurate but functional
DEFAULT_FRACTIONS = [
    (0.02, 0.02),   # top-left
    (0.50, 0.02),   # top-center
    (0.98, 0.02),   # top-right
    (0.02, 0.98),   # bottom-left
    (0.50, 0.98),   # bottom-center
    (0.98, 0.98),   # bottom-right
]


class PocketMap:
    def __init__(self):
        self.pockets = {}  # name -> (cx, cy)
        self._load()

    def _load(self):
        if COORDS_FILE.exists():
            with open(COORDS_FILE) as f:
                self.pockets = {k: tuple(v) for k, v in json.load(f).items()}

    def is_calibrated(self) -> bool:
        return len(self.pockets) == 6

    def calibrate(self, frame):
        """
        Interactive calibration: click each pocket on the frame.
        Call once when camera is in final position.
        """
        clicks = []
        clone = frame.copy()
        remaining = POCKET_NAMES.copy()

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and remaining:
                name = remaining.pop(0)
                clicks.append((name, (x, y)))
                cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(clone, name, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                print(f"  Marked: {name} at ({x}, {y})")
                cv2.imshow("calibrate_pockets", clone)

        print("\nPocket calibration — click each pocket in this order:")
        for n in POCKET_NAMES:
            print(f"  {n}")
        print("Press any key when done.\n")

        cv2.imshow("calibrate_pockets", clone)
        cv2.setMouseCallback("calibrate_pockets", on_click)
        cv2.waitKey(0)
        cv2.destroyWindow("calibrate_pockets")

        self.pockets = {name: pos for name, pos in clicks}
        COORDS_FILE.parent.mkdir(exist_ok=True)
        with open(COORDS_FILE, "w") as f:
            json.dump(self.pockets, f, indent=2)
        print(f"Saved pocket coordinates to {COORDS_FILE}")

    def get(self, name: str) -> tuple | None:
        return self.pockets.get(name)

    def estimate_from_frame(self, frame) -> dict:
        """Fallback: estimate pocket positions from frame dimensions."""
        h, w = frame.shape[:2]
        return {
            name: (int(fx * w), int(fy * h))
            for name, (fx, fy) in zip(POCKET_NAMES, DEFAULT_FRACTIONS)
        }

    def nearest_pocket(self, pos: tuple) -> str:
        """Return the name of the pocket closest to pos (cx, cy)."""
        px, py = pos
        return min(
            self.pockets,
            key=lambda name: (self.pockets[name][0] - px) ** 2 + (self.pockets[name][1] - py) ** 2
        )
