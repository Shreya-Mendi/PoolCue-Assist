"""
Laser dot tracker using OpenCV HSV color masking.
Detects the brightest spot matching the laser color in a camera frame.
Supports green (default) or red lasers — set in config/settings.json.
"""

import cv2
import numpy as np


# HSV ranges — tune these for your specific laser + lighting conditions
# Run with DEBUG=True to see the mask live and adjust
GREEN_LOWER = np.array([40, 100, 150])
GREEN_UPPER = np.array([80, 255, 255])

# Red wraps around 0/180 in HSV so needs two ranges
RED_LOWER_1 = np.array([0, 120, 150])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 150])
RED_UPPER_2 = np.array([180, 255, 255])

MIN_AREA = 5     # ignore noise smaller than this (pixels²)
MAX_AREA = 500   # ignore large bright regions (not a laser dot)


class LaserTracker:
    def __init__(self, color="green", debug=False):
        self.color = color
        self.debug = debug

    def find_dot(self, frame) -> tuple | None:
        """
        Detect laser dot in BGR frame.
        Returns (cx, cy) pixel position or None if not found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.color == "green":
            mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        else:
            mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
            mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
            mask = cv2.bitwise_or(mask1, mask2)

        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if self.debug:
            cv2.imshow("laser_mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Pick the contour with area in valid range, closest to expected dot size
        valid = [c for c in contours if MIN_AREA < cv2.contourArea(c) < MAX_AREA]
        if not valid:
            return None

        best = max(valid, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def draw(self, frame, dot_pos: tuple) -> np.ndarray:
        """Draw a circle at the detected laser dot position."""
        if dot_pos:
            cv2.circle(frame, dot_pos, 8, (0, 255, 255), 2)
            cv2.circle(frame, dot_pos, 2, (0, 255, 255), -1)
        return frame
