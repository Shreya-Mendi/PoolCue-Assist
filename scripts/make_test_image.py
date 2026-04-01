"""
Generate a synthetic test image of a pool table with printed ball cutouts.
Saves to data/test_table.jpg — used with: python src/main.py --test-image data/test_table.jpg

Balls are drawn as colored circles with numbers, matching the print_balls.html colors.
Layout mimics a mid-game 8-ball table seen from overhead.
"""

import cv2
import numpy as np
from pathlib import Path

OUT = Path(__file__).parents[1] / "data" / "test_table.jpg"

# Table dimensions (pixels) — 640×480 overhead view
W, H = 640, 480
TABLE_GREEN = (34, 100, 34)   # BGR dark green felt

# Ball definitions: (label, center_x, center_y, bgr_color, is_stripe)
BALLS = [
    # cue ball
    ("cue",  160, 240, (230, 230, 230), False),
    # solids
    ("1",    380, 180, (0, 190, 230),   False),  # yellow (BGR)
    ("2",    420, 210, (168, 79,  26),  False),  # blue
    ("3",    460, 180, (0,   0,  204),  False),  # red
    ("4",    400, 260, (173, 13, 106),  False),  # purple
    ("5",    440, 290, (0,  96,  224),  False),  # orange
    ("8",    500, 240, (30,  30,  30),  False),  # black
    # stripes
    ("9",    320, 320, (0, 190, 230),   True),   # yellow stripe
    ("10",   360, 350, (168, 79,  26),  True),   # blue stripe
    ("11",   400, 380, (0,   0,  204),  True),   # red stripe
    ("13",   440, 340, (0,  96,  224),  True),   # orange stripe
    ("15",   480, 310, (0,   0, 128),   True),   # maroon stripe
]

RADIUS = 22  # ~2.5cm at this scale


def draw_ball(img, label, cx, cy, color, is_stripe):
    # Shadow
    cv2.circle(img, (cx + 3, cy + 3), RADIUS, (10, 40, 10), -1)

    if is_stripe:
        # White base
        cv2.circle(img, (cx, cy), RADIUS, (220, 220, 220), -1)
        # Color stripe band (horizontal rect clipped to circle)
        stripe_h = RADIUS // 2
        for dy in range(-stripe_h, stripe_h):
            x_span = int((RADIUS**2 - dy**2) ** 0.5)
            cv2.line(img, (cx - x_span, cy + dy), (cx + x_span, cy + dy), color, 1)
        cv2.circle(img, (cx, cy), RADIUS, (180, 180, 180), 2)
    else:
        cv2.circle(img, (cx, cy), RADIUS, color, -1)
        cv2.circle(img, (cx, cy), RADIUS, (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40)), 2)

    # White number circle
    cv2.circle(img, (cx, cy), 9, (255, 255, 255), -1)

    # Number text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = label
    scale = 0.38 if len(text) == 2 else 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), font, scale, (40, 40, 40), thickness, cv2.LINE_AA)

    # Highlight
    cv2.circle(img, (cx - RADIUS // 3, cy - RADIUS // 3), RADIUS // 5, (255, 255, 255), -1)


def main():
    img = np.full((H, W, 3), TABLE_GREEN, dtype=np.uint8)

    # Table border
    cv2.rectangle(img, (20, 20), (W - 20, H - 20), (30, 80, 10), 6)

    # Pockets (6 circles)
    pockets = [(20, 20), (W // 2, 20), (W - 20, 20),
               (20, H - 20), (W // 2, H - 20), (W - 20, H - 20)]
    for px, py in pockets:
        cv2.circle(img, (px, py), 14, (10, 10, 10), -1)

    # Felt texture (subtle noise)
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for label, cx, cy, color, is_stripe in BALLS:
        draw_ball(img, label, cx, cy, color, is_stripe)

    OUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT), img)
    print(f"Saved: {OUT}  ({W}x{H})")
    print("Run: python src/main.py --test-image data/test_table.jpg")


if __name__ == "__main__":
    main()
