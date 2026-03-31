"""
Aim guidance module.
Compares the laser dot position to the ideal shot line and tells
the player how to correct their aim via speaker + LCD.

Ideal shot line: from cue ball through target ball, extended toward pocket.
Laser dot: where the player is currently pointing the cue.
Error: perpendicular distance from dot to ideal line (pixels).
"""

import numpy as np
import time


LOCKED_THRESHOLD = 18    # pixels — within this = "locked in"
GUIDANCE_COOLDOWN = 1.2  # seconds between spoken corrections


def _point_to_line_distance(point, line_start, line_end) -> float:
    """Perpendicular distance from point to infinite line through line_start→line_end."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return float("inf")
    # Signed distance (positive = right of line direction, negative = left)
    return ((dy * px - dx * py + x2 * y1 - y2 * x1) /
            np.sqrt(dx ** 2 + dy ** 2))


def _ideal_aim_point(cue_pos, target_pos) -> tuple:
    """
    Return the point on the cue side of target ball where the laser should be.
    This is the ghost ball position: same direction as cue→target, at cue ball radius distance.
    Simplified here to just the target ball center — good enough for aim guidance.
    """
    return target_pos


class AimGuide:
    def __init__(self, speaker, lcd=None, threshold=LOCKED_THRESHOLD):
        self.speaker = speaker
        self.lcd = lcd
        self.threshold = threshold
        self._last_spoken = 0
        self._locked_since = None
        self.locked = False

    def update(self, laser_dot, cue_pos, target_pos, pocket_pos) -> str:
        """
        Given current laser dot position and shot geometry, provide guidance.
        Returns status string: "locked", "left", "right", "no_dot"

        Args:
            laser_dot: (cx, cy) from LaserTracker, or None
            cue_pos: cue ball pixel position
            target_pos: target ball pixel position
            pocket_pos: destination pocket pixel position
        """
        if laser_dot is None:
            self._locked_since = None
            self.locked = False
            self._update_lcd("AIM: no laser")
            return "no_dot"

        # Signed distance: positive = dot is right of ideal line, negative = left
        signed_dist = _point_to_line_distance(laser_dot, cue_pos, target_pos)
        abs_dist = abs(signed_dist)

        if abs_dist <= self.threshold:
            if self._locked_since is None:
                self._locked_since = time.time()
            # Require 0.5s stable lock before announcing
            if time.time() - self._locked_since > 0.5:
                if not self.locked:
                    self.locked = True
                    self.speaker.say("Locked in — shoot now")
                self._update_lcd(f"AIM: LOCKED {int(abs_dist)}px")
            return "locked"
        else:
            self._locked_since = None
            self.locked = False
            now = time.time()
            if now - self._last_spoken > GUIDANCE_COOLDOWN:
                self._last_spoken = now
                if signed_dist > 0:
                    self.speaker.say("Move left")
                    self._update_lcd(f"AIM: LEFT  {int(abs_dist)}px")
                    return "left"
                else:
                    self.speaker.say("Move right")
                    self._update_lcd(f"AIM: RIGHT {int(abs_dist)}px")
                    return "right"

        return "adjusting"

    def reset(self):
        """Call after each shot to reset lock state."""
        self._locked_since = None
        self.locked = False

    def _update_lcd(self, msg: str):
        if self.lcd:
            try:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(msg[:16].ljust(16))
            except Exception:
                pass
