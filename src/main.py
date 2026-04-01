"""
PoolCue Vision Assist — Main Orchestrator
AIPI 590 Final Project

Auto-starts on boot via systemd. No keyboard/SSH required after setup.

Loop:
  1. Camera captures frame
  2. YOLO detects all balls
  3. On table-state change → Claude recommends shot → speaker announces
  4. Laser tracker watches aim → AimGuide talks player onto the line
  5. Button press → IMU captures stroke → stroke model grades it
  6. Repeat
"""

import sys
import time
import json
import cv2
from pathlib import Path

# --- Hardware imports (Pi only) ---
try:
    import RPi.GPIO as GPIO
    from RPLCD.i2c import CharLCD
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("[WARN] RPi.GPIO not found — running in laptop debug mode")

# --- Project imports ---
sys.path.insert(0, str(Path(__file__).parent))

from vision.detector import BallDetector
from vision.laser_tracker import LaserTracker
from vision.pocket_map import PocketMap
from llm.shot_advisor import ShotAdvisor
from audio.speaker import Speaker
from aim_guidance import AimGuide
from game_state import GameState

# Reuse midterm stroke grading
sys.path.insert(0, str(Path(__file__).parent))
try:
    from imu_helpers import IMUReader
    import joblib
    STROKE_MODEL = joblib.load(Path(__file__).parents[1] / "models" / "stroke_model.pkl")
    HAS_IMU = True
except Exception as e:
    HAS_IMU = False
    print(f"[WARN] IMU/stroke model not loaded: {e}")

# --- Config ---
CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

BUTTON_PIN = CFG.get("button_pin", 18)
LASER_PIN  = CFG.get("laser_pin", 25)   # optional GPIO laser control
LASER_COLOR = CFG.get("laser_color", "green")
LCD_ADDR   = CFG.get("lcd_address", "0x27")
TABLE_CHANGE_THRESHOLD = CFG.get("table_change_threshold", 2)  # balls diff to re-query LLM


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    if CFG.get("laser_gpio_controlled", False):
        GPIO.setup(LASER_PIN, GPIO.OUT)
        GPIO.output(LASER_PIN, GPIO.HIGH)  # laser on by default


def setup_lcd():
    addr = int(LCD_ADDR, 16) if isinstance(LCD_ADDR, str) else LCD_ADDR
    lcd = CharLCD(i2c_expander="PCF8574", address=addr, port=1, cols=16, rows=2)
    lcd.clear()
    return lcd


def lcd_write(lcd, line1="", line2=""):
    if lcd is None:
        return
    try:
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1[:16].ljust(16))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2[:16].ljust(16))
    except Exception:
        pass


def balls_changed(prev: dict, curr: dict, threshold: int) -> bool:
    """Return True if enough balls have appeared/disappeared to re-query LLM."""
    prev_set = set(prev.keys())
    curr_set = set(curr.keys())
    return len(prev_set.symmetric_difference(curr_set)) >= threshold


def capture_stroke_features(imu):
    """Collect 1-second IMU window and extract features (midterm logic)."""
    import numpy as np
    samples = []
    start = time.time()
    while time.time() - start < 1.0:
        data = imu.get_data()
        samples.append(data)
        time.sleep(0.01)

    ax = [s["ax"] for s in samples]
    ay = [s["ay"] for s in samples]
    az = [s["az"] for s in samples]
    gy = [s["gy"] for s in samples]
    gz = [s["gz"] for s in samples]

    accel_mag = np.sqrt(np.array(ax)**2 + np.array(ay)**2 + np.array(az)**2)
    return [[
        float(np.max(accel_mag)),
        float(np.mean(gy)),
        float(np.var(gy)),
        float(np.mean(gz)),
        float(np.var(gz)),
        len(samples)
    ]]


def main():
    print("PoolCue Vision Assist starting...")

    # --- Init hardware ---
    lcd = None
    if HAS_GPIO:
        setup_gpio()
        try:
            lcd = setup_lcd()
            lcd_write(lcd, "PoolCue Vision", "Starting...")
        except Exception as e:
            print(f"[WARN] LCD init failed: {e}")

    # --- Init components ---
    speaker = Speaker()
    speaker.say("Pool Cue Vision Assist ready", block=True)

    detector = BallDetector(conf_threshold=CFG.get("detection_conf", 0.45), imgsz=CFG.get("imgsz", 320))
    laser_tracker = LaserTracker(color=LASER_COLOR)
    pocket_map = PocketMap()
    advisor = ShotAdvisor(model=CFG.get("llm_model", "gpt-4o-mini"))
    aim_guide = AimGuide(speaker=speaker, lcd=lcd)
    game = GameState(speaker=speaker, mode=CFG.get("game_mode", "8ball"))

    # --- Camera ---
    cam_index = CFG.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        speaker.say("Camera not found", block=True)
        lcd_write(lcd, "ERROR", "Camera missing")
        sys.exit(1)

    # --- Pocket calibration ---
    ret, frame = cap.read()
    if not pocket_map.is_calibrated():
        speaker.say("Click each pocket to calibrate", block=True)
        lcd_write(lcd, "CALIBRATE", "Click pockets")
        pocket_map.calibrate(frame)
        speaker.say("Calibration saved", block=True)

    pockets = pocket_map.pockets if pocket_map.is_calibrated() else pocket_map.estimate_from_frame(frame)

    # --- IMU setup ---
    imu = None
    if HAS_IMU and HAS_GPIO:
        try:
            imu = IMUReader()
        except Exception as e:
            print(f"[WARN] IMU init failed: {e}")

    print("System ready. Press button to capture stroke.")
    lcd_write(lcd, "READY", "Watching table")

    prev_balls = {}
    current_rec = None
    frame_h, frame_w = 480, 640

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]

            # --- Ball detection ---
            balls = detector.detect(frame)

            # --- Update game state ---
            game.update(balls)

            # --- Re-query LLM if table changed ---
            if balls_changed(prev_balls, balls, TABLE_CHANGE_THRESHOLD) and len(balls) > 1:
                prev_balls = balls.copy()
                remaining = game.remaining_for_player()
                shootable = {k: v for k, v in balls.items() if k in remaining or k == "cue"}

                if len(shootable) > 1:  # need at least cue + one target
                    lcd_write(lcd, "Thinking...", "")
                    try:
                        current_rec = advisor.recommend(
                            balls=shootable,
                            pockets=pockets,
                            game_state={
                                "mode": game.mode,
                                "player_type": game.current_player_type,
                                "frame_w": frame_w,
                                "frame_h": frame_h,
                            }
                        )
                        speaker.say(current_rec["spoken"])
                        ball_label = current_rec.get("ball", "?")
                        pocket_label = current_rec.get("pocket", "?")[:6]
                        diff = current_rec.get("difficulty", "?")
                        lcd_write(lcd, f"{ball_label} -> {pocket_label}", f"Diff: {diff}/5")
                    except Exception as e:
                        print(f"[LLM error] {e}")
                        lcd_write(lcd, "LLM error", str(e)[:16])

            # --- Aim guidance ---
            if current_rec and "cue" in balls:
                target_label = current_rec.get("ball")
                pocket_name = current_rec.get("pocket")

                if target_label in balls and pocket_name in pockets:
                    laser_dot = laser_tracker.find_dot(frame)
                    aim_guide.update(
                        laser_dot=laser_dot,
                        cue_pos=balls["cue"],
                        target_pos=balls[target_label],
                        pocket_pos=pockets[pocket_name]
                    )

            # --- Button press → stroke capture ---
            button_pressed = (
                HAS_GPIO and GPIO.input(BUTTON_PIN) == GPIO.LOW
            )

            if button_pressed and imu and current_rec:
                aim_guide.reset()
                lcd_write(lcd, "Shoot!", "Capturing...")
                time.sleep(0.1)  # debounce

                features = capture_stroke_features(imu)
                label = STROKE_MODEL.predict(features)[0]
                prob = STROKE_MODEL.predict_proba(features)[0]
                good_prob = prob[1] if len(prob) > 1 else prob[0]

                stroke_text = "GOOD" if label == 1 else "BAD"
                conf_pct = int(good_prob * 100)

                if label == 1:
                    speaker.say(f"Good stroke, {conf_pct} percent confidence")
                else:
                    speaker.say("Bad stroke — you twisted the cue")

                lcd_write(lcd, f"Stroke: {stroke_text}", f"Conf: {conf_pct}%")
                game.log_shot(current_rec, stroke_text, aim_guide._last_spoken)
                time.sleep(1.5)

            # Optional: show annotated frame on HDMI/display
            if CFG.get("show_display", False):
                annotated = detector.draw(frame, balls)
                laser_dot = laser_tracker.find_dot(frame)
                if laser_dot:
                    laser_tracker.draw(annotated, laser_dot)
                cv2.imshow("PoolCue Vision Assist", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)  # ~20fps loop

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if HAS_GPIO:
            GPIO.cleanup()
        lcd_write(lcd, "Goodbye!", "")
        print(f"Shot log: {game.shot_log}")


if __name__ == "__main__":
    main()
