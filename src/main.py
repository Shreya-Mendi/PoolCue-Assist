"""
PoolCue Vision Assist — Demo Mode
AIPI 590 Final Project

Flow:
  1st button press → detect balls → LLM shot recommendation → speak it
  2nd button press → IMU stroke capture (1s) → grade stroke → LEDs + buzzer + speak result
"""

import sys
import argparse
import time
import math
import json
import joblib
import numpy as np
import cv2
from pathlib import Path

# Hardware imports (Pi only)
try:
    from RPLCD.i2c import CharLCD
    HAS_LCD = True
except ImportError:
    HAS_LCD = False

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

sys.path.insert(0, str(Path(__file__).parent))

from vision.detector import BallDetector
from vision.pocket_map import PocketMap
from llm.shot_advisor import ShotAdvisor
from audio.speaker import Speaker
from game_state import GameState

CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

LCD_ADDR    = CFG.get("lcd_address", "0x27")
BUTTON_PIN  = CFG.get("button_pin", 18)
LED_GREEN   = 17
LED_RED     = 27
BUZZER_PIN  = 22
TRIG_PIN    = 23
ECHO_PIN    = 24
PWM_FREQ    = 1000
SAMPLE_RATE = 100
WINDOW_S    = 1.0
CUE_HEIGHT_LOW  = 8   # cm — too low
CUE_HEIGHT_HIGH = 20  # cm — too high

STROKE_MODEL_PATH = Path(__file__).parents[1] / "models" / "stroke_model.pkl"
STROKE_FEATURES   = ["peak_accel", "mean_gyro_y", "var_gyro_y", "mean_gyro_z", "var_gyro_z", "duration"]

# IMU sensor (Pi only)
try:
    from imu_helpers import sensor, load_cal
    HAS_IMU = True
except Exception:
    HAS_IMU = False


# ── LCD helpers ──────────────────────────────────────────────────────────────

def setup_lcd():
    addr = int(LCD_ADDR, 16) if isinstance(LCD_ADDR, str) else LCD_ADDR
    lcd = CharLCD(i2c_expander="PCF8574", address=addr, port=1, cols=16, rows=2)
    lcd.clear()
    return lcd


def lcd_write(lcd, line1="", line2=""):
    if lcd is None:
        print(f"LCD: {line1} | {line2}")
        return
    try:
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1[:16].ljust(16))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2[:16].ljust(16))
    except Exception as e:
        print(f"LCD: {line1} | {line2}  (err: {e})")


# ── GPIO helpers ─────────────────────────────────────────────────────────────

def setup_gpio():
    if not HAS_GPIO:
        return None, None, None
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LED_GREEN,  GPIO.OUT)
    GPIO.setup(LED_RED,    GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(23, GPIO.OUT)  # HC-SR04 TRIG
    GPIO.setup(24, GPIO.IN)   # HC-SR04 ECHO
    pwm_green  = GPIO.PWM(LED_GREEN,  PWM_FREQ)
    pwm_red    = GPIO.PWM(LED_RED,    PWM_FREQ)
    pwm_buzzer = GPIO.PWM(BUZZER_PIN, 2000)
    pwm_green.start(0)
    pwm_red.start(0)
    pwm_buzzer.start(0)
    return pwm_green, pwm_red, pwm_buzzer


def measure_distance_cm():
    """Read HC-SR04 cue height in cm. Returns None on timeout/no GPIO."""
    if not HAS_GPIO:
        return None
    try:
        GPIO.output(23, False); time.sleep(0.01)
        GPIO.output(23, True);  time.sleep(0.00001); GPIO.output(23, False)
        t = time.time()
        while GPIO.input(24) == 0:
            if time.time() - t > 0.04: return None
        t1 = time.time()
        while GPIO.input(24) == 1:
            if time.time() - t1 > 0.04: return None
        return round((time.time() - t1) * 34300 / 2, 1)
    except Exception:
        return None


def button_pressed():
    if not HAS_GPIO:
        return False
    return GPIO.input(BUTTON_PIN) == GPIO.LOW


def set_leds(pwm_green, pwm_red, good_prob):
    if pwm_green is None:
        return
    pwm_green.ChangeDutyCycle(int(good_prob * 100))
    pwm_red.ChangeDutyCycle(int((1.0 - good_prob) * 100))


def beep_good(pwm_buzzer):
    if pwm_buzzer is None:
        return
    for _ in range(2):
        pwm_buzzer.ChangeDutyCycle(70)
        time.sleep(0.1)
        pwm_buzzer.ChangeDutyCycle(0)
        time.sleep(0.08)


def beep_bad(pwm_buzzer):
    if pwm_buzzer is None:
        return
    pwm_buzzer.ChangeDutyCycle(30)
    time.sleep(0.4)
    pwm_buzzer.ChangeDutyCycle(0)


# ── Stroke helpers ───────────────────────────────────────────────────────────

def extract_features(accel_list, gyro_list):
    accel_mag = [math.sqrt(a["x"]**2 + a["y"]**2 + a["z"]**2) for a in accel_list]
    gyro_y    = [g["y"] for g in gyro_list]
    gyro_z    = [g["z"] for g in gyro_list]
    return [
        max(accel_mag),
        float(np.mean(gyro_y)),
        float(np.var(gyro_y)),
        float(np.mean(gyro_z)),
        float(np.var(gyro_z)),
        len(accel_list),
    ]


def _tilt_feedback(gyro_buf):
    """Return a short tilt/stability comment based on gyro data. None if stable."""
    if not gyro_buf:
        return None
    gx = [g["x"] for g in gyro_buf]
    gy = [g["y"] for g in gyro_buf]
    gz = [g["z"] for g in gyro_buf]
    roll_var  = float(np.var(gx))
    pitch_var = float(np.var(gy))
    yaw_var   = float(np.var(gz))
    mean_roll  = float(np.mean(gx))
    mean_pitch = float(np.mean(gy))

    issues = []
    if roll_var > 500:
        issues.append("side-to-side wobble — keep your elbow tucked")
    if pitch_var > 500:
        issues.append("up-down tilt — try to keep the cue level")
    if yaw_var > 300:
        issues.append("twisting — rotate from the shoulder, not the wrist")
    if abs(mean_roll) > 30:
        issues.append("cue tilted sideways — aim straighter")
    if abs(mean_pitch) > 30:
        issues.append("cue angled too steeply — flatten it out")
    return issues[0] if issues else None


def capture_and_grade_stroke(speaker, lcd, clf, cal, pwm_green, pwm_red, pwm_buzzer):
    """Capture 1s of IMU data, check cue height via HC-SR04, grade + give tilt feedback."""
    dist = None  # always defined

    if not HAS_IMU:
        speaker.say("No IMU found — can't grade your stroke, but you look great out there!", block=True)
        return "unknown"

    # ── Cue height check via HC-SR04 (before swing) ──────────────
    dist = measure_distance_cm()
    if dist is not None:
        print(f"[HC-SR04] Cue height: {dist:.1f}cm")
        if dist < CUE_HEIGHT_LOW:
            msg = f"Your cue is only {int(dist)} centimeters up — raise it before you swing!"
            lcd_write(lcd, f"Too low {dist}cm", "Raise cue!")
        elif dist > CUE_HEIGHT_HIGH:
            msg = f"You're {int(dist)} centimeters high — bring the cue down closer to the table."
            lcd_write(lcd, f"Too high {dist}cm", "Lower cue!")
        else:
            msg = f"Height looks good at {int(dist)} centimeters. Now swing!"
            lcd_write(lcd, f"Height OK {dist}cm", "Swing now!")
        speaker.say(msg, block=True)
    else:
        print("[HC-SR04] No reading — skipping height check")
        lcd_write(lcd, "Swing now!", "1 second...")
        speaker.say("Alright, swing when you're ready!", block=True)

    print("[STROKE] Recording 1 second of IMU data...")
    set_leds(pwm_green, pwm_red, 0.5)

    accel_buf, gyro_buf = [], []
    t0 = time.time()
    while time.time() - t0 < WINDOW_S:
        try:
            a = sensor.get_accel_data()
            g = sensor.get_gyro_data()
        except Exception as e:
            print(f"[IMU read error] {e}")
            time.sleep(1.0 / SAMPLE_RATE)
            continue

        # Apply calibration only if bias values are significant
        if cal:
            bias_acc  = cal.get("acc_bias", {})
            bias_gyro = cal.get("gyro_bias", {})
            if any(abs(bias_acc.get(k, 0)) > 0.01 for k in ["x", "y", "z"]):
                for k in a: a[k] -= bias_acc.get(k, 0)
            if any(abs(bias_gyro.get(k, 0)) > 0.01 for k in ["x", "y", "z"]):
                for k in g: g[k] -= bias_gyro.get(k, 0)

        accel_buf.append(a)
        gyro_buf.append(g)
        time.sleep(1.0 / SAMPLE_RATE)

    if len(accel_buf) < 5:
        speaker.say("Hmm, I barely got any data from the sensor. Let's try that again.", block=True)
        lcd_write(lcd, "IMU read fail", "Try again")
        return "unknown"

    import pandas as pd
    feats     = extract_features(accel_buf, gyro_buf)
    feats_df  = pd.DataFrame([feats], columns=STROKE_FEATURES)
    proba     = clf.predict_proba(feats_df)[0]
    good_prob = proba[1]
    label     = "GOOD" if good_prob >= 0.5 else "BAD"
    score_pct = int(good_prob * 100)

    # ── Tilt / stability feedback from gyro ──────────────────────
    tilt_note = _tilt_feedback(gyro_buf)

    set_leds(pwm_green, pwm_red, good_prob)
    if label == "GOOD":
        beep_good(pwm_buzzer)
        if score_pct >= 90:
            base = f"Silky smooth! {score_pct} percent — that's shark-level."
        else:
            base = f"Nice stroke! {score_pct} percent. You're getting there."
        if tilt_note:
            base += f" One tip: watch the {tilt_note}."
        speaker.say(base, block=True)
    else:
        beep_bad(pwm_buzzer)
        if score_pct < 30:
            base = f"Yikes, {score_pct} percent. Did you sneeze mid-swing?"
        else:
            base = f"{score_pct} percent. Not bad, not great."
        if tilt_note:
            base += f" I noticed {tilt_note}."
        else:
            base += " Work on that follow-through!"
        speaker.say(base, block=True)

    dist_str = f"{dist:.0f}cm" if dist is not None else "N/A"
    lcd_write(lcd, f"{label} {score_pct}%", f"Ht:{dist_str} Press btn")
    print(f"[STROKE] {label} score={score_pct}% dist={dist_str} tilt={tilt_note}")
    return label


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image", type=str, default=None,
                        help="Path to a static image file — skips camera")
    parser.add_argument("--test-images-dir", type=str, default=None,
                        help="Directory of images to cycle through on each button press")
    args = parser.parse_args()

    print("PoolCue Vision Assist starting...")

    # LCD
    lcd = None
    if HAS_LCD:
        try:
            lcd = setup_lcd()
            lcd_write(lcd, "PoolCue Vision", "Starting...")
        except Exception as e:
            print(f"[WARN] LCD init failed: {e}")

    # GPIO
    pwm_green, pwm_red, pwm_buzzer = setup_gpio()

    # Speaker
    speaker = Speaker()
    speaker.say("Hey! I'm Cue, your personal pool shark in a box. Give me a sec to warm up!", block=True)

    # Stroke model
    clf = None
    cal = None
    if STROKE_MODEL_PATH.exists():
        try:
            clf = joblib.load(STROKE_MODEL_PATH)
            if HAS_IMU:
                cal = load_cal()
            print("[OK] Stroke model loaded")
        except Exception as e:
            print(f"[WARN] Stroke model load failed: {e}")
    else:
        print(f"[WARN] Stroke model not found at {STROKE_MODEL_PATH}")

    # Vision + LLM
    detector  = BallDetector(conf_threshold=CFG.get("detection_conf", 0.45), imgsz=CFG.get("imgsz", 320))
    pocket_map = PocketMap()
    advisor   = ShotAdvisor(model=CFG.get("llm_model", "gpt-4o-mini"))
    game      = GameState(speaker=speaker, mode=CFG.get("game_mode", "8ball"))

    # Camera or test image
    test_frame = None
    test_images = []   # cyclic list for --test-images-dir
    test_img_idx = 0
    cap = None
    if args.test_images_dir:
        import glob
        paths = sorted(glob.glob(f"{args.test_images_dir}/*.jpg") +
                       glob.glob(f"{args.test_images_dir}/*.png"))
        if not paths:
            print(f"[ERROR] No images found in {args.test_images_dir}")
            sys.exit(1)
        test_images = [cv2.imread(p) for p in paths]
        test_frame = test_images[0].copy()
        print(f"[TEST MODE] Cycling through {len(test_images)} images")
        frame = test_frame
    elif args.test_image:
        test_frame = cv2.imread(args.test_image)
        if test_frame is None:
            print(f"[ERROR] Could not load test image: {args.test_image}")
            sys.exit(1)
        print(f"[TEST MODE] Using static image: {args.test_image}")
        frame = test_frame
    else:
        cap = cv2.VideoCapture(CFG.get("camera_index", 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            speaker.say("Camera not found", block=True)
            lcd_write(lcd, "ERROR", "No camera")
            sys.exit(1)
        ret, frame = cap.read()

    # Pocket calibration
    if not pocket_map.is_calibrated():
        speaker.say("Click each pocket to calibrate", block=True)
        lcd_write(lcd, "CALIBRATE", "Click pockets")
        pocket_map.calibrate(frame)
        speaker.say("Calibration saved", block=True)

    pockets = pocket_map.pockets if pocket_map.is_calibrated() else pocket_map.estimate_from_frame(frame)

    print("Ready — press button for shot recommendation, press again to grade stroke.")
    lcd_write(lcd, "READY", "Press for shot")
    speaker.say("Alright, I'm locked in. Show me what you've got — press the button!", block=True)

    # State machine: "recommend" → "grade" → "recommend" → ...
    state = "recommend"
    btn_was_pressed = False
    last_rec = None

    # Grab a valid frame once at start for camera mode
    last_good_frame = frame

    try:
        while True:
            # Check button FIRST before potentially slow frame grab
            btn_now = button_pressed()
            edge = btn_now and not btn_was_pressed  # rising edge (press)

            # Grab frame — cycle through test images on each recommend press
            if test_images:
                frame = test_images[test_img_idx].copy()
            elif test_frame is not None:
                frame = test_frame.copy()
            else:
                ret, f = cap.read()
                if ret:
                    last_good_frame = f
                frame = last_good_frame

            if edge:
                if state == "recommend":
                    # ── Shot recommendation ──────────────────────────────────
                    print("[BUTTON] Getting shot recommendation...")
                    lcd_write(lcd, "Detecting...", "")

                    frame_h, frame_w = frame.shape[:2]
                    balls = detector.detect(frame)

                    # Re-seed game state on each new recommendation so cycling
                    # through test images doesn't accumulate false "sunk" events
                    if test_images or test_frame is not None:
                        game._seeded = False
                    game.update(balls)
                    print(f"[DETECT] Found: {list(balls.keys())}")

                    remaining = game.remaining_for_player()
                    # Build shootable set: all detected balls that are valid targets
                    # If cue ball not detected (common with test images), include all balls
                    # so the LLM still gets enough context to recommend a shot
                    has_cue = "cue" in balls
                    if has_cue:
                        shootable = {k: v for k, v in balls.items() if k in remaining or k == "cue"}
                    else:
                        # No cue detected — pass all balls; LLM will pick best target
                        shootable = {k: v for k, v in balls.items() if k in remaining}

                    # Advance to next test image AFTER detection
                    if test_images:
                        test_img_idx = (test_img_idx + 1) % len(test_images)

                    if len(shootable) == 0:
                        # No balls detected — skip this image, stay in recommend
                        speaker.say("I can't see any balls here. Reset the table and press again!", block=True)
                        lcd_write(lcd, "No balls found", "Reset table")
                    else:
                        diff_words = {1: "Easy", 2: "Pretty easy", 3: "Medium", 4: "Tricky", 5: "Tough one"}
                        lcd_write(lcd, "Asking Cue...", "")
                        try:
                            rec = advisor.recommend(
                                balls=shootable,
                                pockets=pockets,
                                game_state={
                                    "mode": game.mode,
                                    "player_type": game.current_player_type,
                                    "frame_w": frame_w,
                                    "frame_h": frame_h,
                                }
                            )
                            last_rec = rec
                            diff = rec.get("difficulty", 3)
                            spoken = rec.get("spoken", "I recommend your next shot.")
                            speaker.say(spoken, block=True)
                            ball_label   = rec.get("ball", "?")
                            pocket_label = rec.get("pocket", "?")[:8]
                            diff_label   = diff_words.get(diff, "?")
                            print(f"[REC] {spoken} | reason={rec.get('reason','')}")
                            # LCD updates immediately; prompt plays non-blocking so no extra wait
                            lcd_write(lcd,
                                f"B{ball_label}->{pocket_label}",
                                f"{diff_label} {diff}/5 Swing!"
                            )
                            speaker.say("Take your shot, then press the button to grade it!")
                            state = "grade"
                            print("[STATE] → grade")
                        except Exception as e:
                            print(f"[LLM error] {e}")
                            lcd_write(lcd, "LLM error", "Press to retry")
                            speaker.say("My brain just glitched. Press the button to try again!", block=True)
                            # Stay in recommend state

                elif state == "grade":
                    # ── Stroke grading ───────────────────────────────────────
                    print("[BUTTON] Grading stroke...")
                    lcd_write(lcd, "Grading...", "Hold still!")
                    if clf is not None:
                        try:
                            label = capture_and_grade_stroke(
                                speaker, lcd, clf, cal,
                                pwm_green, pwm_red, pwm_buzzer
                            )
                            if last_rec:
                                game.log_shot(last_rec, label, aim_error_px=0)
                        except Exception as e:
                            print(f"[STROKE error] {e}")
                            speaker.say("Something went wrong grading your stroke. Let's go again!", block=True)
                            lcd_write(lcd, "Grade error", "")
                    else:
                        speaker.say("Stroke model not loaded — skipping grade.", block=True)
                        lcd_write(lcd, "No model", "Skipping")

                    # Reset game state for next image/cycle
                    game._seeded = False
                    last_rec = None
                    state = "recommend"
                    time.sleep(0.5)  # brief pause so button release is clean
                    lcd_write(lcd, "READY", "Press for shot")
                    speaker.say("Press the button for your next shot recommendation!")
                    print("[STATE] → recommend")

            btn_was_pressed = btn_now

            if CFG.get("show_display", False):
                balls = detector.detect(frame)
                annotated = detector.draw(frame, balls)
                cv2.imshow("PoolCue Vision Assist", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if HAS_GPIO:
            GPIO.cleanup()
        lcd_write(lcd, "Goodbye!", "")


if __name__ == "__main__":
    main()
