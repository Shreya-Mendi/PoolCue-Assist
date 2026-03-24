# realtime.py
# Real-time stroke classification with:
#   - Button-triggered start/stop (self-contained, no laptop needed)
#   - LCD display showing score, result, stroke count
#   - PWM LED feedback (green = good, red = bad)
#   - Buzzer audio feedback
#   - HC-SR04 ultrasonic sensor for cue height measurement
#
# Wiring:
#   Button:     GPIO 18 → button → GND  (internal pull-up)
#   Buzzer:     GPIO 22 → buzzer+ → GND
#   HC-SR04:    TRIG → GPIO 23, ECHO → GPIO 24, VCC → 5V, GND → GND
#   LCD (I2C):  SDA → GPIO 2, SCL → GPIO 3, VCC → 5V, GND → GND
#   Green LED:  GPIO 17 → 330Ω → LED → GND
#   Red LED:    GPIO 27 → 330Ω → LED → GND
#
# Usage:
#   python3 realtime.py

import time
import math
import sys
import joblib
import numpy as np
import pandas as pd
import RPi.GPIO as GPIO
from pathlib import Path

# Allow imports from src/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_helpers import sensor, load_cal

ROOT = Path(__file__).resolve().parent.parent

# ── Try to import LCD library (I2C 16x2) ────────────────────────────────────
try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)
    LCD_AVAILABLE = True
except Exception:
    try:
        from RPLCD.i2c import CharLCD
        lcd = CharLCD('PCF8574', 0x3F, cols=16, rows=2)  # alternate I2C address
        LCD_AVAILABLE = True
    except Exception:
        LCD_AVAILABLE = False
        print("WARNING: LCD not found. Continuing without LCD.")

# ── Pin config ───────────────────────────────────────────────────────────────
LED_GREEN   = 17
LED_RED     = 27
BUTTON_PIN  = 18
BUZZER_PIN  = 22
TRIG_PIN    = 23
ECHO_PIN    = 24
PWM_FREQ    = 1000

# ── Sampling config ──────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 100
WINDOW_S        = 1.0
TRIGGER_ACCEL   = 2.0

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_FILE = ROOT / "models" / "stroke_model.pkl"
FEATURES   = ["peak_accel", "mean_gyro_y", "var_gyro_y", "mean_gyro_z", "var_gyro_z", "duration"]

# ── Cue height thresholds (cm) ───────────────────────────────────────────────
CUE_HEIGHT_MIN = 5    # too low
CUE_HEIGHT_MAX = 25   # too high
CUE_HEIGHT_OK_MIN = 8
CUE_HEIGHT_OK_MAX = 18


def lcd_print(line1, line2=""):
    """Write two lines to LCD, clearing first."""
    if not LCD_AVAILABLE:
        print(f"LCD: {line1} | {line2}")
        return
    try:
        lcd.clear()
        lcd.write_string(line1[:16])
        lcd.crlf()
        lcd.write_string(line2[:16])
    except Exception:
        pass


def beep_good(pwm_buzzer):
    """Two short high beeps = good stroke."""
    for _ in range(2):
        pwm_buzzer.ChangeDutyCycle(70)
        time.sleep(0.1)
        pwm_buzzer.ChangeDutyCycle(0)
        time.sleep(0.08)


def beep_bad(pwm_buzzer):
    """One long low buzz = bad stroke."""
    pwm_buzzer.ChangeDutyCycle(30)
    time.sleep(0.4)
    pwm_buzzer.ChangeDutyCycle(0)


def beep_ready(pwm_buzzer):
    """Startup confirmation beep."""
    pwm_buzzer.ChangeDutyCycle(50)
    time.sleep(0.15)
    pwm_buzzer.ChangeDutyCycle(0)


def measure_distance_cm():
    """Read HC-SR04 distance in cm. Returns None on timeout."""
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.01)
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN) == 0:
        if time.time() > timeout:
            return None
    pulse_start = time.time()

    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN) == 1:
        if time.time() > timeout:
            return None
    pulse_end = time.time()

    duration = pulse_end - pulse_start
    distance = (duration * 34300) / 2
    return round(distance, 1)


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


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_GREEN,  GPIO.OUT)
    GPIO.setup(LED_RED,    GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(TRIG_PIN,   GPIO.OUT)
    GPIO.setup(ECHO_PIN,   GPIO.IN)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    pwm_green  = GPIO.PWM(LED_GREEN,   PWM_FREQ)
    pwm_red    = GPIO.PWM(LED_RED,     PWM_FREQ)
    pwm_buzzer = GPIO.PWM(BUZZER_PIN,  2000)  # 2kHz tone

    pwm_green.start(0)
    pwm_red.start(0)
    pwm_buzzer.start(0)

    return pwm_green, pwm_red, pwm_buzzer


def set_leds(pwm_green, pwm_red, good_prob):
    pwm_green.ChangeDutyCycle(int(good_prob * 100))
    pwm_red.ChangeDutyCycle(int((1.0 - good_prob) * 100))


def wait_for_button_press():
    """Block until button is pressed (active LOW)."""
    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.05)


def main():
    print("Loading model...")
    clf = joblib.load(MODEL_FILE)
    cal = load_cal()
    if cal is None:
        print("WARNING: No calibration file. Run calibrate_imu.py first.")
    else:
        print("Calibration loaded.")

    pwm_green, pwm_red, pwm_buzzer = setup_gpio()

    # Startup sequence — wait for first button press to begin
    lcd_print("Pool Cue Assist", "Press btn start")
    beep_ready(pwm_buzzer)
    wait_for_button_press()
    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
        time.sleep(0.05)
    lcd_print("Ready!", "")

    stroke_count = 0
    good_count   = 0

    print("\nReady. Press button to capture a stroke. Ctrl+C to quit.\n")

    try:
        while True:
            # ── Wait for button press ────────────────────────────────────────
            wait_for_button_press()

            # ── Measure cue height before stroke ────────────────────────────
            dist = measure_distance_cm()
            height_warning = ""
            if dist is not None:
                if dist < CUE_HEIGHT_OK_MIN:
                    height_warning = f"Too low {dist}cm"
                elif dist > CUE_HEIGHT_OK_MAX:
                    height_warning = f"Too high {dist}cm"
                else:
                    height_warning = f"Ht:{dist}cm OK"

            lcd_print("Recording...", height_warning)
            set_leds(pwm_green, pwm_red, 0.5)  # both dim while recording

            # ── Collect stroke window ────────────────────────────────────────
            accel_buf, gyro_buf = [], []
            t0 = time.time()
            while time.time() - t0 < WINDOW_S:
                a = sensor.get_accel_data()
                g = sensor.get_gyro_data()
                if cal:
                    for k in a: a[k] -= cal["acc_bias"].get(k, 0)
                    for k in g: g[k] -= cal["gyro_bias"].get(k, 0)
                accel_buf.append(a)
                gyro_buf.append(g)
                time.sleep(1.0 / SAMPLE_RATE_HZ)

            # ── Classify ─────────────────────────────────────────────────────
            feats    = extract_features(accel_buf, gyro_buf)
            feats_df = pd.DataFrame([feats], columns=FEATURES)
            proba    = clf.predict_proba(feats_df)[0]
            good_prob = proba[1]

            stroke_count += 1
            if good_prob >= 0.5:
                good_count += 1
                label = "GOOD"
            else:
                label = "BAD"
                beep_bad(pwm_buzzer)

            set_leds(pwm_green, pwm_red, good_prob)

            # ── LCD output ───────────────────────────────────────────────────
            score_pct = int(good_prob * 100)
            lcd_print(
                f"{label} Sc:{score_pct}%",
                f"Good:{good_count}/{stroke_count}"
            )

            # ── Console output ───────────────────────────────────────────────
            dist_str = f"{dist}cm" if dist is not None else "N/A"
            print(f"Stroke {stroke_count:3d}: {label}  score={score_pct}%  "
                  f"height={dist_str}  good={good_count}/{stroke_count}")

            # Wait for button release before accepting next press
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        lcd_print("Goodbye!", "")
        pwm_green.stop()
        pwm_red.stop()
        pwm_buzzer.stop()
        GPIO.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
