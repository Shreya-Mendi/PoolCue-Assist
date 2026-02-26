# realtime.py
# Real-time stroke classification with PWM LED gradient feedback.
#
# Green LED brightens  → stroke is more like "good"
# Red LED brightens    → stroke is more like "bad"
# Both LEDs fade as motion settles
#
# Usage:
#   python3 realtime.py

import time
import math
import joblib
import numpy as np
import RPi.GPIO as GPIO
from imu_helpers import sensor, load_cal

# ── Pin config ─────────────────────────────────────────────────────────────
LED_GREEN = 17
LED_RED   = 27
PWM_FREQ  = 1000   # Hz — high enough to avoid visible flicker

# ── Sampling config ─────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 100
WINDOW_S        = 1.0          # rolling window length in seconds
STEP_S          = 0.2          # how often to re-evaluate (rolling step)
TRIGGER_ACCEL   = 2.0          # accel magnitude to start a stroke window

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_FILE = "stroke_model.pkl"
FEATURES   = ["peak_accel", "mean_gyro_y", "var_gyro_y", "mean_gyro_z", "var_gyro_z", "duration"]


def extract_features(accel_list, gyro_list):
    """Must match collect_data.py exactly."""
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


def score_to_pwm(good_prob):
    """
    Convert good-stroke probability (0.0–1.0) into LED duty cycles.
    Returns (green_duty, red_duty) each in range 0–100.
    """
    green_duty = int(good_prob * 100)
    red_duty   = int((1.0 - good_prob) * 100)
    return green_duty, red_duty


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_GREEN, GPIO.OUT)
    GPIO.setup(LED_RED,   GPIO.OUT)
    pwm_green = GPIO.PWM(LED_GREEN, PWM_FREQ)
    pwm_red   = GPIO.PWM(LED_RED,   PWM_FREQ)
    pwm_green.start(0)
    pwm_red.start(0)
    return pwm_green, pwm_red


def set_leds(pwm_green, pwm_red, good_prob):
    green_duty, red_duty = score_to_pwm(good_prob)
    pwm_green.ChangeDutyCycle(green_duty)
    pwm_red.ChangeDutyCycle(red_duty)


def main():
    print("Loading model...")
    clf = joblib.load(MODEL_FILE)
    cal = load_cal()
    if cal is None:
        print("WARNING: No calibration file. Run calibrate_imu.py first.")
    else:
        print("Calibration loaded.")

    pwm_green, pwm_red = setup_gpio()
    print(f"Ready. Waiting for stroke (trigger accel > {TRIGGER_ACCEL})...")
    print("Press Ctrl+C to quit.\n")

    # Rolling buffer — keeps last WINDOW_S seconds of data
    accel_buf = []
    gyro_buf  = []
    max_buf   = int(WINDOW_S * SAMPLE_RATE_HZ)

    last_eval_time = time.time()

    try:
        while True:
            # ── Read one sample ──────────────────────────────────────────
            a = sensor.get_accel_data()
            g = sensor.get_gyro_data()
            if cal:
                for k in a: a[k] -= cal["acc_bias"].get(k, 0)
                for k in g: g[k] -= cal["gyro_bias"].get(k, 0)

            accel_buf.append(a)
            gyro_buf.append(g)

            # Keep buffer at rolling window size
            if len(accel_buf) > max_buf:
                accel_buf.pop(0)
                gyro_buf.pop(0)

            # ── Check for motion trigger ─────────────────────────────────
            current_mag = math.sqrt(a["x"]**2 + a["y"]**2 + a["z"]**2)
            in_motion   = current_mag > TRIGGER_ACCEL

            # ── Re-evaluate at STEP_S intervals during/after motion ──────
            now = time.time()
            if in_motion or (now - last_eval_time < 1.0):
                if now - last_eval_time >= STEP_S and len(accel_buf) >= 10:
                    feats     = extract_features(accel_buf, gyro_buf)
                    import pandas as pd
                    feats_df  = pd.DataFrame([feats], columns=["peak_accel","mean_gyro_y","var_gyro_y","mean_gyro_z","var_gyro_z","duration"])
                    proba     = clf.predict_proba(feats_df)[0]
                    good_prob = proba[1]          # probability of class 1 (good)
                    set_leds(pwm_green, pwm_red, good_prob)
                    last_eval_time = now
                    label = "GOOD" if good_prob >= 0.5 else "BAD "
                    print(f"\r  {label}  score={good_prob*100:5.1f}%  "
                          f"green={int(good_prob*100):3d}%  red={int((1-good_prob)*100):3d}%   ",
                          end="", flush=True)
            else:
                # No motion — fade both LEDs off gently
                set_leds(pwm_green, pwm_red, 0.5)   # neutral: both dim

            time.sleep(1.0 / SAMPLE_RATE_HZ)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pwm_green.stop()
        pwm_red.stop()
        GPIO.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
