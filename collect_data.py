# collect_data.py
# Collect labeled stroke data for training the ML model.
#
# Usage:
#   python3 collect_data.py --label 1 --count 20   # good strokes
#   python3 collect_data.py --label 0 --count 20   # bad strokes

import time
import math
import csv
import os
import argparse
import numpy as np
from imu_helpers import sensor, load_cal

SAMPLE_RATE_HZ = 100   # samples per second
WINDOW_S       = 1.0   # how long each stroke window is
CSV_FILE       = "stroke_data.csv"
CSV_HEADER     = ["peak_accel", "mean_gyro_y", "var_gyro_y", "mean_gyro_z", "var_gyro_z", "duration", "label"]


def record_window(cal):
    """Record WINDOW_S seconds of IMU data and return (accel_list, gyro_list)."""
    accel_list, gyro_list = [], []
    t0 = time.time()
    while time.time() - t0 < WINDOW_S:
        a = sensor.get_accel_data()
        g = sensor.get_gyro_data()
        if cal:
            for k in a: a[k] -= cal["acc_bias"].get(k, 0)
            for k in g: g[k] -= cal["gyro_bias"].get(k, 0)
        accel_list.append(a)
        gyro_list.append(g)
        time.sleep(1.0 / SAMPLE_RATE_HZ)
    return accel_list, gyro_list


def extract_features(accel_list, gyro_list):
    """
    Compute features from one stroke window.
    Must match exactly what realtime.py uses.
    """
    accel_mag = [math.sqrt(a["x"]**2 + a["y"]**2 + a["z"]**2) for a in accel_list]
    gyro_y    = [g["y"] for g in gyro_list]
    gyro_z    = [g["z"] for g in gyro_list]

    peak_accel  = max(accel_mag)
    mean_gyro_y = float(np.mean(gyro_y))
    var_gyro_y  = float(np.var(gyro_y))
    mean_gyro_z = float(np.mean(gyro_z))
    var_gyro_z  = float(np.var(gyro_z))
    duration    = len(accel_list)

    return [peak_accel, mean_gyro_y, var_gyro_y, mean_gyro_z, var_gyro_z, duration]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, choices=[0, 1], required=True,
                        help="1 = good stroke, 0 = bad stroke")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of samples to collect")
    args = parser.parse_args()

    cal = load_cal()
    if cal is None:
        print("WARNING: No calibration file found. Run calibrate_imu.py first.")
    else:
        print("Calibration loaded.")

    # Create CSV with header if it doesn't exist yet
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"Created {CSV_FILE}")
    else:
        print(f"Appending to existing {CSV_FILE}")

    label_name = "GOOD" if args.label == 1 else "BAD"
    print(f"\nCollecting {args.count} {label_name} stroke samples.")
    print("Press ENTER when ready for each stroke, then perform the motion.\n")

    for i in range(args.count):
        input(f"  [{i+1}/{args.count}] Press ENTER, then take a {label_name} stroke...")
        accel_list, gyro_list = record_window(cal)
        feats = extract_features(accel_list, gyro_list)
        row = feats + [args.label]
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"    Saved: peak_accel={feats[0]:.3f}  var_gyro_y={feats[2]:.4f}  var_gyro_z={feats[4]:.4f}")

    print(f"\nDone. {args.count} samples saved to {CSV_FILE}.")


if __name__ == "__main__":
    main()
