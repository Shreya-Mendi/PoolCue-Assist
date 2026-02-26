# calibrate_imu.py
import time
import json
from mpu6050 import mpu6050

s = mpu6050(0x68)
N = 500

print("Keep the cue perfectly still on a flat surface.")
print("Calibration starts in 3 seconds...")
time.sleep(3)

acc_sum  = {'x': 0, 'y': 0, 'z': 0}
gyro_sum = {'x': 0, 'y': 0, 'z': 0}

for i in range(N):
    a = s.get_accel_data()
    g = s.get_gyro_data()
    for k in a: acc_sum[k]  += a[k]
    for k in g: gyro_sum[k] += g[k]
    time.sleep(0.01)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N} samples collected...")

acc_bias  = {k: acc_sum[k]  / N for k in acc_sum}
gyro_bias = {k: gyro_sum[k] / N for k in gyro_sum}

cal = {'acc_bias': acc_bias, 'gyro_bias': gyro_bias}

with open('imu_calibration.json', 'w') as f:
    json.dump(cal, f, indent=2)

print("\nCalibration complete. Saved to imu_calibration.json")
print(f"  Accel bias : {acc_bias}")
print(f"  Gyro bias  : {gyro_bias}")
