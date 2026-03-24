# imu_check.py
from mpu6050 import mpu6050
import time
s = mpu6050(0x68)
for i in range(20):
    print("accel:", s.get_accel_data(), "gyro:", s.get_gyro_data())
    time.sleep(0.2)