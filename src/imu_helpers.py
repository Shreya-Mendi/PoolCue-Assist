# imu_helpers.py
import math
import json
from pathlib import Path
from mpu6050 import mpu6050

sensor = mpu6050(0x68)

_ROOT = Path(__file__).resolve().parent.parent

def load_cal(fname=None):
    path = Path(fname) if fname else _ROOT / 'imu_calibration.json'
    try:
        return json.load(open(path))
    except:
        return None

def mag(a):
    return math.sqrt(a['x']**2 + a['y']**2 + a['z']**2)

def read_once(apply_cal=True, cal=None):
    a = sensor.get_accel_data()
    g = sensor.get_gyro_data()
    if apply_cal and cal:
        for k in a: a[k] -= cal['acc_bias'].get(k, 0)
        for k in g: g[k] -= cal['gyro_bias'].get(k, 0)
    return a, g
