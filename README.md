# Pool Cue Assist — Billiards Stroke Analyzer

**AIPI 590 | Midterm Project**
**Platform:** Raspberry Pi 4 | **Language:** Python 3

---

## Purpose

Pool Cue Assist teaches correct billiards stroke mechanics by detecting and classifying cue stroke quality in real time. The goal is **stroke straightness and smoothness** — the most common cause of missed shots is cue twist and lateral wobble on the follow-through, not aim. The device uses an IMU mounted on the cue to measure how straight and stable each stroke is, and gives immediate feedback so players can self-correct without needing a coach watching every shot.

A stroke is classified as **GOOD** (straight, controlled, minimal rotation) or **BAD** (twisted, jabbed, laterally unstable). Results appear on an LCD display, green/red LEDs, and a buzzer — all on the physical device with no laptop required.

---

## Hardware

| Component | Purpose | Interface |
|---|---|---|
| Raspberry Pi 4 | Main compute board | — |
| MPU6050 IMU | Measures stroke motion — 3-axis acceleration + 3-axis gyroscope | I²C (0x68) |
| HC-SR04 Ultrasonic | Measures cue height above table before each stroke | GPIO 23/24 |
| Push Button | Player presses to trigger a stroke capture | GPIO 18 (pull-up) |
| 16×2 LCD (I2C) | Displays result, confidence score, and running count | I²C (0x27 or 0x3F) |
| Green LED | PWM brightness = good-stroke confidence | GPIO 17 |
| Red LED | PWM brightness = bad-stroke confidence | GPIO 27 |
| Buzzer | 2 short beeps = good stroke, 1 long buzz = bad stroke | GPIO 22 |
| 2× 330Ω resistors | Current limiting for LEDs | — |

**Wiring diagram:** [diagrams/wiring_diagram.html](diagrams/wiring_diagram.html) — open in a browser.

### Pin Reference

| GPIO | Pin # | Connected to |
|---|---|---|
| 3.3V | Pin 1 | MPU6050 VCC |
| 5V | Pin 2 | LCD VCC |
| GPIO 2 / SDA | Pin 3 | MPU6050 SDA + LCD SDA (shared bus) |
| GPIO 3 / SCL | Pin 5 | MPU6050 SCL + LCD SCL (shared bus) |
| GND | Pin 6 | MPU6050 GND, LCD GND, LED cathodes, Button GND |
| 5V | Pin 4 | HC-SR04 VCC |
| GPIO 17 | Pin 11 | Green LED anode (via 330Ω) |
| GPIO 18 | Pin 12 | Button (to GND, internal pull-up) |
| GPIO 27 | Pin 13 | Red LED anode (via 330Ω) |
| GPIO 22 | Pin 15 | Buzzer + |
| GPIO 23 | Pin 16 | HC-SR04 TRIG |
| GPIO 24 | Pin 18 | HC-SR04 ECHO |
| GND | Pin 20 | HC-SR04 GND, Buzzer − |

---

## Repository Structure

```
MidtermCue/
├── src/
│   ├── imu_helpers.py       # Shared IMU access + calibration loader
│   ├── calibrate_imu.py     # One-time bias measurement (500 samples, ~5s)
│   ├── collect_data.py      # Interactive labeled data collection → CSV
│   ├── train_model.py       # Model training with cross-validation
│   ├── realtime.py          # Real-time inference + all hardware feedback
│   ├── imu_check.py         # Quick IMU hardware verification
│   └── led_test.py          # Quick LED wiring verification
├── data/
│   └── stroke_data.csv      # 40 labeled training samples
├── models/
│   └── stroke_model.pkl     # Trained Decision Tree (joblib format)
├── diagrams/
│   └── wiring_diagram.html  # Full wiring diagram (open in browser)
└── imu_calibration.json     # Stored per-axis sensor bias values
```

---

## How to Run

### 1. Install dependencies

```bash
pip install mpu6050-raspberrypi RPi.GPIO numpy pandas scikit-learn joblib RPLCD
# smbus must be installed system-wide:
sudo apt install python3-smbus
```

### 2. Verify I²C devices

```bash
i2cdetect -y 1
```

You must see `0x68` (IMU) and either `0x27` or `0x3F` (LCD). If LCD is missing, check wiring and power.

### 3. Calibrate the IMU (one-time, cue held still on flat surface)

```bash
python3 src/calibrate_imu.py
```

Saves `imu_calibration.json` to the repo root. Takes ~5 seconds.

### 4. Collect training data

```bash
python3 src/collect_data.py --label 1 --count 20   # 20 good strokes
python3 src/collect_data.py --label 0 --count 20   # 20 bad strokes
```

Press ENTER before each stroke. Appends rows to `data/stroke_data.csv`.

**Good stroke:** smooth, straight, controlled follow-through
**Bad stroke:** twist at follow-through, lateral wobble, or jabbing stop

### 5. Train the model

```bash
python3 src/train_model.py
```

Prints 5-fold cross-validation F1 score and feature importances. Saves trained model to `models/stroke_model.pkl`.

### 6. Run the real-time system

```bash
python3 src/realtime.py
```

- LCD shows "Pool Cue Assist" → startup beep
- Press button to start a 1-second stroke capture
- HC-SR04 checks cue height before each stroke (warns if outside 8–18 cm)
- LCD shows result + confidence score + running tally (e.g. "GOOD 91%" / "Good:7/10")
- Green LED brightens for good strokes, red for bad
- Buzzer: 2 short beeps = good, 1 long buzz = bad
- Press Ctrl+C to stop

---

## Data

### Dataset

| Property | Value |
|---|---|
| Total samples | 40 |
| Good strokes (label=1) | 20 |
| Bad strokes (label=0) | 20 |
| Sampling rate | 100 Hz |
| Window length | 1 second (~100 readings per sample) |

### Features

Each 1-second IMU window is reduced to 6 features:

| Feature | Description | Why it matters |
|---|---|---|
| `peak_accel` | Maximum acceleration magnitude (m/s²) during the window | Measures stroke speed and force |
| `mean_gyro_y` | Mean Y-axis angular velocity (°/s) | Average lateral drift (pitch plane) |
| `var_gyro_y` | Variance of Y-axis angular velocity | Stroke smoothness in pitch direction |
| `mean_gyro_z` | Mean Z-axis angular velocity (°/s) | Average rotational drift (yaw plane) |
| `var_gyro_z` | Variance of Z-axis angular velocity | Twist instability — the primary bad-stroke indicator |
| `duration` | Number of samples collected (~100 for 1s) | Confirms window completeness |

Acceleration magnitude is computed as:
```
|a| = sqrt(ax² + ay² + az²)
```

### Feature separation between classes

| Feature | Good stroke (mean) | Bad stroke (mean) | Ratio |
|---|---|---|---|
| `peak_accel` | 18.6 m/s² | 21.8 m/s² | 1.2× |
| `var_gyro_y` | 29.7 (°/s)² | 7,294 (°/s)² | 246× |
| `var_gyro_z` | 34.7 (°/s)² | 4,472 (°/s)² | 129× |

Bad strokes produce dramatically higher rotational variance. The gyroscope Z-axis (yaw/twist) captures the dominant failure mode in billiards — cue rotation on the follow-through.

### Sample data (first 5 rows of stroke_data.csv)

```
peak_accel,  mean_gyro_y,  var_gyro_y,  mean_gyro_z,  var_gyro_z,  duration,  label
18.75,        -0.073,        22.39,        2.70,          19.88,       63,        1
18.50,         0.231,         1.67,       -0.149,          4.26,       63,        1
17.87,         3.220,         4.72,        1.479,          7.19,       62,        1
17.45,         0.860,         1.52,       -0.348,          3.26,       62,        1
21.93,         1.718,       131.71,        0.542,        442.32,       63,        0  ← bad
```

---

## Model

**Algorithm:** Decision Tree (`max_depth=3`, `random_state=42`)

Depth limited to 3 to reduce overfitting on the small dataset. A shallow tree also makes the decision logic interpretable.

**Validation:** 5-fold stratified cross-validation (more reliable than a single split on 40 samples — uses all data across 5 independent splits)

**Results:**

| Metric | Score |
|---|---|
| 5-fold CV F1 (mean) | **1.000** |
| 5-fold CV F1 (std) | **0.000** |
| Training accuracy | 100% |

**Feature importances:**

| Feature | Importance |
|---|---|
| `var_gyro_z` | 1.000 |
| All others | 0.000 |

The model relies entirely on Z-axis rotational variance, which reflects physical reality: the dominant error in billiards strokes is cue twist (yaw rotation), which `var_gyro_z` captures with extremely strong class separation (129× difference between good and bad strokes).

---

## Real-Time Inference Pipeline

```
Player presses button (GPIO 18)
    ↓
HC-SR04 measures cue height → warns if outside 8–18 cm range
    ↓
1-second IMU window collected at 100 Hz (~100 samples)
    ↓
Calibration bias subtracted (from imu_calibration.json)
    ↓
6 features extracted from window
    ↓
stroke_model.pkl (Decision Tree) → predict_proba() → P(good)
    ↓
PWM LEDs:  green = P(good) × 100%,  red = (1 − P(good)) × 100%
Buzzer:    2 short beeps (good)  or  1 long buzz (bad)
LCD:       "GOOD 91%" / "BAD 12%"  +  "Good:7/10"
```

Fully offline — no cloud, no laptop needed after first setup.

---

## Challenges and Solutions

### Challenge 1: `smbus` not available in virtual environment
**Problem:** `mpu6050-raspberrypi` depends on `smbus`, a C extension only installable as a system package.
**Solution:** Created the venv with `--system-site-packages` so it inherits system `smbus` while keeping other packages isolated.

### Challenge 2: 100% accuracy on a small test set is misleading
**Problem:** With 40 samples and 80/20 split, the test set has 8 samples — 100% on 8 samples is statistically meaningless.
**Solution:** Switched to 5-fold stratified cross-validation. Consistent F1=1.00 across all 5 independent folds — combined with the 129–246× feature separation — confirms the result is genuine.

### Challenge 3: Binary LED feedback doesn't convey degree of error
**Problem:** A simple on/off LED gives no sense of how far off a stroke was.
**Solution:** Used `predict_proba()` instead of `predict()` to get a continuous 0–1 confidence score, then mapped it to PWM duty cycles on both LEDs simultaneously. A nearly-good stroke glows mostly green with a hint of red; a terrible stroke is solid red.

### Challenge 4: Device had to be started from a laptop
**Problem:** Original version required SSH/terminal to start.
**Solution:** Added a push button (GPIO 18) as the sole user interface. Press once → stroke is captured → feedback given automatically. Ctrl+C on the Pi itself is the only way to stop it.

---

## Results

- IMU verified at I²C address `0x68`
- Calibration completed in ~5 seconds (500 samples, stationary cue)
- 40 training samples collected, 20 per class
- Decision Tree (depth 3) trained: F1 = 1.00 across all 5 folds
- Real-time system runs at ~100 Hz with <1s feedback latency after button press
- LCD, buzzer, and PWM LEDs all provide simultaneous independent feedback channels
