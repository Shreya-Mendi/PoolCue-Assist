# Billiards Stroke Analyzer — Midterm Project Report
**AIPI 590 | Week 6 Midterm**
**Platform:** Raspberry Pi 4 | **Language:** Python 3

---

## 1. Project Overview

This project implements a real-time assistive device for billiards players that uses an inertial measurement unit (IMU) to classify pool cue strokes as **good** (smooth, straight) or **bad** (unstable, twisted) and provides immediate visual feedback through two LEDs. The system is fully offline — no cloud services are used at any stage.

The core pipeline is:

```
Microphone (IMU) → Feature Extraction → ML Classifier → PWM LED Feedback
```

The device satisfies all four rubric pillars:

| Pillar | Implementation |
|---|---|
| **Sensor** | MPU6050 IMU (accelerometer + gyroscope) via I²C |
| **Control Loop** | Continuous rolling-window sampling at 100 Hz |
| **Machine Learning** | Decision Tree trained on real stroke data |
| **Actuation** | PWM-driven dual LED (green = good, red = bad) |

---

## 2. Hardware

| Component | Detail |
|---|---|
| Raspberry Pi 4 | Main compute board |
| MPU6050 IMU | 3-axis accelerometer + 3-axis gyroscope, I²C address 0x68 |
| Green LED | GPIO 17, PWM-driven |
| Red LED | GPIO 27, PWM-driven |
| 2× 330Ω resistors | Current limiting for LEDs |

**Wiring:**
- MPU6050 VCC → 3.3V (Pin 1)
- MPU6050 GND → GND (Pin 6)
- MPU6050 SDA → GPIO 2 / Pin 3
- MPU6050 SCL → GPIO 3 / Pin 5
- GPIO 17 → 330Ω → Green LED anode → GND
- GPIO 27 → 330Ω → Red LED anode → GND

**Interactive wiring diagram:** [wiring_diagram.html](wiring_diagram.html) — open in a browser; hover/click each component to highlight its connections.

---

## 3. Software Architecture

### 3.1 File Structure

```
MidtermCue/
├── imu_helpers.py       # Shared IMU access + calibration loader
├── calibrate_imu.py     # One-time bias measurement (500 samples)
├── collect_data.py      # Interactive labeled data collection → CSV
├── train_model.py       # Model training with cross-validation
├── realtime.py          # Real-time inference + PWM LED control
├── imu_calibration.json # Stored sensor bias values
├── stroke_data.csv      # 40 labeled training samples
└── stroke_model.pkl     # Trained Decision Tree (joblib format)
```

### 3.2 Dependencies

```
mpu6050-raspberrypi    # MPU6050 Python driver
RPi.GPIO               # GPIO + PWM control
numpy                  # Feature computation
pandas                 # CSV loading for training
scikit-learn           # DecisionTreeClassifier, cross-validation
joblib                 # Model serialization
smbus (system)         # I²C backend (python3-smbus via apt)
```

The project uses a Python virtual environment (`.venv`) created with `--system-site-packages` to access the system-installed `smbus` package required by the MPU6050 driver.

---

## 4. Methodology

### 4.1 IMU Calibration

Before data collection, a one-time calibration procedure was run with the cue held stationary. 500 samples were averaged to compute per-axis bias offsets for both the accelerometer and gyroscope:

| Axis | Accel Bias (m/s²) | Gyro Bias (°/s) |
|---|---|---|
| X | -2.002 | -0.866 |
| Y | -2.695 | +0.750 |
| Z | -9.282 | +0.007 |

These biases are subtracted from every subsequent reading during both data collection and real-time inference, ensuring consistent measurements regardless of the cue's resting orientation.

### 4.2 Feature Extraction

Each stroke is captured as a 1-second window sampled at 100 Hz (≈100 readings). Five features are extracted per window:

| Feature | Description | Why it matters |
|---|---|---|
| `peak_accel` | Maximum acceleration magnitude (m/s²) | Measures stroke force/speed |
| `mean_gyro_y` | Mean Y-axis angular velocity (°/s) | Captures average lateral drift |
| `var_gyro_y` | Variance of Y-axis angular velocity | Measures stroke smoothness (pitch) |
| `mean_gyro_z` | Mean Z-axis angular velocity (°/s) | Captures average rotational drift |
| `var_gyro_z` | Variance of Z-axis angular velocity | Measures twist/rotation instability |

Acceleration magnitude is computed as:

```
|a| = sqrt(ax² + ay² + az²)
```

### 4.3 Data Collection

40 labeled stroke samples were collected interactively using `collect_data.py`:

- **20 good strokes (label=1):** Smooth, straight, controlled strokes
- **20 bad strokes (label=0):** Intentionally flawed strokes including twisting at follow-through, jabbing and stopping abruptly, and lateral wobbling

**Observed feature separation between classes:**

| Feature | Good (mean) | Bad (mean) | Ratio |
|---|---|---|---|
| `peak_accel` | 18.64 | 21.76 | 1.2× |
| `var_gyro_y` | 29.67 | 7,293.66 | 246× |
| `var_gyro_z` | 34.68 | 4,471.78 | 129× |

The rotational variance features show extremely strong class separation — bad strokes produce dramatically more angular instability than good strokes.

### 4.4 Model Training

A **Decision Tree Classifier** (`max_depth=3`) was trained using scikit-learn. Depth was limited to 3 to reduce overfitting risk on the small dataset.

**Validation method:** 5-fold stratified cross-validation (more reliable than a single train/test split on 40 samples).

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

The model relies entirely on Z-axis rotational variance. This reflects the physical reality: bad strokes in billiards are predominantly caused by cue twist (yaw rotation), which `var_gyro_z` captures cleanly.

### 4.5 Real-Time Inference

The `realtime.py` control loop runs continuously at 100 Hz:

1. **Read** calibrated IMU sample
2. **Buffer** last 1 second of readings (rolling window, max 100 samples)
3. **Trigger** — if current acceleration magnitude exceeds threshold (`TRIGGER_ACCEL = 2.0 m/s²`), motion is detected
4. **Evaluate** — every 0.2 seconds during/after motion, extract features and call `clf.predict_proba()`
5. **Actuate** — set PWM duty cycles on both LEDs proportional to the good-stroke probability score

The PWM gradient feedback:

```python
green_duty = int(good_prob * 100)   # 0–100%
red_duty   = int((1 - good_prob) * 100)
```

This means as the stroke quality continuously improves (e.g. the user corrects their form mid-sequence), the green LED brightens and the red LED dims in real time — not just a binary on/off.

---

## 5. Challenges and Solutions

### Challenge 1: `smbus` not available in virtual environment
**Problem:** The `mpu6050-raspberrypi` package depends on `smbus`, which is a C extension only available as a system package (`python3-smbus`). A standard venv could not see it.
**Solution:** Created the venv with `--system-site-packages` flag so it inherits the system `smbus` while keeping all other packages isolated.

### Challenge 2: `sr.Microphone()` argument error (Lab 5, separate)
**Problem:** An earlier lab passed `sample_rate=16000` to `sr.Microphone()`, which does not accept that argument in the installed version of SpeechRecognition.
**Solution:** Removed the argument — `sr.Microphone()` uses the system default device and the Google backend resamples internally.

### Challenge 3: Misleading 100% accuracy on small test set
**Problem:** With only 40 samples and an 80/20 split, the test set had just 8 samples — 100% on 8 samples provides no statistical confidence.
**Solution:** Switched to 5-fold stratified cross-validation, which uses all 40 samples for evaluation across 5 independent splits. The consistent F1=1.00 across all folds — combined with the observed 128–246× feature separation between classes — confirms the result is genuine and not a statistical artifact.

### Challenge 4: Binary feedback is less informative for coaching
**Problem:** A simple "good/bad" LED flash gives no sense of degree — a stroke that's slightly off looks the same as one that's completely wrong.
**Solution:** Used `predict_proba()` instead of `predict()` to get a continuous confidence score (0.0–1.0), then mapped it to PWM duty cycles on both LEDs simultaneously. This creates a smooth green↔red gradient that reflects stroke quality continuously.

---

## 6. Results Summary

- **IMU verified** via `i2cdetect -y 1` (device at address `0x68`)
- **Calibration** completed in ~5 seconds (500 samples, stationary cue)
- **Data collected:** 40 samples, 20 per class, balanced dataset
- **Model trained:** Decision Tree, depth=3, F1=1.00 (5-fold CV)
- **Real-time system:** Running at ~100 Hz with 0.2s feedback latency
- **LED feedback:** Smooth PWM gradient reflecting live stroke quality score

---

## 7. How to Run

```bash
# 1. Activate environment
cd ~/Documents/MidtermCue

# 2. Calibrate (cue stationary)
.venv/bin/python calibrate_imu.py

# 3. Collect training data
.venv/bin/python collect_data.py --label 1 --count 20   # good strokes
.venv/bin/python collect_data.py --label 0 --count 20   # bad strokes

# 4. Train model
.venv/bin/python train_model.py

# 5. Run real-time inference
.venv/bin/python realtime.py
```

---

## 8. Possible Extensions

- Will definitely be adding a **Computer vision model to add assistance on what balls to target and game prediction**
- **More classes:** Add a "miscue" class (tip misses contact point) — would require gyro spike detection
- **Stability score display:** Stream the score to a small OLED display instead of just LEDs
- **Wake phrase:** Add audio trigger (Vosk) so the device only activates when the user says "ready" — avoiding false triggers from walking near the table
- **Data augmentation:** Add small Gaussian noise to training samples to improve robustness to session-to-session variation in how the cue is held

