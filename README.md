# PoolCue Vision Assist

**AIPI 590 | Midterm + Final Project**
**Platform:** Raspberry Pi 4 | **Language:** Python 3

---

## Overview

A two-part billiards assistant built on a Raspberry Pi 4:

**Midterm — Stroke Analyzer**
Mounts an IMU on the cue. Detects and classifies stroke quality (good vs bad) using a trained Decision Tree. Feedback via LCD, LEDs, and buzzer.

**Final — Vision Assistant**
Adds a Pi Camera Module, USB speaker, and laser pointer on the cue tip. Detects all balls on the table using a YOLOv11n model trained locally on a 1,200-image Roboflow dataset (CueTor Billiards). Passes table state to a GPT-4o-mini via Duke OIT LiteLLM proxy for shot recommendation. Tracks the laser dot on the cue tip and guides the player onto the correct aim line in real time. IMU stroke grading from the midterm is reused after each shot.

**The full loop:**
> Plug in Pi → camera watches table → LLM recommends a shot → USB speaker announces it → laser guides aim → button press → IMU grades stroke → repeat

---

## Hardware

| Component | Purpose | Interface |
|---|---|---|
| Raspberry Pi 4 | Main compute | — |
| Pi Camera Module (CSI) | Ball detection + laser tracking | CSI ribbon cable |
| USB Speaker | TTS audio output | USB-A port |
| Laser pointer (cue tip) | Aim marker | Taped to cue |
| MPU6050 IMU | Stroke quality grading | I²C (0x68) |
| Push Button | Trigger stroke capture | GPIO 18 |
| 16×2 LCD (I2C) | Shot recommendation display | I²C (0x27) |
| Green LED | Good stroke confidence (PWM) | GPIO 17 |
| Red LED | Bad stroke confidence (PWM) | GPIO 27 |
| Buzzer | Stroke audio feedback | GPIO 22 |
| HC-SR04 Ultrasonic | Cue height check | GPIO 23/24 |

**Wiring diagram:** [diagrams/wiring_diagram.html](diagrams/wiring_diagram.html)

---

## Repository Structure

```
PoolCue-Assist/
├── src/
│   ├── main.py                  # Full orchestrator — entry point
│   ├── aim_guidance.py          # Laser dot vs ideal shot line → audio corrections
│   ├── game_state.py            # Track balls sunk, turns, shot log
│   ├── vision/
│   │   ├── detector.py          # YOLOv11 ball detection
│   │   ├── laser_tracker.py     # OpenCV HSV laser dot detection
│   │   └── pocket_map.py        # Pocket coordinate calibration
│   ├── llm/
│   │   └── shot_advisor.py      # Claude API shot recommendation
│   ├── audio/
│   │   └── speaker.py           # pyttsx3 TTS wrapper
│   │
│   │   ── Midterm (reused) ──
│   ├── imu_helpers.py
│   ├── calibrate_imu.py
│   ├── collect_data.py
│   ├── train_model.py
│   └── realtime.py
├── train/
│   ├── colab_train.ipynb        # YOLOv11 training notebook (run in Google Colab)
│   └── merge_datasets.md        # Instructions to merge Roboflow datasets
├── models/
│   ├── stroke_model.pkl         # Midterm stroke classifier
│   └── pool_vision/
│       └── best.pt              # YOLOv11 weights (download after training)
├── config/
│   ├── settings.json            # All tunable settings
│   └── pocket_coords.json       # Auto-generated after pocket calibration
├── scripts/
│   ├── poolassist.service       # systemd unit for auto-boot
│   ├── install.sh               # One-time Pi setup script
│   └── export_model.py          # Export best.pt → ONNX for Pi
├── data/
│   └── stroke_data.csv          # Midterm training data
├── diagrams/
│   └── wiring_diagram.html
├── imu_calibration.json
└── requirements.txt
```

---

## Setup

### 1. Train the vision model

**Option A — Google Colab (recommended, free T4 GPU, ~15 min)**
1. Upload the dataset zip to your Google Drive
2. Open [train/colab_train.ipynb](train/colab_train.ipynb) in Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Update the dataset path cell to point to your Drive zip
5. Run all cells — downloads `best.pt` at the end
6. Place `best.pt` in `models/pool_vision/best.pt`

**Option B — Local Mac (Apple MPS GPU, ~30-40 min)**
```bash
python3 scripts/train_local.py
```
Weights saved to `models/pool_vision/yolo11n_pool/weights/best.pt` automatically.

### 2. One-time Pi setup

```bash
git clone https://github.com/Shreya-Mendi/PoolCue-Assist
cd PoolCue-Assist
bash scripts/install.sh
```

Then add your Anthropic API key:
```bash
sudo nano /etc/systemd/system/poolassist.service
# Edit: Environment=ANTHROPIC_API_KEY=your_key_here
sudo systemctl daemon-reload
```

### 3. Enable audio output

```bash
sudo raspi-config
# System Options → Audio → 3.5mm jack
```

Test:
```bash
espeak "pool cue vision assist ready"
```

### 4. Pocket calibration (once, with camera in final position)

```bash
python3 src/main.py
# First run: a window opens — click each of the 6 pockets in order
# Coordinates saved to config/pocket_coords.json automatically
```

### 5. Auto-boot

After `install.sh` runs `systemctl enable poolassist`, the program starts automatically on every boot. No SSH or terminal needed.

---

## Running Manually

```bash
python3 src/main.py
```

For debug mode with live annotated video (connect Pi to HDMI monitor):
```bash
# Set "show_display": true in config/settings.json first
python3 src/main.py
```

---

## Configuration

All settings in [config/settings.json](config/settings.json):

| Key | Default | Description |
|---|---|---|
| `game_mode` | `"8ball"` | Game type for LLM context |
| `camera_index` | `0` | Camera device index |
| `detection_conf` | `0.45` | YOLO confidence threshold |
| `laser_color` | `"green"` | `"green"` or `"red"` |
| `button_pin` | `18` | GPIO pin for stroke button |
| `lcd_address` | `"0x27"` | I2C address for LCD |
| `table_change_threshold` | `2` | Ball diff needed to re-query LLM |
| `show_display` | `false` | Show annotated video on HDMI |

---

## Vision Model

**Architecture:** YOLOv11 Nano (`yolo11n`) — fastest inference, small enough for Pi CPU

**Training data:** Merged Roboflow datasets:
- [Billiard Ball Detection v6](https://universe.roboflow.com/billiard-ball-data-set/billiard-ball-detection-aeo1m/dataset/6)
- [Pool Ball Detection by Ben Gann](https://universe.roboflow.com/ben-gann-lscqy/pool-ball-detection/dataset/2)
- [8 Ball Pool by skylep](https://universe.roboflow.com/skylep/8-ball-pool-fmk6g/dataset/8)

**Classes:** `cue`, `1`–`15` (16 total)

**Training:** 80 epochs, 640×640, augmentation (HSV shift, horizontal flip, mosaic)

---

## Midterm Reference

**Algorithm:** Decision Tree (`max_depth=3`)
**Features:** `peak_accel`, `mean_gyro_y`, `var_gyro_y`, `mean_gyro_z`, `var_gyro_z`, `duration`
**Validation:** 5-fold CV F1 = 1.00 (129× class separation on `var_gyro_z`)

See original midterm README content preserved below the final project sections.

---

## Citing AI Assistance

This project used Claude (Anthropic) for code generation assistance.
Claude Code — Anthropic, 2025. https://claude.ai/code
