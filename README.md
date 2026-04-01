# PoolCue Vision Assist

**AIPI 590 | Final Project — Shreya Mendi**
**Platform:** Raspberry Pi 4B | **Language:** Python 3.13

---

## Project Idea & Definition of Success

A real-time billiards coaching assistant that runs entirely on a Raspberry Pi 4. The device watches a pool table through an overhead camera, detects all balls using a custom-trained YOLO model, asks a large language model for the best next shot, and speaks the recommendation out loud. After the player shoots, an IMU on the cue grades the stroke quality and an ultrasonic sensor checks cue height — all delivered through LEDs, a buzzer, a 16×2 LCD, and a neural TTS voice ("Cue").

**Success criteria:**
- Ball detection runs at usable speed on Pi 4 CPU (no GPU)
- LLM produces a clear, natural shot recommendation each press
- Stroke is graded correctly using the midterm Decision Tree model
- Every sensor produces audible + visual feedback in real time
- System auto-boots on Pi startup without a monitor or keyboard

---

## Rubric Alignment

| Rubric Item | Implementation |
|---|---|
| **Idea statement & definition of success (20pts)** | Billiards coaching edge device — see above |
| **Sensor identification for collecting data (10pts)** | Pi Camera (vision), MPU6050 IMU (stroke), HC-SR04 ultrasonic (cue height), GPIO button (input) |
| **Physical user interface (10pts)** | 16×2 I²C LCD shows ball→pocket + difficulty on every recommendation and stroke score after grading |
| **Model development & deployment (10pts)** | YOLOv11n trained on 1,200-image Roboflow dataset, exported to ONNX, runs via `onnxruntime` on Pi CPU. Midterm Decision Tree stroke classifier reused. |
| **Physical environment manipulation (10pts)** | Green/Red PWM LEDs reflect stroke score, active buzzer gives good/bad beeps, LCD updates in real time |
| **Analysis / conclusions (20pts)** | See presentation slides; stroke features, YOLO mAP, and LLM prompt engineering discussed |
| **Code organization (5pts)** | Modular `src/` layout, all config in `settings.json`, systemd auto-boot, inline comments throughout |
| **Presentation (15pts)** | Live demo: button press → detection → LLM voice → stroke grade → LED feedback |

---

## Full Pipeline

```
Boot → systemd starts main.py
  → ONNX model loads (YOLOv11n)
  → "I'm Cue" spoken via Piper TTS

Press button (GPIO 25)
  → Camera captures frame
  → YOLOv11n detects balls (onnxruntime, 640×640)
  → Game state updated (tracks sunk balls)
  → GPT-4.1-Mini via Duke LiteLLM proxy recommends shot
  → Piper neural TTS speaks recommendation
  → LCD shows: Ball X → Pocket Y | Difficulty N/5

Swing, then press button again
  → HC-SR04 measures cue height → voice feedback ("Too low!", "Good height!")
  → IMU records 1s of accelerometer + gyroscope data
  → Gyro variance analyzed for tilt/wobble → coaching comment spoken
  → Decision Tree classifies stroke → score 0–100%
  → Green/Red LEDs set to score, buzzer beeps
  → LCD shows: GOOD/BAD score% | Height Xcm

→ State resets → "Press button for next shot"
```

---

## Sensors & Hardware

| Component | Interface | GPIO / Address | Purpose |
|---|---|---|---|
| Pi Camera Module (CSI) | CSI ribbon | `camera_index: 0` | Ball detection |
| MPU6050 IMU | I²C | `0x68` | Stroke quality grading |
| HC-SR04 Ultrasonic | GPIO | TRIG=23, ECHO=24 | Cue height check |
| Push Button | GPIO | `25` (active-low) | Trigger recommend / grade |
| 16×2 LCD (PCF8574) | I²C | `0x27` | Shot info + stroke score |
| Green LED | GPIO PWM | `17` | Good stroke confidence |
| Red LED | GPIO PWM | `27` | Bad stroke confidence |
| Active Buzzer | GPIO PWM | `22` | Stroke audio feedback |
| USB Speaker (UACDemoV1.0) | USB / ALSA card 3 | — | Piper TTS voice output |

**Wiring diagram:** [diagrams/wiring_diagram.html](diagrams/wiring_diagram.html)

---

## Models

### Vision — YOLOv11n (ONNX)
- **Architecture:** YOLOv11 Nano — fastest variant, <4MB, runs on Pi CPU
- **Input:** 640×640 RGB, output: 8400 candidate boxes × 20 values
- **Classes:** `cue`, `1`–`15` (16 total)
- **Training data:** ~1,200 images from three merged Roboflow datasets
  - [Billiard Ball Detection v6](https://universe.roboflow.com/billiard-ball-data-set/billiard-ball-detection-aeo1m/dataset/6)
  - [Pool Ball Detection — Ben Gann](https://universe.roboflow.com/ben-gann-lscqy/pool-ball-detection/dataset/2)
  - [8 Ball Pool — skylep](https://universe.roboflow.com/skylep/8-ball-pool-fmk6g/dataset/8)
- **Training:** Google Colab T4 GPU, 80 epochs, 640×640, HSV + flip + mosaic augmentation
- **Deployment:** Exported to `data/best.onnx`, loaded via `onnxruntime` (no PyTorch on Pi)

### Stroke Classifier — Decision Tree (Midterm, reused)
- **Algorithm:** `DecisionTreeClassifier(max_depth=3)` via scikit-learn
- **Features:** `peak_accel`, `mean_gyro_y`, `var_gyro_y`, `mean_gyro_z`, `var_gyro_z`, `duration`
- **Data:** 129 labeled strokes collected via midterm data collection script
- **Validation:** 5-fold CV F1 = 1.00 (primary separator: `var_gyro_z`)
- **File:** `models/stroke_model.pkl`

### LLM — GPT-4.1-Mini via Duke LiteLLM Proxy
- **Endpoint:** `https://litellm.oit.duke.edu`
- **Auth:** API key from `~/.duke_litellm_key` or `DUKE_API_KEY` env var
- **Prompt:** Sends ball pixel coordinates + pocket positions → structured response: `BALL / POCKET / SPOKEN / REASON / DIFFICULTY`
- **Persona:** "Cue" — confident female billiards coach, warm and encouraging

---

## Repository Structure

```
MidtermCue/
├── src/
│   ├── main.py               # Orchestrator — state machine, GPIO, full loop
│   ├── game_state.py         # Tracks balls on table, sunk events, shot log
│   ├── imu_helpers.py        # MPU6050 read + calibration helpers
│   ├── vision/
│   │   ├── detector.py       # YOLOv11n ONNX inference, NMS, draw
│   │   ├── laser_tracker.py  # HSV laser dot detection
│   │   └── pocket_map.py     # Click-to-calibrate 6 pockets, saves JSON
│   ├── llm/
│   │   └── shot_advisor.py   # Duke LiteLLM prompt + response parser
│   └── audio/
│       └── speaker.py        # Piper TTS (primary), pico2wave / espeak fallback
├── models/
│   ├── stroke_model.pkl      # Midterm Decision Tree classifier
│   └── pool_vision/
│       └── best.pt           # YOLOv11n training weights (Colab output)
├── data/
│   ├── best.onnx             # Exported ONNX model for Pi inference
│   ├── stroke_data.csv       # 129-row midterm training dataset
│   ├── test_table.jpg        # Static test image (no camera needed)
│   └── sample_test_images/   # 6 real overhead pool table photos for demo
├── train/
│   └── colab_train.ipynb     # YOLOv11n Colab training notebook
├── config/
│   ├── settings.json         # All tunable config (pins, thresholds, model)
│   └── pocket_coords.json    # Auto-generated on first run
├── scripts/
│   ├── poolassist.service    # systemd unit — auto-boot on Pi startup
│   ├── install.sh            # One-shot Pi setup
│   └── sample_images.py      # Script used to sample 6 test images
├── diagrams/
│   └── wiring_diagram.html   # Full circuit wiring diagram
├── imu_calibration.json      # IMU bias calibration values
└── requirements.txt
```

---

## Setup & Running

### Quick start (no camera — demo mode)

```bash
cd /home/shreyam/Documents/MidtermCue
amixer -c 3 sset PCM 80%
.venv/bin/python3 src/main.py --test-images-dir data/sample_test_images
```

Press button → shot recommendation spoken + LCD updates.
Press button again → stroke grading (IMU + HC-SR04 height check).

### With Pi Camera

```bash
.venv/bin/python3 src/main.py
# First run: click each of the 6 pockets when the calibration window opens
```

### Auto-boot (already enabled)

The systemd service is installed and enabled. On every Pi boot:
```
poolassist.service starts → 8s delay → audio configured → main.py launches
```

Check status:
```bash
sudo systemctl status poolassist
journalctl -u poolassist -f
```

Reinstall after editing service file:
```bash
sudo cp scripts/poolassist.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart poolassist
```

### LLM API key

```bash
echo "YOUR_KEY_HERE" > ~/.duke_litellm_key
chmod 600 ~/.duke_litellm_key
```

---

## Configuration (`config/settings.json`)

| Key | Default | Description |
|---|---|---|
| `game_mode` | `"8ball"` | Game rules passed to LLM |
| `camera_index` | `0` | OpenCV camera index |
| `detection_conf` | `0.45` | YOLO confidence threshold |
| `imgsz` | `320` | YOLO inference resolution (320 = faster on Pi) |
| `button_pin` | `25` | GPIO BCM pin for push button |
| `lcd_address` | `"0x27"` | I²C address of LCD |
| `llm_model` | `"GPT 4.1 Mini"` | Model name sent to Duke LiteLLM |
| `show_display` | `false` | Show annotated OpenCV window (needs HDMI) |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No audio | Check `~/.asoundrc` exists with `hw:3,0`. Run `aplay -l` to confirm card 3 is USB speaker. |
| `Camera not found` | `vcgencmd get_camera` → should show `detected=1`. Run `sudo raspi-config nonint do_camera 0` then reboot. |
| LCD blank | `i2cdetect -y 1` → should show `27`. Check SDA/SCL wires. |
| IMU not found | `i2cdetect -y 1` → should show `68`. Check I²C wiring. `sudo raspi-config` → enable I²C. |
| LLM error | Check `~/.duke_litellm_key` is valid. Test: `curl -H "Authorization: Bearer $(cat ~/.duke_litellm_key)" https://litellm.oit.duke.edu/models` |
| Balls not detected | Confirm `data/best.onnx` exists. Lower `detection_conf` to `0.3` in `settings.json`. |
| Score always 100% | IMU calibration may be zeroing bias. Recalibrate: `.venv/bin/python3 src/calibrate_imu.py` |

---

## Citing AI Assistance

This project used **Claude Code** (Anthropic) for code generation and debugging assistance.
> Claude Code — Anthropic, 2025. https://claude.ai/code
