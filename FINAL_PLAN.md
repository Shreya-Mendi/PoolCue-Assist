# PoolCue Vision Assist — Final Project Plan
**AIPI 590 | Final Project**
**Due: April 4**
**Platform:** Raspberry Pi 4 | **Language:** Python 3

---

## Project Overview

An autonomous billiards assistant that:
1. Watches the table via camera
2. Detects all balls using a pre-trained Roboflow/YOLO vision model
3. Passes table state to Claude (LLM) for shot recommendation
4. Tracks where the player is aiming via a laser dot on the cue tip
5. Talks the player onto the correct aim line via speaker
6. After the shot, the IMU (from midterm) grades the stroke quality
7. Everything runs automatically on boot — plug in and play

---

## Hardware Shopping List

### New for Final (what you need to buy)

| Component | Purpose | Approx Cost | Where |
|---|---|---|---|
| Raspberry Pi Camera Module 3 | Main vision input | $25 | Amazon / Adafruit |
| Camera mount / flexible arm | Position camera overhead or at table edge | $10–15 | Amazon |
| USB mini speaker (or 3.5mm) | Audio feedback / TTS | $10–15 | Amazon |
| Green laser module (5mW, 650nm) | Cue aim laser dot | $5–8 | Amazon |
| Small laser mount / clip | Attach laser to cue tip | $3–5 | Amazon |
| Jumper wires (extra female-female) | Laser GPIO wiring | $5 | Amazon |

**Total estimated cost: ~$60–70**

> If you want the laser Pi-controlled (auto on/off): wire it to GPIO 25 via a 2N2222 transistor + 100Ω resistor.
> If you just want plug-and-play for demo: use a battery-powered laser pointer clipped to the cue — no wiring needed.

### Reused from Midterm

| Component | Reused For |
|---|---|
| Raspberry Pi 4 | Main compute |
| MPU6050 IMU | Stroke quality grading after shot |
| Push button (GPIO 18) | Trigger shot capture |
| 16×2 LCD (I2C) | Display shot recommendation + aim status |
| Green/Red LEDs + resistors | Stroke quality feedback |
| Buzzer | Audio stroke feedback |
| HC-SR04 Ultrasonic | Optional: cue height check |

---

## Software Architecture

```
Boot
  ↓
systemd auto-starts main.py
  ↓
Camera initializes → LCD shows "PoolCue Vision Ready"
  ↓
┌─────────────────────────────────────┐
│           VISION LOOP               │
│  Camera frame → YOLO/Roboflow       │
│  Detect: all balls + positions      │
│  Detect: laser dot (HSV mask)       │
│  Detect: pocket coordinates         │
└─────────────────────────────────────┘
  ↓
Table state → Claude API
  → "Hit the 4 ball into the top-right pocket"
  ↓
Speaker speaks recommendation
LCD shows: "4 BALL → TOP-R"
  ↓
┌─────────────────────────────────────┐
│           AIM GUIDANCE LOOP         │
│  Track laser dot position           │
│  Compare to ideal shot line         │
│  Speaker: "left... left... LOCKED"  │
│  LCD: aim error in pixels           │
└─────────────────────────────────────┘
  ↓
Player presses button → IMU captures stroke
  ↓
stroke_model.pkl grades stroke quality
Speaker: "Good stroke" / "You twisted"
LEDs + buzzer feedback (same as midterm)
  ↓
Loop: detect new table state after shot
```

---

## File Structure (Final)

```
PoolCue-Assist/
├── src/
│   ├── vision/
│   │   ├── detector.py          # Roboflow/YOLO inference → ball positions dict
│   │   ├── laser_tracker.py     # OpenCV HSV masking → laser dot (x, y)
│   │   ├── pocket_map.py        # Pocket coordinate calibration + storage
│   │   └── overlay.py           # Optional: draw shot lines on HDMI output
│   ├── llm/
│   │   └── shot_advisor.py      # Build prompt from table state → Claude API → parse response
│   ├── audio/
│   │   └── speaker.py           # pyttsx3 TTS wrapper (offline, no latency)
│   ├── aim_guidance.py          # Compare laser dot to ideal line → speak corrections
│   ├── game_state.py            # Track balls sunk, whose turn, game mode (8-ball/9-ball)
│   ├── main.py                  # Orchestrator + systemd entry point
│   │
│   │   ── REUSED FROM MIDTERM ──
│   ├── imu_helpers.py
│   ├── calibrate_imu.py
│   ├── collect_data.py
│   ├── train_model.py
│   ├── realtime.py              # Stroke grading (called from main.py)
│   ├── imu_check.py
│   └── led_test.py
├── models/
│   ├── stroke_model.pkl         # Midterm stroke classifier (reused)
│   └── pool_vision/             # YOLO weights exported from Roboflow
│       └── best.pt
├── config/
│   ├── pocket_coords.json       # Stored pocket positions from calibration
│   └── settings.json            # API key path, GPIO pins, thresholds
├── data/
│   └── stroke_data.csv
├── diagrams/
│   ├── wiring_diagram_final.html
│   └── wiring_photo.png
├── scripts/
│   └── poolassist.service       # systemd unit file for auto-boot
├── imu_calibration.json
├── requirements.txt
├── README.md
└── FINAL_PLAN.md
```

---

## Key Modules — What Each Does

### `vision/detector.py`
- Loads YOLO model (exported from Roboflow as ONNX or PyTorch)
- Runs inference on each camera frame
- Returns: `{"cue_ball": (x,y), "1": (x,y), "4": (x,y), ...}`
- Filters by confidence threshold (default 0.5)

### `vision/laser_tracker.py`
```python
# Core logic — detect bright green/red dot
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_green, upper_green)
contours = cv2.findContours(mask, ...)
# Return centroid of largest contour = laser dot position
```

### `llm/shot_advisor.py`
```python
prompt = f"""
You are a billiards coach. Current table (8-ball game):
Player shoots: {player_type}  (solids/stripes)
Balls remaining: {remaining_balls}
Ball positions (pixels, 640x480): {positions}
Pocket positions: {pockets}

Give ONE shot recommendation in this exact format:
SHOT: [ball number] → [pocket name]
REASON: [one sentence]
DIFFICULTY: [1-5]
"""
# Parse response, speak SHOT line, show on LCD
```

### `audio/speaker.py`
```python
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 160)   # slightly slower for clarity
engine.say(text)
engine.runAndWait()
```
Fully offline — no internet needed for TTS.

### `aim_guidance.py`
```python
# Ideal aim line: from cue_ball through target_ball toward pocket
# Laser dot: detected from camera
# Error: perpendicular distance from dot to ideal line (pixels)
if error < 15:
    speak("Locked in — shoot")
elif dot is left of line:
    speak("Move right")
else:
    speak("Move left")
```

---

## Auto-Boot Setup (Plug and Play)

### `scripts/poolassist.service`
```ini
[Unit]
Description=PoolCue Vision Assist
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/PoolCue-Assist
ExecStart=/usr/bin/python3 /home/pi/PoolCue-Assist/src/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Install once:
```bash
sudo cp scripts/poolassist.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable poolassist
sudo systemctl start poolassist
```

After this — plug in Pi → program starts in ~15 seconds. No SSH, no terminal.

---

## API Setup

### Roboflow (vision model)
1. Go to https://universe.roboflow.com/billiard-ball-data-set/billiard-pool
2. Export model → YOLOv8 PyTorch format (or use hosted API)
3. Download `best.pt` → place in `models/pool_vision/`
4. Or use Roboflow hosted inference API with your API key

### Claude API
1. Get key from https://console.anthropic.com
2. Store in `~/.anthropic_key` (never hardcode)
3. `shot_advisor.py` reads it: `os.environ["ANTHROPIC_API_KEY"]`

---

## Dependencies (`requirements.txt`)

```
# Vision
ultralytics>=8.0          # YOLOv8
opencv-python>=4.8
roboflow                  # optional: if using hosted inference

# LLM
anthropic>=0.25

# Audio
pyttsx3

# Hardware (Pi only)
RPi.GPIO
mpu6050-raspberrypi
RPLCD

# Utilities
numpy
Pillow
```

System packages (install via apt):
```bash
sudo apt install python3-smbus espeak libespeak-dev
```
`espeak` is the TTS engine backend for `pyttsx3` on Linux/Pi.

---

## Wiring — New Components

### Laser Module (GPIO-controlled)
```
Pi GPIO 25 → 100Ω resistor → Base of 2N2222 transistor
Emitter → GND
Collector → Laser (−)
Laser (+) → 3.3V
```
> Skip this if using battery-powered laser pointer — just always-on for the demo.

### Camera Module 3
- Connect to Pi CSI ribbon cable port (zero-insertion force connector)
- Enable camera: `sudo raspi-config` → Interface Options → Camera → Enable

### Speaker
- USB speaker: plug in, works immediately
- 3.5mm: plug into Pi audio jack, set as default output

---

## Build Order (What to Do First)

1. **Get hardware** — order camera, speaker, laser (~1 week shipping)
2. **Download + test Roboflow model locally** — verify ball detection works on your laptop
3. **Wire camera + test** — confirm Pi sees camera frames
4. **Build `detector.py`** — ball detection working on Pi
5. **Build `laser_tracker.py`** — detect laser dot in frame
6. **Set up Claude API** — `shot_advisor.py` returning recommendations
7. **Add speaker + TTS** — hear the recommendation spoken
8. **Build `aim_guidance.py`** — laser dot vs ideal line comparison
9. **Wire laser module** (if GPIO-controlled)
10. **Integrate IMU stroke grading** — plug in midterm `realtime.py`
11. **Build `game_state.py`** — track balls sunk between frames
12. **Set up systemd auto-boot**
13. **Test full end-to-end loop**
14. **Calibrate pockets** — run calibration once, save `pocket_coords.json`

---

## Presentation Arc (5 min)

| Section | Time | Content |
|---|---|---|
| Background | 45s | "My midterm taught the Pi to grade your stroke. My final teaches it to watch the table, decide the shot, and guide your aim." |
| Hardware | 30s | Show camera, laser on cue, speaker — new vs reused |
| Vision model | 45s | Roboflow dataset, YOLO inference, what it detects |
| LLM advisor | 45s | Show the prompt structure, Claude output |
| Aim guidance | 45s | Laser dot tracking, how error is computed, audio feedback |
| IMU bridge | 30s | How stroke grading from midterm is reused |
| Demo | 60s | Live: place balls → hear recommendation → aim → shoot → hear stroke grade |
| Challenges + next steps | 30s | Latency, lighting, 9-ball mode, shot history analytics |

---

## Challenges to Anticipate

| Challenge | Mitigation |
|---|---|
| Lighting messes up laser detection | Tune HSV thresholds for your specific laser + room lighting. Green laser easier than red. |
| YOLO too slow on Pi | Use Pi 4 (not 3). Run at 320×320 instead of 640×640. Or use Roboflow hosted API — offloads compute |
| Claude API latency (~1-2s) | Only call API when table state changes (new rack or ball sunk), not every frame |
| Camera angle distorts ball positions | Do a perspective transform (cv2.getPerspectiveTransform) to map to top-down coordinates |
| Speaker not loud enough | USB speaker with its own power is louder than 3.5mm on Pi |

---

## Grading Alignment

| Rubric Item | How This Project Covers It |
|---|---|
| Idea statement + definition of success (20pts) | Clear: player receives correct shot recommendation and gets guided to aim correctly before shooting |
| Sensor identification (10pts) | Camera (vision), IMU (stroke), optional laser module (GPIO output) |
| Physical user interface (10pts) | LCD + push button (reused from midterm) |
| Model development + deployment (10pts) | YOLO vision model (Roboflow) + Claude LLM — both run on/from Pi |
| Physical environment manipulation (10pts) | Laser pointer GPIO on/off, LEDs, buzzer, speaker audio |
| Analysis / conclusions (20pts) | Aim error distribution, shot recommendation accuracy, stroke quality vs aim quality correlation |
| Code organization (5pts) | Modular src/ structure, comments, README |
| Presentation clarity (15pts) | Clear 5-min arc with live demo |
