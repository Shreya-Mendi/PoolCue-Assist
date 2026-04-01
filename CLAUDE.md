# PoolCue Vision Assist — Claude Context

AIPI 590 Final Project. Raspberry Pi 4 billiards assistant.
Camera detects balls → LLM recommends shot → speaker announces it → IMU grades stroke.

---

## Hardware (what is physically connected)

| Component | Interface | Config key / pin |
|---|---|---|
| Pi Camera Module (CSI ribbon) | CSI | `camera_index: 0` |
| USB speaker | USB-A | default audio out |
| MPU6050 IMU | I2C (SDA/SCL) | address 0x68 |
| 16×2 LCD | I2C (SDA/SCL) | `lcd_address: "0x27"` |
| Push button | GPIO 18 | `button_pin: 18` |
| Green LED | GPIO 17 | hardcoded in realtime.py |
| Red LED | GPIO 27 | hardcoded in realtime.py |
| Buzzer | GPIO 22 | hardcoded in realtime.py |
| Laser (optional, tape to cue) | always-on / GPIO 25 | `laser_gpio_controlled: false` |

---

## Repo layout

```
src/
  main.py              — orchestrator, entry point for systemd
  vision/
    detector.py        — YOLOv11n ball detection → {label: (cx,cy)}
    laser_tracker.py   — HSV laser dot detection → (x,y) or None
    pocket_map.py      — click-to-calibrate 6 pockets, saves JSON
  llm/
    shot_advisor.py    — Duke LiteLLM (GPT-4o-mini) shot recommendation
  audio/
    speaker.py         — pyttsx3 TTS, non-blocking thread
  aim_guidance.py      — laser vs ideal line → speak "left/right/locked"
  game_state.py        — 8-ball logic, tracks sunk balls, solids/stripes
  realtime.py          — IMU stroke grading (from midterm, reused)
  imu_helpers.py       — MPU6050 read helpers

models/
  stroke_model.pkl     — midterm stroke classifier
  pool_vision/
    best.pt            — YOLOv11n trained on overhead pool table dataset

config/
  settings.json        — all tunable config (pins, thresholds, model)
  pocket_coords.json   — generated on first run by clicking pockets

data/
  test_table.jpg       — synthetic test image (no camera needed)
  stroke_data.csv      — midterm training data

scripts/
  install.sh           — one-shot Pi setup
  make_test_image.py   — regenerate test_table.jpg
  poolassist.service   — systemd unit (NOT in git, may contain API key)

train/
  colab_train.ipynb    — YOLOv11n training on Colab
```

---

## LLM: Duke OIT LiteLLM proxy

- Base URL: `https://litellm.oit.duke.edu`
- Model: `gpt-4o-mini`
- Auth: API key read from env var `DUKE_API_KEY` or file `~/.duke_litellm_key`
- Client: OpenAI Python SDK pointed at the Duke base URL
- Key value: stored in `~/.duke_litellm_key` on the Pi — never commit it

Set key on Pi:
```bash
echo "sk-rJLrekp329isjdOUvMngOA" > ~/.duke_litellm_key
chmod 600 ~/.duke_litellm_key
```

---

## Running

### Normal (camera):
```bash
cd /home/pi/PoolCue-Assist
python3 src/main.py
```

### Test mode (no camera, static image):
```bash
python3 src/main.py --test-image data/test_table.jpg
```

### Auto-boot (systemd):
```bash
# Install once:
sudo cp scripts/poolassist.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable poolassist
sudo systemctl start poolassist

# Check logs:
journalctl -u poolassist -f
```

The `poolassist.service` file must set `DUKE_API_KEY` in the `[Service]` block:
```ini
[Service]
Environment="DUKE_API_KEY=sk-rJLrekp329isjdOUvMngOA"
```

---

## First-run: pocket calibration

On the first run a window opens showing the camera frame.
Click each of the 6 pockets in order: TL, TC, TR, BL, BC, BR.
Coordinates are saved to `config/pocket_coords.json` — never needs to be done again.

---

## Pi setup from scratch

```bash
# 1. Clone
git clone -b final-project https://github.com/Shreya-Mendi/PoolCue-Assist /home/pi/PoolCue-Assist
cd /home/pi/PoolCue-Assist

# 2. Install everything
bash scripts/install.sh

# 3. Set API key
echo "sk-rJLrekp329isjdOUvMngOA" > ~/.duke_litellm_key
chmod 600 ~/.duke_litellm_key

# 4. Enable camera (if not already)
sudo raspi-config nonint do_camera 0
# Or: sudo nano /boot/config.txt → add: start_x=1, gpu_mem=128

# 5. Test without camera first
python3 src/main.py --test-image data/test_table.jpg

# 6. Test with camera
python3 src/main.py
# → click 6 pockets when prompted (first run only)
```

---

## config/settings.json keys

| Key | Default | What it does |
|---|---|---|
| `game_mode` | `"8ball"` | game rules (8ball only for now) |
| `camera_index` | `0` | cv2 camera index |
| `detection_conf` | `0.45` | YOLO confidence threshold |
| `imgsz` | `320` | YOLO inference resolution (320 for Pi speed) |
| `laser_color` | `"green"` | HSV range for laser tracker |
| `laser_gpio_controlled` | `false` | true = Pi controls laser via transistor on GPIO 25 |
| `button_pin` | `18` | GPIO pin for push button |
| `lcd_address` | `"0x27"` | I2C address of LCD |
| `table_change_threshold` | `2` | min ball count change to trigger LLM re-query |
| `show_display` | `false` | show annotated OpenCV window (needs monitor) |
| `llm_model` | `"gpt-4o-mini"` | model name sent to Duke LiteLLM |

---

## Demo flow (cardboard)

1. Print `diagrams/print_balls.html` → cut out balls → paste on green cardboard
2. Place cardboard flat, Pi Camera overhead
3. Run `python3 src/main.py`
4. Calibrate pockets (first run)
5. System detects balls → speaker announces shot recommendation → LCD shows ball+pocket
6. Move laser dot over the cue ball to aim
7. Press button → swing cue → IMU grades stroke → LEDs + speaker give feedback
8. Move/remove a ball → system re-queries LLM for next shot

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Camera not found` | `vcgencmd get_camera` — should show `supported=1 detected=1`. Run `sudo raspi-config nonint do_camera 0` then reboot |
| `FileNotFoundError: best.pt` | `ls models/pool_vision/` — if missing, pull from GitHub or retrain in Colab |
| No audio | `aplay -l` to list devices. `sudo raspi-config nonint do_audio 1` for 3.5mm jack, `2` for HDMI |
| LCD blank | Check `i2cdetect -y 1` — should show device at 0x27. Check SDA/SCL wires |
| IMU not found | `i2cdetect -y 1` — should show 0x68. Check I2C wiring and `sudo raspi-config` I2C enable |
| LLM error | Check `~/.duke_litellm_key` exists and key is valid. Test: `curl -H "Authorization: Bearer $(cat ~/.duke_litellm_key)" https://litellm.oit.duke.edu/models` |
| Slow detection | Confirm `imgsz: 320` in settings.json (not 640). Pi 4 gets ~3fps at 320 |

---

## Branches

| Branch | Purpose |
|---|---|
| `main` | midterm code only |
| `final-project` | full vision + LLM pipeline |
| `test` | test image + --test-image flag |
| `optimizations` | Pi performance experiments |
