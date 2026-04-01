"""
PoolCue Vision Assist — Demo Mode
AIPI 590 Final Project

Points camera at cardboard with printed ball images.
YOLO detects balls → LLM recommends shot → speaker announces it.
Automatically re-announces whenever the ball layout changes.
"""

import sys
import time
import json
import cv2
from pathlib import Path

# Hardware imports (Pi only)
try:
    from RPLCD.i2c import CharLCD
    HAS_LCD = True
except ImportError:
    HAS_LCD = False

sys.path.insert(0, str(Path(__file__).parent))

from vision.detector import BallDetector
from vision.pocket_map import PocketMap
from llm.shot_advisor import ShotAdvisor
from audio.speaker import Speaker
from game_state import GameState

CONFIG_PATH = Path(__file__).parents[1] / "config" / "settings.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

LCD_ADDR = CFG.get("lcd_address", "0x27")
TABLE_CHANGE_THRESHOLD = CFG.get("table_change_threshold", 2)


def setup_lcd():
    addr = int(LCD_ADDR, 16) if isinstance(LCD_ADDR, str) else LCD_ADDR
    lcd = CharLCD(i2c_expander="PCF8574", address=addr, port=1, cols=16, rows=2)
    lcd.clear()
    return lcd


def lcd_write(lcd, line1="", line2=""):
    if lcd is None:
        return
    try:
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1[:16].ljust(16))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2[:16].ljust(16))
    except Exception:
        pass


def balls_changed(prev: dict, curr: dict, threshold: int) -> bool:
    prev_set = set(prev.keys())
    curr_set = set(curr.keys())
    return len(prev_set.symmetric_difference(curr_set)) >= threshold


def main():
    print("PoolCue Vision Assist starting...")

    # LCD
    lcd = None
    if HAS_LCD:
        try:
            lcd = setup_lcd()
            lcd_write(lcd, "PoolCue Vision", "Starting...")
        except Exception as e:
            print(f"[WARN] LCD init failed: {e}")

    # Components
    speaker = Speaker()
    speaker.say("Pool Cue Vision Assist ready", block=True)

    detector  = BallDetector(conf_threshold=CFG.get("detection_conf", 0.45), imgsz=CFG.get("imgsz", 320))
    pocket_map = PocketMap()
    advisor   = ShotAdvisor(model=CFG.get("llm_model", "gpt-4o-mini"))
    game      = GameState(speaker=speaker, mode=CFG.get("game_mode", "8ball"))

    # Camera
    cap = cv2.VideoCapture(CFG.get("camera_index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        speaker.say("Camera not found", block=True)
        lcd_write(lcd, "ERROR", "No camera")
        sys.exit(1)

    # Pocket calibration (click once, saved forever)
    ret, frame = cap.read()
    if not pocket_map.is_calibrated():
        speaker.say("Click each pocket to calibrate", block=True)
        lcd_write(lcd, "CALIBRATE", "Click pockets")
        pocket_map.calibrate(frame)
        speaker.say("Calibration saved", block=True)

    pockets = pocket_map.pockets if pocket_map.is_calibrated() else pocket_map.estimate_from_frame(frame)

    print("Ready — point camera at the cardboard.")
    lcd_write(lcd, "READY", "Show me the table")

    prev_balls = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]
            balls = detector.detect(frame)
            game.update(balls)

            # Re-query LLM whenever ball layout changes
            if balls_changed(prev_balls, balls, TABLE_CHANGE_THRESHOLD) and len(balls) > 1:
                prev_balls = balls.copy()
                remaining = game.remaining_for_player()
                shootable = {k: v for k, v in balls.items() if k in remaining or k == "cue"}

                if len(shootable) > 1:
                    lcd_write(lcd, "Thinking...", "")
                    try:
                        rec = advisor.recommend(
                            balls=shootable,
                            pockets=pockets,
                            game_state={
                                "mode": game.mode,
                                "player_type": game.current_player_type,
                                "frame_w": frame_w,
                                "frame_h": frame_h,
                            }
                        )
                        speaker.say(rec["spoken"])
                        lcd_write(lcd,
                            f"{rec.get('ball','?')} -> {rec.get('pocket','?')[:6]}",
                            f"Diff: {rec.get('difficulty','?')}/5"
                        )
                        print(f"[REC] {rec['spoken']} | {rec.get('reason','')}")
                    except Exception as e:
                        print(f"[LLM error] {e}")
                        lcd_write(lcd, "LLM error", "")

            # Show annotated frame
            if CFG.get("show_display", False):
                annotated = detector.draw(frame, balls)
                cv2.imshow("PoolCue Vision Assist", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.1)  # ~10fps — enough for static cardboard demo

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        lcd_write(lcd, "Goodbye!", "")


if __name__ == "__main__":
    main()
