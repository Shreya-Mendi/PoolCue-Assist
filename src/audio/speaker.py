"""
TTS speaker using Piper TTS — neural, natural-sounding female voice, fully offline.
Falls back to pico2wave then espeak if Piper is unavailable.
Speaks are non-blocking by default so the vision loop keeps running.
"""

import threading
import subprocess
import tempfile
import os
from pathlib import Path

PIPER_BIN   = Path.home() / "piper" / "piper"
PIPER_MODEL = Path.home() / "piper-voices" / "en_US-lessac-medium.onnx"


class Speaker:
    def __init__(self, rate=155, volume=1.0):
        self._lock = threading.Lock()
        self._speaking = False
        self._backend = self._detect_backend()
        print(f"[TTS] Using backend: {self._backend}")

    def _detect_backend(self) -> str:
        if PIPER_BIN.exists() and PIPER_MODEL.exists():
            return "piper"
        result = subprocess.run(["which", "pico2wave"], capture_output=True)
        if result.returncode == 0:
            return "pico2wave"
        return "espeak"

    def say(self, text: str, block=False):
        """Speak text. Non-blocking by default."""
        if block:
            self._speak(text)
        else:
            t = threading.Thread(target=self._speak, args=(text,), daemon=True)
            t.start()

    def _speak(self, text: str):
        with self._lock:
            self._speaking = True
            try:
                if self._backend == "piper":
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav = f.name
                    proc = subprocess.run(
                        [str(PIPER_BIN), "--model", str(PIPER_MODEL),
                         "--output_file", wav],
                        input=text.encode(),
                        capture_output=True
                    )
                    subprocess.run(["aplay", "-q", wav], capture_output=True)
                    os.unlink(wav)
                elif self._backend == "pico2wave":
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav = f.name
                    subprocess.run(
                        ["pico2wave", "-l", "en-US", "-w", wav, text],
                        capture_output=True
                    )
                    subprocess.run(["aplay", "-q", wav], capture_output=True)
                    os.unlink(wav)
                else:
                    subprocess.run(
                        ["espeak", text, "-v", "en-us", "-s", "145", "-a", "118"],
                        capture_output=True
                    )
            except Exception as e:
                print(f"[TTS error] {e}")
            finally:
                self._speaking = False

    def wait(self):
        """Block until current speech is done."""
        while self._speaking:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking
