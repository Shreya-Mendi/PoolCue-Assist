"""
TTS speaker wrapper using pyttsx3 (offline, works on Pi via espeak backend).
Speaks are non-blocking by default so the vision loop keeps running.
Call speaker.wait() before starting stroke capture if you need to sync.
"""

import threading
import pyttsx3


class Speaker:
    def __init__(self, rate=155, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self._lock = threading.Lock()
        self._speaking = False

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
            self.engine.say(text)
            self.engine.runAndWait()
            self._speaking = False

    def wait(self):
        """Block until current speech is done."""
        while self._speaking:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking
