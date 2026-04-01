#!/usr/bin/env python3
"""
Live camera feed over HTTP (MJPEG).
Open http://10.194.64.167:8080 in your laptop browser.
"""
import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from picamera2 import Picamera2
import cv2
import numpy as np

PORT = 8080
frame_lock = threading.Lock()
latest_frame = None

def capture_loop():
    global latest_frame
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("Camera started")
    while True:
        frame = picam2.capture_array()
        # picamera2 returns RGB, convert to BGR for cv2 JPEG encode
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, jpg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = jpg.tobytes()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
<html><body style="margin:0;background:#000">
<h2 style="color:#fff;font-family:sans-serif;padding:8px">PoolCue Camera Feed</h2>
<img src="/stream" style="width:100%;max-width:800px;display:block;margin:auto">
</body></html>""")
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        data = latest_frame
                    if data:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(data)
                        self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print(f"Live feed at http://10.194.64.167:{PORT}")
    print("Open that URL in your laptop browser. Ctrl+C to stop.")
    HTTPServer(('0.0.0.0', PORT), StreamHandler).serve_forever()
