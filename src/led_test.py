# led_test.py
import RPi.GPIO as GPIO
import time
GREEN = 17
RED = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN, GPIO.OUT)
GPIO.setup(RED, GPIO.OUT)
try:
    for i in range(3):
        GPIO.output(GREEN, GPIO.HIGH); time.sleep(0.2); GPIO.output(GREEN, GPIO.LOW)
        GPIO.output(RED, GPIO.HIGH); time.sleep(0.2); GPIO.output(RED, GPIO.LOW)
finally:
    GPIO.cleanup()