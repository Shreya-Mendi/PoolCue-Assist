#!/bin/bash
# Run once on the Pi to set up auto-boot.
# Usage: bash scripts/install.sh

set -e

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-smbus espeak libespeak-dev

echo "Installing Python packages..."
pip3 install -r /home/pi/PoolCue-Assist/requirements.txt

echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

echo "Setting audio output to 3.5mm jack..."
sudo raspi-config nonint do_audio 1

echo "Installing systemd service..."
sudo cp /home/pi/PoolCue-Assist/scripts/poolassist.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable poolassist

echo ""
echo "Done. Edit ANTHROPIC_API_KEY in /etc/systemd/system/poolassist.service"
echo "Then run: sudo systemctl start poolassist"
echo "On next reboot the program will start automatically."
