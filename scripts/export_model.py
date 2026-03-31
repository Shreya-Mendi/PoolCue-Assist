"""
Export trained YOLOv11 model to ONNX for faster CPU inference on Raspberry Pi.
Run this on your laptop after downloading best.pt from Colab.

Usage:
    python3 scripts/export_model.py
"""

from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parents[1] / "models" / "pool_vision" / "best.pt"
ONNX_PATH  = MODEL_PATH.with_suffix(".onnx")


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Train the model first using train/colab_train.ipynb")
        return

    print(f"Loading {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    print("Exporting to ONNX...")
    model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    print(f"\nExported: {ONNX_PATH}")
    print("\nClass names in this model:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")
    print("\nCopy the class list above into src/vision/detector.py if needed.")
    print("On Pi, the ONNX model will run faster than best.pt for CPU inference.")


if __name__ == "__main__":
    main()
