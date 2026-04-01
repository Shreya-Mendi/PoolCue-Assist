"""
Copy N random images from dataset/test/images → data/sample_test_images/
so they can be committed to git and used with --test-image on the Pi.

Usage:
  python scripts/sample_test_images.py          # default 6 images
  python scripts/sample_test_images.py --n 10
"""

import argparse
import random
import shutil
from pathlib import Path

SRC = Path(__file__).parents[1] / "dataset" / "test" / "images"
DST = Path(__file__).parents[1] / "data" / "sample_test_images"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    images = sorted(SRC.glob("*.jpg")) + sorted(SRC.glob("*.png"))
    if not images:
        print(f"No images found in {SRC}")
        return

    random.seed(args.seed)
    chosen = random.sample(images, min(args.n, len(images)))

    DST.mkdir(parents=True, exist_ok=True)
    for img in chosen:
        shutil.copy(img, DST / img.name)
        print(f"  copied {img.name}")

    print(f"\nSampled {len(chosen)} images → {DST}")
    print("Run one with:")
    print(f"  python src/main.py --test-image data/sample_test_images/{chosen[0].name}")


if __name__ == "__main__":
    main()
