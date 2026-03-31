# Merging Billiard Ball Datasets on Roboflow

The goal is to combine multiple pool ball datasets into one large merged dataset,
then export it for YOLOv11 training in Colab.

---

## Recommended Datasets to Merge

| Dataset | URL | Images | Notes |
|---|---|---|---|
| Billiard Ball Detection v6 | https://universe.roboflow.com/billiard-ball-data-set/billiard-ball-detection-aeo1m/dataset/6 | ~129 | Individual ball numbers |
| Pool Ball Detection (Ben Gann) | https://universe.roboflow.com/ben-gann-lscqy/pool-ball-detection/dataset/2 | ~986 | Largest single source |
| 8 Ball Pool (skylep) | https://universe.roboflow.com/skylep/8-ball-pool-fmk6g/dataset/8 | varies | Good variety |
| Pool Bot V2 | https://universe.roboflow.com/siv-poolbot/pool_bot_v2 | varies | Overhead angle |

---

## How to Merge on Roboflow

1. **Create a free Roboflow account** at roboflow.com
2. **Create a new project**: New Project → Object Detection → name it `billiard-pool-merged`
3. **Fork each dataset above** into your workspace:
   - Open each dataset URL
   - Click **Fork** (top right) → select your workspace
4. **Upload into your merged project**:
   - In your merged project → Upload → select images + labels from each forked dataset
   - Repeat for each dataset
5. **Remap class names** (critical):
   - Go to Classes & Tags in your project
   - Make sure all datasets use the same label names: `cue`, `1`, `2`, ... `15`
   - Merge any aliases (e.g. `white` → `cue`, `eight` → `8`)
6. **Generate a version**:
   - Dataset → Generate New Version
   - Split: 80% train / 10% val / 10% test
   - Preprocessing: Auto-orient, Resize to 640×640
   - Augmentation: Flip (horizontal), Brightness ±25%, Blur (up to 1px)
7. **Copy your API key** from Settings → API Keys
8. **Paste into colab_train.ipynb** — `API_KEY`, `WORKSPACE`, `PROJECT`, `VERSION`

---

## Class Name Standardization

After merging, your `data.yaml` should have these classes in order:

```yaml
names:
  0: cue
  1: '1'
  2: '2'
  3: '3'
  4: '4'
  5: '5'
  6: '6'
  7: '7'
  8: '8'
  9: '9'
  10: '10'
  11: '11'
  12: '12'
  13: '13'
  14: '14'
  15: '15'
```

After training, run the last cell in colab_train.ipynb to print your actual class order,
then copy it into `src/vision/detector.py` BALL_CLASSES list.

---

## After Downloading best.pt

Place it at:
```
models/pool_vision/best.pt
```

The detector will find it automatically.
