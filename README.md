# DEXI Hand Gesture Model Training

Tools for building a dataset, extracting hand landmarks, and training a lightweight gesture classifier (MediaPipe landmarks + scikit-learn Random Forest). The resulting model file is designed to be loaded by the [dexi_hand_gesture](https://github.com/DroneBlocks/dexi_hand_gesture) node running on board any DEXI model.

## Pipeline Overview

```
 1. Capture          2. Annotate              3. Train               4. Deploy
 webcam images  ->   landmark JSON files  ->  gesture_classifier  ->  copy .joblib to
 + augmentation      (MediaPipe)              .joblib bundle         the drone
```

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Capture | `dataset_builder.py` | Webcam | `data/built_dataset/<gesture>/*.jpg` |
| 2. Annotate | `create_annotations.py` | `data/built_dataset/` | `data/annotations/<gesture>.json` |
| 3. Train | `train_model.py` | `data/annotations/` | `data/gesture_classifier.joblib` |

## Setup

### Requirements

- Python **3.11+** (the scripts use the built-in `tomllib`)
- A webcam (for step 1)

```bash
pip install -r requirements.txt
```

### Directory layout

```
repo/
├── scripts/
│   ├── config.toml
│   ├── dataset_builder.py
│   ├── create_annotations.py
│   └── train_model.py
└── data/
    ├── internal/
    │   └── hand_landmarker.task      # MediaPipe model (already in the repo)
    ├── built_dataset/                # created by step 1
    ├── annotations/                  # created by step 2
    ├── landmarks_cache.npz           # created by step 3
    └── gesture_classifier.joblib     # final output
```

### Config

`scripts/config.toml` lists which gestures require **two hands**. Everything not listed is treated as a single-hand gesture.

```toml
multi_hand_gestures = ["example_two_hand_gesture"]
```

Names must match your gesture folder names exactly. `annotate.py` and `train.py` both read this file, and they open it by relative path, so **run them from inside `scripts/`**.

## Step 1: Capture Data

```bash
cd scripts
python dataset_builder.py
```

Enter a gesture name when prompted. Images are saved to `data/built_dataset/<gesture>/`. Each capture saves the cropped hand image plus several randomly augmented variants (flip, brightness, contrast, rotation, hue/saturation, noise, blur).

| Key | Action |
|-----|--------|
| `s` | Save a single capture |
| `r` | Start/stop auto-record (3 s countdown, then a burst of captures) |
| `z` | Undo the last capture (and its augmentations) |
| `n` | Switch to a new gesture name |
| `m` | Toggle one-hand / two-hand crop mode |
| `q` | Quit |

Tips:

- For two-hand gestures, press `m` so the crop covers both hands. Captures only save when both hands are detected.
- Record a `no_gesture` class (relaxed hands, random poses, transitions). Without it the classifier has nothing to predict when no gesture is being made.
- Vary lighting, distance, background, and people where you can.
- Capture roughly the same number of images for each gesture to keep the classes balanced.

## Step 2: Annotate (Extract Landmarks)

```bash
python create_annotations.py
```

Runs MediaPipe over every image in `data/built_dataset/` and writes one JSON file per gesture to `data/annotations/`. Each entry contains the 21 hand landmarks, bounding boxes, the gesture label, and handedness. Images are skipped (and counted in the output) if no hand is found or too few hands are detected for that gesture.

## Step 3: Train

```bash
python train_model.py
```

1. Loads all annotation JSON files (or `data/landmarks_cache.npz` if it exists).
2. Normalizes landmarks:
   - **Single hand:** translated to the wrist and scaled by wrist-to-middle-finger-base distance, so features are position and size invariant.
   - **Two hands:** centered on the midpoint between wrists and scaled by the distance between them.
3. Trains two Random Forests (300 trees): a **single-hand** model and a **two-hand** model. The two-hand model includes a `none` class built from any frame where two hands were visible but no multi-hand gesture was performed.
4. Prints a classification report on a held-out 15% test split.
5. Saves both models to `data/gesture_classifier.joblib`.

**Important:** the landmark cache is not invalidated automatically. If you add or change data, **delete `data/landmarks_cache.npz`** before retraining.

Tunable values (trees, depth, test split, per-class cap, whether to include `no_gesture`) are at the top of `train.py`.

## Step 4: Deploy

The output is `data/gesture_classifier.joblib`, a dictionary bundle:

```python
{
    "single": <RandomForestClassifier or None>,
    "multi": <RandomForestClassifier or None>,
    "multi_hand_gestures": [...],
    "no_multi_label": "none",
}
```

To use it on the drone:

Copy `gesture_classifier.joblib` to your DEXI (e.g. via `scp` or dragging and dropping through VSCode) into the `~/dexi_ws/dexi_hand_gesture/data` directory.


## Known Caveats

- **Optimistic test scores.** Each capture produces an original plus several augmented copies, and train/test splitting is random per image, so near-duplicates of the same capture can land in both sets. Real-world accuracy will usually be lower than the report suggests. Test with live data before trusting it.
- **Windows-specific camera setup.** `capture.py` opens the camera with `cv2.CAP_DSHOW`. On Linux/macOS, remove that argument (and the `MJPG` FOURCC line if it causes issues).
- **Single-hand gestures with two hands in frame.** `annotate.py` labels every detected hand in an image with that image's gesture. Keep only the performing hand in frame when capturing single-hand gestures.
- **Image-flip augmentation** mirrors the hand, so gestures that depend on left/right handedness may be affected.