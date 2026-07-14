---
title: Space Station Safety Objects Detection
sdk: gradio
python_version: 3.12.12
app_file: app.py
pinned: true
license: mit
tags:
- object-detection
- yolov8
- safety
- space-station
- duality-ai
- hackathon
- computer-vision
- real-time
sdk_version: 6.12.0
---
## Space Station
![Space Station](https://github.com/Arbaz-hai/Space-Station-Safety-Object-Detection-YOLO-/blob/af41bd49c401ca261b22200b240f237c34c141cf/Space%20Station.png)

#  Space Station Safety Object Detection

> **Duality AI Hackathon 2026 **
> Real-time detection of 7 critical safety objects in space station environments using YOLOv8m trained on synthetic digital twin data — with live webcam inference at **≤ 2 second latency**.

---

## 📌 Problem Statement

Submission for the **Duality AI Space Station Challenge: Safety Object Detection #2**.
The goal is to train a robust object detection model that identifies **7 vital safety objects** under:

- Varied lighting conditions (dark, light, cluttered, hallway)
- Object occlusions and overlaps
- Diverse camera angles and distances

Data was generated using **Falcon**, Duality AI's photorealistic digital twin simulation platform.

---

## 🎯 Detectable Classes

| # | Class | Icon | Description |
|---|-------|------|-------------|
| 0 | OxygenTank | 🫧 | Pressurized crew oxygen supply |
| 1 | NitrogenTank | ⚗️ | Inert nitrogen pressure systems |
| 2 | FirstAidBox | 🩹 | Emergency medical kit |
| 3 | FireAlarm | 🚨 | Smoke/heat detection sensor |
| 4 | SafetySwitchPanel | ⚡ | Electrical isolation panel |
| 5 | EmergencyPhone | 📞 | Direct mission control line |
| 6 | FireExtinguisher | 🧯 | CO₂ suppression unit |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **mAP@0.5** | **82.14%** |
| mAP@0.5:0.95 | 63.73% |
| Precision | 95.1% |
| Recall | 81.4% |
| F1 Score | 82.9% |
| Inference Speed (GPU, 480 px) | ~180 ms |
| Inference Speed (CPU, 480 px) | ~900 ms |

### Per-Class AP@0.5

| Class | AP@0.5 |
|-------|--------|
| FireExtinguisher | 87.9% |
| OxygenTank | 84.4% |
| NitrogenTank | 84.3% |
| FirstAidBox | 82.8% |
| SafetySwitchPanel | 81.3% |
| FireAlarm | 79.8% |
| EmergencyPhone | 74.4% |

---

## Real-Time Latency

This app is optimised for live webcam detection at **≤ 2 seconds end-to-end latency**.

| Mode | Inference Size | Device | Inference | End-to-End |
|------|---------------|--------|-----------|------------|
| Live stream | 480 × 480 | GPU (T4) | ~180 ms | **~700 ms** |
| Live stream | 480 × 480 | CPU | ~900 ms | **~1.4 s** |
| Upload / Capture | 640 × 640 | GPU | ~300 ms | on-demand |
| Upload / Capture | 640 × 640 | CPU | ~2.5 s | on-demand |

### Speed Optimisations Applied

- **480 px live inference** — 3.5× fewer pixels than original 896 px training size
- **FP16 half-precision** — enabled automatically on any CUDA GPU (~2× speedup)
- **Model warmup** — 2 silent passes on startup so CUDA kernels are compiled before first request
- **Fast draw mode** — glow overlay skipped in live mode (~15 ms saved per frame)
- **500 ms stream interval** — prevents frame backlog in Gradio streaming queue
- **Lightweight live badge** — replaces heavy per-box HTML summary cards in live mode

---

## 🧠 Model Architecture

- **Architecture**: YOLOv8m (medium)
- **Parameters**: 25.8M
- **Input Size**: 896 × 896 (training) / 640 px (upload) / 480 px (live)
- **Training Epochs**: 180
- **Optimizer**: AdamW (lr=0.0005, cosine annealing)
- **GPU**: Tesla T4 (15.6 GB VRAM)
- **Training Time**: ~6.9 hours

### Key Training Settings

```yaml
epochs: 180
batch: auto (3)         # AutoBatch @ ~50% VRAM
imgsz: 896
optimizer: AdamW
lr0: 0.0005
cos_lr: true
warmup_epochs: 4
mosaic: 1.0
mixup: 0.2
copy_paste: 0.2
hsv_v: 0.3             # Brightness jitter for lighting variation
hsv_s: 0.5
flipud: 0.3
degrees: 15.0
shear: 2.0
amp: true              # Mixed precision (fp16)
dropout: 0.1
patience: 40
```

---

## 📁 Folder Structure

```
space-station-safety-detection/
├── app.py                   # Main Gradio application
├── utils.py                 # Inference helpers, drawing, latency optimisations
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── deploy_check.py          # Pre-deploy validator + latency benchmark
├── best.pt                  # ← Place your trained model here (52 MB)
└── examples/                # Optional sample images
    ├── sample1.jpg
    ├── sample2.jpg
    └── sample3.jpg
```

> ⚠️ **`best.pt` is NOT included due to file size (52 MB). See the Model Weights section below.**

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://huggingface.co/spaces/ArbazDevHive/Space_Station_Safety_Object_Detection
cd Space_Station_Safety_Object_Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your model weights

```bash
cp /path/to/your/best.pt ./best.pt
```

### 5. Validate before running

```bash
python deploy_check.py
```

This checks all required files, imports, and runs a **latency benchmark** to confirm live-mode is within the 2 s target on your hardware.

### 6. Run the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## ☁️ Deploying on Hugging Face Spaces

### Option A — Direct Upload (recommended for large model files)

1. Go to [huggingface.co/new-space](https://huggingface.co/spaces/ArbazDevHive/Space_Station_Safety_Object_Detection)
2. Choose **Gradio** as the SDK
3. Upload all files via the web UI:
   - `app.py`
   - `utils.py`
   - `requirements.txt`
   - `README.md`
   - `deploy_check.py`
   - `best.pt` ← **Upload this separately via the Files tab**
4. The Space will auto-build and launch.

### Option B — Git LFS (for best.pt)

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add .
git commit -m "Initial commit: Space Station Safety Detection"
git push
```

### Option C — Load from Hugging Face Hub

If your model is hosted on the Hub, replace `load_model()` in `utils.py`:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="your-username/your-model-repo", filename="best.pt")
model = YOLO(path)
```

### Hardware Recommendation

For live detection within 1 s, use a **GPU-enabled Space** (T4 Small or better).
CPU Spaces will still hit the ≤2 s target but with less headroom.

---

## 🖥️ App Tabs

| Tab | Description |
|-----|-------------|
| 🎯 Detection Console | Upload an image or paste from clipboard. Full 640 px quality. |
| 📸 Webcam Capture | Snap a single frame from your webcam, then scan. |
| 🔴 Live Detection | Continuous real-time webcam stream at ≤2 s latency. |
| 🧠 Model Info | Architecture details, class cards, per-class AP scores. |
| 📖 How It Works | Pipeline walkthrough and latency breakdown table. |

---

## ⚙️ Expected Output

When you upload an image or stream webcam and trigger detection:

1. **Annotated Image** — Original image with neon bounding boxes, corner accents, and confidence labels drawn for each detected object.

2. **Detection Summary** — Shows:
   - Total objects found
   - Number of unique classes
   - Inference time (ms)
   - Per-detection row with icon, class name, confidence bar, and score

3. **Live Badge** (live mode only) — Lightweight pill-style summary showing class counts and inference time. Updated every frame without blocking the stream.

---

## 🔬 Dataset Details

Generated using **Duality AI Falcon** digital twin simulator:

| Split | Images | Labels |
|-------|--------|--------|
| Train | 1,769 | 1,769 |
| Val   | 338   | 338   |
| Test  | 1,408 | 1,408 |

### Class Distribution (Training Set)

| Class | Instances |
|-------|-----------|
| NitrogenTank | 1,553 |
| OxygenTank | 1,422 |
| FireExtinguisher | 766 |
| FirstAidBox | 705 |
| EmergencyPhone | 422 |
| SafetySwitchPanel | 410 |
| FireAlarm | 323 |

Scenarios include: `light_uncluttered`, `cluttered_room`, `cluttered_hallway`, `dark_unclutter`, `dark_clutter`, `light_cluttered`

---

## 📝 Notes

- The model was trained **exclusively** on the provided dataset. No test images were used during training (as per hackathon rules).
- `EmergencyPhone` has the lowest AP (74.4%) likely due to fewer training samples (422 instances) and its small, wall-mounted form factor.
- For best upload/webcam quality, use images at 640 px or higher resolution.
- Live mode intentionally uses 480 px inference to guarantee the ≤2 s latency target.
- The app supports upload, clipboard paste, webcam capture, and live streaming.

---

## Test Result
![Sample 1](https://github.com/Arbaz-hai/Space-Station-Safety-Object-Detection-YOLO-/blob/be7606632deb7c77fcd1443b91722659e324c09f/test_predictions_grid%20(1).png)

## 🤝 Credits

- **Data Platform**: [Duality AI Falcon](https://falcon.duality.ai)
- **Model Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Demo Framework**: [Gradio](https://gradio.app)
- **Hackathon**: Duality AI Space Station Challenge #2 — Gurugram 2026

---

*Built with ❤️ for the Duality AI Hackathon · Gurugram 2026*
