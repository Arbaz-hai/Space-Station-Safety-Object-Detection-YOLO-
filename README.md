
# 🚀 Space Safety Object Detection using YOLO

## 📌 Project Description
Space Safety Object Detection is an AI-based project that uses YOLO (You Only Look Once) deep learning model to detect objects in space such as satellites, rockets, asteroids, and space debris.

The system can detect objects from images, videos, and live webcam feed in real-time, helping improve space safety and monitoring systems.

---

## 🎯 Objectives
- Detect space-related objects using YOLOv8
- Perform real-time object detection
- Improve space safety monitoring
- Demonstrate AI applications in aerospace
- Deploy model locally or on cloud

---

## 🧠 Model Information
- Model: YOLOv8 (Ultralytics)
- Framework: PyTorch
- Task: Object Detection

---

## 📂 Dataset Structure
dataset/
 ├── train/
 │    ├── images
 │    ├── labels
 │
 ├── valid/
 │    ├── images
 │    ├── labels

Annotation format:
class x_center y_center width height

---

## ⚙️ Installation

pip install ultralytics opencv-python torch torchvision numpy

---

## ▶️ Usage

### Train Model
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)

### Image Prediction
model.predict("test.jpg", show=True)

### Video Prediction
model.predict("video.mp4", show=True)

### Live Webcam
model.predict(source=0, show=True)

---

## 📊 Applications
- Space safety monitoring
- Satellite tracking
- Space debris detection
- Aerospace AI research

---

## 🛠️ Technologies
Python, YOLOv8, PyTorch, OpenCV, NumPy

---

## 👨‍💻 Author
Arbaz
B.Tech CSE
Greater Noida Institute of Technology (GNIT)
Guru Gobind Singh Indraprastha University

---

⭐ Give star on GitHub if helpful!
