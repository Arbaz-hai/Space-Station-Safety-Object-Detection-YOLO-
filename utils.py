"""
utils.py — Inference helpers for Space Station Safety Object Detection
"""

import os
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

# ── Class Configuration ──────────────────────────────────────────────────────
CLASS_NAMES = [
    "OxygenTank",
    "NitrogenTank",
    "FirstAidBox",
    "FireAlarm",
    "SafetySwitchPanel",
    "EmergencyPhone",
    "FireExtinguisher",
]

# Neon color palette for each class (BGR for OpenCV, RGB for display)
CLASS_COLORS_RGB = [
    (0,   210, 255),   # 0 OxygenTank        — cyan
    (180,  80, 255),   # 1 NitrogenTank       — purple
    (0,   255, 130),   # 2 FirstAidBox        — green
    (255,  60,  80),   # 3 FireAlarm          — red-neon
    (255, 170,   0),   # 4 SafetySwitchPanel  — amber
    (0,   160, 255),   # 5 EmergencyPhone     — blue
    (255,  80,   0),   # 6 FireExtinguisher   — orange
]

CLASS_ICONS = {
    "OxygenTank":        "🫧",
    "NitrogenTank":      "⚗️",
    "FirstAidBox":       "🩹",
    "FireAlarm":         "🚨",
    "SafetySwitchPanel": "⚡",
    "EmergencyPhone":    "📞",
    "FireExtinguisher":  "🧯",
}

CLASS_DESCRIPTIONS = {
    "OxygenTank":        "Pressurized oxygen supply for crew respiration.",
    "NitrogenTank":      "Inert nitrogen gas used in pressure systems.",
    "FirstAidBox":       "Emergency medical kit for crew injuries.",
    "FireAlarm":         "Smoke/heat sensor that triggers station alerts.",
    "SafetySwitchPanel": "Critical electrical isolation & override panel.",
    "EmergencyPhone":    "Direct line to mission control & emergency crew.",
    "FireExtinguisher":  "CO₂ suppressant for onboard fire emergencies.",
}

MODEL_PATH = "best.pt"


def get_model_path() -> str | None:
    """Return the path to best.pt if it exists, else None."""
    candidates = [
        MODEL_PATH,
        "models/best.pt",
        "weights/best.pt",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def load_model():
    """Load the YOLO model. Returns (model, error_message)."""
    try:
        from ultralytics import YOLO
        path = get_model_path()
        if path is None:
            return None, (
                "⚠️  Model weights not found.  "
                "Please upload `best.pt` to the Space root directory."
            )
        model = YOLO(path)
        return model, None
    except ImportError:
        return None, "❌  `ultralytics` is not installed."
    except Exception as e:
        return None, f"❌  Failed to load model: {e}"


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL (RGB) to OpenCV (BGR) ndarray."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV (BGR) to PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def draw_fancy_boxes(
    image: Image.Image,
    boxes,           # ultralytics Results.boxes
    conf_threshold: float = 0.25,
) -> Image.Image:
    """
    Draw neon bounding boxes with class labels on a PIL image.
    Returns the annotated PIL image.
    """
    cv_img = pil_to_cv2(image)
    h, w = cv_img.shape[:2]

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        if cls_id >= len(CLASS_NAMES):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = CLASS_NAMES[cls_id]
        color_rgb = CLASS_COLORS_RGB[cls_id]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Outer glow effect — draw thick semi-transparent rect
        overlay = cv_img.copy()
        cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color_bgr, 4)
        cv2.addWeighted(overlay, 0.35, cv_img, 0.65, 0, cv_img)

        # Main bounding box
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color_bgr, 2)

        # Corner accents
        corner_len = max(12, int(min(x2 - x1, y2 - y1) * 0.15))
        thick = 3
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(cv_img, (cx, cy), (cx + dx * corner_len, cy), color_bgr, thick)
            cv2.line(cv_img, (cx, cy), (cx, cy + dy * corner_len), color_bgr, thick)

        # Label background
        text = f"{label}  {conf:.0%}"
        font_scale = max(0.45, min(0.65, (x2 - x1) / 280))
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
        pad = 5
        label_y1 = max(y1 - th - 2 * pad, 0)
        label_y2 = y1
        label_x2 = min(x1 + tw + 2 * pad, w)

        # Semi-transparent filled label box
        label_bg = cv_img.copy()
        cv2.rectangle(label_bg, (x1, label_y1), (label_x2, label_y2), color_bgr, -1)
        cv2.addWeighted(label_bg, 0.75, cv_img, 0.25, 0, cv_img)

        # Text
        cv2.putText(
            cv_img, text,
            (x1 + pad, y1 - pad),
            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return cv2_to_pil(cv_img)


def run_inference(
    model,
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold:  float = 0.45,
) -> tuple[Image.Image, list[dict], float]:
    """
    Run YOLO detection on a PIL image.

    Returns:
        annotated_image  — PIL Image with drawn boxes
        detections       — list of {class, confidence, bbox}
        inference_ms     — inference time in ms
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    t0 = time.perf_counter()
    results = model.predict(
        source=np.array(image),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xyxy   = [round(v, 1) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class_id":   cls_id,
                    "class":      CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                    "confidence": round(conf * 100, 1),
                    "bbox":       xyxy,
                    "icon":       CLASS_ICONS.get(CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "", "🔍"),
                    "color":      CLASS_COLORS_RGB[cls_id] if cls_id < len(CLASS_COLORS_RGB) else (200, 200, 200),
                })
        annotated = draw_fancy_boxes(image, r.boxes, conf_threshold)
    else:
        annotated = image.copy()

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return annotated, detections, inference_ms


def build_detection_summary(detections: list[dict], inference_ms: float) -> str:
    """Build an HTML summary card for the detections panel."""
    if not detections:
        return """
<div class="no-detection-card">
  <div class="no-det-icon">🔭</div>
  <div class="no-det-text">No safety objects detected above threshold.</div>
  <div class="no-det-sub">Try lowering the confidence threshold or uploading a clearer image.</div>
</div>
"""
    # Count by class
    from collections import Counter
    counts = Counter(d["class"] for d in detections)
    total  = len(detections)

    rows = ""
    for d in detections:
        r, g, b = d["color"]
        conf_bar_w = int(d["confidence"])
        rows += f"""
<div class="det-row">
  <span class="det-icon">{d["icon"]}</span>
  <div class="det-info">
    <span class="det-label" style="color:rgb({r},{g},{b})">{d["class"]}</span>
    <div class="conf-bar-bg">
      <div class="conf-bar-fill" style="width:{conf_bar_w}%;background:rgb({r},{g},{b})"></div>
    </div>
  </div>
  <span class="det-conf" style="color:rgb({r},{g},{b})">{d["confidence"]}%</span>
</div>"""

    summary_pills = " ".join(
        f'<span class="pill">{icon} {cls} ×{cnt}</span>'
        for (cls, cnt), icon in zip(counts.items(), [CLASS_ICONS.get(c, "🔍") for c in counts])
    )

    return f"""
<div class="summary-header">
  <div class="summary-stat">
    <span class="stat-num">{total}</span>
    <span class="stat-lbl">Objects Found</span>
  </div>
  <div class="summary-stat">
    <span class="stat-num">{len(counts)}</span>
    <span class="stat-lbl">Unique Classes</span>
  </div>
  <div class="summary-stat">
    <span class="stat-num">{inference_ms:.0f}ms</span>
    <span class="stat-lbl">Inference Time</span>
  </div>
</div>
<div class="pill-row">{summary_pills}</div>
<div class="det-list">{rows}</div>
"""
