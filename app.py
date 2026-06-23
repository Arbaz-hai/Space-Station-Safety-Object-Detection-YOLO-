import gradio as gr
from PIL import Image
import traceback
import os
import datetime

from utils import (
    CLASS_NAMES, CLASS_COLORS_RGB, CLASS_ICONS, CLASS_DESCRIPTIONS,
    load_model, run_inference, build_detection_summary,
)

_MODEL, _MODEL_ERROR = load_model()

SAVE_DIR = "saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)

CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════════════════
   GLOBAL RESET & BASE
═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Exo+2:wght@300;400;500;600&display=swap');
:root {
  --bg-deep:      #020818;
  --bg-card:      rgba(6, 20, 50, 0.85);
  --bg-card2:     rgba(10, 30, 70, 0.7);
  --border:       rgba(0, 180, 255, 0.18);
  --border-glow:  rgba(0, 200, 255, 0.45);
  --cyan:         #00d4ff;
  --purple:       #a855f7;
  --green:        #10b981;
  --amber:        #f59e0b;
  --red:          #ef4444;
  --text-primary: #e2f0ff;
  --text-muted:   #7090b0;
  --font-head:    'Orbitron', monospace;
  --font-body:    'Exo 2', sans-serif;
  --radius:       14px;
  --radius-sm:    8px;
  --glow-cyan:    0 0 20px rgba(0,212,255,0.3), 0 0 60px rgba(0,212,255,0.1);
  --glow-purple:  0 0 20px rgba(168,85,247,0.3);
}
* { box-sizing: border-box; }
body, .gradio-container {
  background: var(--bg-deep) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  min-height: 100vh;
}
/* Starfield background */
.gradio-container::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(0,100,200,0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(120,40,200,0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 60% 80%, rgba(0,200,150,0.05) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}
/* ═══════════════════════════════════════════════════════════════
   HEADER BANNER
═══════════════════════════════════════════════════════════════ */
#header-banner {
  background: linear-gradient(135deg,
    rgba(0,10,30,0.98) 0%,
    rgba(5,20,60,0.98) 50%,
    rgba(0,10,30,0.98) 100%);
  border: 1px solid var(--border-glow);
  border-radius: var(--radius);
  padding: 32px 40px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--glow-cyan), inset 0 1px 0 rgba(255,255,255,0.05);
}
#header-banner::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -10%;
  width: 40%;
  height: 200%;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.04), transparent);
  transform: skewX(-15deg);
  animation: sweep 6s infinite linear;
}
@keyframes sweep {
  0%   { left: -40%; }
  100% { left: 110%; }
}
.header-inner {
  display: flex;
  align-items: center;
  gap: 24px;
  position: relative;
  z-index: 1;
}
.header-badge {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  flex-shrink: 0;
  box-shadow: var(--glow-cyan);
}
.header-text h1 {
  font-family: var(--font-head) !important;
  font-size: 1.6rem !important;
  font-weight: 700 !important;
  color: var(--cyan) !important;
  letter-spacing: 0.05em !important;
  margin: 0 0 6px 0 !important;
  text-shadow: 0 0 30px rgba(0,212,255,0.5);
}
.header-text p {
  color: var(--text-muted) !important;
  font-size: 0.9rem !important;
  margin: 0 !important;
  line-height: 1.5 !important;
}
.header-tags {
  display: flex;
  gap: 8px;
  margin-top: 10px;
  flex-wrap: wrap;
}
.htag {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-family: var(--font-head);
  letter-spacing: 0.04em;
  border: 1px solid;
}
.htag-cyan   { color: var(--cyan);   border-color: var(--cyan);   background: rgba(0,212,255,0.08); }
.htag-purple { color: var(--purple); border-color: var(--purple); background: rgba(168,85,247,0.08); }
.htag-green  { color: var(--green);  border-color: var(--green);  background: rgba(16,185,129,0.08); }
.htag-amber  { color: var(--amber);  border-color: var(--amber);  background: rgba(245,158,11,0.08); }
.header-stats {
  margin-left: auto;
  display: flex;
  gap: 20px;
  flex-shrink: 0;
}
.hstat { text-align: center; }
.hstat-num {
  display: block;
  font-family: var(--font-head);
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--cyan);
  text-shadow: var(--glow-cyan);
}
.hstat-lbl {
  display: block;
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
/* ═══════════════════════════════════════════════════════════════
   GLASS CARD
═══════════════════════════════════════════════════════════════ */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  padding: 24px;
  margin-bottom: 20px;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
  border-color: rgba(0,180,255,0.35);
  box-shadow: 0 8px 32px rgba(0,100,200,0.15);
}
.section-title {
  font-family: var(--font-head) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--cyan) !important;
  margin: 0 0 16px 0 !important;
  padding-bottom: 10px !important;
  border-bottom: 1px solid var(--border) !important;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-title::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 14px;
  background: var(--cyan);
  border-radius: 2px;
  box-shadow: 0 0 8px var(--cyan);
}
/* ═══════════════════════════════════════════════════════════════
   INPUT & OUTPUT PANELS
═══════════════════════════════════════════════════════════════ */
.upload-zone label,
.gradio-image label {
  font-family: var(--font-head) !important;
  font-size: 0.75rem !important;
  color: var(--cyan) !important;
  letter-spacing: 0.08em !important;
}
.gradio-image,
.gradio-image > div {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: rgba(0,10,30,0.6) !important;
  overflow: hidden !important;
}
.gradio-image:hover {
  border-color: var(--border-glow) !important;
}
/* ═══════════════════════════════════════════════════════════════
   SLIDERS / CONTROLS
═══════════════════════════════════════════════════════════════ */
input[type=range] { accent-color: var(--cyan) !important; }
.gradio-slider label,
.gradio-number label {
  font-family: var(--font-body) !important;
  font-size: 0.82rem !important;
  color: var(--text-primary) !important;
}
/* ═══════════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════════ */
#detect-btn {
  background: linear-gradient(135deg, #0080cc, #6020cc) !important;
  border: 1px solid var(--cyan) !important;
  border-radius: var(--radius-sm) !important;
  color: white !important;
  font-family: var(--font-head) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  padding: 14px 28px !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: var(--glow-cyan) !important;
  text-transform: uppercase !important;
  width: 100% !important;
}
#detect-btn:hover {
  background: linear-gradient(135deg, #00a0ff, #8040ee) !important;
  box-shadow: 0 0 30px rgba(0,212,255,0.5), 0 0 60px rgba(0,212,255,0.2) !important;
  transform: translateY(-1px) !important;
}
#clear-btn {
  background: transparent !important;
  border: 1px solid rgba(100,150,200,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-muted) !important;
  font-family: var(--font-body) !important;
  font-size: 0.8rem !important;
  padding: 10px 20px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  width: 100% !important;
}
#clear-btn:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}
#save-btn {
  background: linear-gradient(135deg, #065f46, #064e3b) !important;
  border: 1px solid var(--green) !important;
  border-radius: var(--radius-sm) !important;
  color: white !important;
  font-family: var(--font-head) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  padding: 12px 20px !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 0 14px rgba(16,185,129,0.25) !important;
  text-transform: uppercase !important;
  width: 100% !important;
}
#save-btn:hover {
  background: linear-gradient(135deg, #059669, #047857) !important;
  box-shadow: 0 0 28px rgba(16,185,129,0.45) !important;
  transform: translateY(-1px) !important;
}
/* Live indicator pulse */
@keyframes livepulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}
.live-dot {
  display: inline-block;
  width: 8px; height: 8px;
  background: var(--red);
  border-radius: 50%;
  margin-right: 6px;
  animation: livepulse 1s infinite;
  vertical-align: middle;
}
/* ═══════════════════════════════════════════════════════════════
   DETECTION SUMMARY HTML OUTPUT
═══════════════════════════════════════════════════════════════ */
.summary-header {
  display: flex;
  gap: 16px;
  padding: 16px;
  background: rgba(0,50,100,0.3);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  margin-bottom: 14px;
  justify-content: space-around;
}
.summary-stat { text-align: center; }
.stat-num {
  display: block;
  font-family: var(--font-head);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--cyan);
  text-shadow: 0 0 10px rgba(0,212,255,0.4);
}
.stat-lbl {
  display: block;
  font-size: 0.68rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 2px;
}
.pill-row {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;
}
.pill {
  padding: 4px 12px;
  border-radius: 20px;
  background: rgba(0,100,200,0.2);
  border: 1px solid rgba(0,180,255,0.2);
  font-size: 0.75rem;
  color: var(--cyan);
  white-space: nowrap;
}
.det-list { display: flex; flex-direction: column; gap: 10px; }
.det-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: var(--bg-card2);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  transition: border-color 0.2s;
}
.det-row:hover { border-color: rgba(0,200,255,0.3); }
.det-icon { font-size: 1.2rem; flex-shrink: 0; }
.det-info { flex: 1; }
.det-label {
  display: block;
  font-family: var(--font-head);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  margin-bottom: 4px;
}
.conf-bar-bg {
  height: 4px;
  background: rgba(255,255,255,0.08);
  border-radius: 2px;
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease;
  box-shadow: 0 0 6px currentColor;
}
.det-conf {
  font-family: var(--font-head);
  font-size: 0.85rem;
  font-weight: 700;
  flex-shrink: 0;
  min-width: 50px;
  text-align: right;
}
.no-detection-card {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-muted);
}
.no-det-icon { font-size: 2.5rem; margin-bottom: 12px; }
.no-det-text {
  font-family: var(--font-head);
  font-size: 0.85rem;
  color: var(--text-primary);
  margin-bottom: 6px;
}
.no-det-sub { font-size: 0.78rem; }
/* ═══════════════════════════════════════════════════════════════
   SAVE STATUS
═══════════════════════════════════════════════════════════════ */
.save-success {
  background: rgba(16,185,129,0.1);
  border: 1px solid rgba(16,185,129,0.35);
  border-radius: var(--radius-sm);
  padding: 12px 16px;
  color: #6ee7b7;
  font-size: 0.82rem;
  margin-top: 10px;
  font-family: var(--font-head);
  letter-spacing: 0.04em;
}
.save-error {
  background: rgba(239,68,68,0.1);
  border: 1px solid rgba(239,68,68,0.35);
  border-radius: var(--radius-sm);
  padding: 12px 16px;
  color: #fca5a5;
  font-size: 0.82rem;
  margin-top: 10px;
}
/* ═══════════════════════════════════════════════════════════════
   MODEL INFO PANEL
═══════════════════════════════════════════════════════════════ */
.model-info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}
.mi-item {
  padding: 12px 16px;
  background: rgba(0,30,70,0.5);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}
.mi-key {
  display: block;
  font-size: 0.68rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 4px;
}
.mi-val {
  display: block;
  font-family: var(--font-head);
  font-size: 0.82rem;
  color: var(--cyan);
  font-weight: 600;
}
.class-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
}
.class-badge {
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  border: 1px solid;
  font-size: 0.72rem;
  font-family: var(--font-head);
  display: flex;
  align-items: center;
  gap: 6px;
  letter-spacing: 0.04em;
}
/* ═══════════════════════════════════════════════════════════════
   HOW IT WORKS
═══════════════════════════════════════════════════════════════ */
.how-steps { display: flex; gap: 12px; flex-wrap: wrap; }
.how-step {
  flex: 1;
  min-width: 140px;
  padding: 16px;
  background: rgba(0,30,70,0.5);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  text-align: center;
  position: relative;
}
.step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px; height: 28px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  font-family: var(--font-head);
  font-size: 0.72rem;
  font-weight: 700;
  color: white;
  margin-bottom: 10px;
}
.step-icon { font-size: 1.5rem; display: block; margin-bottom: 6px; }
.step-title {
  font-family: var(--font-head);
  font-size: 0.72rem;
  color: var(--cyan);
  letter-spacing: 0.06em;
  margin-bottom: 4px;
  display: block;
}
.step-desc { font-size: 0.72rem; color: var(--text-muted); line-height: 1.4; }
/* ═══════════════════════════════════════════════════════════════
   PERFORMANCE METRICS
═══════════════════════════════════════════════════════════════ */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.metric-card {
  padding: 14px;
  background: rgba(0,30,60,0.6);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  text-align: center;
}
.metric-val {
  display: block;
  font-family: var(--font-head);
  font-size: 1.3rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.metric-lbl {
  display: block;
  font-size: 0.65rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 4px;
}
/* ═══════════════════════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════════════════════ */
#footer {
  border-top: 1px solid var(--border);
  padding: 24px 0;
  margin-top: 32px;
  text-align: center;
}
#footer p {
  color: var(--text-muted) !important;
  font-size: 0.78rem !important;
  margin: 4px 0 !important;
  line-height: 1.6 !important;
}
#footer strong { color: var(--cyan) !important; }
.footer-badges {
  display: flex; gap: 10px; justify-content: center;
  margin-top: 12px; flex-wrap: wrap;
}
.fbadge {
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-family: var(--font-head);
  letter-spacing: 0.06em;
}
.fbadge-cyan   { background: rgba(0,212,255,0.1);   border: 1px solid rgba(0,212,255,0.3);   color: var(--cyan);   }
.fbadge-purple { background: rgba(168,85,247,0.1);  border: 1px solid rgba(168,85,247,0.3);  color: var(--purple); }
.fbadge-green  { background: rgba(16,185,129,0.1);  border: 1px solid rgba(16,185,129,0.3);  color: var(--green);  }
/* ═══════════════════════════════════════════════════════════════
   ERROR / WARNING
═══════════════════════════════════════════════════════════════ */
.model-error {
  background: rgba(239,68,68,0.1);
  border: 1px solid rgba(239,68,68,0.35);
  border-radius: var(--radius-sm);
  padding: 14px 18px;
  color: #fca5a5;
  font-size: 0.82rem;
  line-height: 1.5;
}
/* ═══════════════════════════════════════════════════════════════
   GRADIO OVERRIDES
═══════════════════════════════════════════════════════════════ */
.gradio-container .prose { color: var(--text-primary) !important; }
.gradio-container .tab-nav { border-bottom: 1px solid var(--border) !important; }
.gradio-container .tab-nav button {
  font-family: var(--font-head) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.06em !important;
  color: var(--text-muted) !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}
.gradio-container .tabitem { padding: 20px 0 !important; }
footer { display: none !important; }
"""


# ── HTML Components ───────────────────────────────────────────────────────────
def make_header_html() -> str:
    return """
<div id="header-banner">
  <div class="header-inner">
    <div class="header-badge"></div>
    <div class="header-text">
      <h1>Space Station Safety Detection</h1>
      <p>AI-powered safety equipment recognition · YOLOv8m · Synthetic Digital Twin Data · Duality AI</p>
      <div class="header-tags">
        <span class="htag htag-cyan">YOLOv8m</span>
        <span class="htag htag-purple">7 Classes</span>
        <span class="htag htag-green">mAP@0.5: 82.1%</span>
        <span class="htag htag-amber">Hackathon 2026</span>
      </div>
    </div>
    <div class="header-stats">
      <div class="hstat">
        <span class="hstat-num">82.1%</span>
        <span class="hstat-lbl">mAP@0.5</span>
      </div>
      <div class="hstat">
        <span class="hstat-num">63.7%</span>
        <span class="hstat-lbl">mAP@0.5:95</span>
      </div>
      <div class="hstat">
        <span class="hstat-num">6.3ms</span>
        <span class="hstat-lbl">Inference</span>
      </div>
    </div>
  </div>
</div>
"""


def make_model_info_html() -> str:
    classes_html = ""
    colors = [
        "#00d4ff","#a855f7","#10b981","#ef4444",
        "#f59e0b","#3b82f6","#f97316",
    ]
    for i, cls in enumerate(CLASS_NAMES):
        icon = CLASS_ICONS.get(cls, "🔍")
        color = colors[i]
        desc = CLASS_DESCRIPTIONS.get(cls, "")
        classes_html += f"""
<div class="class-badge" style="border-color:{color}30;background:{color}10;color:{color}" title="{desc}">
  {icon} {cls}
</div>"""

    model_status = (
        '<span style="color:#10b981"> Loaded</span>'
        if _MODEL else
        '<span style="color:#ef4444"> Not Found</span>'
    )

    return f"""
<div class="glass-card">
  <div class="section-title">Model Architecture</div>
  <div class="model-info-grid">
    <div class="mi-item"><span class="mi-key">Architecture</span><span class="mi-val">YOLOv8m</span></div>
    <div class="mi-item"><span class="mi-key">Status</span><span class="mi-val">{model_status}</span></div>
    <div class="mi-item"><span class="mi-key">Parameters</span><span class="mi-val">25.8M</span></div>
    <div class="mi-item"><span class="mi-key">Input Size</span><span class="mi-val">640×640</span></div>
    <div class="mi-item"><span class="mi-key">Training Epochs</span><span class="mi-val">80</span></div>
    <div class="mi-item"><span class="mi-key">Optimizer</span><span class="mi-val">AdamW</span></div>
    <div class="mi-item"><span class="mi-key">Train Images</span><span class="mi-val">1,769</span></div>
    <div class="mi-item"><span class="mi-key">Val Images</span><span class="mi-val">338</span></div>
  </div>
  <div class="section-title" style="margin-top:20px">Performance Metrics</div>
  <div class="metrics-grid">
    <div class="metric-card"><span class="metric-val">82.1%</span><span class="metric-lbl">mAP@0.5</span></div>
    <div class="metric-card"><span class="metric-val">63.7%</span><span class="metric-lbl">mAP@0.5:95</span></div>
    <div class="metric-card"><span class="metric-val">91.4%</span><span class="metric-lbl">Precision</span></div>
    <div class="metric-card"><span class="metric-val">75.9%</span><span class="metric-lbl">Recall</span></div>
  </div>
  <div class="section-title" style="margin-top:20px">Detectable Classes</div>
  <div class="class-grid">{classes_html}</div>
</div>
"""


def make_how_it_works_html() -> str:
    return """
<div class="glass-card">
  <div class="section-title">How It Works</div>
  <div class="how-steps">
    <div class="how-step">
      <span class="step-num">1</span>
      <span class="step-icon"></span>
      <span class="step-title">Upload / Webcam</span>
      <span class="step-desc">Upload an image or use webcam for live capture</span>
    </div>
    <div class="how-step">
      <span class="step-num">2</span>
      <span class="step-icon"></span>
      <span class="step-title">Preprocessing</span>
      <span class="step-desc">Image resized to 640×640 and normalized for inference</span>
    </div>
    <div class="how-step">
      <span class="step-num">3</span>
      <span class="step-icon"></span>
      <span class="step-title">YOLOv8 Inference</span>
      <span class="step-desc">Model runs forward pass detecting all 7 safety objects</span>
    </div>
    <div class="how-step">
      <span class="step-num">4</span>
      <span class="step-icon"></span>
      <span class="step-title">NMS Filtering</span>
      <span class="step-desc">Non-max suppression removes duplicate detections</span>
    </div>
    <div class="how-step">
      <span class="step-num">5</span>
      <span class="step-icon"></span>
      <span class="step-title">Save Results</span>
      <span class="step-desc">Annotated image + JSON metadata saved to disk</span>
    </div>
  </div>
</div>
"""


def make_footer_html() -> str:
    return """
<div id="footer">
  <p><strong>Space Station Safety Object Detection</strong> · Duality AI Hackathon 2026</p>
  <p>Trained on synthetic data from Falcon Digital Twin Simulator · YOLOv8m · Ultralytics</p>
  <div class="footer-badges">
    <span class="fbadge fbadge-cyan">YOLOv8m</span>
    <span class="fbadge fbadge-purple">Gradio</span>
    <span class="fbadge fbadge-green">Hugging Face Spaces</span>
    <span class="fbadge fbadge-cyan">Duality AI Falcon</span>
  </div>
</div>
"""


# ── Shared detection logic ────────────────────────────────────────────────────
def _run_detection(image: Image.Image, conf_threshold: float, iou_threshold: float):
    """Core detection — returns (annotated_image, summary_html, detections, inf_ms)."""
    if image is None:
        return None, '<div class="no-detection-card"><div class="no-det-icon">📷</div><div class="no-det-text">Please provide an image first.</div></div>', [], 0

    if _MODEL is None:
        return None, f'<div class="model-error">{_MODEL_ERROR}</div>', [], 0

    try:
        image = image.resize((640, 640))
        annotated, detections, inf_ms = run_inference(_MODEL, image, conf_threshold, iou_threshold)
        summary_html = build_detection_summary(detections, inf_ms)
        return annotated, summary_html, detections, inf_ms
    except Exception:
        tb = traceback.format_exc()
        error_html = f'<div class="model-error"> Inference error:<br><pre style="font-size:0.7rem;overflow:auto">{tb}</pre></div>'
        return image, error_html, [], 0


# ── Inference Functions ───────────────────────────────────────────────────────
def detect_objects(image: Image.Image, conf_threshold: float, iou_threshold: float):
    """Upload tab handler."""
    annotated, summary_html, _, _ = _run_detection(image, conf_threshold, iou_threshold)
    return annotated, summary_html


def detect_from_webcam(webcam_image: Image.Image, conf_threshold: float, iou_threshold: float):
    """Webcam tab handler — runs on each captured frame."""
    if webcam_image is None:
        return (
            None,
            '<div class="no-detection-card"><div class="no-det-icon">📸</div>'
            '<div class="no-det-text">Point your webcam and click Capture.</div></div>',
        )
    annotated, summary_html, _, _ = _run_detection(webcam_image, conf_threshold, iou_threshold)
    return annotated, summary_html


def live_predict(webcam_stream_image: Image.Image, conf_threshold: float, iou_threshold: float):
    """Live streaming handler — called on every frame from gr.Image(streaming=True)."""
    if webcam_stream_image is None:
        return None
    try:
        img = webcam_stream_image.resize((640, 640))
        annotated, _, _ = run_inference(_MODEL, img, conf_threshold, iou_threshold)
        return annotated
    except Exception:
        return webcam_stream_image


def save_result(annotated_image: Image.Image, summary_html: str):
    """Save the annotated image and a companion metadata text file."""
    if annotated_image is None:
        return '<div class="save-error"> No result to save. Run detection first.</div>'

    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(SAVE_DIR, f"detection_{ts}.jpg")
        meta_path = os.path.join(SAVE_DIR, f"detection_{ts}.txt")

        annotated_image.save(img_path, "JPEG", quality=95)

        with open(meta_path, "w") as f:
            f.write(f"Timestamp : {ts}\n")
            f.write(f"Image path: {img_path}\n\n")
            # Strip HTML tags for plain-text metadata
            import re
            plain = re.sub(r"<[^>]+>", " ", summary_html)
            plain = re.sub(r"\s+", " ", plain).strip()
            f.write("Detection Summary:\n" + plain + "\n")

        return (
            f'<div class="save-success"> Saved! &nbsp;'
            f'<span style="opacity:0.7">{img_path}</span></div>'
        )
    except Exception as e:
        return f'<div class="save-error"> Save failed: {e}</div>'


# ── Example images ────────────────────────────────────────────────────────────
EXAMPLE_IMAGES = []
for p in ["examples/sample1.jpg", "examples/sample2.jpg", "examples/sample3.jpg"]:
    from pathlib import Path
    if Path(p).exists():
        EXAMPLE_IMAGES.append([p, 0.25, 0.45])


# ── Build Gradio UI ───────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Exo 2"),
)

_EMPTY_SUMMARY = """
<div class="no-detection-card">
  <div class="no-det-icon">🛸</div>
  <div class="no-det-text">Awaiting image scan...</div>
  <div class="no-det-sub">Upload an image and press SCAN to begin detection.</div>
</div>"""

_WEBCAM_EMPTY = """
<div class="no-detection-card">
  <div class="no-det-icon">📸</div>
  <div class="no-det-text">Webcam ready.</div>
  <div class="no-det-sub">Capture a frame or enable Live Mode to begin detection.</div>
</div>"""

with gr.Blocks(title="Space Station Safety Detection", css=CUSTOM_CSS, theme=theme) as demo:

    # ── Header ─────────────────────────────────────────────────────────────
    gr.HTML(make_header_html())

    # Shared state for save functionality
    last_annotated = gr.State(None)
    last_summary   = gr.State(_EMPTY_SUMMARY)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — Upload & Detection
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("  Detection Console"):
            with gr.Row(equal_height=False):

                # Left — inputs
                with gr.Column(scale=4):
                    gr.HTML('<div class="glass-card" style="padding-bottom:8px">'
                            '<div class="section-title">Input Image</div>')
                    input_image = gr.Image(
                        type="pil",
                        label="",
                        elem_classes=["upload-zone"],
                        height=420,
                        sources=["upload", "clipboard"],
                    )
                    gr.HTML('</div>')

                    with gr.Row():
                        conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence Threshold",
                            info="Minimum confidence score to accept a detection",
                        )
                        iou_slider = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                            label="IoU Threshold (NMS)",
                            info="Controls overlap suppression between boxes",
                        )

                    with gr.Row():
                        detect_btn = gr.Button("⚡  SCAN FOR SAFETY OBJECTS", elem_id="detect-btn", variant="primary")
                        clear_btn  = gr.Button("✕  Clear", elem_id="clear-btn")

                    with gr.Row():
                        save_btn_upload = gr.Button("  SAVE RESULT", elem_id="save-btn")

                    save_status_upload = gr.HTML(value="")

                    if EXAMPLE_IMAGES:
                        gr.Examples(
                            examples=EXAMPLE_IMAGES,
                            inputs=[input_image, conf_slider, iou_slider],
                            outputs=[],
                            label="Sample Images",
                        )

                # Right — outputs
                with gr.Column(scale=5):
                    gr.HTML('<div class="glass-card" style="padding-bottom:8px">'
                            '<div class="section-title">Detection Output</div>')
                    output_image_upload = gr.Image(type="pil", label="", height=420, interactive=False)
                    gr.HTML('</div>')

                    gr.HTML('<div class="glass-card"><div class="section-title">Detection Summary</div>')
                    summary_upload = gr.HTML(value=_EMPTY_SUMMARY)
                    gr.HTML('</div>')

            # Wire upload tab
            detect_btn.click(
                fn=detect_objects,
                inputs=[input_image, conf_slider, iou_slider],
                outputs=[output_image_upload, summary_upload],
            ).then(
                fn=lambda img, html: (img, html),
                inputs=[output_image_upload, summary_upload],
                outputs=[last_annotated, last_summary],
            )

            clear_btn.click(
                fn=lambda: (None, None, _EMPTY_SUMMARY, ""),
                inputs=[],
                outputs=[input_image, output_image_upload, summary_upload, save_status_upload],
            )

            save_btn_upload.click(
                fn=save_result,
                inputs=[last_annotated, last_summary],
                outputs=[save_status_upload],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — Webcam Capture
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("  Webcam Capture"):
            gr.HTML("""
<div class="glass-card" style="padding:14px 20px;margin-bottom:16px">
  <span style="font-size:0.8rem;color:var(--text-muted)">
  📷 &nbsp;Use the webcam below to capture a single frame, then click
  <strong style="color:var(--cyan)">SCAN FRAME</strong> to run detection.
  </span>
</div>""")
            with gr.Row(equal_height=False):

                with gr.Column(scale=4):
                    webcam_input = gr.Image(
                        type="pil",
                        label="Webcam Feed",
                        sources=["webcam"],
                        height=400,
                        elem_classes=["upload-zone"],
                    )

                    with gr.Row():
                        wc_conf = gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                                            label="Confidence Threshold")
                        wc_iou  = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                                            label="IoU Threshold (NMS)")

                    with gr.Row():
                        wc_scan_btn = gr.Button("  SCAN FRAME", elem_id="detect-btn", variant="primary")
                        wc_clear_btn = gr.Button("  Clear", elem_id="clear-btn")

                    with gr.Row():
                        wc_save_btn = gr.Button("  SAVE RESULT", elem_id="save-btn")

                    wc_save_status = gr.HTML(value="")

                with gr.Column(scale=5):
                    gr.HTML('<div class="glass-card" style="padding-bottom:8px">'
                            '<div class="section-title">Webcam Detection Output</div>')
                    wc_output = gr.Image(type="pil", label="", height=400, interactive=False)
                    gr.HTML('</div>')

                    gr.HTML('<div class="glass-card"><div class="section-title">Detection Summary</div>')
                    wc_summary = gr.HTML(value=_WEBCAM_EMPTY)
                    gr.HTML('</div>')

            wc_scan_btn.click(
                fn=detect_from_webcam,
                inputs=[webcam_input, wc_conf, wc_iou],
                outputs=[wc_output, wc_summary],
            ).then(
                fn=lambda img, html: (img, html),
                inputs=[wc_output, wc_summary],
                outputs=[last_annotated, last_summary],
            )

            wc_clear_btn.click(
                fn=lambda: (None, None, _WEBCAM_EMPTY, ""),
                inputs=[],
                outputs=[webcam_input, wc_output, wc_summary, wc_save_status],
            )

            wc_save_btn.click(
                fn=save_result,
                inputs=[last_annotated, last_summary],
                outputs=[wc_save_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — Live Webcam Stream
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("🔴  Live Detection"):
            gr.HTML("""
<div class="glass-card" style="padding:14px 20px;margin-bottom:16px">
  <span style="font-size:0.8rem;color:var(--text-muted)">
  <span class="live-dot"></span>
  <strong style="color:var(--red)">LIVE MODE</strong> — Detection runs on every frame from your webcam in real time.
  Click <strong style="color:var(--cyan)">Start</strong> in the video feed to begin streaming.
  Use the sliders to tune sensitivity.
  </span>
</div>""")

            with gr.Row():
                with gr.Column(scale=1):
                    live_conf = gr.Slider(minimum=0.05, maximum=0.95, value=0.30, step=0.05,
                                          label="Confidence Threshold")
                    live_iou  = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step=0.05,
                                          label="IoU Threshold (NMS)")
                    gr.HTML("""
<div style="margin-top:12px;padding:14px;background:rgba(239,68,68,0.08);
     border:1px solid rgba(239,68,68,0.25);border-radius:8px;
     font-size:0.75rem;color:#fca5a5;line-height:1.6">
  <strong>Tips for Live Mode:</strong><br>
  • Higher confidence (0.4+) reduces false positives<br>
  • Good lighting improves accuracy<br>
  • Hold objects steady for best results<br>
  • Press <em>Stop</em> to pause the stream
</div>""")

                    live_save_btn    = gr.Button("  SAVE CURRENT FRAME", elem_id="save-btn")
                    live_save_status = gr.HTML(value="")

                with gr.Column(scale=2):
                    live_input = gr.Image(
                        type="pil",
                        label="Live Webcam",
                        sources=["webcam"],
                        streaming=True,
                        height=480,
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="glass-card" style="padding-bottom:8px">'
                            '<div class="section-title"><span class="live-dot"></span>Live Output</div>')
                    live_output = gr.Image(type="pil", label="", height=480, interactive=False)
                    gr.HTML('</div>')

            # Streaming inference — fires on each frame
            live_input.stream(
                fn=live_predict,
                inputs=[live_input, live_conf, live_iou],
                outputs=[live_output],
                time_limit=120,
                stream_every=0.1,   # ~10 FPS
            )

            # Save current live frame
            live_save_btn.click(
                fn=save_result,
                inputs=[live_output, gr.State("Live detection frame")],
                outputs=[live_save_status],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — Model Info
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("  Model Info"):
            gr.HTML(make_model_info_html())

        
        with gr.TabItem("  How It Works"):
            gr.HTML(make_how_it_works_html())
            gr.HTML("""
<div class="glass-card">
  <div class="section-title">Dataset Overview</div>
  <div class="model-info-grid">
    <div class="mi-item"><span class="mi-key">Data Source</span><span class="mi-val">Duality AI Falcon</span></div>
    <div class="mi-item"><span class="mi-key">Type</span><span class="mi-val">Synthetic Digital Twin</span></div>
    <div class="mi-item"><span class="mi-key">Train Split</span><span class="mi-val">1,769 images</span></div>
    <div class="mi-item"><span class="mi-key">Val Split</span><span class="mi-val">338 images</span></div>
    <div class="mi-item"><span class="mi-key">Test Split</span><span class="mi-val">1,408 images</span></div>
    <div class="mi-item"><span class="mi-key">Label Format</span><span class="mi-val">YOLO TXT</span></div>
    <div class="mi-item"><span class="mi-key">Lighting Variants</span><span class="mi-val">Dark / Light / Mixed</span></div>
    <div class="mi-item"><span class="mi-key">Scenarios</span><span class="mi-val">Cluttered / Hallway / Uncluttered</span></div>
  </div>
  <div class="section-title" style="margin-top:20px">Training Highlights</div>
  <div style="color:var(--text-muted);font-size:0.82rem;line-height:1.8">
    <p>• <strong style="color:var(--cyan)">Auto-batch</strong> selected batch size 7 for optimal VRAM usage (~55%) on Tesla T4.</p>
    <p>• <strong style="color:var(--cyan)">Cosine annealing</strong> LR schedule with 3 warm-up epochs.</p>
    <p>• <strong style="color:var(--cyan)">Heavy augmentation</strong>: mosaic (1.0), mixup (0.15), copy-paste (0.1), HSV jitter, rotation, shear, perspective.</p>
    <p>• <strong style="color:var(--cyan)">Mixed precision (AMP)</strong> training for 2× speed on GPU.</p>
    <p>• <strong style="color:var(--cyan)">80 epochs</strong> completed in ~93 minutes. Best checkpoint at epoch 78.</p>
  </div>
</div>
""")

    gr.HTML(make_footer_html())


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        ssr_mode=False,
    )
