import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent

REQUIRED_FILES = ["app.py", "utils.py", "requirements.txt", "README.md"]
OPTIONAL_FILES = ["best.pt", ".gitattributes"]

print("=" * 60)
print("  Space Station Safety Detection — Deploy Checker")
print("=" * 60)

all_ok = True

print("\n📁 Required files:")
for f in REQUIRED_FILES:
    exists = (ROOT / f).exists()
    status = "✅" if exists else "❌"
    size = f"  ({(ROOT/f).stat().st_size/1024:.1f} KB)" if exists else ""
    print(f"  {status}  {f}{size}")
    if not exists:
        all_ok = False

print("\n📦 Optional files:")
for f in OPTIONAL_FILES:
    exists = (ROOT / f).exists()
    status = "✅" if exists else "⚠️ "
    size = f"  ({(ROOT/f).stat().st_size/1e6:.1f} MB)" if exists else ""
    print(f"  {status}  {f}{size}")

# Check best.pt specifically
model_path = ROOT / "best.pt"
if not model_path.exists():
    print("\n⚠️  WARNING: best.pt not found!")
    print("   The app will show a model error until you add best.pt.")
    print("   Options:")
    print("   1. Copy best.pt to the project root.")
    print("   2. Use Git LFS: git lfs track '*.pt' && git add best.pt")
    print("   3. Load from Hugging Face Hub (see README).")

# Check imports
print("\n🔍 Checking Python imports:")
try:
    import gradio
    print(f"  ✅  gradio {gradio.__version__}")
except ImportError:
    print("  ❌  gradio not installed")
    all_ok = False

try:
    import ultralytics
    print(f"  ✅  ultralytics {ultralytics.__version__}")
except ImportError:
    print("  ❌  ultralytics not installed")
    all_ok = False

try:
    import cv2
    print(f"  ✅  opencv {cv2.__version__}")
except ImportError:
    print("  ❌  opencv-python not installed")
    all_ok = False

try:
    import PIL
    print(f"  ✅  Pillow {PIL.__version__}")
except ImportError:
    print("  ❌  Pillow not installed")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("  ✅  ALL CHECKS PASSED — Ready to deploy!")
else:
    print("  ⚠️   Some checks failed. See above for details.")
print("=" * 60)
