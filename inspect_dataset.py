"""
inspect_dataset.py - look inside the dataset folder before training.

Usage (from the project folder):
    python inspect_dataset.py data
where "data" is the folder that contains the unzipped dataset.
Copy the whole output and send it if something does not work.
"""
import sys
from pathlib import Path
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

if len(sys.argv) < 2:
    print("Usage: python inspect_dataset.py <dataset_folder>")
    sys.exit(1)

root = Path(sys.argv[1])
if not root.exists():
    print(f"Folder not found: {root.resolve()}")
    sys.exit(1)

# ---- 1. folder tree (3 levels deep, max 80 lines) ----
print(f"Dataset folder: {root.resolve()}\n")
print("FOLDER TREE (images directly inside each folder):")
lines = 0
for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
    depth = len(d.relative_to(root).parts)
    if depth > 3:
        continue
    n_img = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    n_sub = sum(1 for p in d.iterdir() if p.is_dir())
    print("  " * depth + f"{d.name}/   [{n_img} images, {n_sub} subfolders]")
    lines += 1
    if lines >= 80:
        print("  ... (truncated)")
        break

# ---- 2. totals ----
imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
total_mb = sum(p.stat().st_size for p in imgs) / 1e6
print(f"\nTOTAL IMAGES: {len(imgs)}   ({total_mb:.0f} MB of images)")
print("File extensions:", dict(Counter(p.suffix.lower() for p in imgs)))
print("Course minimum of 2,000 images:", "OK" if len(imgs) >= 2000 else "NOT MET")

# ---- 3. image sizes (sample) ----
try:
    from PIL import Image
    sizes, modes = Counter(), Counter()
    step = max(1, len(imgs) // 300)
    for p in imgs[::step]:
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    print("Most common image sizes (sample):", sizes.most_common(5))
    print("Color modes (sample):", dict(modes))
except ImportError:
    print("Pillow not installed yet (pip install pillow) - skipping image size check.")

# ---- 4. other files (metadata, notebooks, ...) ----
others = [p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.suffix.lower() not in IMG_EXTS]
print("\nNon-image files:", [str(p) for p in others[:20]] if others else "none")
print("\nExample image paths:")
for p in imgs[:: max(1, len(imgs) // 6)][:6]:
    print("  ", p.relative_to(root))
